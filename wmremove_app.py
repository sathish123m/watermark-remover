#!/usr/bin/env python3
"""
wmremove_app.py - Watermark Remover as a touch-friendly app for Android.

Run inside Termux:      python wmremove_app.py
Then open in Chrome:    http://localhost:8080

Self-contained: stdlib + numpy + ffmpeg only (no OpenCV). The page lets
you pick a video, draw boxes over watermarks with your finger, or
auto-detect static logos, then removes them and saves the cleaned video.

Engine is identical to wmremove-lite.py (verified) with two changes:
frame dimensions come from the decoded first frame (so rotated phone
videos map correctly), and processing reports progress.
"""

import base64
import datetime
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

# everything the app keeps lives here (uploads, cleaned videos, database);
# override with the DATA_DIR environment variable (e.g. in Docker)
DATA_DIR = os.environ.get("DATA_DIR") or os.path.expanduser(
    "~/watermark-remover-data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
THUMB_DIR = os.path.join(DATA_DIR, "thumbs")
DB_PATH = os.path.join(DATA_DIR, "app.db")
for d in (DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, THUMB_DIR):
    os.makedirs(d, exist_ok=True)

# optional extras: OCR supercharges auto-detection when available
try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None
try:
    from rapidocr_onnxruntime import RapidOCR as _RapidOCR
    _OCR = _RapidOCR()
except Exception:
    _OCR = None

# OpenCV gives noticeably better inpainting when available (e.g. on a
# computer); on Termux the numpy diffusion fallback is used instead.
if _cv2 is not None:
    def diffuse_fill(img, mask, iters=120):
        if not mask.any():
            return img
        img8 = np.clip(img, 0, 255).astype(np.uint8)
        out = _cv2.inpaint(img8, (mask * 255).astype(np.uint8), 6,
                           _cv2.INPAINT_TELEA)
        return out.astype(np.float32)
else:
    def diffuse_fill(img, mask, iters=120):
        if not mask.any():
            return img
        img = coons_init(img.astype(np.float32).copy(), mask)
        m = mask
        for _ in range(iters):
            avg = (np.roll(img, 1, 0) + np.roll(img, -1, 0) +
                   np.roll(img, 1, 1) + np.roll(img, -1, 1)) * 0.25
            img[m] = avg[m]
        return img

PORT = 8080
TMP = tempfile.mkdtemp(prefix="wmapp_")
JOBS = {}          # job id -> dict(status, progress, message, out, error)
UPLOADS = {}       # upload id -> {path, name, w, h, fps}


# ==========================================================================
# engine (copied from wmremove-lite.py)
# ==========================================================================

def probe_fps(path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate,duration",
           "-of", "json", path]
    out = subprocess.run(cmd, capture_output=True, text=True,
                         check=True).stdout
    info = json.loads(out)["streams"][0]
    num, den = info["r_frame_rate"].split("/")
    return float(num) / float(den), float(info.get("duration") or 0)


class Reader:
    def __init__(self, path, w, h):
        self.w, self.h = w, h
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", path,
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE)

    def __iter__(self):
        n = self.w * self.h * 3
        while True:
            buf = self.proc.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(
                self.h, self.w, 3).copy()
        self.proc.stdout.close()
        self.proc.wait()


def median_blur(img, k):
    out = img.astype(np.float32)
    half = k // 2
    for axis in (0, 1):
        stack = [np.roll(out, s, axis=axis)
                 for s in range(-half, half + 1)]
        out = np.median(np.stack(stack), axis=0)
    return out


def dilate_box(mask, r):
    out = mask
    for axis in (0, 1):
        out = np.maximum.reduce(
            [np.roll(out, s, axis=axis) for s in range(-r, r + 1)])
    return out


def blur3(img):
    out = img.astype(np.float32)
    for axis in (0, 1):
        out = (np.roll(out, 1, axis=axis) + 2 * out +
               np.roll(out, -1, axis=axis)) * 0.25
    return out


def text_mask(gray, thresh=16):
    diff = np.clip(gray.astype(np.float32) - median_blur(gray, 15), 0, None)
    return dilate_box(diff > thresh, 3)


def coons_init(img, mask):
    h, w = mask.shape
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    top = img[max(0, y0 - 1), x0:x1 + 1]
    bot = img[min(h - 1, y1 + 1), x0:x1 + 1]
    left = img[y0:y1 + 1, max(0, x0 - 1)]
    right = img[y0:y1 + 1, min(w - 1, x1 + 1)]
    if y0 == 0:
        top = bot
    if y1 == h - 1:
        bot = top
    if x0 == 0:
        left = right
    if x1 == w - 1:
        right = left
    ty = (ys - y0 + 0.5) / (y1 - y0 + 1.0)
    tx = (xs - x0 + 0.5) / (x1 - x0 + 1.0)
    hb = top[xs - x0] * (1 - tx)[:, None] + bot[xs - x0] * tx[:, None]
    vb = left[ys - y0] * (1 - ty)[:, None] + right[ys - y0] * ty[:, None]
    wh, wv = (1 - ty), (1 - tx)
    blend = (hb * wh[:, None] + vb * wv[:, None]) / (wh + wv)[:, None]
    img[ys, xs] = blend
    return img


def track(gray, tpl_hp, prev, move, min_conf=0.25):
    th, tw = tpl_hp.shape
    H, W = gray.shape
    m = move + 10
    cx0, cx1 = max(0, prev[0] - m), min(W - tw, prev[0] + m)
    cy0, cy1 = max(0, prev[1] - m), min(H - th, prev[1] + m)
    if cx1 < cx0 or cy1 < cy0:
        return prev[0], prev[1], 0.0
    wx0, wy0 = max(0, cx0 - 16), max(0, cy0 - 16)
    wx1, wy1 = min(W, cx1 + tw + 16), min(H, cy1 + th + 16)
    win_raw = gray[wy0:wy1, wx0:wx1]
    win = (win_raw - median_blur(win_raw, 15)).astype(np.float32)
    t = tpl_hp - tpl_hp.mean()
    sw = np.lib.stride_tricks.sliding_window_view(win, (th, tw))
    num = np.einsum("ijkl,kl->ij", sw, t, optimize=True)
    c = np.cumsum(np.cumsum(win * win, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    gh, gw = sw.shape[:2]
    S = (c[th:th + gh, tw:tw + gw] - c[0:gh, tw:tw + gw] -
         c[th:th + gh, 0:gw] + c[0:gh, 0:gw])
    denom = np.sqrt(np.maximum(S * (t * t).sum(), 1e-6))
    ncc = num / denom
    i, j = np.unravel_index(np.argmax(ncc), ncc.shape)
    conf = float(ncc[i, j])
    if conf < min_conf:
        return prev[0], prev[1], conf
    return wx0 + int(j), wy0 + int(i), conf


def soft_noise(rng, shape, sigma):
    """Spatially smooth grain, capped so it can never read as speckle.

    Per-pixel white noise looks like flickering particles; real grain is
    softer, so the noise is blurred and rescaled to the target level.
    """
    n = blur3(rng.normal(0.0, 1.0, shape).astype(np.float32))
    s = float(n.std())
    if s < 1e-6:
        return np.zeros(shape, np.float32)
    return n * (sigma / s)


def grain_sigmas(bgr, ring):
    out = []
    for ch in range(3):
        g = bgr[:, :, ch].astype(np.float32)
        res = (g - blur3(g))[ring]
        if res.size < 200:
            out.append(0.0)
            continue
        mad = np.median(np.abs(res - np.median(res)))
        out.append(float(1.4826 * mad / 0.80))
    return out


# ==========================================================================
# database
# ==========================================================================

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def db_init():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS videos (
            id TEXT PRIMARY KEY, name TEXT, uploaded_at TEXT,
            width INT, height INT, duration REAL, size_bytes INT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, video_id TEXT, started_at TEXT,
            status TEXT, boxes TEXT, moving TEXT, out_name TEXT,
            frames INT)""")


db_init()


# ==========================================================================
# app-specific: processing with progress, auto-detect
# ==========================================================================

def grab_frame(path, t):
    """Decode one frame at time t as a JPEG image (bytes)."""
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(t), "-i", path,
         "-frames:v", "1", "-f", "mjpeg", "-q:v", "3", "-"],
        capture_output=True, check=True).stdout


def decode_jpeg(data):
    if _cv2 is None or not data:
        return None
    arr = np.frombuffer(data, np.uint8)
    return _cv2.imdecode(arr, _cv2.IMREAD_COLOR)


def ocr_texts(img):
    """[{text, conf, box:[x,y,w,h]}] for readable text in a frame."""
    if _OCR is None or img is None:
        return []
    result, _ = _OCR(img)
    out = []
    for box, text, conf in (result or []):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, y0 = max(0, int(min(xs)) - 4), max(0, int(min(ys)) - 4)
        x1 = int(max(xs)) + 4
        y1 = int(max(ys)) + 4
        out.append({"text": str(text), "conf": float(conf),
                    "box": [x0, y0, x1 - x0, y1 - y0]})
    return out


def _norm_text(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _plausible(box, w, h):
    """Watermark-plausible text box (not tiny, not huge, not center-only)."""
    x, y, bw, bh = box
    if bw < 0.02 * w or bw > 0.7 * w:
        return False
    if bh < 0.015 * h or bh > 0.4 * h:
        return False
    return True


def _text_like(a, b):
    """Fuzzy text match (OCR often misreads a letter or two)."""
    na, nb = _norm_text(a), _norm_text(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    import difflib
    return difflib.SequenceMatcher(None, na, nb).ratio() >= 0.7


def autodetect(inp):
    """Find watermarks: OCR text at the START and END of the video (the same
    text at a shifted position = a drifting watermark), plus the
    temporal-static pass for borders and solid logos.
    Returns {"boxes": [[x,y,w,h]...], "moving": [x,y,w,h] or None}."""
    fps, duration = probe_fps(inp)
    w, h = display_dims(inp)
    boxes = []
    moving = None

    texts_a = [t for t in ocr_texts(decode_jpeg(grab_frame(inp, 0)))
               if t["conf"] >= 0.55 and _plausible(t["box"], w, h)]
    texts_b = [t for t in ocr_texts(
                   decode_jpeg(grab_frame(inp, max(0, duration - 0.3))))
               if t["conf"] >= 0.55 and _plausible(t["box"], w, h)]

    def center(b):
        return (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)

    for ta in texts_a:
        match = None
        for tb in texts_b:
            if _text_like(tb["text"], ta["text"]):
                match = tb
                break
        if match is None:
            boxes.append(ta["box"])       # unreadable at the end; suggest it
            continue
        (ax, ay), (bx, by) = center(ta["box"]), center(match["box"])
        if abs(ax - bx) + abs(ay - by) >= 8:
            if moving is None:
                moving = ta["box"]        # same text, drifted -> moving
        else:
            boxes.append(ta["box"])       # same text, same place -> static

    # static pass (borders, solid logos, burned-in overlays)
    for b in auto_detect_boxes(inp, max_boxes=4):
        if moving is None or _iou(b, moving) < 0.2:
            boxes.append(b)

    # dedupe overlapping suggestions
    dedup = []
    for b in sorted(boxes, key=lambda b: -(b[2] * b[3])):
        if all(_iou(b, k) < 0.3 for k in dedup):
            dedup.append(b)
    return {"boxes": dedup[:6], "moving": moving}

def display_dims(inp):
    """Dimensions of decoded frames (ffmpeg bakes rotation, so a 90°
    rotated phone video swaps width/height compared to codec dims)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries",
         "stream=width,height,side_data_list,tags",
         "-of", "json", inp],
        capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    rot = 0
    for sd in st.get("side_data_list") or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    tr = (st.get("tags") or {}).get("rotate")
    if tr:
        try:
            rot = int(float(tr))
        except ValueError:
            pass
    if rot % 180 != 0:
        w, h = h, w
    return w, h


def process_video(inp, outp, regions, track_box, move, progress):
    """regions: list of (x,y,w,h); track_box: (x,y,w,h) or None."""
    fps, duration = probe_fps(inp)
    w, h = display_dims(inp)
    reader = Reader(inp, w, h)
    it = iter(reader)
    try:
        first = next(it)
    except StopIteration:
        raise RuntimeError("could not decode any frames")
    h, w = first.shape[:2]        # actual frame dims are the truth

    static_rois = []
    for (x, y, rw, rh) in regions:
        pad = 6
        x0r, y0r = max(0, x - pad), max(0, y - pad)
        x1r, y1r = min(w, x + rw + pad), min(h, y + rh + pad)
        ox, oy = x - x0r, y - y0r
        mask = np.zeros((y1r - y0r, x1r - x0r), bool)
        mask[oy:oy + rh, ox:ox + rw] = True
        static_rois.append((x0r, y0r, mask))

    tpl_hp = None
    if track_box:
        tx, ty, tw_, th_ = track_box
        pad = 5
        tx0, ty0 = max(0, tx - pad), max(0, ty - pad)
        tw_ = min(w - tx0, tw_ + 2 * pad)
        th_ = min(h - ty0, th_ + 2 * pad)
        tpl_crop = first[ty0:ty0 + th_, tx0:tx0 + tw_].astype(
            np.float32).mean(axis=2)
        tpl_hp = tpl_crop - median_blur(tpl_crop, 15)
        if text_mask(tpl_crop).mean() < 0.03:
            raise RuntimeError(
                "no text-like watermark found in the moving-watermark box")
        pos, bw, bh = (tx0, ty0), tw_, th_

    writer = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", f"{fps:.6f}", "-i", "-",
         "-i", inp,
         "-map", "0:v:0", "-map", "1:a?",
         "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-c:a", "copy",
         "-movflags", "+faststart", outp],
        stdin=subprocess.PIPE)

    rng = np.random.default_rng(0)
    grain = None
    s_grain = [grain_sigmas(first[y:y + m.shape[0], x:x + m.shape[1]], ~m)
               for (x, y, m) in static_rois]
    est_total = max(1, int(round(duration * fps)))
    n = 0
    misses = 0
    frame = first
    while frame is not None:
        for (x, y, mask), sig in zip(static_rois, s_grain):
            rh, rw = mask.shape
            crop = frame[y:y + rh, x:x + rw]
            fixed = diffuse_fill(crop, mask, iters=150)
            noise = np.zeros((rh, rw, 3), np.float32)
            for ch, s in enumerate(sig):
                if s > 0.5:
                    noise[:, :, ch] = soft_noise(
                        rng, (rh, rw), min(s, 1.5) * 0.6) * mask
            frame[y:y + rh, x:x + rw] = np.clip(
                fixed + noise, 0, 255).astype(np.uint8)

        if track_box:
            gray = frame.astype(np.float32).mean(axis=2)
            nx_, ny_, conf = track(gray, tpl_hp, pos, move)
            if conf >= 0.25:
                pos = (nx_, ny_)
            else:
                misses += 1
            px, py = pos
            crop = frame[py:py + bh, px:px + bw]
            g = crop.astype(np.float32).mean(axis=2)
            m = text_mask(g)
            if grain is None:
                grain = grain_sigmas(crop, ~m)
            fixed = diffuse_fill(crop.astype(np.float32), m, iters=90)
            noise = np.zeros((bh, bw, 3), np.float32)
            for ch, s in enumerate(grain):
                if s > 0.5:
                    noise[:, :, ch] = soft_noise(
                        rng, (bh, bw), min(s, 1.5) * 0.6) * m
            frame[py:py + bh, px:px + bw] = np.clip(
                fixed + noise, 0, 255).astype(np.uint8)

        writer.stdin.write(frame.tobytes())
        n += 1
        progress(min(99, int(100 * n / est_total)))
        frame = next(it, None)

    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError("encoding failed")
    progress(100)
    return n, misses, w, h, fps


def auto_detect_boxes(inp, max_boxes=4):
    """Suggest static-watermark boxes: pixels that never change over time
    AND sit on strong edges. Pure numpy implementation."""
    fps, duration = probe_fps(inp)
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", inp],
        capture_output=True, text=True, check=True)
    st = json.loads(proc.stdout)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    total = max(1, int(duration * fps))
    step = max(1, total // 40)
    frames = []
    rd = Reader(inp, w, h)
    i = 0
    for f in rd:
        if i % step == 0:
            frames.append(f.astype(np.float32).mean(axis=2))
            if len(frames) >= 40:
                break
        i += 1
    if len(frames) < 4:
        return []
    stack = np.stack(frames)
    t_std = stack.std(axis=0)
    t_mean = stack.mean(axis=0)
    gx = np.abs(np.diff(t_mean, axis=1, prepend=t_mean[:, :1]))
    gy = np.abs(np.diff(t_mean, axis=0, prepend=t_mean[:1, :]))
    edge = gx + gy
    cand = ((t_std < 3.0) & (edge > 18)).astype(np.uint8)
    # also catch solid static areas (solid logo bodies)
    solid = (t_std < 1.5).astype(np.uint8)
    cand = np.maximum(cand, solid)
    # downsample 4x for cheap connected components
    ds = cand[::4, ::4]
    H, W = ds.shape
    seen = np.zeros_like(ds, bool)
    comps = []
    for yy in range(H):
        for xx in range(W):
            if ds[yy, xx] and not seen[yy, xx]:
                q = deque([(yy, xx)])
                seen[yy, xx] = True
                x0 = x1 = xx
                y0 = y1 = yy
                area = 0
                while q:
                    cy, cx = q.popleft()
                    area += 1
                    x0, x1 = min(x0, cx), max(x1, cx)
                    y0, y1 = min(y0, cy), max(y1, cy)
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < H and 0 <= nx < W and ds[ny, nx]
                                and not seen[ny, nx]):
                            seen[ny, nx] = True
                            q.append((ny, nx))
                if area >= 25:
                    comps.append((area, x0, y0, x1, y1))
    if not comps:
        return []
    DH, DW = ds.shape
    scored = []
    for area, x0, y0, x1, y1 in comps:
        bw = (x1 - x0 + 1) * 4
        bh = (y1 - y0 + 1) * 4
        if bw * bh > 0.15 * w * h:
            continue
        cx = (x0 + x1) / 2 / DW
        cy = (y0 + y1) / 2 / DH
        border = min(cx, 1 - cx) < 0.28 or min(cy, 1 - cy) < 0.28
        score = area * (2.5 if border else 1.0)
        scored.append((score, x0 * 4, y0 * 4, bw, bh))
    scored.sort(reverse=True)
    return [list(b) for _, *b in scored[:max_boxes]]


def run_job(job_id, inp, outp, regions, track_box, move, video_id=None):
    try:
        n, misses, w, h, fps = process_video(
            inp, outp, regions, track_box, move,
            lambda p: JOBS[job_id].update(progress=p))
        JOBS[job_id].update(status="done", progress=100, out=outp,
                            message=f"cleaned ({n} frames)")
        with db() as c:
            c.execute("UPDATE jobs SET status='done', frames=?, out_name=? "
                      "WHERE id=?", (n, os.path.basename(outp), job_id))
    except Exception as e:
        JOBS[job_id].update(status="error", error=str(e))
        with db() as c:
            c.execute("UPDATE jobs SET status='error' WHERE id=?", (job_id,))


# ==========================================================================
# http server
# ==========================================================================

PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>Watermark Remover</title>
<style>
:root{--bg:#111;--card:#1d222b;--ac:#4f8cff;--ok:#39c26d}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
body{margin:0;font-family:system-ui;background:var(--bg);color:#eee;
  padding-bottom:40px}
.wrap{max-width:640px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:8px 0 2px}
.sub{color:#9aa3b2;font-size:13px;margin-bottom:18px}
.card{background:var(--card);border-radius:14px;padding:16px;margin:12px 0}
button{border:0;border-radius:12px;padding:14px 18px;font-size:16px;
  background:var(--ac);color:#fff;width:100%;margin-top:10px;font-weight:600}
button.ghost{background:#2a3140;color:#dfe6f2}
button:disabled{opacity:.45}
#pick{display:none}
.picklabel{display:block;border:2px dashed #4a5468;border-radius:14px;
  padding:34px 10px;text-align:center;color:#aeb8c9;font-size:16px}
canvas{width:100%;touch-action:none;border-radius:10px;background:#000;
  display:block}
.hint{color:#8f99ab;font-size:13px;line-height:1.5}
.boxrow{display:flex;align-items:center;gap:8px;background:#2a3140;
  border-radius:10px;padding:8px 12px;margin:6px 0;font-size:14px}
.boxrow .del{margin-left:auto;background:#e5484d;width:34px;height:34px;
  padding:0;border-radius:10px;font-size:18px}
.boxrow .mv{background:#2f6fd0;width:auto;padding:8px 12px;font-size:13px}
.boxrow .mv.on{background:var(--ok)}
.bar{height:14px;background:#2a3140;border-radius:8px;overflow:hidden}
.bar>div{height:100%;width:0;background:var(--ok);transition:width .5s}
video{width:100%;border-radius:10px;margin-top:10px;background:#000}
a.btn{display:block;text-align:center;text-decoration:none;background:var(--ok);
  color:#fff;border-radius:12px;padding:15px;font-size:16px;font-weight:600}
.hidden{display:none}
</style></head><body><div class="wrap">
<h1>💧 Watermark Remover</h1>
<div class="sub">pick a video · mark the watermark · remove · save
&nbsp;·&nbsp;<a href="/library" style="color:#7fb0ff">📚 Library</a></div>

<div class="card" id="step1">
  <label class="picklabel" for="pick">▶️ &nbsp;Tap to choose a video</label>
  <input type="file" id="pick" accept="video/*">
</div>

<div class="card hidden" id="step2">
  <canvas id="cv"></canvas>
  <div class="hint" style="margin-top:8px">
    👆 Drag your finger over the watermark to draw a box (add as many as you
    need). Or tap <b>Auto-detect</b>. If a logo <b>moves</b>, draw one box on
    it and mark that box as “moves”.
  </div>
  <div id="boxes"></div>
  <button id="auto">✨ Auto: detect &amp; remove</button>
  <button class="ghost" id="detect">🔍 Suggest boxes (fixed logos)</button>
  <button class="ghost" id="go">🪄 Remove marked watermarks</button>
</div>

<div class="card hidden" id="step3">
  <div id="msg" class="hint">Working…</div>
  <div class="bar" style="margin-top:8px"><div id="fill"></div></div>
</div>

<div class="card hidden" id="step4">
  <div class="hint" id="done_msg"></div>
  <video id="prev" controls playsinline></video>
  <a class="btn" id="save" href="#">💾 Save cleaned video</a>
  <button class="ghost" id="again">↩️ Clean another video</button>
</div>
</div>

<script>
let state = {id:null, name:'', vw:0, vh:0, img:null,
             boxes:[], job:null};

const $ = s => document.querySelector(s);
const cv = $('#cv'), ctx = cv.getContext('2d');

$('#pick').addEventListener('change', async e => {
  const f = e.target.files[0];
  if (!f) return;
  $('#step1').classList.add('hidden');
  $('#step2').classList.remove('hidden');
  const r = await fetch('/api/upload', {method:'POST',
    headers:{'X-Filename': encodeURIComponent(f.name)}, body: f});
  const j = await r.json();
  state.id = j.id; state.name = j.name;
  const fr = await fetch('/api/frame?id=' + state.id);
  const blob = await fr.blob();
  state.img = await createImageBitmap(blob);
  state.vw = state.img.width; state.vh = state.img.height;
  cv.width = state.vw; cv.height = state.vh;
  state.boxes = [];
  render(); listBoxes();
});

function render() {
  if (!state.img) return;
  ctx.drawImage(state.img, 0, 0);
  ctx.lineWidth = Math.max(2, state.vw / 220);
  ctx.setLineDash([]);
  state.boxes.forEach((b, i) => {
    ctx.strokeStyle = b.moving ? '#39c26d' : '#4f8cff';
    ctx.strokeRect(b.x, b.y, b.w, b.h);
  });
  if (drag) {
    ctx.strokeStyle = '#ffd54a'; ctx.setLineDash([12, 8]);
    const r = norm(drag);
    ctx.strokeRect(r.x, r.y, r.w, r.h);
  }
}
function norm(d) {
  return {x: Math.min(d.x0, d.x1), y: Math.min(d.y0, d.y1),
          w: Math.abs(d.x1 - d.x0), h: Math.abs(d.y1 - d.y0)};
}
function pos(e) {
  const r = cv.getBoundingClientRect();
  return {x: (e.clientX - r.left) * state.vw / r.width,
          y: (e.clientY - r.top) * state.vh / r.height};
}
let drag = null;
cv.addEventListener('pointerdown', e => {
  if (!state.img) return;
  const p = pos(e); drag = {x0:p.x, y0:p.y, x1:p.x, y1:p.y};
  cv.setPointerCapture(e.pointerId); e.preventDefault();
});
cv.addEventListener('pointermove', e => {
  if (!drag) return;
  const p = pos(e); drag.x1 = p.x; drag.y1 = p.y; render();
});
cv.addEventListener('pointerup', () => {
  if (!drag) return;
  const r = norm(drag); drag = null;
  if (r.w > 12 && r.h > 12) {
    state.boxes.push({x:r.x, y:r.y, w:r.w, h:r.h, moving:false});
    listBoxes();
  }
  render();
});
function listBoxes() {
  const host = $('#boxes');
  host.innerHTML = '';
  state.boxes.forEach((b, i) => {
    const row = document.createElement('div');
    row.className = 'boxrow';
    const mv = document.createElement('button');
    mv.className = 'mv' + (b.moving ? ' on' : '');
    mv.textContent = b.moving ? '✓ moves' : 'moves?';
    mv.onclick = () => {
      state.boxes.forEach(x => x.moving = false);
      b.moving = !b.moving; listBoxes(); render();
    };
    const info = document.createElement('span');
    info.textContent = `box ${i+1}: ${Math.round(b.w)}×${Math.round(b.h)}`;
    const del = document.createElement('button');
    del.className = 'del'; del.textContent = '✕';
    del.onclick = () => {state.boxes.splice(i,1); listBoxes(); render();};
    row.append(mv, info, del);
    host.append(row);
  });
}
$('#detect').onclick = async () => {
  $('#detect').textContent = '🔍 detecting…';
  const r = await fetch('/api/autodetect', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({id: state.id})});
  const j = await r.json();
  $('#detect').textContent = '🔍 Suggest boxes (fixed logos)';
  (j.boxes || []).forEach(b =>
    state.boxes.push({x:b[0], y:b[1], w:b[2], h:b[3], moving:false}));
  render(); listBoxes();
};
$('#auto').onclick = async () => {
  $('#auto').textContent = '✨ detecting…';
  const r = await fetch('/api/autodetect', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({id: state.id})});
  const j = await r.json();
  $('#auto').textContent = '✨ Auto: detect & remove';
  state.boxes = [];
  (j.boxes || []).forEach(b =>
    state.boxes.push({x:b[0], y:b[1], w:b[2], h:b[3], moving:false}));
  if (j.moving) {
    state.boxes = state.boxes.filter(b =>
      !(Math.abs(b.x - j.moving[0]) < 30 && Math.abs(b.y - j.moving[1]) < 30));
    state.boxes.push({x:j.moving[0], y:j.moving[1], w:j.moving[2],
                      h:j.moving[3], moving:true});
  }
  render(); listBoxes();
  if (!state.boxes.length) {
    alert('Nothing detected automatically — draw a box over the watermark yourself');
    return;
  }
  startProcessing();
};
function startProcessing() {
  $('#step2').classList.add('hidden');
  $('#step3').classList.remove('hidden');
  fetch('/api/start', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({id: state.id, boxes: state.boxes})})
  .then(r => r.json())
  .then(j => { state.job = j.job; poll(); });
}
$('#go').onclick = () => {
  if (!state.boxes.length) { alert('Draw a box over the watermark first'); return; }
  startProcessing();
};
async function poll() {
  const r = await fetch('/api/status?job=' + state.job);
  const j = await r.json();
  $('#fill').style.width = (j.progress || 0) + '%';
  $('#msg').textContent = j.status === 'error'
    ? '❌ ' + (j.error || 'failed')
    : (j.status === 'done' ? '✅ Done!' : 'Removing watermark… ' + (j.progress||0) + '%');
  if (j.status === 'done') {
    $('#step3').classList.add('hidden');
    $('#step4').classList.remove('hidden');
    $('#prev').src = '/api/video?job=' + state.job;
    $('#save').href = '/api/video?job=' + state.job + '&save=1';
    $('#save').setAttribute('download',
      state.name.replace(/\.[^.]+$/, '') + '_clean.mp4');
  } else if (j.status === 'error') {
    $('#again').classList.remove('hidden');
  } else {
    setTimeout(poll, 900);
  }
}
$('#again').onclick = () => location.reload();
</script></body></html>"""


LIBRARY_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Video Library</title>
<style>
body{margin:0;font-family:system-ui;background:#111;color:#eee}
.wrap{max-width:900px;margin:0 auto;padding:16px}
h1{font-size:22px} a{color:#7fb0ff}
.card{background:#1d222b;border-radius:14px;padding:14px;margin:12px 0;
  display:flex;gap:14px}
.card img{width:180px;border-radius:8px;background:#000;object-fit:cover}
.meta{flex:1}
.name{font-weight:700;font-size:16px;word-break:break-all}
.dim{color:#9aa3b2;font-size:13px;margin:4px 0 10px}
.btnrow a{display:inline-block;background:#39c26d;color:#fff;border-radius:9px;
  padding:8px 14px;text-decoration:none;font-size:14px;margin-right:8px}
.btnrow a.orig{background:#2a3140}
.back{display:inline-block;margin-bottom:6px;text-decoration:none;
  color:#7fb0ff;font-size:14px}
.empty{color:#8f99ab;padding:30px;text-align:center}
</style></head><body><div class="wrap">
<a class="back" href="/">&larr; Watermark Remover</a>
<h1>📚 Video Library</h1>
<div class="dim">every video ever uploaded is stored here</div>
%s
</div></body></html>"""


def library_html():
    with db() as c:
        rows = c.execute(
            """SELECT v.id, v.name, v.uploaded_at, v.width, v.height,
                      v.duration, v.size_bytes,
                      (SELECT COUNT(*) FROM jobs j
                        WHERE j.video_id = v.id AND j.status='done')
                          AS n_clean
               FROM videos v ORDER BY v.uploaded_at DESC""").fetchall()
    if not rows:
        return LIBRARY_PAGE % '<div class="empty">no videos yet</div>'
    cards = []
    for r in rows:
        cleans = ""
        if r["n_clean"]:
            cleans = ('<a href="/api/file?kind=cleaned">cleaned ⬇</a>'
                      )
        cards.append(f"""
<div class="card">
  <img src="/api/thumb?id={r['id']}" alt="">
  <div class="meta">
    <div class="name">{r['name']}</div>
    <div class="dim">{r['uploaded_at'].replace('T',' ')} &middot;
      {r['width']}x{r['height']} &middot;
      {int(r['duration'])}s &middot; {r['size_bytes']/1e6:.1f} MB
      {('&middot; ' + str(r['n_clean']) + ' cleaned version(s)') if r['n_clean'] else ''}
    </div>
    <div class="btnrow">
      <a class="orig" href="/api/file?kind=orig&id={r['id']}">original ⬇</a>
      {cleans}
    </div>
    <div class="cleanlist" id="cl_{r['id']}"></div>
  </div>
</div>""")
    # fill in per-job cleaned links
    with db() as c:
        jobs = c.execute(
            "SELECT id, video_id, started_at, frames FROM jobs "
            "WHERE status='done' ORDER BY started_at DESC").fetchall()
    links = {}
    for j in jobs:
        links.setdefault(j["video_id"], []).append(
            f'<div style="margin-top:8px"><a href="/api/file?kind=clean&job={j["id"]}" '
            f'style="color:#7fb0ff">cleaned {j["started_at"].replace("T"," ")} '
            f'({j["frames"]} frames) ⬇</a></div>')
    html = LIBRARY_PAGE % "".join(cards)
    for vid, l in links.items():
        html = html.replace(f'<div class="cleanlist" id="cl_{vid}"></div>',
                            "".join(l))
    return html


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif u.path == "/library":
            body = library_html().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif u.path == "/api/thumb":
            vid = (q.get("id") or [""])[0]
            thumb = os.path.join(THUMB_DIR, vid + ".jpg")
            if not os.path.exists(thumb):
                with db() as c:
                    row = c.execute("SELECT id FROM videos WHERE id=?",
                                    (vid,)).fetchone()
                if not row:
                    return self._json({"error": "unknown id"}, 404)
                src = os.path.join(UPLOAD_DIR, vid + ".mp4")
                data = grab_frame(src, 0)
                if not data:
                    return self._json({"error": "no frame"}, 404)
                with open(thumb, "wb") as fh:
                    fh.write(data)
            data = open(thumb, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif u.path == "/api/file":
            kind = (q.get("kind") or [""])[0]
            if kind == "orig":
                vid = (q.get("id") or [""])[0]
                with db() as c:
                    row = c.execute("SELECT name FROM videos WHERE id=?",
                                    (vid,)).fetchone()
                if not row:
                    return self._json({"error": "unknown id"}, 404)
                path = os.path.join(UPLOAD_DIR, vid + ".mp4")
                fname = row["name"]
            elif kind in ("clean", "cleaned"):
                jid = (q.get("job") or [""])[0]
                with db() as c:
                    row = c.execute(
                        "SELECT j.video_id, j.out_name, v.name FROM jobs j "
                        "JOIN videos v ON v.id = j.video_id "
                        "WHERE j.id=? AND j.status='done'", (jid,)).fetchone()
                if not row:
                    return self._json({"error": "unknown job"}, 404)
                path = os.path.join(OUTPUT_DIR, row["out_name"])
                fname = row["name"].rsplit(".", 1)[0] + "_clean.mp4"
            else:
                return self._json({"error": "bad kind"}, 400)
            if not os.path.exists(path):
                return self._json({"error": "file missing"}, 404)
            data = open(path, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Disposition",
                             f'attachment; filename="{fname}"')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif u.path == "/api/frame":
            up = UPLOADS.get(q["id"][0])
            if not up:
                return self._json({"error": "unknown id"}, 404)
            out = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", up["path"],
                 "-frames:v", "1", "-f", "mjpeg", "-q:v", "3", "-"],
                capture_output=True, check=True).stdout
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        elif u.path == "/api/status":
            job = JOBS.get(q["job"][0], {})
            self._json(job)
        elif u.path == "/api/video":
            job = JOBS.get(q["job"][0])
            if not job or job.get("status") != "done":
                return self._json({"error": "not ready"}, 404)
            data = open(job["out"], "rb").read()
            size = len(data)
            try:
                rng = self.headers.get("Range")
                if rng and not q.get("save"):
                    # browsers stream <video> with range requests
                    a, _, b = rng.replace("bytes=", "").partition("-")
                    a = int(a or 0)
                    b = min(int(b) if b else size - 1, size - 1)
                    chunk = data[a:b + 1]
                    self.send_response(206)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Content-Range",
                                     f"bytes {a}-{b}/{size}")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk)
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Content-Length", str(size))
                    self.send_header("Accept-Ranges", "bytes")
                    if q.get("save"):
                        self.send_header("Content-Disposition",
                                         'attachment; filename="clean.mp4"')
                    self.end_headers()
                    self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser closed the stream early; harmless
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        from urllib.parse import urlparse
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        if u.path == "/api/upload":
            name = self.headers.get("X-Filename", "video.mp4")
            uid = uuid.uuid4().hex[:12]
            path = os.path.join(UPLOAD_DIR, uid + ".mp4")
            with open(path, "wb") as fh:
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    fh.write(chunk)
                    remaining -= len(chunk)
            try:
                w, h = display_dims(path)
                fps, duration = probe_fps(path)
            except Exception:
                w = h = duration = 0
            with db() as c:
                c.execute(
                    "INSERT INTO videos VALUES (?,?,?,?,?,?,?)",
                    (uid, name,
                     datetime.datetime.now().isoformat(timespec="seconds"),
                     w, h, duration, os.path.getsize(path)))
            UPLOADS[uid] = {"path": path, "name": name}
            self._json({"id": uid, "name": name})
        elif u.path == "/api/autodetect":
            req = json.loads(self.rfile.read(length) or b"{}")
            up = UPLOADS.get(req.get("id"))
            if not up:
                return self._json({"error": "unknown id"}, 404)
            self._json(autodetect(up["path"]))
        elif u.path == "/api/start":
            req = json.loads(self.rfile.read(length) or b"{}")
            up = UPLOADS.get(req.get("id"))
            if not up:
                return self._json({"error": "unknown id"}, 404)
            boxes = req.get("boxes") or []
            regions = [[int(b["x"]), int(b["y"]),
                        int(b["w"]), int(b["h"])]
                       for b in boxes if not b.get("moving")]
            moving = [b for b in boxes if b.get("moving")]
            track = ([int(moving[0]["x"]), int(moving[0]["y"]),
                      int(moving[0]["w"]), int(moving[0]["h"])]
                     if moving else None)
            jid = uuid.uuid4().hex[:12]
            outp = os.path.join(OUTPUT_DIR, jid + ".mp4")
            JOBS[jid] = {"status": "running", "progress": 0}
            with db() as c:
                c.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)",
                          (jid, req.get("id"),
                           datetime.datetime.now().isoformat(
                               timespec="seconds"),
                           "running", json.dumps(boxes),
                           json.dumps(track), None, None))
            threading.Thread(
                target=run_job, daemon=True,
                args=(jid, up["path"], outp, regions, track, 8,
                      req.get("id"))).start()
            self._json({"job": jid})
        else:
            self._json({"error": "not found"}, 404)


def _open_browser_later(port):
    """On Termux this opens Chrome with the app; elsewhere it's a no-op."""
    time.sleep(1.0)
    try:
        subprocess.run(["termux-open-url", f"http://localhost:{port}"],
                       capture_output=True, timeout=5)
    except Exception:
        pass


def main():
    # cloud hosts (Render/Railway/Fly) set PORT and need 0.0.0.0 binding
    port = int(os.environ.get("PORT") or
               (sys.argv[1] if len(sys.argv) > 1 else PORT))
    host = os.environ.get("HOST", "127.0.0.1")
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"Watermark Remover running on http://{host}:{port}")
    if host == "127.0.0.1":
        threading.Thread(target=_open_browser_later, args=(port,),
                         daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
