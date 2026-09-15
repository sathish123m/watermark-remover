#!/usr/bin/env python3
"""
wmremove-lite.py - watermark/logo remover with NO OpenCV dependency.
Needs only python3 + numpy + ffmpeg  ->  runs on Termux (Android).

Supports:
  --region X Y W H   (repeatable) static watermark rectangles
  --track  X Y W H   moving watermark: position in the FIRST frame;
                     re-located every frame by template matching

Video is decoded/encoded by ffmpeg pipes; pixels are processed in numpy.
"""

import argparse
import json
import subprocess
import sys

import numpy as np


# --------------------------------------------------------------------------
# ffmpeg plumbing
# --------------------------------------------------------------------------

def probe(path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,r_frame_rate",
           "-of", "json", path]
    out = subprocess.run(cmd, capture_output=True, text=True,
                         check=True).stdout
    info = json.loads(out)["streams"][0]
    num, den = info["r_frame_rate"].split("/")
    return int(info["width"]), int(info["height"]), float(num) / float(den)


class Reader:
    """Yields BGR uint8 frames from ffmpeg rawvideo stdout."""

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
            # copy(): frombuffer over bytes is read-only and we edit frames
            yield np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3).copy()
        self.proc.stdout.close()
        self.proc.wait()


class Writer:
    """Raw BGR frames in, original file as 2nd input for audio copy."""

    def __init__(self, path, out, w, h, fps, crf, preset):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", f"{fps:.6f}", "-i", "-",
             "-i", path,
             "-map", "0:v:0", "-map", "1:a?",
             "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
             "-pix_fmt", "yuv420p", "-c:a", "copy",
             "-movflags", "+faststart", out],
            stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        self.proc.stdin.close()
        return self.proc.wait()


# --------------------------------------------------------------------------
# numpy image ops (no cv2)
# --------------------------------------------------------------------------

def median_blur(img, k):
    """Separable median filter - background estimate for text masks."""
    out = img.astype(np.float32)
    half = k // 2
    for axis in (0, 1):
        stack = [np.roll(out, s, axis=axis) for s in range(-half, half + 1)]
        out = np.median(np.stack(stack), axis=0)
    return out


def dilate_box(mask, r):
    """Binary dilation with a square (2r+1) box, separable via maximum."""
    out = mask
    for axis in (0, 1):
        out = np.maximum.reduce(
            [np.roll(out, s, axis=axis) for s in range(-r, r + 1)])
    return out


def blur3(img):
    """[1 2 1]/4 separable blur (grain high-pass helper)."""
    out = img.astype(np.float32)
    for axis in (0, 1):
        out = (np.roll(out, 1, axis=axis) + 2 * out +
               np.roll(out, -1, axis=axis)) * 0.25
    return out


def high_pass(img):
    return img.astype(np.float32) - blur3(img)


def text_mask(gray, thresh=25):
    """Bright semi-transparent text: pixels above the local background."""
    diff = np.clip(gray.astype(np.float32) - median_blur(gray, 15), 0, None)
    return dilate_box(diff > thresh, 2)


def coons_init(img, mask):
    """Seed masked pixels by blending the colors around the mask box."""
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


def diffuse_fill(img, mask, iters=120):
    """Laplace-style inpainting: average neighbors inside the mask.

    The mask is seeded with coons_init first so even larger regions
    converge quickly and smoothly.
    """
    if not mask.any():
        return img
    img = coons_init(img.astype(np.float32).copy(), mask)
    m = mask
    for _ in range(iters):
        avg = (np.roll(img, 1, 0) + np.roll(img, -1, 0) +
               np.roll(img, 1, 1) + np.roll(img, -1, 1)) * 0.25
        img[m] = avg[m]
    return img


def track(gray, tpl_hp, prev, move, min_conf=0.25):
    """Normalized cross-correlation of the median-high-passed template
    against a window around the previous position.

    The high-pass is gray minus a 15px median background (same as the
    OpenCV version): it suppresses smooth scene content and keeps the
    glyphs, which makes the correlation robust.
    Returns (x, y, confidence).
    """
    th, tw = tpl_hp.shape
    H, W = gray.shape
    m = move + 10
    # candidate top-left positions within +/- m of the previous position
    cx0, cx1 = max(0, prev[0] - m), min(W - tw, prev[0] + m)
    cy0, cy1 = max(0, prev[1] - m), min(H - th, prev[1] + m)
    if cx1 < cx0 or cy1 < cy0:
        return prev[0], prev[1], 0.0
    # crop with a margin so the median filter's wrap-around edges stay
    # away from the candidate positions
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
    # sliding-window index -> global template top-left
    return (wx0 + int(j), wy0 + int(i), conf)


def grain_sigmas(bgr, ring):
    """Robust per-channel grain level from unmasked ring pixels."""
    out = []
    for ch in range(3):
        res = high_pass(bgr[:, :, ch])[ring]
        if res.size < 200:
            out.append(0.0)
            continue
        mad = np.median(np.abs(res - np.median(res)))
        # empirical: high_pass keeps ~0.80 of white-noise sigma (MAD-based)
        out.append(float(1.4826 * mad / 0.80))
    return out


# --------------------------------------------------------------------------
# main pipeline
# --------------------------------------------------------------------------

def run(inp, outp, regions, track_box, move, crf, preset):
    w, h, fps = probe(inp)
    print(f"video: {w}x{h} @ {fps:.3f} fps")

    static_rois = []
    for (x, y, rw, rh) in regions:
        pad = 6
        # ROI extends `pad` beyond the region on every side (clamped to the
        # frame); the erase mask sits at its true offset inside the ROI
        x0r, y0r = max(0, x - pad), max(0, y - pad)
        x1r, y1r = min(w, x + rw + pad), min(h, y + rh + pad)
        ox, oy = x - x0r, y - y0r
        mask = np.zeros((y1r - y0r, x1r - x0r), bool)
        mask[oy:oy + rh, ox:ox + rw] = True
        static_rois.append((x0r, y0r, mask))

    reader = iter(Reader(inp, w, h))
    try:
        first = next(reader)
    except StopIteration:
        sys.exit("error: could not decode any frames")

    # tracked-watermark template from the first frame
    tpl = None
    if track_box:
        tx, ty, tw, th = track_box
        pad = 5
        tx, ty = max(0, tx - pad), max(0, ty - pad)
        tw = min(w - tx, tw + 2 * pad)
        th = min(h - ty, th + 2 * pad)
        tpl_crop = first[ty:ty + th, tx:tx + tw].astype(np.float32).mean(axis=2)
        tpl_hp = tpl_crop - median_blur(tpl_crop, 15)
        if text_mask(tpl_crop).mean() < 0.03:
            sys.exit("error: no text-like watermark at the --track position")
        pos = (tx, ty)
        bw, bh = tw, th

    writer = Writer(inp, outp, w, h, fps, crf, preset)
    rng = np.random.default_rng(0)
    grain = None          # tracked-area grain (calibrated on frame 1)
    s_grain = []          # per-static-ROI grain
    for (x, y, mask) in static_rois:
        s_grain.append(grain_sigmas(first[y:y + mask.shape[0],
                                       x:x + mask.shape[1]], ~mask))

    n = 0
    misses = 0
    frame = first
    while frame is not None:
        # 1) static regions
        for (x, y, mask), sig in zip(static_rois, s_grain):
            rh, rw = mask.shape
            crop = frame[y:y + rh, x:x + rw]
            fixed = diffuse_fill(crop, mask, iters=150)
            noise = np.zeros((rh, rw, 3), np.float32)
            for ch, s in enumerate(sig):
                if s > 0.5:
                    noise[:, :, ch] = rng.normal(0, s, (rh, rw)) * mask
            frame[y:y + rh, x:x + rw] = np.clip(fixed + noise,
                                                0, 255).astype(np.uint8)

        # 2) tracked moving watermark
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
                    noise[:, :, ch] = rng.normal(0, s, (bh, bw)) * m
            frame[py:py + bh, px:px + bw] = np.clip(fixed + noise,
                                                    0, 255).astype(np.uint8)

        writer.write(frame)
        n += 1
        if n % 25 == 0:
            sys.stdout.write(f"\r  frame {n}")
            sys.stdout.flush()
        frame = next(reader, None)
    print()
    if track_box and misses:
        print(f"note: low tracking confidence on {misses} frame(s)")
    if writer.close() != 0:
        sys.exit("error: ffmpeg encoding failed")
    print(f"done: {outp}  ({n} frames)")


def main():
    ap = argparse.ArgumentParser(
        description="Watermark/logo remover for Termux/Android "
                    "(numpy + ffmpeg only, no OpenCV).")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--region", type=int, nargs=4, action="append",
                    metavar=("X", "Y", "W", "H"),
                    help="static watermark rectangle (repeatable)")
    ap.add_argument("--track", type=int, nargs=4,
                    metavar=("X", "Y", "W", "H"),
                    help="moving watermark: position in the FIRST frame")
    ap.add_argument("--move", type=int, default=8,
                    help="max watermark drift per frame in px (default 8)")
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--preset", default="medium")
    args = ap.parse_args()

    if not args.region and not args.track:
        sys.exit("error: give at least one --region or a --track box")

    run(args.input, args.output, args.region or [], args.track,
        args.move, args.crf, args.preset)


if __name__ == "__main__":
    main()
