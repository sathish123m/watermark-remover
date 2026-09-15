#!/usr/bin/env python3
"""
wmremove.py - remove static watermarks/logos from videos with content-aware
inpainting (OpenCV Telea / Navier-Stokes), so the area is rebuilt from the
surrounding image instead of being blurred or covered with a box.

Only the watermark pixels are touched. Everything else is passed through
pixel-for-pixel (one clean re-encode). Audio is copied without re-encoding.

Usage examples
--------------
  # Auto-detect the watermark and remove it
  python wmremove.py input.mp4 output.mp4

  # Remove a logo at a known position (x y width height, top-left origin)
  python wmremove.py input.mp4 output.mp4 --region 940 40 300 80

  # Several logos at once
  python wmremove.py input.mp4 output.mp4 --region 40 40 200 60 --region 940 980 300 60

  # See what the tool detected / where it will erase, without processing
  python wmremove.py input.mp4 output.mp4 --preview
"""

import argparse
import json
import math
import subprocess
import sys

import cv2
import numpy as np


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def probe(path):
    """Return (width, height, fps, nb_frames) using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames",
        "-of", "json", path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    info = json.loads(out)["streams"][0]
    num, den = info["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    frames = int(info.get("nb_frames") or 0)
    return int(info["width"]), int(info["height"]), fps, frames


def iter_sampled_frames(path, n_samples):
    """Yield up to n_samples frames, evenly spread over the video.

    Reads sequentially with grab() instead of CAP_PROP_POS_FRAMES seeking:
    seeking lands on keyframes only for many codecs, which would return the
    same frame many times and break the temporal statistics.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"error: cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = n_samples
    step = max(1, total // max(1, n_samples))
    idx = 0
    while True:
        if idx % step == 0:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
            idx += 1
            if (idx // step) >= n_samples:
                break
        else:
            if not cap.grab():
                break
            idx += 1
    cap.release()


# --------------------------------------------------------------------------
# mask construction
# --------------------------------------------------------------------------

def mask_from_regions(frame_shape, regions, pad=4):
    """Binary uint8 mask (255 = erase) covering the given rectangles."""
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for (x, y, rw, rh) in regions:
        x, y = max(0, x), max(0, y)
        x2, y2 = min(w, x + rw), min(h, y + rh)
        mask[y:y2, x:x2] = 255
    if pad > 0:
        mask = cv2.dilate(mask, np.ones((1 + 2 * pad, 1 + 2 * pad), np.uint8))
    return mask


def auto_detect_mask(path, samples=48):
    """
    Detect a static overlay (logo/watermark) in a mostly-moving video.

    Idea: pixels belonging to a static watermark have very low variance
    across time AND strong edges in the temporal average image. Real scene
    content may be static in places too, so we prefer candidates near the
    frame border, where logos almost always live.
    """
    frames = list(iter_sampled_frames(path, samples))
    if len(frames) < 4:
        sys.exit("error: not enough frames to analyse for auto-detection; "
                 "use --region x y w h instead")

    h, w = frames[0].shape[:2]
    stack = np.stack([f.astype(np.float32) for f in frames])          # (N,h,w,3)
    t_std = stack.std(axis=0).mean(axis=2)                            # temporal std
    t_mean = stack.mean(axis=0).astype(np.uint8)                      # temporal mean

    edges = cv2.Canny(cv2.cvtColor(t_mean, cv2.COLOR_BGR2GRAY), 60, 180)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))

    # 1) pixels that never change and hold an edge.  A burned-in overlay is
    # pixel-identical over time, so its temporal std is ~0; the thresholds
    # stay well below typical film-grain / sensor-noise floors so noisy
    # backgrounds are not mistaken for overlay.
    static_edges = ((t_std < 3.0) & (edges > 0)).astype(np.uint8) * 255

    # 2) pixels that never change at all (solid logo bodies without edges)
    solid_static = (t_std < 1.5).astype(np.uint8) * 255

    cand = cv2.bitwise_or(static_edges, solid_static)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE,
                            np.ones((9, 9), np.uint8))

    # connected-component filtering
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    margin = 0.22  # "near border" band as a fraction of the frame size
    bx, by = int(w * margin), int(h * margin)
    kept = np.zeros_like(cand)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < 60:
            continue
        near_border = (x < bx or y < by or
                       x + cw > w - bx or y + ch > h - by)
        # non-border blobs must be much bigger to be trusted
        if not near_border and area < 0.002 * w * h:
            continue
        kept[labels == i] = 255

    if kept.max() == 0:
        sys.exit("error: no static watermark detected automatically.\n"
                 "Specify it manually with:  --region x y w h")

    # grow the mask to swallow anti-aliased logo edges and soft shadows
    mask = cv2.dilate(kept, np.ones((15, 15), np.uint8))
    covered = mask.mean() / 255.0
    if covered > 0.12:
        sys.exit(f"error: auto-detection wants to erase {covered:.0%} of the "
                 "frame, which is not a watermark.\n"
                 "Specify the region manually with:  --region x y w h")
    return mask


# --------------------------------------------------------------------------
# processing
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# moving-watermark tracking
# --------------------------------------------------------------------------

def track_position(frame_diff, template_diff, prev, search=8, min_conf=0.25):
    """Locate the watermark in a high-passed frame near its last position.

    Matches against the whole frame, then keeps only candidate positions
    within `search` px of the previous one. (Matching a small cropped
    window directly is not portable: when the window is smaller than the
    template, OpenCV 5 silently swaps the two instead of failing.)
    Returns (x, y, confidence); on low confidence the previous position is
    returned with the measured confidence.
    """
    th, tw = template_diff.shape[:2]
    h, w = frame_diff.shape
    res = cv2.matchTemplate(frame_diff, template_diff, cv2.TM_CCOEFF_NORMED)
    x0 = max(0, prev[0] - search)
    y0 = max(0, prev[1] - search)
    x1 = min(res.shape[1] - 1, prev[0] + search)
    y1 = min(res.shape[0] - 1, prev[1] + search)
    if x1 < x0 or y1 < y0:
        return prev[0], prev[1], 0.0
    window = res[y0:y1 + 1, x0:x1 + 1]
    _, max_val, _, max_loc = cv2.minMaxLoc(window)
    if max_val < min_conf:
        return prev[0], prev[1], max_val
    return x0 + max_loc[0], y0 + max_loc[1], max_val


def hp(gray):
    """Background-subtracted image (keeps text/edges, drops smooth bg)."""
    return cv2.subtract(gray, cv2.medianBlur(gray, 15))


def text_mask(gray_crop, thresh=25):
    """Mask of bright semi-transparent text inside a crop.

    The text is brighter than the local background, so subtracting a
    median-blurred background estimate (kernel wider than the stroke)
    isolates the glyphs; a small dilation covers anti-aliased edges.
    """
    diff = cv2.subtract(gray_crop, cv2.medianBlur(gray_crop, 15))
    m = (diff > thresh).astype(np.uint8) * 255
    return cv2.dilate(m, np.ones((5, 5), np.uint8))


def mask_rois(mask, pad=24):
    """One (x, y, w, h) box per connected mask component, with margin.

    Inpainting each component separately keeps the working area small even
    when several logos sit at opposite corners of the frame.
    """
    h, w = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    rois = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < 4:
            continue
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + cw + pad)
        y1 = min(h, y + ch + pad)
        rois.append((x0, y0, x1 - x0, y1 - y0))
    return rois


def _hp(x):
    """High-pass residual (details/grain) of an image."""
    x = x.astype(np.float32)
    return x - cv2.GaussianBlur(x, (3, 3), 1.0)


def _hp_unit_std():
    """std() of the high-pass residual of unit white noise (calibration)."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 1.0, (256, 256)).astype(np.float32)
    return float(_hp(noise).std())


_HP_C = _hp_unit_std()


def estimate_grain_sigma(bgr_roi, ring):
    """Robust per-channel grain level from unmasked ring pixels.

    The ring is the band of real footage around the watermark inside the
    ROI crop; MAD keeps the estimate stable when the ring contains edges.
    """
    sigmas = []
    for ch in range(3):
        res = _hp(bgr_roi[:, :, ch])[ring]
        if res.size < 200:
            sigmas.append(0.0)
            continue
        mad = np.median(np.abs(res - np.median(res)))
        sigmas.append(float(1.4826 * mad / _HP_C))
    return sigmas


def process_tracked(inp, outp, box, method, crf, preset, move,
                    static_mask=None):
    """Remove a slowly-moving watermark.

    box is (x, y, w, h) at the START of the video; each frame the logo is
    re-located by template matching near its previous position, and only
    the glyph pixels (bright against local background) are inpainted.
    static_mask (optional) is erased as well, fixed in place.
    """
    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        sys.exit(f"error: cannot open video: {inp}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    inpaint_flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS

    ok, first = cap.read()
    if not ok:
        sys.exit(f"error: could not decode any frames from {inp}")

    x, y, bw, bh = box
    pad = 5
    x, y = max(0, x - pad), max(0, y - pad)
    bw = min(w - x, bw + 2 * pad)
    bh = min(h - y, bh + 2 * pad)
    template = cv2.cvtColor(first[y:y + bh, x:x + bw], cv2.COLOR_BGR2GRAY)
    template_diff = hp(template)
    # keep only the bright glyphs in the template so the dark background
    # around the text does not confuse the matcher
    tpl_mask = text_mask(template)
    if tpl_mask.mean() < 8:
        sys.exit("error: no text-like watermark found at the given "
                 "--track position; check the coordinates")

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
        "-r", f"{fps:.6f}", "-i", "-",
        "-i", inp,
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        outp,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # optional static erase regions (borders, a second fixed logo, ...)
    rois = []
    if static_mask is not None:
        rois = [(x, y, static_mask[y:y + rh, x:x + rw])
                for (x, y, rw, rh) in mask_rois(static_mask)]
        grain_static = {}
    rng = np.random.default_rng(0)
    pos = (x, y)
    grain_sigmas = None
    radius = 6
    n = 0
    misses = 0
    frame = first
    while frame is not None:
        for bx, by, roi_mask in rois:
            rh, rw = roi_mask.shape[:2]
            roi = frame[by:by + rh, bx:bx + rw]
            frame[by:by + rh, bx:bx + rw] = cv2.inpaint(roi, roi_mask,
                                                        radius, inpaint_flag)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nx_, ny_, conf = track_position(hp(gray), template_diff, pos,
                                        search=move)
        if conf >= 0.25:
            pos = (nx_, ny_)
        else:
            misses += 1         # keep previous position
        px, py = pos
        crop = frame[py:py + bh, px:px + bw]
        m = text_mask(gray[py:py + bh, px:px + bw])
        fixed = cv2.inpaint(crop, m, radius, inpaint_flag)
        # grain matching (same idea as the static path)
        if grain_sigmas is None:
            grain_sigmas = estimate_grain_sigma(crop, m == 0)
        if any(s > 0.5 for s in grain_sigmas):
            fixed = fixed.astype(np.float32)
            for ch, s in enumerate(grain_sigmas):
                if s <= 0.5:
                    continue
                g = rng.normal(0.0, s, (bh, bw)).astype(np.float32)
                layer = fixed[:, :, ch]
                layer[m > 0] += g[m > 0]
            fixed = np.clip(fixed, 0, 255).astype(np.uint8)
        frame[py:py + bh, px:px + bw] = fixed
        proc.stdin.write(frame.tobytes())
        n += 1
        if total and (n % 50 == 0 or n == total):
            pct = 100.0 * n / max(1, total)
            sys.stdout.write(f"\r  frame {n}/{total}  ({pct:.0f}%) "
                             f"logo at {pos[0]},{pos[1]}")
            sys.stdout.flush()
        ok, frame = cap.read()
    print()
    proc.stdin.close()
    cap.release()
    if proc.wait() != 0:
        sys.exit("error: ffmpeg encoding failed")
    if misses:
        print(f"note: low tracking confidence on {misses} frame(s); "
              "last known position was used there")
    print(f"done: {outp}  ({n} frames, {w}x{h} @ {fps:.3f} fps)")


def process(inp, outp, mask, method, crf, preset):
    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        sys.exit(f"error: cannot open video: {inp}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    inpaint_flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS

    # only inpaint the bounding boxes of the mask components -> big speedup
    rois = [(x, y, mask[y:y + rh, x:x + rw]) for (x, y, rw, rh) in mask_rois(mask)]

    # encoder: raw BGR frames in, original file as 2nd input for audio
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
        "-r", f"{fps:.6f}", "-i", "-",
        "-i", inp,
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        outp,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    ok, first = cap.read()
    if not ok:
        proc.stdin.close()
        proc.wait()
        sys.exit(f"error: could not decode any frames from {inp}")

    # calibrate per-ROI grain level once: the inpainted patch must carry the
    # same film grain / sensor noise as the surrounding footage, or it reads
    # as a suspiciously smooth "edited" spot
    rng = np.random.default_rng(0)
    grain = []
    for bx, by, roi_mask in rois:
        bh, bw = roi_mask.shape[:2]
        crop = first[by:by + bh, bx:bx + bw]
        grain.append(estimate_grain_sigma(crop, roi_mask == 0))

    radius = 6
    n = 0
    frame = first
    while frame is not None:
        for (bx, by, roi_mask), sigmas in zip(rois, grain):
            bh, bw = roi_mask.shape[:2]
            roi = frame[by:by + bh, bx:bx + bw]
            fixed = cv2.inpaint(roi, roi_mask, radius, inpaint_flag)
            m = roi_mask > 0
            if any(s > 0.5 for s in sigmas):
                fixed = fixed.astype(np.float32)
                for ch, s in enumerate(sigmas):
                    if s <= 0.5:
                        continue
                    g = rng.normal(0.0, s, (bh, bw)).astype(np.float32)
                    layer = fixed[:, :, ch]
                    layer[m] += g[m]
                fixed = np.clip(fixed, 0, 255).astype(np.uint8)
            frame[by:by + bh, bx:bx + bw] = fixed
        proc.stdin.write(frame.tobytes())
        n += 1
        if total and (n % 50 == 0 or n == total):
            pct = 100.0 * n / max(1, total)
            sys.stdout.write(f"\r  frame {n}/{total}  ({pct:.0f}%)")
            sys.stdout.flush()
        ok, frame = cap.read()
    print()
    proc.stdin.close()
    cap.release()
    if proc.wait() != 0:
        sys.exit("error: ffmpeg encoding failed")
    print(f"done: {outp}  ({n} frames, {w}x{h} @ {fps:.3f} fps)")


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Remove static watermarks/logos from a video using "
                    "content-aware inpainting (no blur, no black box).")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--region", type=int, nargs=4, action="append",
                    metavar=("X", "Y", "W", "H"),
                    help="watermark rectangle (top-left origin); "
                         "repeatable for several logos")
    ap.add_argument("--auto", action="store_true", default=True,
                    help="auto-detect the watermark when no --region is "
                         "given (default)")
    ap.add_argument("--track", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                    help="remove a MOVING watermark: its position at the "
                         "START of the video (top-left origin). It is "
                         "re-located by template matching every frame")
    ap.add_argument("--move", type=int, default=8,
                    help="max watermark drift per frame, in pixels "
                         "(default 8)")
    ap.add_argument("--mask", metavar="PNG",
                    help="use a custom erase mask image (white = erase, "
                         "black = keep) instead of --region/auto")
    ap.add_argument("--method", choices=["telea", "ns"], default="telea",
                    help="inpainting algorithm (telea: fast and clean, "
                         "ns: smoother for big areas; default telea)")
    ap.add_argument("--crf", type=int, default=17,
                    help="output quality, lower = better (default 17, "
                         "visually lossless-ish)")
    ap.add_argument("--preset", default="medium",
                    help="x264 speed preset (default medium)")
    ap.add_argument("--preview", action="store_true",
                    help="write preview.png + mask.png showing what will be "
                         "erased, then exit")
    return ap.parse_args()


def main():
    args = parse_args()

    for r in (args.region or []):
        x, y, w, h = r
        if w <= 0 or h <= 0:
            sys.exit("error: region width/height must be positive")
        if x < 0 or y < 0:
            sys.exit("error: region coordinates must be >= 0 "
                     "(origin is the top-left corner)")

    static_mask = None
    if args.mask:
        m = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if m is None:
            sys.exit(f"error: cannot read mask image: {args.mask}")
        static_mask = (m > 127).astype(np.uint8) * 255
    elif args.region:
        first = next(iter_sampled_frames(args.input, 1))
        static_mask = mask_from_regions(first.shape, args.region)
    elif not args.track:
        print("auto-detecting static watermark ...")
        static_mask = auto_detect_mask(args.input)

    if args.preview:
        first = next(iter_sampled_frames(args.input, 1))
        overlay = first.copy()
        if static_mask is not None:
            overlay[static_mask > 0] = (0, 0, 255)
            for x, y, w, h in mask_rois(static_mask, pad=0):
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if args.track:
            x, y, w, h = args.track
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(overlay, "tracked logo", (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        preview = cv2.addWeighted(first, 0.7, overlay, 0.3, 0) \
            if static_mask is not None else overlay
        cv2.imwrite("preview.png", preview)
        if static_mask is not None:
            cv2.imwrite("mask.png", static_mask)
        print("wrote preview.png (red = static erase, blue box = tracked "
              "logo start position)")
        return

    if args.track:
        x, y, w, h = args.track
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            sys.exit("error: --track needs x y width height, all positive")
        print(f"tracking moving watermark from x={x} y={y} w={w} h={h} "
              f"(max drift {args.move}px/frame)")
        process_tracked(args.input, args.output, args.track, args.method,
                        args.crf, args.preset, args.move, static_mask)
        return

    if static_mask is None:
        sys.exit("error: nothing to erase -- give --region, --mask, "
                 "--track or enable auto-detection")
    covered = static_mask.mean() / 255.0
    boxes = mask_rois(static_mask, pad=0)
    desc = ", ".join(f"[x={x} y={y} w={w} h={h}]" for x, y, w, h in boxes)
    print(f"erase regions: {desc} ({covered:.1%} of frame)")

    process(args.input, args.output, mask, args.method,
            args.crf, args.preset)


if __name__ == "__main__":
    main()
