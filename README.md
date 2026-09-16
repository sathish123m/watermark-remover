# 💧 Watermark Remover

Remove watermarks and logos from videos — **without blurring or making it
look edited**. Erased areas are rebuilt from the surrounding picture using
content-aware inpainting, and the local film grain is matched so the patch
blends into the footage. Audio is copied untouched.

Runs as a **web app** (pick video → mark → remove → save) or from the
**command line**. Works on a laptop, on Android via Termux, and in Docker.

## Features

- **✨ One-tap auto mode** — OCR reads the video's first and last frame,
  finds text watermarks, and detects whether a logo **drifts/moves** (the
  anti-removal trick used by TikTok-style apps) — a moving watermark is
  tracked frame-by-frame and erased where it actually is.
- **Manual mode** — drag boxes over one or several watermarks with mouse
  or finger; mark a box as "moves" for drifting logos.
- **Static auto-detect** — temporal analysis finds fixed logos, burned-in
  overlays and black borders/letterboxing.
- **Quality first** — Telea inpainting (OpenCV) with a pure-numpy
  fallback, text-shaped masks (not rectangles), grain matching, and
  pixel-identical pass-through outside the erased areas.
- **📚 Video library** — every upload and every cleaned version is stored
  in SQLite and browsable/downloadable from the Library page.

## Run locally (Mac/Linux/Windows)

Requirements: Python 3.9+, ffmpeg on PATH.

```bash
pip install -r requirements.txt
python wmremove_app.py          # opens http://localhost:8080
```

Data (uploads, cleaned videos, `app.db`) is kept in
`~/watermark-remover-data/` — set `DATA_DIR` to change it.

Environment variables: `PORT` (default 8080), `HOST` (default 127.0.0.1),
`DATA_DIR`.

## Run in Docker

```bash
docker build -t watermark-remover .
docker run -p 8080:8080 -v wm-data:/data watermark-remover
```

## Deploy

The included Dockerfile reads `PORT` and binds `0.0.0.0`, so it works on
Render / Railway / Fly.io / any container host. Mount a volume at `/data`
to keep the video library between deploys.

### Render

1. New + → **Blueprint** → pick this repo → Render reads `render.yaml`
   (or: New + → Web Service → Docker → add a 5 GB disk mounted at `/data`).
2. Use **at least the Starter plan**. The free plan (512 MB, no disk) is
   too small for video processing — jobs will stall and the library
   resets on every deploy.
3. Deploy. First build takes ~5–10 minutes (installs ffmpeg + OCR).

### Fly.io

```bash
fly launch          # picks up fly.toml
fly scale memory 1024
fly deploy
```

> ⚠️ **If you deploy publicly, add an auth layer** (e.g. the platform's
> basic auth or a reverse proxy). As shipped, anyone who can reach the
> page can upload and process videos.

### Honest performance expectations

Processing is CPU-bound and single-threaded per job. Expect roughly:

- laptop: a 15 s clip in ~2–4 min
- Render/Fly starter (shared vCPU): ~3× slower than a laptop
- uploads are limited by your connection — the app shows a live upload
  progress bar, and processing shows frame count + ETA

The job watchdog fails a job with a clear message instead of hanging
forever if the server runs out of memory.

## Android (Termux)

Install Termux from [F-Droid](https://f-droid.org/en/packages/com.termux/),
then:

```bash
pkg install -y python ffmpeg python-numpy
termux-setup-storage      # tap Allow
python wmremove_app.py    # Chrome opens with the app
```

OpenCV and the OCR engine are optional on Termux — the app automatically
falls back to its pure-numpy engine. A one-tap home-screen launcher is in
[`termux/`](termux/), along with `wmremove-lite.py`, a no-dependency CLI
version. Full notes: [termux/README.md](termux/README.md).

## Command line

```bash
# static watermark at a known spot (x y width height)
python wmremove.py input.mp4 output.mp4 --region 940 40 300 80

# moving watermark (position in the FIRST frame)
python wmremove.py input.mp4 output.mp4 --track 38 39 190 26

# arbitrary erase mask (white = erase)
python wmremove.py input.mp4 output.mp4 --mask mask.png

# see what would be erased before processing
python wmremove.py input.mp4 output.mp4 --preview
```

`wmremove.py` needs OpenCV (`pip install opencv-python-headless`).

## How it works

1. **Decode** frames through an ffmpeg raw-video pipe (rotation metadata
   is baked so phone videos map correctly).
2. **Find the watermark** — for auto mode: OCR at two timestamps (same
   text in different places ⇒ moving watermark), plus temporal statics
   for fixed overlays. For manual mode: your boxes.
3. **Erase per frame** — a text-shaped mask (median-background
   subtraction) covers only the logo pixels; Telea/diffusion inpainting
   rebuilds them from the surroundings; grain matching re-injects the
   local noise level so the patch doesn't read as "edited".
4. **Track** moving logos by normalized cross-correlation of the
   background-subtracted template, with confidence checks.
5. **Encode** with x264 (CRF 17–18) and copy the audio stream untouched.

## Limitations

- Watermarks must be reasonably opaque/visible; fully invisible ones
  can't be located automatically (draw a box instead).
- Very large watermarks (a third of the frame or more) leave visibly
  synthesized content — inpainting can only guess what was behind.
- Auto-detect for moving logos relies on OCR, so it works for **text**
  watermarks; moving logos without text need a manual box + "moves".
- Processing is CPU-bound: roughly real-time×0.5 on a laptop, slower on
  phones.
