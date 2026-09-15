# Android (Termux) setup

Run the full Watermark Remover app on your phone — offline, no PC.

## Install

1. Install **Termux** from
   [F-Droid](https://f-droid.org/en/packages/com.termux/) (not the Play
   Store — that build is outdated).
2. In Termux:
   ```bash
   pkg install -y python ffmpeg python-numpy
   termux-setup-storage     # tap Allow on the dialog
   ```
3. Copy `wmremove_app.py` from this repo to the phone (Drive, USB,
   Telegram, ...) into the Download folder, then:
   ```bash
   cp ~/storage/downloads/wmremove_app.py ~/
   python ~/wmremove_app.py
   ```
   Chrome opens with the app.

OpenCV and the OCR engine are optional — without them the app uses its
built-in pure-numpy engine (auto-detect for text watermarks needs the OCR
extra, which is not available on Termux; manual boxes and the static
auto-detect work fine).

## Daily use

Open Termux → `python ~/wmremove_app.py` (or press ↑ then Enter) → pick a
video → draw a box, or tap **Auto: detect & remove** → Save.

## One-tap home-screen icon (optional)

1. Install [Termux:Widget](https://f-droid.org/en/packages/com.termux.widget/)
   from F-Droid.
2. ```bash
   mkdir -p ~/.shortcuts
   cp start-app.sh ~/.shortcuts/"Watermark Remover"
   ```
3. Long-press the home screen → Widgets → Termux:Widget → drag the
   shortcut out.

## `wmremove-lite.py`

A minimal no-frills CLI version with the same pure-numpy engine — useful
if you prefer the terminal or want the smallest possible file:

```bash
python wmremove-lite.py input.mp4 output.mp4 --track 38 39 190 26
```

## Notes

- Phone CPUs are slow: expect 10–20 minutes for a 15-second clip. Keep
  the phone charging, screen on, and Termux in the foreground.
- Data lives in `~/watermark-remover-data/` on the phone.
