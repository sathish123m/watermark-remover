#!/data/data/com.termux/files/usr/bin/sh
# One-tap launcher for the Watermark Remover (used with Termux:Widget).
# Kills any old instance, starts the app; the app opens Chrome by itself.
pkill -f wmremove_app.py 2>/dev/null
sleep 1
nohup python "$HOME/wmremove_app.py" >/dev/null 2>&1 &
