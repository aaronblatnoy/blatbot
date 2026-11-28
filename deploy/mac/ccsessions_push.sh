#!/bin/bash
# Pushes a compact snapshot of this Mac's Claude Code sessions to black-sky (Blatbot reads it there).
# Run every minute by launchd (com.aaron.blatbot-ccsessions). Pushes, never pulls, so a sleeping Mac
# just stops updating the file and Blatbot reports "Mac last seen at X".
set -u
DIR="$HOME/.local/share/blatbot-ccsessions"
PY="${CCSESSIONS_PYTHON:-$(command -v python3)}"
SNAP="$("$PY" "$DIR/ccsessions.py" snapshot --label Mac --hours 24)" || exit 1
printf '%s' "$SNAP" | ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 black-sky \
  'umask 077; d="$HOME/blatbot/ccsessions"; mkdir -p "$d" && cat > "$d/mac.json.tmp" && mv "$d/mac.json.tmp" "$d/mac.json"'
