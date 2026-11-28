#!/bin/bash
# One-time install on the Mac: copies the snapshot script, writes the launchd job, loads it.
# Usage: bash deploy/mac/install_ccsessions.sh   (from a checkout of this repo)
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="$HOME/.local/share/blatbot-ccsessions"
LABEL="com.aaron.blatbot-ccsessions"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
PY="${CCSESSIONS_PYTHON:-$(command -v python3)}"
mkdir -p "$DIR"
cp "$HERE/../../inkbox_claude/gate/ccsessions.py" "$DIR/ccsessions.py"
cp "$HERE/ccsessions_push.sh" "$DIR/ccsessions_push.sh"
chmod +x "$DIR/ccsessions_push.sh"
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$DIR/ccsessions_push.sh</string></array>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/usr/bin:/bin:/usr/local/bin:$(dirname "$PY")</string>
    <key>CCSESSIONS_PYTHON</key><string>$PY</string>
  </dict>
  <key>StartInterval</key><integer>60</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardErrorPath</key><string>$DIR/push.err</string>
  <key>StandardOutPath</key><string>$DIR/push.out</string>
</dict></plist>
PL
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "loaded $LABEL; first push log: $DIR/push.err"
