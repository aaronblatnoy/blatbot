#!/bin/sh
# State lives on the /data volume: exec dir, logs, media, standing instructions, Claude home.
set -e
mkdir -p "$HOME" "$INKBOX_CLAUDE_HOME/exec" "$INKBOX_CLAUDE_HOME/logs" "$INKBOX_CLAUDE_HOME/media"
[ -f "$INKBOX_CLAUDE_HOME/standing.md" ] || : > "$INKBOX_CLAUDE_HOME/standing.md"
exec inkbox-claude run
