#!/bin/sh
# State lives on the /data volume: exec dir, logs, media, standing instructions, Claude home.
set -e
mkdir -p "$HOME" "$INKBOX_CLAUDE_HOME/exec" "$INKBOX_CLAUDE_HOME/logs" "$INKBOX_CLAUDE_HOME/media"
[ -f "$INKBOX_CLAUDE_HOME/standing.md" ] || : > "$INKBOX_CLAUDE_HOME/standing.md"
# The Definite Optimists brain is Gondroid's memory: an HTTP MCP server, reached through the
# stdio bridge, signed in with Gondroid's own brain key (DO_BRAIN_KEY, a Coolify secret).
python - <<'PY'
import json, os
path = os.path.join(os.environ["HOME"], ".claude.json")
try:
    cfg = json.load(open(path))
except Exception:
    cfg = {}
servers = cfg.setdefault("mcpServers", {})
key = os.environ.get("DO_BRAIN_KEY", "").strip()
if key:
    servers["do-brain"] = {"type": "stdio", "command": "python",
                           "args": ["/app/scripts/http_mcp_bridge.py", os.environ["DO_BRAIN_MCP_URL"]],
                           "env": {"MCP_BRIDGE_AUTH": "Bearer " + key}}
else:
    servers.pop("do-brain", None)
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
os.write(fd, json.dumps(cfg).encode()); os.close(fd)
PY
exec inkbox-claude run
