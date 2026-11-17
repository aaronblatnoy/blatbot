"""Stdio-to-HTTP bridge for one stateless MCP server.

Blatbot's ToolBox only launches stdio MCP servers. This relays each JSON-RPC line
from stdin to a streamable-HTTP MCP endpoint and writes the reply to stdout.
Works for servers that answer with plain JSON or a single SSE event per request
and keep no session (stateless_http), such as the Definite Optimists brain.

Usage: http_mcp_bridge.py <url>   (Authorization header from MCP_BRIDGE_AUTH)
"""
import json
import os
import sys
import urllib.error
import urllib.request

URL = sys.argv[1]
AUTH = os.environ.get("MCP_BRIDGE_AUTH", "")


def post(msg: dict):
    # A named User-Agent: Cloudflare (the brain's public Worker) refuses the bare Python-urllib one (error 1010).
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
               "User-Agent": "inkbox-claude-mcp-bridge/1.0"}
    if AUTH:
        headers["Authorization"] = AUTH
    req = urllib.request.Request(URL, data=json.dumps(msg).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read().decode()
            ctype = r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return {"jsonrpc": "2.0", "id": msg.get("id"),
                "error": {"code": -32000, "message": f"HTTP {e.code}: {e.read().decode()[:300]}"}}
    except Exception as e:
        return {"jsonrpc": "2.0", "id": msg.get("id"), "error": {"code": -32000, "message": str(e)[:300]}}
    if not body.strip():
        return None
    if "text/event-stream" in ctype:
        data = [l[5:].strip() for l in body.splitlines() if l.startswith("data:")]
        body = data[-1] if data else ""
        if not body:
            return None
    return json.loads(body)


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        reply = post(msg)
        if "id" in msg and reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
