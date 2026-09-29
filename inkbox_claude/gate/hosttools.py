"""Read-only view of the machine Blatbot runs on. Fixed commands only, no shell:
the owner can ask what is running without giving any model a terminal."""

from __future__ import annotations

import asyncio
from typing import Any, Dict

COMMANDS: Dict[str, list] = {
    "containers": ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Ports}}"],
    "services": ["systemctl", "--user", "list-units", "--type=service", "--state=running", "--no-pager", "--no-legend"],
    "system_services": ["systemctl", "list-units", "--type=service", "--state=running", "--no-pager", "--no-legend"],
    "uptime_load": ["uptime"],
    "disk": ["df", "-h", "/", "/home"],
    "memory": ["free", "-h"],
    "gpus": ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
}

DESCRIPTION = ("WHAT IS RUNNING ON THIS SERVER (the machine Blatbot itself runs on, called black-sky): running "
               "containers, running services, uptime and load, disk, memory, GPUs. Read-only; pick which parts.")

SCHEMA = {"type": "object", "properties": {
    "parts": {"type": "array", "items": {"type": "string", "enum": list(COMMANDS)},
              "description": "which of: " + ", ".join(COMMANDS) + ". Omit for all."}}}


async def _run(cmd: list) -> str:
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=20)
        text = out.decode("utf-8", "replace").strip()
        return text or "(nothing)"
    except FileNotFoundError:
        return f"({cmd[0]} is not installed here)"
    except Exception as exc:
        return f"(failed: {exc})"


async def host_status(args: Dict[str, Any]) -> str:
    parts = [p for p in (args.get("parts") or list(COMMANDS)) if p in COMMANDS]
    outs = await asyncio.gather(*[_run(COMMANDS[p]) for p in parts])
    return "\n\n".join(f"## {p}\n{o}" for p, o in zip(parts, outs))


def sdk_server() -> Any:
    """The same tool as an in-process MCP server for the Claude Code executor."""
    from claude_agent_sdk import create_sdk_mcp_server, tool

    @tool("host_status", DESCRIPTION, SCHEMA)
    async def _host_status(args: Dict[str, Any]) -> Dict[str, Any]:
        return {"content": [{"type": "text", "text": await host_status(args)}]}

    return create_sdk_mcp_server(name="host", version="1.0.0", tools=[_host_status])
