"""`inkbox-claude scopes ...`: list, check and sync the capability registry."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Dict, List


def _print_list() -> int:
    from .gate import scopes as sc
    for sysname, desc in sc.SCOPE_TREE_SYSTEMS.items():
        print(f"{sysname}: {desc[:90]}")
        for g in sc.SCOPE_TREE_ALWAYS.get(sysname, []):
            print(f"    always -> {g} ({len(sc.SCOPES[g]['tools'])} tools)")
        for leaf, (q, grants) in sc.SCOPE_TREE_LEAVES.get(sysname, {}).items():
            print(f"    {leaf}? -> {', '.join(grants)}")
    print()
    for name, v in sc.SCOPES.items():
        kind = "read " if v.get("read") else "write"
        src = f"server {v['server']} patterns {v['include']}" if v.get("server") and v.get("include") else "explicit"
        print(f"{kind} {name:24} {len(v['tools']):3} tools  [{src}]")
    return 0


def _print_check() -> int:
    from .gate import scopes as sc
    problems = sc.registry_problems()
    for p in problems:
        print("PROBLEM:", p)
    print("ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


async def _sync(servers: List[str]) -> Dict[str, List[str]]:
    from .gate.jevagent import ToolBox, mcp_config_from_claude_json
    cfg = mcp_config_from_claude_json()
    resolved: Dict[str, List[str]] = {}
    async with ToolBox(None, cfg) as box:
        for name in servers:
            if name not in cfg:
                print(f"FAIL {name}: not in ~/.claude.json mcpServers", file=sys.stderr)
                continue
            if cfg[name].get("type") not in (None, "stdio") or not cfg[name].get("command"):
                print(f"SKIP {name}: only stdio servers can be resolved (type={cfg[name].get('type')})", file=sys.stderr)
                continue
            try:
                s = await asyncio.wait_for(box._session(name), 60)
                r = await asyncio.wait_for(s.list_tools(), 30)
                resolved[name] = [f"mcp__{name}__{t.name}" for t in r.tools]
                print(f"OK   {name}: {len(resolved[name])} tools")
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {name}: {type(exc).__name__}: {str(exc)[:160]}", file=sys.stderr)
    return resolved


def _do_sync() -> int:
    from .gate import scopes as sc
    reg = sc.load_registry()
    servers = sorted({str(v["server"]) for v in (reg.get("scopes") or {}).values() if v.get("server") and not v.get("tools")})
    print("resolving", ", ".join(servers))
    fresh = asyncio.run(_sync(servers))
    resolved = sc.load_resolved()
    resolved.update(fresh)
    with open(sc.RESOLVED_PATH, "w", encoding="utf-8") as fh:
        json.dump(resolved, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {sc.RESOLVED_PATH}")
    scopes = sc.build_scopes(reg, resolved)
    rc = 0
    for name, v in scopes.items():
        if v.get("server") and not v["tools"]:
            print(f"PROBLEM: scope {name} resolved to no tools on {v['server']}", file=sys.stderr)
            rc = 1
        elif v.get("server"):
            print(f"{name:24} {len(v['tools']):3} tools: " + ", ".join(t.split('__')[-1] for t in v["tools"])[:200])
    # explicit tool lists: warn when the live server no longer exposes a tool
    for name, v in scopes.items():
        if v.get("tools") and not v.get("server"):
            for t in v["tools"]:
                srv = t.split("__")[1] if t.startswith("mcp__") else None
                if srv in resolved and t not in resolved[srv]:
                    print(f"WARN: scope {name} lists {t}, which {srv} does not expose", file=sys.stderr)
    return rc


def run(args: Any) -> int:
    if args.scopes_command == "list":
        return _print_list()
    if args.scopes_command == "check":
        return _print_check()
    if args.scopes_command == "sync":
        return _do_sync()
    return 2
