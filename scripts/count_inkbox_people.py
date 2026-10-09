#!/usr/bin/env python3
"""Count, per channel and distinct total, how many people Blatbot's Inkbox
identity has ever spoken to — without printing a single name, number, or
address, and without writing or sending anything.

Runs ONLY the collection step the console's people-sync uses
(`console.directory.InkboxSDKDirectoryClient` paging every channel), then the
same dedupe key `console.sync.sync_people` uses (`gate.store._trust_key`), and
prints counts. It never calls `sync_people` itself, so it never touches the
gate database or the console's people table.

Usage:
    INKBOX_API_KEY=... INKBOX_IDENTITY=... .venv/bin/python scripts/count_inkbox_people.py

Reads INKBOX_API_KEY / INKBOX_IDENTITY / INKBOX_BASE_URL the same way the
gateway does (see inkbox_claude/config.py). Run this yourself against the real
account — it is intentionally not run as part of the test suite or by any
Claude Code session in this repo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    try:
        from inkbox import Inkbox
    except ImportError:
        print("inkbox SDK is not installed in this environment.", file=sys.stderr)
        return 2

    from inkbox_claude.config import inkbox_client_kwargs
    from inkbox_claude.console.directory import build_directory_client
    from inkbox_claude.console.sync import _CHANNEL_METHODS
    from inkbox_claude.gate.store import _trust_key

    api_key = os.getenv("INKBOX_API_KEY", "").strip()
    identity_name = os.getenv("INKBOX_IDENTITY", "").strip()
    base_url = os.getenv("INKBOX_BASE_URL", "").strip() or None
    if not api_key or not identity_name:
        print("Set INKBOX_API_KEY and INKBOX_IDENTITY first.", file=sys.stderr)
        return 2

    client_kwargs = inkbox_client_kwargs(api_key, base_url)
    inkbox = Inkbox(**client_kwargs)
    identity = inkbox.get_identity(identity_name)
    directory = build_directory_client(inkbox, identity)

    distinct_keys: set[str] = set()
    print("Channel counts (rows seen while paging, not distinct people):")
    for channel, method_name in _CHANNEL_METHODS:
        method = getattr(directory, method_name, None)
        if method is None:
            print(f"  {channel}: (no method)")
            continue
        count = 0
        try:
            for row in method():
                count += 1
                handle = str(row.get("handle") or "").strip()
                if handle:
                    distinct_keys.add(_trust_key(handle))
        except Exception as exc:  # noqa: BLE001 - report, keep going on other channels
            print(f"  {channel}: FAILED ({exc.__class__.__name__})")
            continue
        print(f"  {channel}: {count}")

    print(f"\nDistinct people (deduped by trust key, across all channels): {len(distinct_keys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
