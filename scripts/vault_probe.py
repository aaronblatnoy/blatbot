#!/usr/bin/env python3
"""Probe the knowledge-scope traversal (TaskPicker.judge_vault_tree) against a
real vault, without changing anything in it.

Prints only scope names, note paths, probabilities and timings for each
question given -- never note text. Repeatable: pass the vault path and the
questions on the command line or in a file; nothing about a real vault's
content is hardcoded here.

Usage:
  python scripts/vault_probe.py --vault /path/to/vault "question one" "question two"
  python scripts/vault_probe.py --vault /path/to/vault --questions-file questions.txt

Requires TYPESAFE_API_KEY in the environment (load it the way the repo's own
docs do, e.g. `set -a; . ~/blatbot/.env; set +a` in the same shell, never
echoed).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vault", required=True, help="path to the vault root")
    ap.add_argument("--questions-file", help="a text file, one question per line")
    ap.add_argument("questions", nargs="*", help="questions given directly on the command line")
    args = ap.parse_args()

    questions = list(args.questions)
    if args.questions_file:
        with open(args.questions_file, encoding="utf-8") as f:
            questions += [line.strip() for line in f if line.strip()]
    if not questions:
        print("no questions given (use positional args or --questions-file)", file=sys.stderr)
        return 2

    os.environ["BLATBOT_VAULT_DIR"] = os.path.abspath(args.vault)

    from inkbox_claude.gate.taskpick import TaskPicker
    from inkbox_claude.gate import vaultscopes

    api_key = (os.getenv("TYPESAFE_API_KEY") or "").strip()
    if not api_key:
        print("TYPESAFE_API_KEY is not set in this shell", file=sys.stderr)
        return 2
    picker = TaskPicker(api_key=api_key)

    vaultscopes.discover_cached(force=True)  # one fresh discovery pass, cached for every question below

    async def run() -> None:
        for q in questions:
            t0 = time.monotonic()
            res = await picker.judge_vault_tree(prompt=q, summary=q[:160])
            elapsed = time.monotonic() - t0
            print(f"question: {q!r}")
            print(f"  scopes: {res.get('scopes')}")
            print(f"  notes: {res.get('notes')}")
            probs = res.get("probabilities") or {}
            top = sorted(probs.items(), key=lambda kv: -kv[1])[:8]
            print(f"  top probabilities: {[(k, round(v, 2)) for k, v in top]}")
            print(f"  timings: {res.get('timings')}  total: {elapsed:.2f}s")
            print(f"  reason: {res.get('reason')}")
            print()

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
