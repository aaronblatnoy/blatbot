"""What Aaron's Claude Code sessions are up to, on this machine and (from a pushed snapshot) on his Mac.

Read-only and stdlib only, so the same file runs inside Blatbot on black-sky and, standalone, on the
Mac, where a launchd job pushes `snapshot --label Mac` to black-sky every minute (see
deploy/mac/install_ccsessions.sh). It reads Claude Code's own files and nothing else:

  ~/.claude/sessions/<pid>.json        live-process registry: pid, session id, cwd, busy or idle
  ~/.claude/projects/*/<id>.jsonl      each session's transcript

Only short excerpts leave a transcript: the latest ask, the last reply, a few recent turns, all cut to
a few hundred characters with obvious secrets masked. Nothing here is sent to any model service; the
gate hands the text to the run that asked, exactly as it does for any other tool result.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # display timezone; the Mac pusher may run on a Python without zoneinfo data
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _TZ = None

TAIL_BYTES = 1_500_000        # how far back from the end of a transcript to look for the latest turns
HEAD_BYTES = 200_000          # how far in from the start to look for the first ask
ASK_CHARS = 240
GIST_CHARS = 280
TURN_CHARS = 260
STALE_AFTER_S = 300           # a pushed Mac snapshot older than this is reported as "last seen", not current
WORKING_IF_WROTE_WITHIN_S = 90  # no live-process record (headless runs): treat a very recent write as working
MAX_SESSIONS = 25

_NOT_AN_ASK = ("<system-reminder>", "<task-notification", "<command-", "<local-command", "Caveat:",
               "[Request interrupted", "<user-prompt-submit-hook", "This session is being continued",
               "Base directory for this skill", "<bash-", "<channel-")
_SYSTEM_BLOCK = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)
_SECRETS = re.compile(
    r"(sk-[A-Za-z0-9_\-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|xox[abp]-[A-Za-z0-9\-]{10,}|AKIA[0-9A-Z]{16}"
    r"|eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{10,}|(?i:bearer)\s+[A-Za-z0-9._\-]{16,}"
    r"|(?i:(?:token|secret|password|passwd|api[_-]?key)\s*[=:]\s*)\S{6,})")


def enabled() -> bool:
    """Per instance, off unless CLAUDE_SESSIONS_ENABLED is set: Blatbot's profile turns it on, Gondroid's turns it off."""
    return str(os.environ.get("CLAUDE_SESSIONS_ENABLED") or "").strip().lower() in ("1", "true", "yes", "on")


def _claude_dir() -> Path:
    return Path(os.environ.get("CCSESSIONS_CLAUDE_DIR") or (Path.home() / ".claude"))


def state_dir() -> Path:
    base = os.environ.get("CCSESSIONS_DIR") or os.path.join(os.environ.get("INKBOX_CLAUDE_HOME") or "~/blatbot", "ccsessions")
    return Path(os.path.expanduser(base))


def _clip(text: str, n: int) -> str:
    text = _SECRETS.sub("[masked]", " ".join((text or "").split()))
    return text if len(text) <= n else text[: n - 1].rstrip() + "..."


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(value: str) -> float:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def _clock(ts: float, with_day: bool = False) -> str:
    dt = datetime.fromtimestamp(ts, _TZ) if _TZ else datetime.fromtimestamp(ts)
    stamp = dt.strftime("%a %H:%M" if with_day else "%H:%M")
    return stamp + (" ET" if _TZ else "")


def _ago(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds}s ago"
    if seconds < 5400:
        return f"{round(seconds / 60)} min ago"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h ago"
    return f"{seconds // 86400} d ago"


# ---------------------------------------------------------------------------
# reading one transcript
# ---------------------------------------------------------------------------

def _lines(path: Path, tail: bool, limit: int) -> List[str]:
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            if tail and size > limit:
                fh.seek(size - limit)
                fh.readline()  # drop the partial first line
            data = fh.read(limit if not tail else size)
    except OSError:
        return []
    return data.decode("utf-8", "replace").splitlines()


def _loads(line: str) -> Optional[Dict[str, Any]]:
    if not line.startswith("{"):
        return None
    try:
        return json.loads(line)
    except ValueError:
        return None


def _user_text(msg: Dict[str, Any]) -> Optional[str]:
    """What the human typed in this user line, or None when it is a tool result or harness text."""
    if msg.get("isMeta") or msg.get("isSidechain"):
        return None
    content = (msg.get("message") or {}).get("content")
    if isinstance(content, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return None
        content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    if not isinstance(content, str):
        return None
    text = _SYSTEM_BLOCK.sub("", content).strip()
    if not text or text.startswith(_NOT_AN_ASK):
        return None
    return text


def _assistant_parts(msg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"text": "", "tool": ""}
    blocks = (msg.get("message") or {}).get("content")
    if isinstance(blocks, str):
        out["text"] = blocks
        return out
    for b in blocks or []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text" and b.get("text", "").strip():
            out["text"] = b["text"]
        elif b.get("type") == "tool_use":
            inp = b.get("input") or {}
            hint = inp.get("description") or inp.get("command") or inp.get("file_path") or inp.get("pattern") or inp.get("prompt") or ""
            out["tool"] = f"{b.get('name', 'tool')}: {_clip(str(hint), 90)}" if hint else str(b.get("name", "tool"))
    return out


def read_transcript(path: Path, turns: int = 6, turn_chars: int = TURN_CHARS) -> Dict[str, Any]:
    """Compact facts about one transcript: asks, last reply, last tool, recent turns, title, cwd."""
    info = _read_tail(path, turns, turn_chars, TAIL_BYTES)
    if not info["asks"] and path.stat().st_size > TAIL_BYTES:   # a long run of tool output: look further back
        info = _read_tail(path, turns, turn_chars, TAIL_BYTES * 6)
    return info


def _read_tail(path: Path, turns: int, turn_chars: int, tail_bytes: int) -> Dict[str, Any]:
    info: Dict[str, Any] = {"asks": [], "turns": [], "gist": "", "last_tool": "", "title": "", "cwd": "",
                            "entrypoint": "", "first_ask": ""}
    asks: List[Dict[str, Any]] = []
    recent: List[Dict[str, Any]] = []
    tail = _lines(path, True, tail_bytes)
    for raw in tail:
        d = _loads(raw)
        if not d:
            continue
        typ = d.get("type")
        if typ == "ai-title" and d.get("aiTitle"):
            info["title"] = d["aiTitle"]
        if d.get("cwd"):
            info["cwd"] = d["cwd"]
        if d.get("entrypoint"):
            info["entrypoint"] = d["entrypoint"]
        ts = d.get("timestamp") or ""
        if typ == "user":
            text = _user_text(d)
            if text:
                asks.append({"at": ts, "text": _clip(text, ASK_CHARS)})
                recent.append({"role": "aaron", "at": ts, "text": _clip(text, turn_chars)})
        elif typ == "assistant" and not d.get("isSidechain"):
            parts = _assistant_parts(d)
            if parts["tool"]:
                info["last_tool"] = parts["tool"]
            if parts["text"].strip():
                info["gist"] = _clip(parts["text"], GIST_CHARS)
                recent.append({"role": "claude", "at": ts, "text": _clip(parts["text"], turn_chars)})
    info["asks"] = asks[-3:]
    info["turns"] = recent[-turns:]
    for raw in _lines(path, False, HEAD_BYTES):
        d = _loads(raw)
        if d and d.get("type") == "user":
            text = _user_text(d)
            if text:
                info["first_ask"] = _clip(text, ASK_CHARS)
                break
    if not info["title"]:
        for raw in reversed(tail):
            if '"last-prompt"' in raw:
                d = _loads(raw)
                if d and d.get("lastPrompt"):
                    info["title"] = _clip(d["lastPrompt"], 80)
                    break
    return info


# ---------------------------------------------------------------------------
# the live-process registry and the snapshot
# ---------------------------------------------------------------------------

def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        return True
    except (OSError, ValueError, TypeError):
        return False


def _registry() -> Dict[str, Dict[str, Any]]:
    """session id -> its newest live registry record (dead processes are dropped)."""
    out: Dict[str, Dict[str, Any]] = {}
    for p in (_claude_dir() / "sessions").glob("*.json"):
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        sid = rec.get("sessionId")
        if not sid or not _pid_alive(rec.get("pid")):
            continue
        if sid not in out or (rec.get("updatedAt") or rec.get("startedAt") or 0) > (out[sid].get("updatedAt") or out[sid].get("startedAt") or 0):
            out[sid] = rec
    return out


def _project_label(cwd: str, project_dir: str) -> str:
    home = str(Path.home())
    if cwd:
        return "~" + cwd[len(home):] if cwd.startswith(home) else cwd
    return project_dir


def snapshot(label: str, hours: float = 24, turns: int = 6, now: Optional[float] = None) -> Dict[str, Any]:
    now = now or time.time()
    reg = _registry()
    cutoff = now - hours * 3600
    found: List[Dict[str, Any]] = []
    for path in (_claude_dir() / "projects").glob("*/*.jsonl"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        sid = path.stem
        live = reg.get(sid)
        if mtime < cutoff and not live:
            continue
        found.append({"sid": sid, "path": path, "mtime": mtime, "live": live})
    found.sort(key=lambda r: r["mtime"], reverse=True)
    sessions: List[Dict[str, Any]] = []
    for r in found[: MAX_SESSIONS * 2]:
        info = read_transcript(r["path"], turns=turns)
        live = r["live"]
        entry = info["entrypoint"] or (live or {}).get("entrypoint") or ""
        if not info["asks"] and not live:
            continue
        idle_s = now - r["mtime"]
        if live:
            raw_status = live.get("status") or ""
            if raw_status:
                state = "working" if raw_status == "busy" else "idle"
            else:   # headless runs keep no busy/idle flag: a very recent write means working
                state = "working" if idle_s < WORKING_IF_WROTE_WITHIN_S else "idle"
        else:
            state = "working" if idle_s < WORKING_IF_WROTE_WITHIN_S else "ended"
        sessions.append({
            "id": r["sid"],
            "name": (live or {}).get("name", ""),
            "title": info["title"],
            "cwd": info["cwd"],
            "project": _project_label(info["cwd"], r["path"].parent.name),
            "entrypoint": entry,
            "automated": entry in ("sdk-py", "sdk-ts"),   # Blatbot's own runs; a headless `claude -p` of Aaron's still counts
            "state": state,
            "process_alive": bool(live),
            "started_at": _iso((live or {}).get("startedAt", 0) / 1000) if live and live.get("startedAt") else "",
            "last_activity": _iso(r["mtime"]),
            "latest_ask": (info["asks"][-1]["text"] if info["asks"] else ""),
            "latest_ask_at": (info["asks"][-1]["at"] if info["asks"] else ""),
            "gist": info["gist"],
            "last_tool": info["last_tool"],
            "first_ask": info["first_ask"],
            "asks": info["asks"],
            "turns": info["turns"],
        })
    return {"machine": label, "generated_at": _iso(now), "window_hours": hours, "sessions": sessions[:MAX_SESSIONS]}


# ---------------------------------------------------------------------------
# the Mac snapshot, as pushed to black-sky
# ---------------------------------------------------------------------------

def mac_file() -> Path:
    return state_dir() / "mac.json"


def load_mac(now: Optional[float] = None) -> Dict[str, Any]:
    """The last snapshot the Mac pushed, with how old it is. Never raises."""
    now = now or time.time()
    path = mac_file()
    try:
        data = json.loads(path.read_text())
        received = path.stat().st_mtime
    except (OSError, ValueError):
        return {"present": False}
    try:
        generated = _parse_iso(data.get("generated_at", ""))
    except ValueError:
        generated = received
    data["present"] = True
    data["seen_at"] = min(received, generated) if generated else received
    data["age_s"] = now - data["seen_at"]
    data["stale"] = data["age_s"] > STALE_AFTER_S
    return data


# ---------------------------------------------------------------------------
# what the Blatbot tool returns
# ---------------------------------------------------------------------------

def _line_for(n: int, s: Dict[str, Any], machine: str, now: float, as_of: float) -> str:
    last = _parse_iso(s["last_activity"])
    lead = {"working": "WORKING", "idle": "idle", "ended": "no live process"}.get(s["state"], s["state"])
    if machine == "Mac" and as_of < now - STALE_AFTER_S:
        lead = f"{lead} (as of the last Mac snapshot)"
    title = s.get("title") or s.get("name") or ""
    head = f"{n}. [{machine}] {lead} | {s['project']} | last activity {_clock(last)} ({_ago(now - last)}) | id {s['id'][:8]}"
    if title:
        head += f" | {title}"
    out = [head]
    if s.get("latest_ask"):
        out.append(f"   Aaron's latest ask: {s['latest_ask']}")
    if s["state"] == "working" and s.get("last_tool"):
        out.append(f"   Doing now: {s['last_tool']}")
    if s.get("gist"):
        out.append(f"   Last said: {s['gist']}")
    return "\n".join(out)


def _detail(s: Dict[str, Any], machine: str, now: float) -> str:
    last = _parse_iso(s["last_activity"])
    lines = [f"[{machine}] session {s['id']}" + (f" ({s['name']})" if s.get("name") else ""),
             f"Title: {s.get('title') or '(none)'}",
             f"Working directory: {s.get('cwd') or s['project']}",
             f"State: {s['state']} (process {'alive' if s.get('process_alive') else 'not running'}); "
             f"last activity {_clock(last, True)} ({_ago(now - last)})"]
    if s.get("started_at"):
        lines.append(f"Started: {_clock(_parse_iso(s['started_at']), True)}")
    if s.get("first_ask"):
        lines.append(f"First ask: {s['first_ask']}")
    if s.get("asks"):
        lines.append("Recent asks:")
        for a in s["asks"]:
            when = _clock(_parse_iso(a["at"][:19] + "Z"), True) if a.get("at") else ""
            lines.append(f"  - {when} {a['text']}")
    if s.get("last_tool"):
        lines.append(f"Last tool use: {s['last_tool']}")
    if s.get("turns"):
        lines.append("Last turns, oldest first:")
        for t in s["turns"]:
            lines.append(f"  {'Aaron' if t['role'] == 'aaron' else 'Claude'}: {t['text']}")
    return "\n".join(lines)


def report(args: Dict[str, Any], now: Optional[float] = None) -> str:
    """The tool's answer. args: hours (default 6), machine (all|mac|black-sky), session (id prefix, name or
    part of a directory: gives that one session in detail), include_automated (Blatbot's own runs)."""
    now = now or time.time()
    hours = min(max(float(args.get("hours") or 6), 0.1), 72.0)
    which = str(args.get("machine") or "all").lower().replace("_", "-")
    want = str(args.get("session") or "").strip().lower()
    show_auto = bool(args.get("include_automated"))
    notes: List[str] = []
    pool: List[Dict[str, Any]] = []   # (machine, session, as_of)

    if which in ("all", "mac"):
        mac = load_mac(now)
        if not mac.get("present"):
            notes.append("Mac: no snapshot has arrived from the Mac yet (its push job may not be loaded or the Mac is off). "
                         "Nothing is known about Mac sessions.")
        else:
            seen = mac["seen_at"]
            if mac["stale"]:
                notes.append(f"Mac last seen at {_clock(seen, True)} ({_ago(now - seen)}). The Mac may be asleep or offline; "
                             "its sessions below are as they were then, not current.")
            else:
                notes.append(f"Mac: snapshot from {_clock(seen)} ({_ago(now - seen)}).")
            for s in mac.get("sessions", []):
                pool.append({"machine": "Mac", "s": s, "as_of": seen})
    if which in ("all", "black-sky", "blacksky", "server"):
        try:
            local = snapshot("black-sky", hours=max(hours, 1), turns=12, now=now)
            for s in local["sessions"]:
                pool.append({"machine": "black-sky", "s": s, "as_of": now})
            notes.append(f"black-sky: read live at {_clock(now)}.")
        except Exception as exc:  # never break the run over this
            notes.append(f"black-sky: could not read its own sessions ({exc}).")

    cutoff = now - hours * 3600
    rows = [r for r in pool if _parse_iso(r["s"]["last_activity"]) >= cutoff or r["s"]["state"] == "working"
            or r["s"].get("process_alive")]
    hidden = sum(1 for r in rows if r["s"].get("automated"))
    if not show_auto and not want:
        rows = [r for r in rows if not r["s"].get("automated")]
    rows.sort(key=lambda r: _parse_iso(r["s"]["last_activity"]), reverse=True)

    if want:
        hits = [r for r in pool if r["s"]["id"].lower().startswith(want) or want in (r["s"].get("name") or "").lower()
                or want in (r["s"].get("project") or "").lower() or want in (r["s"].get("title") or "").lower()]
        hits.sort(key=lambda r: _parse_iso(r["s"]["last_activity"]), reverse=True)
        if not hits:
            return "\n".join(notes + [f"No session matches '{want}' in the last {hours:g} hours."])
        body = [_detail(h["s"], h["machine"], now) for h in hits[:3]]
        if len(hits) > 3:
            body.append(f"({len(hits) - 3} more match; give a longer id prefix to pick one.)")
        return "\n".join(notes) + "\n\n" + "\n\n".join(body)

    out = [f"Claude Code sessions active in the last {hours:g} h (now {_clock(now)}):", *notes, ""]
    if not rows:
        out.append("No Claude Code session has been active in that window on the machines that could be read.")
    for i, r in enumerate(rows, 1):
        out.append(_line_for(i, r["s"], r["machine"], now, r["as_of"]))
    if hidden and not show_auto:
        out.append(f"\n({hidden} automated run(s) of Blatbot's own were left out; ask with include_automated to see them.)")
    out.append("\nExcerpts are cut short on purpose. Give session=<id prefix> for one session in more detail.")
    return "\n".join(out)


def _main(argv: List[str]) -> int:
    if len(argv) >= 2 and argv[1] == "snapshot":
        label = "Mac"
        hours = 24.0
        for i, a in enumerate(argv):
            if a == "--label" and i + 1 < len(argv):
                label = argv[i + 1]
            if a == "--hours" and i + 1 < len(argv):
                hours = float(argv[i + 1])
        json.dump(snapshot(label, hours=hours), sys.stdout, ensure_ascii=False, separators=(",", ":"))
        return 0
    if len(argv) >= 2 and argv[1] == "show":
        print(report({"machine": "black-sky" if "--local" in argv else "all",
                      "hours": 6, "session": next((argv[i + 1] for i, a in enumerate(argv) if a == "--session"), "")}))
        return 0
    print("usage: ccsessions.py snapshot [--label Mac] [--hours 24] | show [--local] [--session ID]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
