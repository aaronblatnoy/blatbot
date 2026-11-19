"""Dev work from chat: Aaron asks, Claude Code makes the change and merges the PR.

Aaron types, in his own direct chat with Blatbot:

    dev <repo>: <what to change>

for example "dev blatbot: make the stop reply say which request stopped" or
"dev aaronblatnoy/some-repo: fix the typo in the README". "dev work <repo>: ..." and
"claude code <repo>: ..." work the same. A repo without an owner is taken under
GATE_DEV_OWNER (default aaronblatnoy).

What happens, by code, never by a model's choice:

* Only the verified approver (is_approver(): his iMessage conversation id or his
  Telegram user id) may start it, and only in a direct chat. Anyone else typing the
  phrase gets one refusal line and nothing runs.
* Blatbot answers with one "started" line, then runs Claude Code headless
  (`claude -p`) in a fresh git worktree under ~/devwork/wt/<id>, on a new branch
  cut from the repo's default branch. Claude Code runs with the user's normal
  Claude Code settings on black-sky (network and gh allowed), and is told to make
  the change, push, open a ready PR, wait for checks and squash-merge when green.
* The run has a hard time limit (GATE_DEV_TIMEOUT_S, default 45 minutes); at the
  limit the process group is killed and the run reports a failure.
* The result is checked against GitHub (gh pr view on the branch), not taken from
  the model's word: Blatbot sends the PR link and a one-line summary, or the failure.
* Repos whose name says "bullpen" run only when his message for that task names
  Bullpen (standing rule: no Bullpen work on black-sky unless he asks for it then).
* One run at a time; a second ask while one runs is told so and nothing starts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_CMD_RE = re.compile(
    r"^\s*(?:dev(?:\s+work)?|claude\s+code)\s+(?:in\s+|on\s+)?"
    r"(?P<repo>[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)?)\s*:\s*(?P<task>\S.*)$",
    re.I | re.S)

_REPO_PART = re.compile(r"^[A-Za-z0-9_.\-]+$")

_lock = asyncio.Lock()


@dataclass
class DevAsk:
    owner: str
    name: str
    task: str
    text: str

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"


@dataclass
class DevResult:
    ok: bool
    pr_url: str = ""
    summary: str = ""
    error: str = ""
    seconds: float = 0.0

    def line(self) -> str:
        if self.ok:
            return f"Merged: {self.pr_url} {self.summary}".strip()
        if self.pr_url:
            return f"Not merged: {self.pr_url} {self.error}".strip()
        return f"Dev work failed: {self.error}".strip()


def match(text: str) -> Optional[DevAsk]:
    """The dev-work phrase, or None. Pure text matching, no model."""
    m = _CMD_RE.match(text or "")
    if not m:
        return None
    repo = m.group("repo").strip().rstrip(".")
    task = m.group("task").strip()
    if "/" in repo:
        owner, name = repo.split("/", 1)
    else:
        owner, name = (os.getenv("GATE_DEV_OWNER") or "aaronblatnoy"), repo
    if name.endswith(".git"):
        name = name[:-4]
    if not (_REPO_PART.match(owner) and _REPO_PART.match(name)) or name in (".", "..") or len(task) < 3:
        return None
    return DevAsk(owner=owner, name=name, task=task, text=text)


def bullpen_blocked(ask: DevAsk) -> bool:
    """A Bullpen repo runs on black-sky only when his message names Bullpen."""
    if "bullpen" not in ask.name.lower():
        return False
    return "bullpen" not in ask.text.lower()


def busy() -> bool:
    return _lock.locked()


def base_dir() -> Path:
    return Path(os.getenv("GATE_DEV_DIR") or (Path.home() / "devwork"))


def _timeout() -> float:
    try:
        return float(os.getenv("GATE_DEV_TIMEOUT_S") or 45 * 60)
    except ValueError:
        return 45 * 60.0


def _bin(name: str) -> str:
    home = Path.home()
    for cand in (home / ".local/bin" / name, home / "bin" / name, Path("/usr/local/bin") / name,
                 Path("/usr/bin") / name):
        if cand.exists():
            return str(cand)
    return shutil.which(name) or name


def _env() -> Dict[str, str]:
    """A clean environment: none of Blatbot's own secrets or settings reach the run."""
    home = str(Path.home())
    keep = {k: os.environ[k] for k in ("LANG", "LC_ALL", "USER", "LOGNAME", "SSH_AUTH_SOCK",
                                       "CLAUDE_CODE_OAUTH_TOKEN") if os.environ.get(k)}
    keep.update({
        "HOME": home,
        "PATH": f"{home}/bin:{home}/.local/bin:/usr/local/bin:/usr/bin:/bin",
        "JEV_NO_DISPATCH": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GH_PROMPT_DISABLED": "1",
        "TERM": "dumb",
    })
    return keep


async def _run(cmd: list, cwd: Optional[str] = None, timeout: float = 300.0) -> tuple:
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd, env=_env(), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=True)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        _kill(proc)
        await proc.wait()
        return 124, "", f"timed out after {int(timeout)}s"
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


def _kill(proc: Any) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _prompt(ask: DevAsk, branch: str, default_branch: str) -> str:
    return f"""You are doing one piece of dev work Aaron asked for by chat. Work only in this
git worktree (repo {ask.slug}, branch {branch}, cut from {default_branch}). Do not touch any
other directory, repo or branch.

His request, in his words:
{ask.task}

Do all of this yourself, without asking anything (nobody can answer):
1. Read the repo's CLAUDE.md / README first if present, then make the change. Keep it to
   what he asked.
2. Run the repo's quick checks if there are any (tests, lint) and fix what you broke.
3. Commit on {branch} with a clear message, then `git push -u origin {branch}`.
4. Open a ready (not draft) PR into {default_branch} with `gh pr create`.
5. Wait for the PR's checks (`gh pr checks --watch`, skip if the repo has none). If a check
   fails, read why, fix it, push again. When checks pass (or there are none), merge with
   `gh pr merge --squash --delete-branch`.
6. Never send email, SMS or chat messages, never deploy or restart services, never force-push
   to {default_branch}.

Your LAST line of output must be exactly one of:
RESULT: <pr url> | <one-line summary of what changed>
FAILED: <one-line reason>
"""


async def _default_branch(mirror: Path) -> str:
    code, out, _ = await _run(["git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"], cwd=str(mirror))
    if code == 0 and out.strip().startswith("origin/"):
        return out.strip().split("/", 1)[1]
    await _run(["git", "remote", "set-head", "origin", "--auto"], cwd=str(mirror))
    code, out, _ = await _run(["git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"], cwd=str(mirror))
    return out.strip().split("/", 1)[1] if code == 0 and "/" in out else "main"


async def _prepare(ask: DevAsk, run_id: str) -> tuple:
    """Mirror clone (kept between runs) + a fresh worktree on a new branch."""
    root = base_dir()
    (root / "repos").mkdir(parents=True, exist_ok=True)
    (root / "wt").mkdir(parents=True, exist_ok=True)
    mirror = root / "repos" / f"{ask.owner}__{ask.name}"
    if not (mirror / ".git").exists():
        code, _, err = await _run(["git", "clone", "-q", f"git@github.com:{ask.slug}.git", str(mirror)], timeout=600)
        if code != 0:
            raise RuntimeError(f"could not clone {ask.slug}: {err.strip()[:200]}")
    code, _, err = await _run(["git", "fetch", "-q", "--prune", "origin"], cwd=str(mirror), timeout=300)
    if code != 0:
        raise RuntimeError(f"could not fetch {ask.slug}: {err.strip()[:200]}")
    default = await _default_branch(mirror)
    branch = f"blatbot/dev-{run_id}"
    wt = root / "wt" / run_id
    code, _, err = await _run(["git", "worktree", "add", "-q", "-b", branch, str(wt), f"origin/{default}"],
                              cwd=str(mirror))
    if code != 0:
        raise RuntimeError(f"could not make a worktree: {err.strip()[:200]}")
    return mirror, wt, branch, default


async def _cleanup(mirror: Path, wt: Path) -> None:
    await _run(["git", "worktree", "remove", "--force", str(wt)], cwd=str(mirror))
    await _run(["git", "worktree", "prune"], cwd=str(mirror))


def _last_line(text: str) -> str:
    for line in reversed((text or "").strip().splitlines()):
        if line.strip():
            return line.strip()
    return ""


async def _pr_state(ask: DevAsk, branch: str) -> Dict[str, Any]:
    code, out, _ = await _run([_bin("gh"), "pr", "list", "-R", ask.slug, "--head", branch, "--state", "all",
                               "--json", "url,state,title,mergedAt", "-L", "1"])
    if code != 0:
        return {}
    try:
        rows = json.loads(out or "[]")
    except ValueError:
        return {}
    return rows[0] if rows else {}


async def run(ask: DevAsk, *, run_id: Optional[str] = None) -> DevResult:
    """Run one dev task end to end. Never raises; a failure comes back as DevResult."""
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    started = time.time()
    async with _lock:
        log_dir = base_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{run_id}.log"
        try:
            mirror, wt, branch, default = await _prepare(ask, run_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[dev %s] prepare failed: %s", run_id, exc)
            return DevResult(ok=False, error=str(exc), seconds=time.time() - started)
        logger.info("[dev %s] %s on %s (from %s), worktree %s", run_id, ask.slug, branch, default, wt)
        cmd = [_bin("claude"), "-p", _prompt(ask, branch, default), "--output-format", "json"]
        model = (os.getenv("GATE_DEV_MODEL") or "").strip()
        if model:
            cmd += ["--model", model]
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(wt), env=_env(), stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=_timeout())
        except asyncio.TimeoutError:
            timed_out = True
            _kill(proc)
            out, _ = await proc.communicate()
        text = (out or b"").decode(errors="replace")
        log_path.write_text(text)
        final = text
        try:
            final = str(json.loads(text).get("result") or "")
        except ValueError:
            pass
        last = _last_line(final)
        pr = await _pr_state(ask, branch)
        await _cleanup(mirror, wt)
        seconds = time.time() - started
        summary = ""
        if last.upper().startswith("RESULT:"):
            summary = last.split("|", 1)[1].strip() if "|" in last else ""
        if pr.get("state") == "MERGED":
            logger.info("[dev %s] merged %s in %ds", run_id, pr.get("url"), seconds)
            return DevResult(ok=True, pr_url=pr.get("url", ""), summary=summary or pr.get("title", ""),
                             seconds=seconds)
        if timed_out:
            err = f"stopped at the {int(_timeout() // 60)} minute limit"
        elif last.upper().startswith("FAILED:"):
            err = last.split(":", 1)[1].strip()
        elif pr:
            err = f"the PR is {str(pr.get('state', '')).lower()}, not merged"
        else:
            err = f"Claude Code ended without a PR (exit {proc.returncode}); log {log_path}"
        logger.warning("[dev %s] not merged: %s", run_id, err)
        return DevResult(ok=False, pr_url=pr.get("url", ""), error=err, seconds=seconds)


def started_line(ask: DevAsk) -> str:
    return f"Started dev work on {ask.slug}. I will send the PR link when it is merged."


def _cli() -> None:
    """Direct executor test: python -m inkbox_claude.gate.devwork 'dev owner/repo: task'"""
    import sys
    logging.basicConfig(level=logging.INFO)
    ask = match(" ".join(sys.argv[1:]))
    if ask is None:
        raise SystemExit("usage: devwork 'dev <owner/repo>: <task>'")
    if bullpen_blocked(ask):
        raise SystemExit("Bullpen repo not named in the message; refused")
    result = asyncio.run(run(ask))
    print(result.line())
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    _cli()
