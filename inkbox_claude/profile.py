"""Instance profiles: one codebase, several assistants (Blatbot, Gondroid, ...).

Every setting of a running instance is an environment variable (see instance.py for the
identity ones, config.py for the gateway ones, gate/settings.py for the gate's). A profile is a
checked-in YAML file that names one instance's settings in one place, so a second assistant
runs from this same code with nothing but a different profile:

    name: gondroid                      # the instance's name (exported as INSTANCE_NAME)
    env_files: [~/gondroid/.env]        # KEY=VALUE files, typically the secrets; optional
    secret_files:                       # VAR: file whose whole content (stripped) is the value
      TELEGRAM_BOT_TOKEN: ~/.config/definite-optimists/gondroid-telegram-token
    env:                                # plain values, checked in
      ASSISTANT_NAME: Gondroid
      INKBOX_BRIDGE_PORT: "8772"

How it is applied: when INSTANCE_PROFILE names a file, importing the package applies it before
any module reads the environment. Nothing ever overrides a variable the process already has
(the real environment wins); after that `env` wins over `secret_files`, which wins over
`env_files`. `~` and `${VAR}` are expanded in file paths and in `env` values. A missing
secret or env file is skipped with a warning, never an exception: a profile must not stop the
process from starting.

With INSTANCE_PROFILE unset nothing happens at all, which is how Blatbot runs today.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Tuple

logger = logging.getLogger(__name__)

PROFILE_ENV = "INSTANCE_PROFILE"
_KNOWN_KEYS = {"name", "env", "env_files", "secret_files", "description"}

# What the last apply() did, for `inkbox-claude profile` and the console: var -> source.
applied: Dict[str, str] = {}
active_path: str = ""


_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: str, extra: Optional[Dict[str, str]] = None) -> str:
    """`~` and `${VAR}` expanded; VAR comes from `extra` first (INSTANCE_DIR), then the
    environment, and an unknown one is left as written."""
    def sub(m: "re.Match[str]") -> str:
        name = m.group(1)
        got = (extra or {}).get(name, os.environ.get(name))
        return m.group(0) if got is None else got
    return os.path.expanduser(_VAR.sub(sub, str(value)))


def read(path: str) -> Dict[str, Any]:
    """The profile file as a dict. Raises ValueError for a file that is not a profile."""
    import yaml
    with open(_expand(path), encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a profile is a mapping")
    unknown = sorted(set(data) - _KNOWN_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown profile keys {unknown} (allowed: {sorted(_KNOWN_KEYS)})")
    for key in ("env", "secret_files"):
        if not isinstance(data.get(key) or {}, dict):
            raise ValueError(f"{path}: {key} must be a mapping")
    if not isinstance(data.get("env_files") or [], list):
        raise ValueError(f"{path}: env_files must be a list")
    data["_dir"] = str(Path(_expand(path)).resolve().parent)
    return data


def parse_env_file(path: str) -> List[Tuple[str, str]]:
    """KEY=VALUE lines of a dotenv file, in order (same rules as daemon._maybe_load_env_file)."""
    out: List[Tuple[str, str]] = []
    for line in Path(_expand(path)).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        if s.startswith("export "):
            s = s[len("export "):]
        key, value = s.split("=", 1)
        out.append((key.strip(), value.strip().strip('"').strip("'")))
    return out


def resolve(profile: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """(var, value, source) for everything the profile sets, highest precedence first.
    ${INSTANCE_DIR} is the folder the profile file is in (read() records it)."""
    out: List[Tuple[str, str, str]] = []
    extra = {"INSTANCE_DIR": str(profile.get("_dir") or "")} if profile.get("_dir") else {}
    name = str(profile.get("name") or "").strip()
    if name:
        out.append(("INSTANCE_NAME", name, "profile name"))
    for key, value in (profile.get("env") or {}).items():
        out.append((str(key), "" if value is None else _expand(value, extra), "profile env"))
    for key, path in (profile.get("secret_files") or {}).items():
        p = _expand(path, extra)
        try:
            out.append((str(key), Path(p).read_text(encoding="utf-8").strip(), f"secret file {p}"))
        except OSError as exc:
            logger.warning("[profile] %s: secret file %s not readable (%s); left unset", key, p, type(exc).__name__)
    for path in profile.get("env_files") or []:
        p = _expand(path, extra)
        try:
            out += [(k, v, f"env file {p}") for k, v in parse_env_file(p)]
        except OSError as exc:
            logger.warning("[profile] env file %s not readable (%s); skipped", p, type(exc).__name__)
    return out


def apply(profile: Dict[str, Any], environ: Optional[MutableMapping[str, str]] = None) -> Dict[str, str]:
    """Set every variable the profile names that the environment does not have yet.
    Returns var -> source for the ones it set."""
    env = os.environ if environ is None else environ
    done: Dict[str, str] = {}
    for key, value, source in resolve(profile):
        if key in env:
            continue
        env[key] = value
        done[key] = source
    return done


def apply_from_environment(environ: Optional[MutableMapping[str, str]] = None) -> Dict[str, str]:
    """Apply the profile INSTANCE_PROFILE names, if any. Never raises."""
    global applied, active_path
    env = os.environ if environ is None else environ
    path = (env.get(PROFILE_ENV) or "").strip()
    if not path:
        return {}
    try:
        done = apply(read(path), env)
    except Exception as exc:  # noqa: BLE001 - a bad profile is logged, not fatal
        logger.error("[profile] could not apply %s: %s", path, exc)
        return {}
    applied, active_path = done, path
    return done


def describe(path: str, environ: Optional[MutableMapping[str, str]] = None) -> List[str]:
    """One line per variable: name, where its value comes from, and whether the environment
    already overrides it. Values are never printed."""
    env = os.environ if environ is None else environ
    lines = []
    seen = set()
    for key, value, source in resolve(read(path)):
        if key in seen:
            lines.append(f"{key:<34} (also in {source}, lower precedence)")
            continue
        seen.add(key)
        note = "  [environment overrides]" if key in env and env.get(key) != value else ""
        lines.append(f"{key:<34} {source}{' (empty)' if value == '' else ''}{note}")
    return lines
