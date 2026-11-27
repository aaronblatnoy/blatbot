"""Instance profiles (inkbox_claude/profile.py): one codebase, several assistants."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from inkbox_claude import profile

ROOT = Path(__file__).resolve().parents[1]
INSTANCES = ROOT / "deploy" / "instances"


def _write(tmp_path: Path, text: str, name: str = "p.yaml") -> str:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_precedence_real_env_then_env_then_secret_then_env_file(tmp_path):
    (tmp_path / "s").write_text("from-secret\n")
    (tmp_path / "e.env").write_text("# c\nA=from-envfile\nB=from-envfile\nC=from-envfile\nexport D='q'\n")
    path = _write(tmp_path, f"""
name: test
env_files: [{tmp_path}/e.env]
secret_files: {{B: {tmp_path}/s, C: {tmp_path}/s}}
env: {{C: from-env, E: "${{INSTANCE_DIR}}/x"}}
""")
    environ = {"A": "real"}
    done = profile.apply(profile.read(path), environ)
    assert environ["A"] == "real" and "A" not in done           # the real environment always wins
    assert environ["B"] == "from-secret"                        # secret file beats env file
    assert environ["C"] == "from-env"                           # env beats secret file
    assert environ["D"] == "q"
    assert environ["E"] == f"{tmp_path.resolve()}/x"           # ${INSTANCE_DIR} = the profile's folder
    assert environ["INSTANCE_NAME"] == "test"


def test_missing_files_are_skipped_not_fatal(tmp_path):
    path = _write(tmp_path, f"env_files: [{tmp_path}/nope.env]\nsecret_files: {{X: {tmp_path}/nope}}\nenv: {{Y: '1'}}\n")
    environ: dict = {}
    profile.apply(profile.read(path), environ)
    assert environ == {"Y": "1"}


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        profile.read(_write(tmp_path, "nmae: typo\n"))


def test_apply_from_environment_unset_does_nothing():
    environ = {"PATH": "/bin"}
    assert profile.apply_from_environment(environ) == {}
    assert environ == {"PATH": "/bin"}


def test_bad_profile_never_raises(tmp_path):
    environ = {"INSTANCE_PROFILE": str(tmp_path / "missing.yaml")}
    assert profile.apply_from_environment(environ) == {}


def test_blatbot_profile_is_todays_environment(tmp_path, monkeypatch):
    """Blatbot's profile loads ~/blatbot/.env and nothing else: on top of the environment the
    live unit already has (that same file), it changes nothing."""
    home = tmp_path / "home"
    (home / "blatbot").mkdir(parents=True)
    (home / "blatbot" / ".env").write_text("ASSISTANT_NAME=Blatbot\nINKBOX_BRIDGE_PORT=8771\n")
    monkeypatch.setattr(profile, "_user_home", lambda: str(home))
    live = dict(profile.parse_env_file(str(home / "blatbot" / ".env")))
    live["CLAUDE_MODEL"] = "sonnet"            # exported by ~/blatbot/watchdog.sh
    environ = dict(live)
    assert profile.apply(profile.read(str(INSTANCES / "blatbot.yaml")), environ) == {"INSTANCE_NAME": "profile name"}
    environ.pop("INSTANCE_NAME")
    assert environ == live


def test_no_profile_means_blatbot_identity():
    env = {k: v for k, v in os.environ.items()
           if k not in ("INSTANCE_PROFILE", "ASSISTANT_NAME", "ASSISTANT_EMAIL", "SERVER_NAME", "GRAPHITI_GROUP")}
    out = subprocess.run([sys.executable, "-c",
                          "from inkbox_claude import instance as i; print(i.ASSISTANT, i.ASSISTANT_EMAIL, i.SERVER_NAME, i.GRAPHITI_GROUP)"],
                         cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout.split()
    assert out == ["Blatbot", "blatbot@inkboxmail.com", "black-sky", "blatbot"]


def test_gondroid_profile_applies_before_modules_read_it(tmp_path):
    """The profile is applied when the package is imported, before instance.py reads it."""
    env = {k: v for k, v in os.environ.items() if k not in ("ASSISTANT_NAME", "INKBOX_BRIDGE_PORT")}
    # A separate HOME for the instance's runs does not move its files: ~ is the account's home.
    env.update({"HOME": str(tmp_path), "INSTANCE_PROFILE": str(INSTANCES / "gondroid.yaml")})
    out = subprocess.run([sys.executable, "-c",
                          "import os; from inkbox_claude import instance as i; "
                          "print(i.ASSISTANT, os.environ['INKBOX_BRIDGE_PORT'], os.environ['GATE_DB_PATH'], "
                          "os.environ['GATE_SCOPES_FILE'])"],
                         cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout.split()
    assert out[:3] == ["Gondroid", "8772", f"{profile._user_home()}/gondroid/gate.db"]
    assert Path(out[3]).is_file()                             # the scope registry it names exists


def test_instance_profiles_parse():
    for p in INSTANCES.glob("*.yaml"):
        data = profile.read(str(p))
        assert data.get("name") == p.stem
