"""Gondroid's people and roles, written into its own gate database. Idempotent; run once after the
first start and again whenever the answerer list changes:

    INSTANCE_PROFILE=~/blatbot/app/deploy/instances/gondroid.yaml \\
        ~/blatbot/app/.venv/bin/python ~/blatbot/app/deploy/instances/gondroid/seed.py

Aaron is the owner through GATE_APPROVER_TELEGRAM_ID (nothing to seed). Every other answerer in
TELEGRAM_ANSWER_ONLY (Stanley Rabkin) gets the role "team": the team brain and the public web, so
their questions run at once instead of waiting for Aaron's yes. Nobody else is given anything."""

from __future__ import annotations

import os
import sys

from inkbox_claude.config import read_config
from inkbox_claude.gate.store import Store

ROLE, ROLE_SCOPES = "team", ["memory_read", "web"]
NAMES = {"8804662817": "Stanley Rabkin"}


def main() -> int:
    store = Store(read_config().gate_db_path)
    store.set_role(ROLE, ROLE_SCOPES, note="Definite Optimists answerers: the team brain and the public web")
    owner = (os.getenv("GATE_APPROVER_TELEGRAM_ID") or "").strip()
    ids = [x.strip() for x in (os.getenv("TELEGRAM_ANSWER_ONLY") or "").split(",") if x.strip() and x.strip() != owner]
    for tid in ids:
        found = store.resolve_handles([f"telegram:{tid}"])
        if found.get("person_id"):
            store.update_person(found["person_id"], role=ROLE)
            print(f"telegram {tid}: {found.get('person') or found['person_id']} -> role {ROLE}")
            continue
        person = store.create_person(display=NAMES.get(tid, f"Telegram {tid}"), role=ROLE)
        store.add_contact(person["id"], "telegram", tid, source="gondroid seed")
        print(f"telegram {tid}: created {person['display']} with role {ROLE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
