"""Backend switch for the suite: with STORE_BACKEND=postgres and
BLATBOT_TEST_PG_DSN set, every `Store(path)` call in the tests is routed to
its own throwaway schema inside the test Postgres database instead of a
sqlite file, so the exact code under test is the code that runs live.

Mechanism: this module patches the `Store` name on
inkbox_claude.gate.store BEFORE any test module does its own
`from inkbox_claude.gate.store import Store` (conftest.py is imported by
pytest ahead of test collection), so every such import -- module-level or
the several inline ones inside test functions -- picks up the routed
subclass. The original sqlite `path` argument a test passes (e.g.
str(tmp_path / "gate.db")) is never opened; it is only used, via a stable
hash, to pick a schema name, so two Store(...) calls with the SAME path
string within one test share state (matching sqlite's "same file = same
data" semantics) while different tests get different schemas.

Nothing here runs, or is imported, unless STORE_BACKEND=postgres is set --
with it unset (the default), the suite runs against sqlite exactly as
before this file existed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

# A handful of tests build a LEGACY sqlite file by hand (raw sqlite3.connect +
# executescript with the pre-people-model trust/people/synced_people tables,
# or a pre-schedules requests table) specifically to exercise store.py's
# one-time sqlite migrations (_migrate_people_model, _migrate_person_keyed_tasks,
# the PRAGMA-table_info ALTER-column dance). Those migrations are deliberately
# sqlite-only -- see SCHEMA_PG's comment in store.py -- a Postgres Store always
# starts from the current final shape, filled by the migration script, so
# there is no equivalent "old file" to migrate in place under Postgres. These
# tests are skipped only under STORE_BACKEND=postgres; they still run (and
# must still pass) under the default sqlite backend.
_SQLITE_ONLY_LEGACY_MIGRATION_TESTS = {
    "test_migration_from_legacy_fixture_tables",
    "test_migration_groups_two_trust_rows_sharing_a_display_name_into_one_person",
    "test_existing_database_migrates_in_place",
}


def pytest_collection_modifyitems(config, items):
    if os.environ.get("STORE_BACKEND") != "postgres":
        return
    import pytest

    skip = pytest.mark.skip(
        reason="sqlite-only legacy migration path; SCHEMA_PG starts from the "
        "final shape and has no equivalent old-file-in-place migration"
    )
    for item in items:
        if item.name in _SQLITE_ONLY_LEGACY_MIGRATION_TESTS:
            item.add_marker(skip)


if os.environ.get("STORE_BACKEND") == "postgres":
    import psycopg

    from inkbox_claude.gate import store as _store_mod

    _BASE_DSN = os.environ["BLATBOT_TEST_PG_DSN"]  # postgresql://user:pass@host:port/db
    _OrigStore = _store_mod.Store
    _seen_schemas: set[str] = set()

    def _schema_for(path: str) -> str:
        return "test_" + hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]

    def _dsn_for_schema(schema: str) -> str:
        sep = "&" if "?" in _BASE_DSN else "?"
        return f"{_BASE_DSN}{sep}options=-c%20search_path%3D{schema}"

    def _reset_schema(schema: str) -> None:
        with psycopg.connect(_BASE_DSN, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
                cur.execute(f'CREATE SCHEMA "{schema}"')

    class _PgRoutedStore(_OrigStore):  # type: ignore[misc]
        def __init__(self, path: str, *a, **kw):
            schema = _schema_for(str(path))
            if schema not in _seen_schemas:
                _reset_schema(schema)
                _seen_schemas.add(schema)
            # Some tests os.remove(path) as cleanup, expecting a real sqlite
            # file to exist there; leave a harmless empty placeholder so that
            # cleanup still succeeds under Postgres routing (never read or
            # written as data -- the real data lives in `schema` above).
            try:
                Path(path).touch(exist_ok=True)
            except OSError:
                pass
            super().__init__(_dsn_for_schema(schema), *a, **kw)

    _store_mod.Store = _PgRoutedStore  # type: ignore[assignment]
