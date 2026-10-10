"""The Postgres connection wrapper: only runs when the suite is pointed at Postgres."""

import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("STORE_BACKEND") != "postgres", reason="postgres backend only")


def _conn():
    from inkbox_claude.gate.pgdb import PGConnection
    return PGConnection(os.environ["BLATBOT_TEST_PG_DSN"])


def test_a_failed_statement_does_not_poison_the_next_one():
    db = _conn()
    with pytest.raises(Exception):
        db.execute("SELECT * FROM a_table_that_does_not_exist")
    assert db.execute("SELECT 1 AS one").fetchone()["one"] == 1
    db.close()


def test_a_dropped_connection_is_reopened():
    db = _conn()
    db._conn.close()
    assert db.execute("SELECT 2 AS two").fetchone()["two"] == 2
    db.close()
