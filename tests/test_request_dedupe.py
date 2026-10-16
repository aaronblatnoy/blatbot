import pytest

from inkbox_claude.gate.store import Store


def _create(store, task_id, *, original_message="same handoff", prompt="same prompt"):
    return store.create_request(
        chat_id="call-1", sender="caller", sender_name="Caller", mode="voice", subject="",
        original_message=original_message, summary="handle the ask", scopes=["web"],
        prompt=prompt, state="approved", task_id=task_id,
    )


@pytest.mark.parametrize(("original_message", "prompt"), [
    ("different transcript", "same prompt"),
    ("same handoff", "different prompt"),
])
def test_matching_message_or_prompt_returns_existing_request(tmp_path, original_message, prompt):
    store = Store(str(tmp_path / "gate.db"))
    task = store.create_task("handle the ask")
    first = _create(store, task["id"])

    duplicate = _create(store, task["id"], original_message=original_message, prompt=prompt)

    assert duplicate.id == first.id
    count = store._db.execute("SELECT COUNT(*) AS count FROM requests").fetchone()["count"]
    assert count == 1


def test_handoff_older_than_ten_minutes_is_created_again(tmp_path):
    store = Store(str(tmp_path / "gate.db"))
    task = store.create_task("handle the ask")
    first = _create(store, task["id"])
    store._db.execute("UPDATE requests SET created_at=created_at-601 WHERE id=?", (first.id,))
    store._db.commit()

    fresh = _create(store, task["id"])

    assert fresh.id != first.id


def test_terminal_non_quota_failure_is_not_deduped(tmp_path):
    store = Store(str(tmp_path / "gate.db"))
    task = store.create_task("handle the ask")
    first = _create(store, task["id"])
    store.set_state(first.id, "failed", status={"ok": False, "error": "network unavailable"})

    fresh = _create(store, task["id"])

    assert fresh.id != first.id


def test_recent_quota_failed_handoff_is_deduped(tmp_path):
    store = Store(str(tmp_path / "gate.db"))
    task = store.create_task("handle the ask")
    first = _create(store, task["id"])
    store.set_state(first.id, "failed", status={"ok": False, "quota_error": True})

    duplicate = _create(store, task["id"])

    assert duplicate.id == first.id
