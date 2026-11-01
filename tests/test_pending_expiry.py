import time

from inkbox_claude.gate.store import Store

DAY = 24 * 3600


def _req(s, state="pending"):
    task = s.create_task("expiry test")
    return s.create_request(chat_id="c1", sender="+15550100001", sender_name="Sam", mode="imessage",
                            subject="", original_message="hi", summary="do a thing", scopes=[],
                            prompt="do a thing", state=state, task_id=task["id"])


def _backdate(s, rid, created_ago, updated_ago):
    now = time.time()
    s._db.execute("UPDATE requests SET created_at=?, updated_at=? WHERE id=?",
                  (now - created_ago, now - updated_ago, rid))
    s._db.commit()


def test_reparked_confirmation_does_not_expire_from_creation_time(tmp_path):
    s = Store(str(tmp_path / "g.db"))
    r = _req(s, state="running")
    s.set_state(r.id, "pending", status={"confirm": {"tool": "x", "args": {}}})  # what ask_confirmation does
    _backdate(s, r.id, 25 * 3600, 60)   # arrived over a day ago, asked a minute ago
    assert s.expire_older_than(DAY) == []
    assert s.get_request(r.id).state == "pending"


def test_revised_request_does_not_expire_from_creation_time(tmp_path):
    s = Store(str(tmp_path / "g.db"))
    r = _req(s)
    _backdate(s, r.id, 23 * 3600, 23 * 3600)
    s.revise_prompt(r.id, "do a thing, but differently")
    s._db.execute("UPDATE requests SET created_at=? WHERE id=?", (time.time() - 25 * 3600, r.id))
    s._db.commit()
    assert s.expire_older_than(DAY) == []
    assert s.get_request(r.id).state == "pending"


def test_request_waiting_a_full_day_still_expires(tmp_path):
    s = Store(str(tmp_path / "g.db"))
    r = _req(s)
    _backdate(s, r.id, 25 * 3600, 25 * 3600)
    assert [x.id for x in s.expire_older_than(DAY)] == [r.id]
    assert s.get_request(r.id).state == "expired"
