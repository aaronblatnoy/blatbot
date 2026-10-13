"""Cadence model: next-run computation and plain-words description, including
daylight-saving and a non-drifting `every`."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from inkbox_claude.gate import cadence


def _ts(text: str, tz: str = "America/New_York") -> float:
    return datetime.fromisoformat(text).replace(tzinfo=ZoneInfo(tz)).timestamp()


def test_once_next_run_and_description():
    cad = {"at": "2026-10-12T14:30:00"}
    after = _ts("2026-10-12T00:00:00")
    nr = cadence.next_run("once", cad, "America/New_York", after)
    assert nr == _ts("2026-10-12T14:30:00")
    assert cadence.describe("once", cad, "America/New_York") == "once on Mon Oct 12 at 2:30 PM"


def test_once_already_passed_returns_none():
    cad = {"at": "2026-10-12T14:30:00"}
    after = _ts("2026-10-12T15:00:00")
    assert cadence.next_run("once", cad, "America/New_York", after) is None


def test_every_description_in_hours():
    assert cadence.describe("every", {"minutes": 240}, "America/New_York") == "every 4 hours"


def test_every_description_in_minutes():
    assert cadence.describe("every", {"minutes": 90}, "America/New_York") == "every 90 minutes"


def test_every_rejects_under_five_minutes():
    with pytest.raises(cadence.CadenceError):
        cadence.validate("every", {"minutes": 4})


def test_every_anchored_does_not_drift_by_run_duration():
    anchor = _ts("2026-10-12T14:30:00")
    cad = {"minutes": 240, "anchor": anchor}
    # A run that ran long and finished at 14:52 still wakes up at the anchored
    # 18:30, not 18:52.
    after = anchor + 22 * 60
    assert cadence.next_run("every", cad, "America/New_York", after) == anchor + 4 * 3600


def test_daily_description_and_next_run():
    cad = {"time": "08:00"}
    assert cadence.describe("daily", cad, "America/New_York") == "daily at 8:00 AM"
    after = _ts("2026-10-12T09:00:00")
    assert cadence.next_run("daily", cad, "America/New_York", after) == _ts("2026-10-13T08:00:00")


def test_weekdays_description_and_skips_weekend():
    cad = {"time": "07:30"}
    assert cadence.describe("weekdays", cad, "America/New_York") == "weekdays at 7:30 AM"
    # 2026-10-16 is a Friday; the next weekday firing after it must be Monday.
    after = _ts("2026-10-16T08:00:00")
    nr = cadence.next_run("weekdays", cad, "America/New_York", after)
    local = datetime.fromtimestamp(nr, ZoneInfo("America/New_York"))
    assert local.weekday() == 0  # Monday
    assert local.strftime("%Y-%m-%d") == "2026-10-19"


def test_weekly_description_and_next_run():
    cad = {"day": "mon", "time": "09:00"}
    assert cadence.describe("weekly", cad, "America/New_York") == "Mondays at 9:00 AM"
    after = _ts("2026-10-12T00:00:00")  # a Monday before 9am
    nr = cadence.next_run("weekly", cad, "America/New_York", after)
    local = datetime.fromtimestamp(nr, ZoneInfo("America/New_York"))
    assert local.weekday() == 0
    assert local.strftime("%H:%M") == "09:00"


def test_cron_falls_back_to_cron_wording():
    cad = {"expr": "*/15 9-17 * * 1,3,5"}
    text = cadence.describe("cron", cad, "America/New_York")
    assert text.startswith("cron ")


def test_daylight_saving_change_handled_like_cron():
    # Spring-forward 2026-03-08 in America/New_York: 2am jumps to 3am, so 2:30am
    # does not exist that day. zoneinfo normalizes the nonexistent local time
    # forward; the point of this test is that cadence does not crash or loop.
    cad = {"time": "02:30"}
    after = _ts("2026-03-07T12:00:00")
    nr = cadence.next_run("daily", cad, "America/New_York", after)
    assert nr is not None
    local = datetime.fromtimestamp(nr, ZoneInfo("America/New_York"))
    assert local.strftime("%Y-%m-%d") == "2026-03-08"
    assert local.strftime("%H:%M") in ("02:30", "03:30")


def test_validate_rejects_bad_cadence_kind():
    with pytest.raises(cadence.CadenceError):
        cadence.validate("hourly", {})


def test_validate_rejects_missing_fields():
    with pytest.raises(cadence.CadenceError):
        cadence.validate("weekly", {"time": "09:00"})


def test_every_without_an_anchor_is_one_interval_later_never_now():
    now = 1_800_000_000.0
    nxt = cadence.next_run("every", {"minutes": 240}, "America/New_York", now)
    assert nxt == now + 240 * 60
    # the value a tick would compute right after firing must be in the future
    assert cadence.next_run("every", {"minutes": 5}, "America/New_York", now) > now
