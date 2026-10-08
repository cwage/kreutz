"""Fast checks on synthetic inputs; no frames needed. The real test is the
July 2025 benchmark (scripts/benchmark.sh), run by hand when the detector
changes."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from kreutz import evaluate, live, reports, search
from kreutz.frames import Sequence

F = timedelta(minutes=15)


class FakeSeq:
    """Just the frame times, for the window and linking logic."""
    def __init__(self, times):
        self.times = times
        self.paths = times


def test_report_coordinate_round_trip():
    seq = Sequence.__new__(Sequence)
    seq.shape = (1920, 2048)
    r, c = seq.report_to_rc(1953.0, 1126.0)
    assert (r, c) == (793.0, 1953.0)
    assert seq.rc_to_report(r, c) == (1953.0, 1126.0)


def test_window_frames_tolerates_one_missing_frame():
    t0 = datetime(2026, 10, 6, 12, 0)
    times = [t0 + F * k for k in range(10) if k != 6]  # frame 6 missing
    idx, back = search.window_frames(FakeSeq(times), last=len(times) - 1, n=8)
    assert len(idx) == 7
    assert back.tolist() == [7, 6, 5, 4, 2, 1, 0]  # oldest first; the missing frame is 3 back


def test_window_frames_rejects_two_missing_frames():
    t0 = datetime(2026, 10, 6, 12, 0)
    times = [t0 + F * k for k in range(10) if k not in (5, 6)]
    with pytest.raises(ValueError):
        search.window_frames(FakeSeq(times), last=len(times) - 1, n=8)


def test_window_thresholds_raise_only_busy_windows():
    quiet = [{"score": s} for s in (6.1, 6.5, 7.0, 6.2)]
    busy = [{"score": 6.0 + 0.1 * k} for k in range(50)]
    thr = evaluate.window_thresholds({"a": quiet, "b": busy}, 6.0, k=10)
    assert thr["a"] == 6.0
    assert thr["b"] == pytest.approx(6.0 + 0.1 * 40)  # the 10th highest of 50


def test_tracks_link_across_a_missing_frame():
    t0 = datetime(2026, 10, 6, 12, 0)
    by_time = {}
    for k in (0, 1, 3):  # window at k=2 missing
        t = t0 + F * k
        by_time[t] = [{"t": t, "time": t.isoformat(), "x": 1000.0 - 5 * k, "y": 800.0 - 2 * k,
                       "vx": -5.0, "vy": -2.0, "score": 8.0, "per_frame": [2] * 8}]
    trs = evaluate.tracks(by_time, 6.0, link=3)
    assert len(trs) == 1 and len(trs[0]) == 3


def test_consistent_rule():
    assert search.consistent([1.5, 2.0, 1.2, 0.5, 1.1, 1.3, 1.0, 1.4])
    assert not search.consistent([5.0, 0.2, 0.1, 0.3, 0.0, 0.2, 0.1, 0.4])


def test_parse_sungrazer_report_page():
    lines = ["report_bc_20260923190827", "Bill Collins", "2026-09-23", "Potential Comet", "Kreutz", "CCOR1",
             "1700 860 774", "1715 854 771", "published",
             "report_eb_20260923005802", "Eryk Banach", "2026-09-23", "Potential Comet", "Kreutz", "CCOR2",
             "0400 1784 1495", "published", "Pagination"]
    out = reports.parse(lines)
    assert [r["reporter"] for r in out] == ["Bill Collins", "Eryk Banach"]
    assert out[0]["telescope"] == "CCOR1" and out[0]["points"] == [["1700", 860.0, 774.0], ["1715", 854.0, 771.0]]
    assert out[0]["posted_utc"] == "2026-09-23T23:08:27Z"  # the stamp is US Eastern (EDT)


def test_swpc_canvas_sun_maps_to_crpix(monkeypatch):
    hdr = {"PC1_1": 0.9159, "PC1_2": 0.4014, "PC2_1": -0.4014, "PC2_2": 0.9159,
           "CRPIX1": 1002.07, "CRPIX2": 968.06, "NAXIS2": 1920}
    t = datetime(2026, 9, 16, 1, 45, 25)
    monkeypatch.setitem(evaluate._HDR, f"{t:%Y%m%dT%H%M}", hdr)
    x, y = evaluate.swpc_canvas_to_report(t, *evaluate.SWPC_SUN)
    assert x == pytest.approx(1001.07, abs=1e-6)
    assert y == pytest.approx(1919 - 967.06, abs=1e-6)
    # one canvas pixel east of the Sun is 1/SWPC_SCALE frame pixels away
    x2, y2 = evaluate.swpc_canvas_to_report(t, evaluate.SWPC_SUN[0] + 10, evaluate.SWPC_SUN[1])
    assert np.hypot(x2 - x, y2 - y) == pytest.approx(10 / evaluate.SWPC_SCALE, rel=1e-3)


def test_prune_keeps_days_not_wholly_past_the_cutoff(tmp_path):
    l1b, sig = tmp_path / "l1b", tmp_path / "sig" / "somekey"
    for root in (l1b, sig):
        for day in ("2026/09/30", "2026/10/04", "2026/10/05", "2026/10/06", "2026/10/07"):
            (root / day).mkdir(parents=True)
            (root / day / "frame.fits").touch()
        (root / "notes").mkdir()  # not a date; left alone
    # 2 days before 2026-10-07 12:00 is 10-05 12:00: 10-05 is only half past
    gone = live.prune(2, datetime(2026, 10, 7, 12, 0), roots=[l1b, sig])
    assert sorted(p.relative_to(tmp_path).as_posix() for p in gone) == [
        "l1b/2026/09/30", "l1b/2026/10/04", "sig/somekey/2026/09/30", "sig/somekey/2026/10/04"]
    for root in (l1b, sig):
        assert sorted(p.name for p in (root / "2026" / "10").iterdir()) == ["05", "06", "07"]
        assert not (root / "2026" / "09").exists()  # emptied month removed
        assert (root / "notes").exists()
