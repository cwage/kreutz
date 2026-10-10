"""Fast checks on synthetic inputs; no frames needed. The real test is the
July 2025 benchmark (scripts/benchmark.sh), run by hand when the detector
changes."""

from datetime import datetime, timedelta, timezone

import json
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


def test_open_tracks_names_the_longest_chain():
    assert live.open_tracks([]) == "0 open tracks"
    blip = [{"x": 100.4, "y": 200.6}]
    comet = [{"x": 1876.0, "y": 1215.0}, {"x": 1868.2, "y": 1223.1}, {"x": 1860.0, "y": 1231.0}]
    assert live.open_tracks([blip]) == "1 open tracks (longest 1 windows, at 100,201)"
    assert live.open_tracks([blip, comet]) == "2 open tracks (longest 3 windows, at 1860,1231)"


def test_status_stages_and_paths():
    pend = {"alert": "/data/live/alerts/20261008T2215_1913_1225.json", "misses": 1, "confirmed": True,
            "track": [{"time": "2026-10-08T21:45:25", "x": 1930.0, "y": 1220.0, "vx": -8.0, "vy": 2.0}],
            "points": [None] * 6, "t": datetime(2026, 10, 8, 23, 0, 25), "x": 1890.0, "y": 1230.0,
            "vx": -8.0, "vy": 2.0, "known": None, "others": [{"reporter": "X"}]}
    recent = [{"kind": "reported", "decided_utc": "2026-10-09T12:16:12", "first": "2026-10-09T11:30:25",
               "last": "2026-10-09T12:00:25", "x": 1876.0, "y": 1215.0, "report": "Jiangao Ruan report_jr_20261009074834 (CCOR1)",
               "how": "position", "alert": "/data/live/alerts/20261009T1200_1876_1215.json",
               },  # an older entry: no 'known' record, only the report string
              {"kind": "star", "decided_utc": "2026-10-09T08:00:00", "first": "2026-10-09T07:00:25",
               "last": "2026-10-09T07:30:25", "x": 10.0, "y": 20.0}]
    chains = [[{"x": 1.0, "y": 2.0}], [{"x": 3.0, "y": 4.0}, {"x": 5.0, "y": 6.0}]]
    pics = {"preview": "../render/20261008T2215_1913_1225/preview.gif", "styles": {}}
    s = live.status(chains, [pend], datetime(2026, 10, 8, 23, 0, 25), recent, datetime(2026, 10, 8, 23, 5),
                    rendered=lambda name: pics if name == "20261008T2215_1913_1225" else None)
    assert (s["open"], s["longest"], s["longest_at"]) == (2, 2, [5.0, 6.0])
    f, = s["following"]
    assert f["stage"] == "confirmed" and f["points"] == 6 and f["others"] == 1
    assert f["gif"] == "alerts/20261008T2215_1913_1225.gif"
    assert f["pictures"] is pics
    assert f["last_seen"] == "2026-10-08T23:00:25"
    a, b = s["recent"]
    assert a["found_utc"] == "2026-10-09T12:16:12" and a["confirmed_utc"] is None  # from the log line itself
    assert b["found_utc"] is None
    assert a["gif"] == "alerts/20261009T1200_1876_1215.gif" and a["pictures"] is None
    assert a["known"]["url"] == "https://sungrazer.nrl.navy.mil/report/reportjr20261009074834"
    assert b["name"] is None and b["gif"] is None
    assert a["report"]["url"].endswith("reportjr20261009074834")
    assert (a["report"]["reporter"], a["report"]["telescope"]) == ("Jiangao Ruan", "CCOR1")
    # an unreported confirmation matched after the fact against a report that appeared since
    reps = [{"id": "report_cw_20261008230000", "reporter": "Me", "telescope": "CCOR1", "group": "Kreutz",
             "kind": "Potential Comet", "posted_utc": "2026-10-09T03:00:00Z",
             "points": [["2245", 1898.0, 1228.0], ["2300", 1890.0, 1230.0], ["2315", 1882.0, 1232.0]]}]
    pend["track"] = [{"time": "2026-10-08T22:45:25", "x": 1898.0, "y": 1228.0, "vx": -8.0, "vy": 2.0},
                     {"time": "2026-10-08T23:00:25", "x": 1890.0, "y": 1230.0, "vx": -8.0, "vy": 2.0}]
    s = live.status([], [pend], datetime(2026, 10, 8, 23, 0, 25), [], datetime(2026, 10, 9, 4, 0), reps=reps)
    f, = s["following"]
    assert f["known"] is None and f["report"]["how"] == "position" and f["report"]["reporter"] == "Me"
    # a decision row whose alert/confirmed lines have left the log window still carries its times
    ended = {"kind": "ended", "decided_utc": "2026-10-09T11:20:00", "first": "2026-10-08T21:45:25",
             "last": "2026-10-09T11:00:25", "x": 1.0, "y": 2.0, "frames": 44,
             "alert": "/data/live/alerts/20261008T2215_1913_1225.json"}
    files = {"alert": {"track": pend["track"], "decided_utc": "2026-10-08T22:35:10"},
             "confirmed": {"decided_utc": "2026-10-08T22:45:30"}}
    s = live.status([], [], datetime(2026, 10, 9, 11, 0, 25), [ended], datetime(2026, 10, 9, 12, 0),
                    files_of=lambda name: files, reps=reps)
    r, = s["recent"]
    assert (r["found_utc"], r["confirmed_utc"]) == ("2026-10-08T22:35:10", "2026-10-08T22:45:30")
    assert r["report"]["reporter"] == "Me"  # matched from the file's full track
    assert f["report"]["url"].endswith("/report/reportcw20261008230000")
    # another instrument's Kreutz report hours later, while the object is still crossing: likely it
    reps = [{"id": "report_eb_20261008225115", "reporter": "Eryk Banach", "telescope": "CCOR2",
             "group": "Kreutz", "kind": "Potential Comet", "posted_utc": "2026-10-09T02:51:15Z",
             "points": [["0200", 1509.0, 1398.0], ["0215", 1502.0, 1394.0], ["0230", 1496.0, 1388.0]]}]
    s = live.status([], [pend], datetime(2026, 10, 8, 23, 0, 25), [], datetime(2026, 10, 9, 4, 0), reps=reps)
    assert s["following"][0]["report"]["how"] == "time"
    assert s["following"][0]["report"]["reporter"] == "Eryk Banach"
    pend["confirmed"] = False  # a candidate that hasn't held gets no time-only guess
    s = live.status([], [pend], datetime(2026, 10, 8, 23, 0, 25), [], datetime(2026, 10, 9, 4, 0), reps=reps)
    assert s["following"][0]["report"] is None


def test_notify_is_quiet_about_known_comets():
    from kreutz import notify
    track = [{"time": "2026-10-09T11:30:25", "x": 1893.0, "y": 1219.0, "vx": -9.0, "vy": -2.0}]
    known = {"reporter": "Jiangao Ruan", "id": "r1", "telescope": "CCOR1",
             "posted_utc": "2026-10-09T11:48:34Z", "kind": "Potential Comet", "how": "position"}
    assert notify.compose({"track": track, "known": known}, "a.json") is None
    title, lines, prio, _ = notify.compose({"kind": "confirmed", "track": track, "known": known,
                                            "frames": 5, "note": "ok"}, "a.confirmed.json")
    assert prio == "low" and "Jiangao Ruan" in title and "2026-10-09 11:48 UTC" in lines[0]
    assert lines[-1].endswith("kreutz.report /data/live/alerts/a.confirmed.json")
    title, _, prio, _ = notify.compose({"track": track, "other_instruments": []}, "a.json")
    assert (title, prio) == ("kreutz: candidate, UNREPORTED", "high")
    title, _, prio, _ = notify.compose({"kind": "confirmed", "track": track, "other_instruments": [],
                                        "frames": 5, "note": "ok"}, "a.confirmed.json")
    assert (title, prio) == ("kreutz: CONFIRMED, report ready", "urgent")


def test_seen_extends_the_track_to_the_last_measurement():
    f = {"track": [{"time": "2026-10-08T22:15:25", "x": 1913.0, "y": 1225.0, "vx": -8.0, "vy": 2.0}],
         "points": [(datetime(2026, 10, 9, 3, 0, 25), 1760.0, 1263.0, 5.0)], "vx": -8.1, "vy": 2.1}
    s = live.seen(f)
    assert s[0] is f["track"][0] and s[-1]["time"] == "2026-10-09T03:00:25" and s[-1]["x"] == 1760.0
    assert live.seen({"track": f["track"], "points": [], "vx": 0, "vy": 0}) == f["track"]


def test_field_gif_levels_survive_shrinking():
    from kreutz import style
    rng = np.random.default_rng(0)
    big = rng.normal(60, 15, (900, 1200)).clip(0, 255).astype(np.uint8)
    small, k = style.shrunk(big, 800)
    assert small.shape == (600, 800) and abs(k - 2 / 3) < 1e-9
    assert len(np.unique(style.coarser(small, 64))) <= 64
    assert style.shrunk(small, 800)[1] == 1.0


def test_state_round_trips_and_expires(tmp_path):
    p = tmp_path / "state.json"
    t0 = datetime(2026, 10, 9, 12, 0, 25)
    done = {t0 - timedelta(minutes=15 * k) for k in range(60)}  # 15 h of windows
    pend = {"alert": "/data/live/alerts/a.json", "track": [{"time": "2026-10-09T11:30:25", "x": 1.0}],
            "points": [(t0, np.float32(1893.5), np.float64(1219.25), np.float32(6.5))], "t": t0,
            "x": 1893.5, "y": 1219.25, "vx": -9.0, "vy": -2.0, "misses": 0, "known": None, "others": [],
            "render_proc": object(), "since_render": 2, "confirmed": True}
    span = timedelta(hours=13)
    live.save_state(done, [[{"x": 1.0}]], [pend], span, p)
    d2, chains, pending = live.load_state(t0 + timedelta(minutes=30), span, p)
    assert len(d2) == 53 and max(d2) == t0 and min(d2) == t0 - span  # trimmed to the span
    assert chains == [[{"x": 1.0}]]
    f, = pending
    assert f["t"] == t0 and f["points"] == [(t0, 1893.5, 1219.25, 6.5)]
    assert "render_proc" not in f and f["since_render"] == 2 and f["confirmed"]
    assert live.load_state(t0 + span + timedelta(minutes=1), span, p) == (set(), [], [])
    assert live.load_state(t0, span, tmp_path / "none.json") == (set(), [], [])


def test_pictures_lists_what_a_render_has(tmp_path):
    assert live.pictures("none", tmp_path) is None
    d = tmp_path / "x"
    (d / "style-soft").mkdir(parents=True)
    (d / "style-tintype").mkdir()
    (d / "preview.gif").write_bytes(b"")
    (d / "style-soft" / "field.gif").write_bytes(b"")
    (d / "style-soft" / "still.png").write_bytes(b"")
    import os
    os.utime(d / "preview.gif", (1_000_000, 1_000_000))
    os.utime(d / "style-soft" / "still.png", (1_000_000, 1_000_000))
    os.utime(d / "style-soft" / "field.gif", (1_700_000_000, 1_700_000_000))  # a style rerun, preview untouched
    pics = live.pictures("x", tmp_path)
    assert pics.pop("made_utc") == datetime.fromtimestamp(1_700_000_000, timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    assert pics == {
        "preview": "../render/x/preview.gif",
        "styles": {"soft": {"field": "../render/x/style-soft/field.gif", "still": "../render/x/style-soft/still.png"}}}


def test_preview_scale_ignores_flash_frames():
    from kreutz import render
    rng = np.random.default_rng(1)
    cube = rng.normal(0, 1, (20, 50, 50)).astype(np.float32)
    cube[9:11] += 500  # two eclipse-flash frames
    cube[0, :5, :5] = np.nan
    lo, hi = render.preview_limits(cube)
    assert -4 < lo < -1.5 and 1.5 < hi < 4  # the noise's own scale, not the flash's
    naive_hi = np.nanpercentile(cube, 99.7)
    assert naive_hi > 100


def test_load_state_backfills_times_from_alert_files(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "LIVE", tmp_path)
    (tmp_path / "alerts").mkdir()
    (tmp_path / "alerts" / "a.json").write_text(json.dumps({"track": [], "decided_utc": "2026-10-09T19:28:00"}))
    (tmp_path / "alerts" / "a.confirmed.json").write_text(json.dumps({"decided_utc": "2026-10-09T19:51:00"}))
    t0 = datetime(2026, 10, 9, 19, 45, 25)
    old = {"alert": str(tmp_path / "alerts" / "a.json"), "track": [], "points": [], "t": t0, "x": 0, "y": 0,
           "vx": 0, "vy": 0, "misses": 0}
    live.save_state({t0}, [], [old], timedelta(hours=13), tmp_path / "state.json")
    _, _, (f,) = live.load_state(t0, timedelta(hours=13), tmp_path / "state.json")
    assert (f["decided_utc"], f["confirmed_utc"]) == ("2026-10-09T19:28:00", "2026-10-09T19:51:00")


def test_crosshair_leaves_the_object_clear():
    from PIL import Image
    from kreutz import style
    im = style.crosshair(Image.fromarray(np.zeros((40, 40), np.uint8)), 20, 20)
    assert im.mode == "RGB"
    assert im.getpixel((20, 20)) == (0, 0, 0)  # the object itself
    assert im.getpixel((20 + style.MARK_GAP + 1, 20)) == style.MARK  # a tick
    assert im.getpixel((20, 20 - style.MARK_GAP - 1)) == style.MARK
    assert im.getpixel((20 + 3, 20)) == (0, 0, 0)  # inside the gap
    # shrunk pictures get a shorter mark, never a vanishing one
    small = style.crosshair(Image.fromarray(np.zeros((10, 10), np.uint8)), 5, 5, scale=0.1)
    assert small.getpixel((5 + 2, 5)) == style.MARK


def test_gif_frames_keep_the_mark_color():
    from PIL import Image
    from kreutz import style
    gray = (np.arange(40 * 40, dtype=np.int32).reshape(40, 40) % 64 * 4).astype(np.uint8)
    im = style.crosshair(Image.fromarray(gray), 20, 20)
    p = style.paletted(im)
    assert p.mode == "P"
    assert p.convert("RGB").getpixel((20 + style.MARK_GAP + 1, 20)) == style.MARK
    assert p.convert("RGB").getpixel((0, 0)) == (0, 0, 0)
    # the caption is white whether the picture is gray or (marked) RGB
    for mode in ("L", "RGB"):
        g = Image.new(mode, (30, 12))
        style.stamp(g, "x", 255)
        r, gg, b = g.convert("RGB").getextrema()  # white-ish, not a red int-as-RGB
        assert r == gg == b and r[1] > 200
