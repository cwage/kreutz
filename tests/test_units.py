"""Fast checks on synthetic inputs; no frames needed. The real test is the
July 2025 benchmark (scripts/benchmark.sh), run by hand when the detector
changes."""

from datetime import datetime, timedelta, timezone

import json
import numpy as np
import pytest

from kreutz import evaluate, lane, live, report, reports, search, vet
from kreutz.frames import flatten, sun_pixel
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
            "points": [(datetime(2026, 10, 8, 21, 45, 25) + timedelta(minutes=15 * i), 1930.0 - 8 * i, 1220.0 + 2 * i, 5.0)
                       for i in range(6)],
            "t": datetime(2026, 10, 8, 23, 0, 25), "x": 1890.0, "y": 1230.0,
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


def test_other_instrument_lookback_is_per_instrument():
    def rep(tel, posted, hhmm):
        return {"id": f"report_x_{posted}", "reporter": "X", "telescope": tel, "group": "Kreutz",
                "kind": "Potential Comet", "posted_utc": posted[:4] + "-" + posted[4:6] + "-" + posted[6:8]
                + "T" + posted[8:10] + ":" + posted[10:12] + ":00Z", "points": [[hhmm, 1.0, 2.0]]}
    first, last = datetime(2026, 10, 10, 8, 0, 25), datetime(2026, 10, 10, 8, 30, 25)
    yesterday_ccor2 = rep("CCOR2", "202610091206", "1030")  # Kaszczyk's, 21 h before: another comet
    yesterday_c3 = rep("C3", "202610091206", "1030")  # C3 sees a Kreutz comet a day ahead of us
    same_morning = rep("CCOR2", "202610101202", "0945")  # an hour after our last window
    got = evaluate.other_kreutz([yesterday_ccor2, yesterday_c3, same_morning], first, last)
    assert [o["telescope"] for o in got] == ["C3", "CCOR2"]
    assert got[1]["posted_utc"].startswith("2026-10-10")


def test_reported_by_extrapolates_along_the_report():
    # Ruan's 2026-10-10 report, measured 09:30-10:00; our chain at 17:30-18:00
    # is the same comet eight hours on, well past the measured span.
    reps = [{"id": "report_jr_20261010062348", "reporter": "Jiangao Ruan", "telescope": "CCOR1", "group": "Kreutz",
             "kind": "Potential Comet", "posted_utc": "2026-10-10T10:23:48Z",
             "points": [["0930", 1849.0, 1198.0], ["0945", 1838.0, 1197.0], ["1000", 1828.0, 1194.0]]}]
    chain = [{"time": "2026-10-10T17:30:25", "x": 1539.0, "y": 1119.0, "vx": -10.0, "vy": -2.5},
             {"time": "2026-10-10T17:45:25", "x": 1530.0, "y": 1116.0, "vx": -10.0, "vy": -2.5},
             {"time": "2026-10-10T18:00:25", "x": 1520.0, "y": 1113.0, "vx": -10.0, "vy": -2.5}]
    k = live.reported_by(chain, datetime(2026, 10, 10, 18, 0, 25), reps)
    assert k and k["id"] == "report_jr_20261010062348" and k["how"] == "position"
    # additional observations of it, posted later, don't displace the discovery
    more = dict(reps[0], id="report_jr_20261010100000", kind="Additional Observations",
                posted_utc="2026-10-10T14:00:00Z", points=[["1230", 1736.0, 1171.0], ["1300", 1717.0, 1166.0]])
    assert live.reported_by(chain, datetime(2026, 10, 10, 18, 0, 25), [more] + reps)["id"] == "report_jr_20261010062348"
    elsewhere = [dict(c, y=c["y"] + 300) for c in chain]
    assert live.reported_by(elsewhere, datetime(2026, 10, 10, 18, 0, 25), reps) is None


def test_follow_up_counts_the_chain_when_the_frame_does_not(monkeypatch):
    import types
    monkeypatch.setattr(live, "measure", lambda *a, **k: [])
    t0 = datetime(2026, 10, 10, 8, 30, 25)
    f = {"alert": "/data/live/alerts/x.json", "track": [], "others": [], "known": None, "points": [],
         "t": t0, "x": 1883.0, "y": 1210.0, "vx": -9.0, "vy": -2.5, "misses": 0}
    seq = types.SimpleNamespace(times=[t0, t0 + timedelta(minutes=15)])
    on_track = [[{"time": seq.times[1].isoformat(), "x": 1874.0, "y": 1208.0, "vx": -9.0, "vy": -2.5, "score": 7.1}]]
    keep = live.follow_up(seq, 1, [{**f, "points": []}], None, chains=on_track)
    assert keep[0]["misses"] == 0 and len(keep[0]["points"]) == 1 and keep[0]["x"] == 1874.0
    off_track = [[dict(on_track[0][0], y=1240.0)]]
    keep = live.follow_up(seq, 1, [{**f, "points": []}], None, chains=off_track)
    assert keep[0]["misses"] == 1 and keep[0]["points"] == []


def test_followed_comet_is_not_alerted_again():
    t0 = datetime(2026, 10, 10, 8, 30, 25)
    pending = [{"t": t0, "x": 1883.0, "y": 1210.0, "vx": -9.0, "vy": -2.5}]
    later = t0 + timedelta(hours=2)  # 8 frames on: predicted (1811, 1190), tolerance 23 px
    assert live.followed({"x": 1820.0, "y": 1192.0}, later, pending)
    assert not live.followed({"x": 1860.0, "y": 1192.0}, later, pending)
    assert not live.followed({"x": 1811.0, "y": 1190.0}, t0 + timedelta(hours=4), pending)  # given up by then


def test_best_report_prefers_a_position_match():
    guess = {"reporter": "Mikolaj Kaszczyk", "id": "report_mk_20261009080647", "telescope": "CCOR2",
             "posted_utc": "2026-10-09T12:06:47Z", "kind": "Potential Comet", "how": "time+lane"}
    reps = [{"id": "report_jr_20261010062348", "reporter": "Jiangao Ruan", "telescope": "CCOR1", "group": "Kreutz",
             "kind": "Potential Comet", "posted_utc": "2026-10-10T10:23:48Z",
             "points": [["0930", 1849.0, 1198.0], ["0945", 1838.0, 1197.0], ["1000", 1828.0, 1194.0]]}]
    track = [{"time": "2026-10-10T17:30:25", "x": 1539.0, "y": 1119.0, "vx": -10.0, "vy": -2.5},
             {"time": "2026-10-10T18:00:25", "x": 1520.0, "y": 1113.0, "vx": -10.0, "vy": -2.5}]
    assert live.best_report(guess, track, reps)["reporter"] == "Jiangao Ruan"
    assert live.best_report(guess, track, [])["reporter"] == "Mikolaj Kaszczyk"
    assert live.best_report(None, track, reps)["how"] == "position"


def test_sun_pixel_comes_from_the_wcs_not_the_reference_pixel():
    from astropy.io import fits
    h = fits.Header()  # the WCS of the 2026-10-12 02:45 frame: the Sun is 40 px from CRPIX
    for k, v in {"NAXIS1": 2048, "NAXIS2": 1920, "WCSAXES": 2, "CTYPE1": "HPLN-ZPN", "CTYPE2": "HPLT-ZPN",
                 "CUNIT1": "deg", "CUNIT2": "deg", "CRPIX1": 1044.9281046400001, "CRPIX2": 950.94145667,
                 "CRVAL1": 0.15712334354340432, "CRVAL2": -0.1467966047586972,
                 "CDELT1": 0.005370007103014486, "CDELT2": 0.005370007103014486,
                 "PC1_1": 0.9010203819677571, "PC1_2": 0.43377675281033323,
                 "PC2_1": -0.43377675281033323, "PC2_2": 0.9010203819677571,
                 "PV1_1": 0.0, "PV1_2": 90.0, "PV1_3": 180.0, "PV2_0": -7.91559035e-07, "PV2_1": 1.00014749,
                 "PV2_2": -0.00566578401, "PV2_3": 0.99620514, "LONPOLE": 180.0, "LATPOLE": 0.0}.items():
        h[k] = v
    assert sun_pixel(h) == pytest.approx((961.9, 1005.7), abs=0.3)
    bare = fits.Header()
    bare["CRPIX1"], bare["CRPIX2"] = 1044.9, 950.9
    assert sun_pixel(bare) == pytest.approx((949.9, 1043.9))


def test_motion_vetting_catches_stationary_and_off_lane():
    samples = [[1950 - 9 * k, 1250 - 2.5 * k, -9.0, -2.5] for k in range(60)]  # a lane from (1950,1250)
    field = lane.build_field((1920, 2048), samples)
    assert lane.off_flow(field, 1600, 1150, -10.0, -2.5, 1920) is None
    assert lane.off_flow(field, 1400, 1086, -15.1, -3.4, 1920) is None  # faster than the flow, the same way
    assert "off the lane flow" in lane.off_flow(field, 1265, 1619, 2.8, -1.4, 1920)  # the Oct 11 glare edge
    pts = [(k, 1265.0, 1619.0, 5.0) for k in range(3)]
    assert vet.Motion(0.2, 0.1, pts, 1.8, -0.6).is_stationary
    assert not vet.Motion(0.2, 0.1, pts[:2], 1.8, -0.6).is_stationary  # too few points: the search's velocity
    assert not vet.Motion(-9.0, -2.5, pts, 1.8, -0.6).is_stationary


def test_report_fit_gives_the_motion_and_the_residual():
    t0 = datetime(2026, 10, 8, 21, 15, 25)
    pts = [(t0 + timedelta(minutes=15 * k), 1949.0 - 9 * k, 1235.0 - 2.5 * k, 5.0) for k in range(5)]
    vx, vy, resid = report.fit(pts)
    assert (vx, vy) == pytest.approx((-9.0, -2.5)) and resid == pytest.approx(0.0, abs=1e-9)
    assert report.block(pts)[1] == "straight-line residual 0.0 px over 5 frames"
    assert report.fit(pts[:1]) == (0.0, 0.0, 0.0)


def test_follow_up_rejects_what_does_not_lie_on_a_line(monkeypatch, tmp_path):
    import types
    t0 = datetime(2026, 10, 11, 6, 0, 25)
    logged, remembered = [], []
    monkeypatch.setattr(live, "log", lambda kind, track, decided, **extra: logged.append((kind, extra.get("note"))))
    monkeypatch.setattr(live, "remember", lambda c, why: remembered.append(why))
    seq = types.SimpleNamespace(times=[t0 + timedelta(minutes=15 * k) for k in range(5)])

    def pending(points):
        return {"alert": str(tmp_path / "x.json"), "track": [], "others": [], "known": None, "points": list(points),
                "t": points[-1][0], "x": points[-1][1], "y": points[-1][2], "vx": -9.0, "vy": -2.5, "misses": 0}
    straight = [(seq.times[k], 1883.0 - 9 * k, 1210.0 - 2.5 * k, 6.0) for k in range(4)]
    # a fifth frame on the line: confirmed
    monkeypatch.setattr(live, "measure", lambda *a, **k: [(seq.times[4], 1883.0 - 36, 1210.0 - 10, 6.0)])
    keep = live.follow_up(seq, 4, [pending(straight)], None)
    assert keep and keep[0]["confirmed"] and logged[-1][0] == "confirmed"
    # a fifth frame 8 px off the line: rejected and dropped, remembered so it isn't raised again
    monkeypatch.setattr(live, "measure", lambda *a, **k: [(seq.times[4], 1883.0 - 36, 1210.0 - 2, 6.0)])
    keep = live.follow_up(seq, 4, [pending(straight)], None)
    assert keep == [] and logged[-1][0] == "rejected" and "residual" in logged[-1][1] and remembered
    # five frames that don't move: rejected as stationary
    still = [(seq.times[k], 1247.0, 1625.0, 6.0) for k in range(4)]
    monkeypatch.setattr(live, "measure", lambda *a, **k: [(seq.times[4], 1247.2, 1625.1, 6.0)])
    keep = live.follow_up(seq, 4, [pending(still)], None)
    assert keep == [] and logged[-1][0] == "rejected" and "stationary" in logged[-1][1]


def test_flatten_takes_out_a_hot_column_and_leaves_noise_alone():
    rng = np.random.default_rng(3)
    s = rng.standard_normal((400, 300)).astype(np.float32)
    s[:50, :] = np.nan  # off the detector
    s[:, 120] += 2.0  # a hot column, like 1806 on 2026-10-14
    s[200, :] += 1.5  # and a readout stripe
    out = flatten(s)
    assert abs(np.nanmedian(out[:, 120])) < 0.15
    assert abs(np.nanmedian(out[200, :])) < 0.15
    # ordinary pixels move by at most the medians' sampling noise
    ok = np.ones(s.shape, bool); ok[:, 120] = False; ok[200, :] = False; ok[:50] = False
    assert np.nanmax(np.abs(out[ok] - s[ok])) < 0.4
    assert np.isnan(out[:50]).all()


def test_lane_distance_gates_the_page_but_not_the_follow_up():
    samples = [[1950 - 9 * k, 1250 - 2.5 * k, -9.0, -2.5] for k in range(60)]  # a lane from (1950,1250)
    field = lane.build_field((1920, 2048), samples)
    assert lane.lane_px(field, 1950, 1250, 1920) == 0.0
    assert lane.lane_px(field, 1600, 1153, 1920) < 5.0  # on the lane, between samples 9 px apart
    assert lane.lane_px(field, 1645, 609, 1920) > 400  # the 2026-10-17 04:02 stray-light ghost
    assert lane.MATCH_PX == 150
    from kreutz import notify
    track = [{"time": "2026-10-17T03:15:25", "x": 1645.0, "y": 609.0, "vx": -3.5, "vy": 0.5}]
    far = {"track": track, "other_instruments": [], "lane_px": 435, "near_lane": False}
    assert notify.compose(far, "a.json") is None  # a candidate off the lane doesn't page
    title, lines, prio, _ = notify.compose(dict(far, kind="confirmed", frames=5, note="ok"), "a.confirmed.json")
    assert prio == "urgent" and "435 px off the Kreutz lane" in lines[0]  # confirmed, it does, and says so
    near = dict(far, lane_px=12, near_lane=True)
    assert notify.compose(near, "a.json")[0] == "kreutz: candidate, UNREPORTED"
    assert notify.compose({"track": track, "other_instruments": []}, "a.json")[2] == "high"  # older file: pages
    pend = {"alert": "/data/live/alerts/20261017T0345_1645_609.json", "misses": 0, "track": track,
            "points": [], "t": datetime(2026, 10, 17, 3, 45, 25), "x": 1645.0, "y": 609.0, "vx": -3.5, "vy": 0.5,
            "known": None, "others": [], "lane_px": 435, "near_lane": False}
    f, = live.status([], [pend], datetime(2026, 10, 17, 3, 45, 25), [], datetime(2026, 10, 17, 4, 0))["following"]
    assert f["stage"] == "candidate" and f["near_lane"] is False and f["lane_px"] == 435


def test_status_keeps_the_match_on_an_ended_object():
    known = {"reporter": "Jiangao Ruan", "id": "report_jr_20261016011229", "telescope": "CCOR2",
             "posted_utc": "2026-10-16T05:12:29Z", "kind": "Potential Comet", "how": "time+lane"}
    ended = {"kind": "ended", "decided_utc": "2026-10-16T19:31:45", "first": "2026-10-16T09:30:25",
             "last": "2026-10-16T19:00:25", "x": 1292.0, "y": 1050.0, "frames": 29, "known": known,
             "alert": "/data/live/alerts/20261016T0930_1658_1165.json"}
    s = live.status([], [], datetime(2026, 10, 16, 19, 30, 25), [ended], datetime(2026, 10, 16, 19, 35))
    d, = s["recent"]
    assert d["kind"] == "ended" and d["report"]["reporter"] == "Jiangao Ruan" and d["report"]["how"] == "time+lane"


def test_tile_pads_a_crop_that_runs_off_the_frame():
    img = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
    t = live.tile(img, 5, 50)
    assert t.shape == (81, 81) and np.isnan(t[0, 40]) and t[40, 40] == img[5, 50]
    assert np.isnan(live.tile(img, 300, 300)).all()


def test_extent_tells_a_line_from_a_point():
    yy, xx = np.mgrid[-25:26, -25:26]
    point = np.exp(-(yy ** 2 + xx ** 2) / (2 * 1.5 ** 2)) * 8
    length, ratio = vet.extent(point, 25, 25)
    assert length < vet.LINE_LEN and ratio < 1.5
    line = np.exp(-((yy - 0.5 * xx) ** 2) / (2 * 1.2 ** 2)) * 6  # a streak across the box
    length, ratio = vet.extent(line, 25, 25)
    assert length > vet.LINE_LEN and ratio > vet.LINE_RATIO


def test_ghost_verdict_needs_a_matching_profile_a_day_earlier():
    # the 2026-10-17 03:45 ghost along its track and the same pixels on the 16th
    now = {k: (v, 0.0, 0.0) for k, v in enumerate([1.5, 1.0, 4.9, 7.4, 7.5, 8.8, 8.6, 2.3])}
    prev = {k: (v, 0.0, 0.0) for k, v in enumerate([1.1, 1.4, 5.2, 7.3, 7.3, 8.6, 8.7, 1.9])}
    assert "same as 24 h earlier" in vet.ghost_verdict(now, prev)
    noise = {k: (v, 0.0, 0.0) for k, v in enumerate([1.2, 2.0, 1.5, 0.9, 2.2, 1.7, 1.1, 1.6])}
    assert vet.ghost_verdict(now, noise) is None  # a comet: nothing there a day earlier
    assert vet.ghost_verdict(now, {k: prev[k] for k in range(4)}) is None  # too few frames to compare


def test_confirmed_follow_up_reanchors_on_a_search_chain_and_says_where_it_was_lost():
    t = datetime(2026, 10, 15, 6, 0, 25)
    f = {"x": 1600.0, "y": 1180.0, "vx": -6.0, "vy": -2.5, "confirmed": True}
    # the comet, 14 px ahead of where the drifting refit predicts it (1576,1170) 4 frames on
    chains = [[{"time": t.isoformat(), "x": 1576.0 - 14, "y": 1170.0, "vx": -7.5, "vy": -2.5, "score": 9.0}]]
    hit = live.reanchored(f, chains, t, 4)
    assert hit is not None and hit[0][1] == 1562.0 and hit[1] == (-7.5, -2.5)
    far = [[dict(chains[0][0], x=1500.0)]]
    assert live.reanchored(f, far, t, 4) is None
    sun = (962.0, 1006.0)  # (row, col)
    assert live.where_lost(2020.0, 1200.0, (1920, 2048), sun) == "edge"  # at the right edge
    assert live.where_lost(1450.0, 1050.0, (1920, 2048), sun) == "edge"  # inside the search's inner radius + margin
    assert live.where_lost(1700.0, 1250.0, (1920, 2048), sun) == "lost"
