"""Fast checks on synthetic inputs; no frames needed. The real test is the
July 2025 benchmark (scripts/benchmark.sh), run by hand when the detector
changes."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import json
import types
import numpy as np
import pytest

from kreutz import evaluate, lane, live, report, reports, search, vet
from kreutz.frames import flatten, sun_pixel
from kreutz.frames import Sequence

F = timedelta(minutes=15)


@pytest.fixture(autouse=True)
def no_orbit_match(monkeypatch):
    """Report matching by position only: the tracks here are made up, and
    with real frames on disk the orbit matcher would test them against
    those frames' WCS."""
    monkeypatch.setattr(live, "ORBIT_MATCH", False)


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


def _noisy_seq(t0, n, shape=(320, 400), source=None, seed=3):
    """A fake Sequence: N(0,1) significance maps, with a 6-sigma source
    moving along source(t) -> (x, y) when given."""
    import types
    rng = np.random.default_rng(seed)
    times = [t0 + timedelta(minutes=15) * k for k in range(n)]
    maps = {}

    def significance(i):
        if i not in maps:
            m = rng.standard_normal(shape).astype(np.float32)
            if source:
                x, y = source(times[i])
                m[int(round(shape[0] - 1 - y)), int(round(x))] += 6.0
            maps[i] = m
        return maps[i]
    return types.SimpleNamespace(times=times, shape=shape, significance=significance)


def test_stack_reads_a_planted_source_and_not_noise():
    from kreutz import stackcheck
    t0 = datetime(2026, 11, 8, 2, 0, 25)
    track = [{"time": (t0 + timedelta(minutes=15) * k).isoformat(), "x": 200.0 - 2 * k, "y": 160.0 - 1.5 * k,
              "vx": -2.0, "vy": -1.5, "per_frame": [1] * 4} for k in (3, 4, 5)]
    where = stackcheck.line_through(track)
    assert where(t0 + timedelta(minutes=15) * 4) == pytest.approx((192.0, 154.0))
    start, end = stackcheck.span(track)
    assert start == t0 and end == t0 + timedelta(minutes=75)
    real = stackcheck.stack(_noisy_seq(t0, 12, source=where), where, start, t0 + timedelta(minutes=165))
    noise = stackcheck.stack(_noisy_seq(t0, 12), where, start, t0 + timedelta(minutes=165))
    assert real.n == 12 and noise.n == 12
    assert real.z_at() > 8 and noise.z_at() < 4
    assert real.z_at(end) > real.z_at(t0 + timedelta(minutes=15))  # grows with frames
    back = stackcheck.Stack.from_dict(json.loads(json.dumps(real.to_dict())))
    assert back.n == 12 and back.z_at() == pytest.approx(real.z_at(), abs=0.05)  # to_dict rounds
    assert back.summary()["n"] == 12 and back.summary()["beaten"] == 0
    assert stackcheck.verdict(real.z_at()) == "real" and stackcheck.verdict(2.0) == "weak"
    assert stackcheck.verdict(None) is None


def test_page_for_and_report_page_once(monkeypatch, tmp_path):
    from kreutz import stackcheck
    t0 = datetime(2026, 11, 8, 2, 0, 25)
    track = [{"time": (t0 + timedelta(minutes=15) * k).isoformat(), "x": 200.0 - 2 * k, "y": 160.0 - 1.5 * k,
              "vx": -2.0, "vy": -1.5, "per_frame": [1] * 4} for k in (3, 4, 5)]
    where = stackcheck.line_through(track)
    seq = _noisy_seq(t0, 16, source=where)
    # the alert's own frames: a stack, but not yet a verdict of real
    st = live.track_stack(seq, track)
    assert st is not None and st.n == 6
    assert live.page_for(False, None, st) is None and live.page_for(True, {"id": "r"}, st) is None
    assert live.page_for(True, None, None) == "watching"
    assert live.page_for(True, None, st) in ("watching", "report")
    monkeypatch.setattr(live, "LIVE", tmp_path)
    (tmp_path / "alerts").mkdir()
    alert = tmp_path / "alerts" / "20261108T0245_190_152.json"
    alert.write_text("{}")
    f = {"alert": str(alert), "track": track, "others": [], "known": None, "points": [],
         "t": datetime.fromisoformat(track[-1]["time"]), "x": 190.0, "y": 152.5, "vx": -2.0, "vy": -1.5,
         "misses": 0, "paged": True, "stack": st.to_dict(), "report_paged": False, "lane_px": 9, "heading_deg": 7}
    sent = []
    monkeypatch.setattr(live.subprocess, "Popen", lambda cmd: sent.append(cmd))
    for i in range(6, 16):
        x, y = where(seq.times[i])
        live.extend_stack(f, seq, i, x, y, "python -m kreutz.notify")
    assert stackcheck.Stack.from_dict(f["stack"]).n == 16
    assert f["report_paged"] is True and len(sent) == 1  # one REPORT page, however far the stack goes on
    rep = json.loads(alert.with_name(alert.stem + ".report.json").read_text())
    assert rep["kind"] == "report" and rep["stack"]["z"] >= stackcheck.REAL_Z and rep["report"] == live.form_lines(track)
    recs = [json.loads(ln) for ln in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in recs] == ["report"] and recs[0]["alert"] == str(alert)
    assert rep["why"].startswith("stack z")
    assert live.report_why(2.0, 100) is None and live.report_why(None, 100) is None  # a negative stack never pages
    # a faint one: the stack never reads real, but the search holds it
    monkeypatch.setattr(live, "REAL_Z", 1000.0)
    monkeypatch.setattr(live, "WEAK_Z", -100.0)
    g = dict(f, alert=str(alert.with_name("faint.json")), stack=st.to_dict(), report_paged=False)
    sent.clear()
    for k, i in enumerate(range(6, 16)):
        x, y = where(seq.times[i])
        g["held_windows"] = k + 1
        live.extend_stack(g, seq, i, x, y, "python -m kreutz.notify")
        assert g["report_paged"] is (k + 1 >= live.HELD_WINDOWS)
    assert len(sent) == 1
    rep = json.loads(alert.with_name("faint.report.json").read_text())
    assert rep["why"].startswith(f"held {live.HELD_WINDOWS} windows") and rep["held_windows"] == live.HELD_WINDOWS


def test_verdict_pages():
    from kreutz import notify
    track = [{"time": "2026-11-08T07:06:09", "x": 636.0, "y": 713.0, "vx": -1.5, "vy": -1.5},
             {"time": "2026-11-08T07:30:10", "x": 633.0, "y": 708.0, "vx": -1.5, "vy": -2.0}]
    base = {"track": track, "other_instruments": [], "known": None, "paged": True, "lane_px": 10, "heading_deg": 9,
            "report": ["0706 636 713", "0730 633 708"]}
    weak = dict(base, page="watching", stack={"n": 6, "z": 3.1, "null_mean": 2.0, "null_std": 0.9, "stack": 4.8, "beaten": 1})
    title, lines, prio, _ = notify.compose(weak, "a.json")
    assert "watching" in title and "no action" in title and prio == "low" and "stack z 3.1" in lines[0]
    strong = dict(weak, page="report", stack=dict(weak["stack"], z=11.2))
    title, lines, prio, _ = notify.compose(strong, "a.json")
    assert title.endswith("REPORT, stack says real") and prio == "urgent" and lines[-1] == "0730 633 708"
    later = dict(base, kind="report", frames=4, note="straight-line residual 0.3 px over 4 frames", stack=strong["stack"])
    title, lines, prio, _ = notify.compose(later, "a.report.json")
    assert "REPORT" in title and prio == "urgent" and "held 4 frames" in lines[0]
    conf = dict(base, kind="confirmed", frames=5, note="straight-line residual 0.3 px over 5 frames",
                stack=strong["stack"], report_paged=False)
    title, _, prio, _ = notify.compose(conf, "a.confirmed.json")
    assert title.endswith("REPORT, confirmed") and prio == "urgent"
    title, _, prio, _ = notify.compose(dict(conf, report_paged=True), "a.confirmed.json")
    assert "measured block" in title and prio == "low"  # the REPORT page already went
    assert notify.compose(dict(weak, paged=False, page=None), "a.json") is None  # held by a gate: as before
    assert notify.compose(dict(weak, other_instruments=[{"reporter": "X", "telescope": "C3"}]), "a.json")[2] == "low"


def test_parse_report_with_no_group_keeps_its_points():
    # Lecci's C2 comet of 2026-11-08: no group, so the points start one field early
    lines = ["report_vl_20261108152830", "Vito Lecci", "2026-11-08", "Potential Comet", "C2",
             "1836 740 1018", "1848 732 1009", "published",
             "report_eb_20261108085320", "Eryk Banach", "2026-11-08", "Retracting Report", "Kreutz", "C2",
             "published", "Pagination"]
    r, q = reports.parse(lines)
    assert r["points"] == [["1836", 740.0, 1018.0], ["1848", 732.0, 1009.0]]
    assert reports.telescope_of(r) == "C2" and r["group"] is None and r["kind"] == "Potential Comet"
    assert q["kind"] == "Retracting Report" and q["points"] == [] and reports.telescope_of(q) == "C2"


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
    assert (title, prio) == ("kreutz CCOR-1: candidate, UNREPORTED", "high")
    title, _, prio, _ = notify.compose({"kind": "confirmed", "track": track, "other_instruments": [],
                                        "frames": 5, "note": "ok"}, "a.confirmed.json")
    assert (title, prio) == ("kreutz CCOR-1: CONFIRMED, report ready", "urgent")


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
    assert k["group"] == "Kreutz"  # the status page names any other group
    # additional observations of it, posted later, don't displace the discovery
    more = dict(reps[0], id="report_jr_20261010100000", kind="Additional Observations",
                posted_utc="2026-10-10T14:00:00Z", points=[["1230", 1736.0, 1171.0], ["1300", 1717.0, 1166.0]])
    assert live.reported_by(chain, datetime(2026, 10, 10, 18, 0, 25), [more] + reps)["id"] == "report_jr_20261010062348"
    elsewhere = [dict(c, y=c["y"] + 300) for c in chain]
    assert live.reported_by(elsewhere, datetime(2026, 10, 10, 18, 0, 25), reps) is None


def test_follow_up_holds_on_the_chain_but_does_not_count_it(monkeypatch):
    import types
    monkeypatch.setattr(live, "measure", lambda *a, **k: [])
    t0 = datetime(2026, 10, 10, 8, 30, 25)
    f = {"alert": "/data/live/alerts/x.json", "track": [], "others": [], "known": None, "points": [],
         "t": t0, "x": 1883.0, "y": 1210.0, "vx": -9.0, "vy": -2.5, "misses": 0}
    seq = types.SimpleNamespace(times=[t0, t0 + timedelta(minutes=15)])
    on_track = [[{"time": seq.times[1].isoformat(), "x": 1874.0, "y": 1208.0, "vx": -9.0, "vy": -2.5, "score": 7.1}]]
    keep = live.follow_up(seq, 1, [{**f, "points": []}], None, chains=on_track)
    assert keep[0]["misses"] == 0 and keep[0]["points"] == [] and keep[0]["x"] == 1874.0
    # held every frame, it never confirms: nothing was measured
    f5 = {**f, "points": [], "misses": 0}
    for i in range(1, 7):
        times = [t0 + timedelta(minutes=15) * j for j in range(i + 1)]
        seq_i = types.SimpleNamespace(times=times)
        chain = [[{"time": times[i].isoformat(), "x": 1883.0 - 9.0 * i, "y": 1210.0 - 2.5 * i,
                   "vx": -9.0, "vy": -2.5, "score": 7.1}]]
        f5, = live.follow_up(seq_i, i, [f5], None, chains=chain)
    assert f5["points"] == [] and not f5.get("confirmed") and f5["misses"] == 0
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


def test_occulted_r_is_where_the_valid_data_starts():
    H, W = 600, 600
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(yy - 300, xx - 290)
    sig = np.where(r < 125, np.nan, 0.0)
    sig[295:298, 200:280] = 6.6  # a row-constant stripe inside the occulter, as CCOR-1's
    assert live.occulted_r(sig, (300, 290), 460) == 130.0
    assert live.occulted_r(np.full((H, W), np.nan), (300, 290), 260) is None


def test_follow_up_ends_behind_the_occulter_and_after_a_feed_gap(monkeypatch, tmp_path):
    import types
    t0 = datetime(2026, 10, 27, 10, 30, 25)
    logged = []
    monkeypatch.setattr(live, "log", lambda kind, track, decided, **extra: logged.append((kind, extra.get("note"))))
    monkeypatch.setattr(live, "measure", lambda *a, **k: [])
    H, W = 1920, 2048
    sig = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    sig[np.hypot(yy - 962, xx - 1006) < 200] = np.nan
    sun_rc = (962.0, 1006.0)  # report (1006, 957)

    def seq_at(t):
        return types.SimpleNamespace(times=[t - timedelta(minutes=15), t], shape=(H, W), sun_rc=lambda: sun_rc,
                                     significance=lambda i: sig)

    def comet(x, y, t, confirmed=True):
        return {"alert": str(tmp_path / "x.json"), "track": [{"time": t.isoformat(), "x": x, "y": y, "vx": -9.0,
                "vy": -4.0}], "others": [], "known": None, "points": [], "t": t, "x": x, "y": y,
                "vx": -9.0, "vy": -4.0, "misses": 0, "confirmed": confirmed}
    # due 213 px from the Sun next frame: inside 200 + OCCULTED_MARGIN, ended
    assert live.follow_up(seq_at(t0 + timedelta(minutes=15)), 1, [comet(1210.0, 1040.0, t0)], None) == []
    assert logged[-1] == ("ended", "behind the occulter")
    # 384 px out and still in view: kept, a miss counted
    keep = live.follow_up(seq_at(t0 + timedelta(minutes=15)), 1, [comet(1329.0, 1164.0, t0)], None)
    assert keep and keep[0]["misses"] == 1
    # the next frame 4.5 h later, after a gap in the feed: ended, not measured
    assert live.follow_up(seq_at(t0 + timedelta(hours=4, minutes=30)), 1, [comet(1700.0, 1300.0, t0)], None) == []
    assert logged[-1][0] == "ended" and "gap" in logged[-1][1]
    # a candidate after an hour's gap: faded
    assert live.follow_up(seq_at(t0 + timedelta(hours=1)), 1, [comet(1700.0, 1300.0, t0, False)], None) == []
    assert logged[-1][0] == "faded" and "gap" in logged[-1][1]


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
    far = {"track": track, "other_instruments": [], "lane_px": 435, "heading_deg": 62, "paged": False}
    assert notify.compose(far, "a.json") is None  # a candidate not moving like a sungrazer doesn't page
    title, lines, prio, _ = notify.compose(dict(far, kind="confirmed", frames=5, note="ok"), "a.confirmed.json")
    assert prio == "low" and "62 deg off sunward" in lines[0]  # confirmed, a quiet message saying why
    assert title == "kreutz CCOR-1: confirmed, unlikely Kreutz"
    near = dict(far, heading_deg=3, paged=True)
    assert notify.compose(near, "a.json")[0] == "kreutz CCOR-1: candidate, UNREPORTED"
    assert notify.compose({"track": track, "other_instruments": []}, "a.json")[2] == "high"  # older file: pages
    old = dict(far, near_lane=False); del old["paged"]; del old["heading_deg"]
    assert notify.compose(old, "a.json") is None  # a file from before the heading gate
    assert "435 px off the Kreutz lane" in notify.compose(dict(old, kind="confirmed", frames=5, note="ok"), "a.confirmed.json")[1][0]
    pend = {"alert": "/data/live/alerts/20261017T0345_1645_609.json", "misses": 0, "track": track,
            "points": [], "t": datetime(2026, 10, 17, 3, 45, 25), "x": 1645.0, "y": 609.0, "vx": -3.5, "vy": 0.5,
            "known": None, "others": [], "lane_px": 435, "heading_deg": 62, "paged": False}
    f, = live.status([], [pend], datetime(2026, 10, 17, 3, 45, 25), [], datetime(2026, 10, 17, 4, 0))["following"]
    assert f["stage"] == "candidate" and f["paged"] is False and f["heading_deg"] == 62


def test_pageworthy_wants_a_sunward_track_at_sungrazer_speed():
    from kreutz.instrument import INSTRUMENTS
    sun = INSTRUMENTS["ccor1"].sun_xy
    comet = {"x": 1519.0, "y": 1422.0, "vx": -8.5, "vy": -4.5}      # Ruan's comet, 2026-10-21 11:00
    satellite = {"x": 1664.0, "y": 415.0, "vx": -7.0, "vy": 2.0}     # the glint of 2026-10-21 07:00
    sideways = {"x": 1406.0, "y": 307.0, "vx": -7.5, "vy": -4.0}     # noise, 2026-10-21 12:45
    creeper = {"x": 1652.0, "y": 608.0, "vx": -1.0, "vy": -2.0}      # the morning ghost
    assert live.pageworthy(comet, sun)
    assert round(live.sunward(comet, sun)[0]) == 14
    assert not live.pageworthy(satellite, sun) and round(live.sunward(satellite, sun)[0]) == 24
    assert not live.pageworthy(sideways, sun)
    assert not live.pageworthy(creeper, sun)  # sunward enough, but 2.2 px/frame


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
    # the 2026-10-15 comet: measured at 9.5 px/frame for ten frames, then the
    # follow-up's step refit drifted to 6 px/frame and it missed four frames
    t0 = datetime(2026, 10, 15, 1, 0, 25)
    pts = [(t0 + timedelta(minutes=15 * k), 1943.0 - 9.5 * k, 1286.0 - 2.5 * k, 6.0) for k in range(10)]
    f = {"x": pts[-1][1], "y": pts[-1][2], "vx": -6.0, "vy": -2.5, "confirmed": True, "points": pts}
    t = t0 + timedelta(minutes=15 * 13)
    true_x, true_y = 1943.0 - 9.5 * 13, 1286.0 - 2.5 * 13
    chains = [[{"time": t.isoformat(), "x": true_x + 2, "y": true_y - 1, "vx": -9.5, "vy": -2.5, "score": 9.0}]]
    hit = live.reanchored(f, chains, t, 4)
    assert hit is not None and hit[0][1] == true_x + 2 and hit[1] == (-9.5, -2.5)
    # something else nearby, moving its own way, isn't taken
    other = [[dict(chains[0][0], vx=-3.0, vy=1.0)]]
    assert live.reanchored(f, other, t, 4) is None
    far = [[dict(chains[0][0], x=true_x - 60)]]
    assert live.reanchored(f, far, t, 4) is None
    # with too few points to fit, the latest step predicts
    g = {"x": 1600.0, "y": 1180.0, "vx": -6.0, "vy": -2.5, "confirmed": True, "points": []}
    near = [[{"time": t.isoformat(), "x": 1562.0, "y": 1170.0, "vx": -7.5, "vy": -2.5, "score": 9.0}]]
    assert live.reanchored(g, near, t, 4) is not None
    sun = (962.0, 1006.0)  # (row, col)
    assert live.where_lost(2020.0, 1200.0, (1920, 2048), sun) == "edge"  # at the right edge
    assert live.where_lost(1450.0, 1050.0, (1920, 2048), sun) == "edge"  # inside the search's inner radius + margin
    assert live.where_lost(1700.0, 1250.0, (1920, 2048), sun) == "lost"


def test_prediction_comes_from_the_recent_line_not_the_last_step():
    # Stark's comet, 2026-10-20: 9 px/frame for an hour of frames, then one
    # off-pointed frame turned the latest step into (-6,-4)
    t0 = datetime(2026, 10, 20, 4, 0, 25)
    pts = [(t0 + timedelta(minutes=15 * k), 1580.0 - 8.7 * k, 1208.0 - 3.2 * k, 40.0) for k in range(12)]
    f = {"x": pts[-1][1], "y": pts[-1][2], "vx": -6.0, "vy": -4.0, "points": pts}
    t = t0 + timedelta(minutes=15 * 17)
    x, y, vx, vy = live.predicted(f, t, 6)
    assert (x, y) == pytest.approx((1580.0 - 8.7 * 17, 1208.0 - 3.2 * 17), abs=0.01)
    assert (vx, vy) == pytest.approx((-8.7, -3.2))
    # only the latest FIT_RECENT points count: an early, slower stretch doesn't drag it
    slow = [(t0 - timedelta(minutes=15 * (k + 1)), 1580.0 + 5.0 * (k + 1), 1208.0, 40.0) for k in range(30)][::-1]
    x2, _, vx2, _ = live.predicted(dict(f, points=slow + pts), t, 6)
    assert x2 == pytest.approx(x, abs=0.01) and vx2 == pytest.approx(-8.7)
    # too few points: the latest step
    assert live.predicted(dict(f, points=pts[:2]), t, 6)[:2] == (pts[-1][1] - 36.0, pts[-1][2] - 24.0)


def test_instruments_name_their_own_frames_and_directories():
    from kreutz.instrument import INSTRUMENTS
    c1, c2 = INSTRUMENTS["ccor1"], INSTRUMENTS["ccor2"]
    assert c1.name_re.search("CCOR1_1B_20261020T201525_V00_NC.fits").group(1) == "20261020T201525"
    assert c2.name_re.search("CCOR2_1A_20261020T201514_V00_NC.fits").group(1) == "20261020T201514"
    assert not c1.name_re.search("CCOR2_1A_20261020T201514_V00_NC.fits")
    assert c1.glob("20261020T2015*") == "CCOR1_1B_20261020T2015*_NC.fits"
    # CCOR-1's directories keep their old names; CCOR-2's are tagged.
    assert (c1.tag, c2.tag) == ("", "-ccor2")
    assert c1.frames_dir.parts[-2:] == ("ccor1", "l1b")
    assert c2.frames_dir.parts[-2:] == ("ccor2", "l1a")
    assert c1.lane_samples.exists() and c1.lane_samples.name == "lane-samples.json"
    assert c2.lane_samples.name == "lane-samples-ccor2.json"


def test_lasco_frames_are_named_by_time_and_kept_apart():
    from kreutz.instrument import INSTRUMENTS
    c2 = INSTRUMENTS["c2"]
    assert c2.name_re.search("C2_05_20261022T031233_23903880.fts").group(1) == "20261022T031233"
    assert not c2.name_re.search("23903880.fts")
    assert c2.glob("20261022T0312*") == "C2_05_20261022T0312*_*.fts"
    assert c2.frames_dir.parts[-2:] == ("c2", "l05") and c2.tag == "-c2"
    assert c2.cadence == timedelta(minutes=12) and c2.shape == (1024, 1024)


def test_lasco_index_keeps_the_synoptic_sequence_only():
    from kreutz import lasco
    text = """  FILE NAME           OBSERVATION     TEL   EXPOSE   NUMBER OF     STARTING    FILTER POLAR   LEB
22999706.fts    2025/08/19  00:00:05   C2    25.1   1024   1024     20      1  Orange  Clear   Normal       0.0000  4179
22999707.fts    2025/08/19  00:12:05   C2    25.8   1024   1024     20      1  Blue    Clear   Normal       0.0000  4179
22999708.fts    2025/08/19  00:24:05   C2    25.1    512    512     20      1  Orange  Clear   Normal       0.0000  4179
22999709.fts    2025/08/19  00:36:05   C2    25.1   1024   1024     20      1  Orange  +60deg  Normal       0.0000  4179
32829961.fts    2025/08/19  00:06:05   C3    17.6   1024   1024     20      1  Clear   Clear   Normal       0.0000  4180
"""
    assert lasco.parse_index(text, "C2") == [(datetime(2025, 8, 19, 0, 0, 5), "22999706.fts")]
    assert lasco.parse_index(text, "C3") == [(datetime(2025, 8, 19, 0, 6, 5), "32829961.fts")]


def test_lasco_north_up_turns_the_roll_away():
    from kreutz import lasco
    img = np.zeros((101, 101), np.float32)
    img[50, 60] = 1.0  # 10 px along +col from the Sun at (50, 50)
    # A roll of 90 deg: that pixel offset is 10 px north on the sky, so
    # north up it's 10 rows up (rows count up, FITS order).
    out = lasco.north_up(img, {"CROTA1": 90.0, "CRPIX1": 51.0, "CRPIX2": 51.0})
    assert np.unravel_index(np.nanargmax(out), out.shape) == (60, 50)
    out = lasco.north_up(img, {"CROTA1": 0.0, "CRPIX1": 51.0, "CRPIX2": 51.0})
    assert out[50, 60] == 1.0


def test_lasco_star_drift_is_the_suns_motion_along_the_ecliptic():
    from kreutz import lasco
    # C3 2026-09-15 05:18: stars measured (-0.02, +0.51) px/frame
    dr, dc = lasco.star_drift(datetime(2026, 9, 15, 5, 18), {"CDELT1": 56.0})
    assert abs(dc / 120 - 0.51) < 0.02 and abs(dr / 120) < 0.03
    # faster at perihelion (early January) than at aphelion (early July):
    # angular speed goes as 1/r^2, ((1 + e) / (1 - e))^2 with e = 0.0167
    jan = lasco.star_drift(datetime(2026, 1, 3), {"CDELT1": 56.0})
    jul = lasco.star_drift(datetime(2026, 7, 4), {"CDELT1": 56.0})
    assert np.hypot(*jan) / np.hypot(*jul) == pytest.approx(1.069, abs=0.005)


def test_frames_off_the_grid_are_kept_a_half_step_apart(tmp_path, monkeypatch):
    from kreutz import frames
    from kreutz.instrument import INSTRUMENTS
    c2 = INSTRUMENTS["c2"]
    monkeypatch.setattr(frames, "INSTRUMENT", c2)
    monkeypatch.setattr(frames, "_NAME", c2.name_re)
    monkeypatch.setattr(frames, "CADENCE", c2.cadence)
    monkeypatch.setattr(frames, "L1B", tmp_path)
    d = tmp_path / "2025/08/19"
    d.mkdir(parents=True)
    # 2025-08-22: a gap, then frames 10.6 min apart, one of them twice
    # (5.6 min after the one before, as after the long gap that evening).
    stamps = ["031205", "032405", "041740", "042817", "043351", "043851"]
    for k, s in enumerate(stamps):
        (d / f"C2_05_20250819T{s}_{22999706 + k}.fts").write_bytes(b"x")  # not empty: empty files are skipped
    got = [frames.frame_time(p) for p in frames.frame_paths(datetime(2025, 8, 19), datetime(2025, 8, 19, 5))]
    assert [f"{t:%H%M%S}" for t in got] == ["031205", "032405", "041740", "042817", "043851"]
    seq = FakeSeq(got)
    assert Sequence.index(seq, datetime(2025, 8, 19, 4, 28)) == 3
    with pytest.raises(ValueError):
        Sequence.index(seq, datetime(2025, 8, 19, 3, 50))


def test_seasonal_lane_takes_the_season_and_widens_where_thin(monkeypatch):
    from kreutz import lane
    from kreutz.instrument import INSTRUMENTS
    monkeypatch.setattr(lane, "INSTRUMENT", INSTRUMENTS["c3"])
    monkeypatch.setattr(lane, "SEASON_MIN", 3)
    # (x, y, vx, vy, score, time): September in two years, one in late
    # December, one in early January (14 days apart across the new year)
    S = [[800, 700, -2, -2, 0, "2024-09-10T00:00:00"], [801, 701, -2, -2, 0, "2025-09-20T00:00:00"],
         [802, 702, -2, -2, 0, "2026-09-14T00:00:00"], [300, 800, 2, -2, 0, "2025-12-25T00:00:00"],
         [301, 801, 2, -2, 0, "2026-01-08T00:00:00"]]
    assert len(lane.season_samples(S, datetime(2026, 9, 15))) == 3
    # Jan 1: two within 15 days, so the window doubles and still has two,
    # then doubles again (60 days) and stops there
    assert len(lane.season_samples(S, datetime(2027, 1, 1))) == 2
    monkeypatch.setattr(lane, "INSTRUMENT", replace(INSTRUMENTS["ccor1"], season_days=None))
    assert lane.season_samples(S, datetime(2026, 9, 15)) == S  # one lane all year


def test_season_sector_follows_the_samples_and_ignores_strays(monkeypatch):
    from kreutz import lane
    from kreutz.instrument import INSTRUMENTS
    monkeypatch.setattr(lane, "INSTRUMENT", INSTRUMENTS["c3"])
    sun = (500.0, 500.0)
    # 30 samples 300 px out at -40 deg +-10, and three strays straight up
    angs = np.radians(np.linspace(-50, -30, 30))
    S = [[500 + 300 * np.cos(a), 500 - 300 * np.sin(a), 0, 0, 0, "2026-09-15T00:00:00"] for a in angs]
    S += [[500, 200, 0, 0, 0, "2026-09-15T00:00:00"]] * 3
    monkeypatch.setattr(lane, "lane_samples", lambda when=None: S)
    center, half = lane.season_sector(datetime(2026, 9, 15), sun)
    assert abs(center + 40) < 1 and abs(half - (10 + lane.SECTOR_MARGIN)) < 1


def test_follow_measures_in_the_instruments_steps_not_the_first_gap(monkeypatch):
    from kreutz import frames, vet
    from kreutz.instrument import INSTRUMENTS
    monkeypatch.setattr(vet, "CADENCE", INSTRUMENTS["c2"].cadence)
    # LASCO-like: a 36-minute gap first, then 12-minute steps; a point
    # source moving (-3, 0) px per 12 minutes
    t0 = datetime(2026, 9, 16, 12, 0)
    times = [t0] + [t0 + timedelta(minutes=36 + 12 * k) for k in range(12)]
    H = W = 200
    yy, xx = np.mgrid[:H, :W]

    def raw(i):
        x = 170 - 3 * (times[i] - t0) / timedelta(minutes=12)
        return (50 * np.exp(-((xx - x) ** 2 + (yy - 100) ** 2) / (2 * frames.PSF_SIGMA ** 2))).astype(np.float32)
    seq = types.SimpleNamespace(times=times, shape=(H, W), raw=raw,
                                _displacement=lambda j, i: (lambda r, c: (0.0, 0.5)))
    last = len(times) - 1
    x_last = 170 - 3 * (times[last] - t0) / timedelta(minutes=12)
    m = vet.follow(seq, last, x_last, H - 1 - 100, -3.0, 0.0)
    assert m.measured and abs(m.vx + 3.0) < 0.2 and abs(m.vy) < 0.2


def test_cosmic_rays_flags_hits_and_streaks_but_not_psf_points():
    from kreutz import frames
    rng = np.random.default_rng(1)
    d = rng.normal(0, 1, (120, 120)).astype(np.float32)
    yy, xx = np.mgrid[-6:7, -6:7]
    d[50:63, 50:63] += 40 * np.exp(-(xx ** 2 + yy ** 2) / (2 * frames.PSF_SIGMA ** 2))  # a comet
    d[20, 20] += 40           # a hit
    d[80, 30:37] += 40        # a streak
    cr = frames.cosmic_rays(d)
    assert cr[20, 20] and cr[80, 33]
    assert not cr[44:69, 44:69].any()


def test_fill_dark_fills_dark_spots_and_keeps_bright_points_and_nans():
    from kreutz import frames
    rng = np.random.default_rng(2)
    raw = (100 + rng.normal(0, 1, (120, 120))).astype(np.float32)
    raw[30:33, 40:43] = 60           # a dark defect
    raw[80, 80] = 160                # a bright point
    raw[:5] = np.nan                 # off the detector
    out = frames.fill_dark(raw)
    assert abs(out[31, 41] - 100) < 3 and abs(out[30, 40] - 100) < 3
    assert out[80, 80] == 160
    assert np.isnan(out[:5]).all() and np.isfinite(out[5:]).all()


def test_status_and_pushes_name_the_instrument_and_its_live_directory():
    from kreutz.instrument import INSTRUMENTS
    c1, c2 = INSTRUMENTS["ccor1"], INSTRUMENTS["ccor2"]
    assert c1.measure_cmd("alerts/a.json") == "docker compose run --rm kreutz kreutz.report /data/live/alerts/a.json"
    assert c2.measure_cmd("alerts/a.json") == \
        "KREUTZ_INSTRUMENT=ccor2 docker compose run --rm kreutz kreutz.report /data/live-ccor2/alerts/a.json"
    s = live.status([], [], datetime(2026, 10, 9, 11, 0, 25), [], datetime(2026, 10, 9, 12, 0))
    assert s["instrument"] == "CCOR-1" and s["measure_prefix"] == c1.measure_cmd("")


def test_page_lists_every_instrument_and_redirects_old_pages(tmp_path):
    from kreutz import page
    (tmp_path / "live").mkdir()
    page.write(tmp_path)
    insts = json.load(open(tmp_path / "instruments.json"))
    assert [(i["key"], i["live"]) for i in insts] == [("ccor1", "live"), ("ccor2", "live-ccor2"),
                                                      ("c2", "live-c2"), ("c3", "live-c3")]
    assert insts[1]["measure_prefix"].startswith("KREUTZ_INSTRUMENT=ccor2 ")
    assert "instruments.json" in (tmp_path / "index.html").read_text()
    assert "url=../index.html" in (tmp_path / "live" / "status.html").read_text()
    assert not (tmp_path / "live-ccor2").exists()  # no watcher has run there: nothing made


def _audit_report():
    t0 = datetime(2026, 10, 16, 10, 0, 25)
    r = {"id": "report_xx_1", "reporter": "X", "group": "Kreutz", "kind": "Potential Comet"}
    return r, [(t0, 1500.0, 1200.0), (t0 + timedelta(minutes=30), 1470.0, 1188.0)]


def _audit_record(kind, alert=None, x=1470.0, y=1188.0):
    rec = {"kind": kind, "decided_utc": "2026-10-16T10:45:00", "first": "2026-10-16T10:00:25",
           "last": "2026-10-16T10:30:25", "x": x, "y": y, "vx": -15.0, "vy": -6.0}
    if alert:
        rec["alert"] = f"/data/live/alerts/{alert}.json"
    return rec


def test_audit_judges_reports_against_what_the_watcher_did():
    from kreutz import audit
    r, pts = _audit_report()
    later = pts[-1][0] + timedelta(hours=3)

    def judge(*recs, fate=None, newest=later, chains=()):
        cands = audit.candidates(list(recs), {"chains": list(chains)})
        return audit.judge(r, pts, cands, fate or {}, newest)[0]

    alert = _audit_record("alert", "a")
    assert judge(alert, fate={"a": "confirmed"}) == "OK"
    assert judge(alert, fate={"a": "faded"}) == "LOST"
    assert judge(_audit_record("star")) == "VETO"
    assert judge(_audit_record("star", x=900.0)) == "MISS"  # nowhere near it
    assert judge() == "MISS"
    assert judge(newest=pts[-1][0] + timedelta(hours=1)) is None  # not due yet
    chain = [{"time": "2026-10-16T10:30:25", "x": 1470.0, "y": 1188.0, "vx": -15.0, "vy": -6.0}]
    assert judge(chains=[chain]) is None  # still being tracked
    # the watcher's own match counts even where positions don't
    far = dict(_audit_record("reported", "b", x=900.0), report="X report_xx_1 (CCOR1)")
    assert judge(far) == "OK"


def test_audit_false_alerts_skip_matched_ones():
    from kreutz import audit
    log = [_audit_record("alert", "a"), _audit_record("faded", "a"),
           _audit_record("alert", "b"), _audit_record("faded", "b"),
           _audit_record("alert", "c"), _audit_record("confirmed", "c")]
    assert set(audit.false_alerts(log)) == {"a", "b"}
    assert set(audit.false_alerts(log, matched={"b"})) == {"a"}


def _stack_seq(maps, t0=datetime(2026, 10, 22, 5, 0, 14)):
    import types
    return types.SimpleNamespace(times=[t0 + timedelta(minutes=15) * i for i in range(len(maps))],
                                 shape=maps[0].shape, significance=lambda i: maps[i])


def test_line_stack_vetoes_a_stripe_but_not_a_moving_point():
    from kreutz import vet
    rng = np.random.default_rng(3)
    H, W = 200, 200
    yy, xx = np.mgrid[0:H, 0:W]
    vx, vy = -4.0, -4.0  # px/frame, report coords (y down)
    x8, y8 = 100.0, 100.0  # where the track is in the last frame
    point, stripe = [], []
    for i in range(8):
        k = i - 7
        x, y = x8 + vx * k, y8 + vy * k
        r, c = H - 1 - y, x
        noise = rng.normal(0, 1, (H, W)).astype(np.float32)
        point.append(noise + 4.0 * np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * 1.5 ** 2)))
        # a static diagonal band lying along the track's direction: the track
        # goes down-left in array terms (row up as y falls, col down with x)
        d = np.abs((yy - (H - 1 - y8)) + (xx - x8)) / np.sqrt(2)
        stripe.append(noise + 2.0 * np.exp(-d ** 2 / (2 * 2.0 ** 2)))
    peak, ratio, length, area = vet.stack_shape(_stack_seq(point), 7, x8, y8, vx, vy)
    assert peak > 8 and ratio < 2.0 and length < 15, (peak, ratio, length, area)
    assert vet.line_stack(_stack_seq(point), 7, x8, y8, vx, vy) is None
    peak, ratio, length, area = vet.stack_shape(_stack_seq(stripe), 7, x8, y8, vx, vy)
    assert ratio >= 4 and length >= 20, (peak, ratio, length, area)
    assert vet.line_stack(_stack_seq(stripe), 7, x8, y8, vx, vy).startswith("a line in its own stack")


def test_through_line_scores_a_line_through_the_track_not_a_point_or_its_tail():
    from kreutz import vet
    rng = np.random.default_rng(5)
    H, W = 200, 200
    yy, xx = np.mgrid[0:H, 0:W]
    vx, vy = -4.0, -4.0
    x8, y8 = 100.0, 100.0
    point, tailed, streak = [], [], []
    for i in range(8):
        k = i - 7
        x, y = x8 + vx * k, y8 + vy * k
        r, c = H - 1 - y, x
        noise = rng.normal(0, 1, (H, W)).astype(np.float32)
        blob = 4.0 * np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * 1.5 ** 2))
        point.append(noise + blob)
        # a tail trailing the point by up to 40 px, against the motion
        along = ((xx - c) * -1 + (yy - r) * -1) / np.sqrt(2)  # px behind the point
        across = np.abs((xx - c) - (yy - r)) / np.sqrt(2)
        tailed.append(noise + blob + ((along > 2) & (along < 40)) * np.exp(-across ** 2 / 4.0))
        # the CCOR-2 streak: faint, broken, along the track's direction
        d = np.abs((yy - (H - 1 - y8)) + (xx - x8)) / np.sqrt(2)
        gaps = (((xx + yy) // 9) % 3 != 0)
        streak.append(noise + 1.0 * gaps * np.exp(-d ** 2 / (2 * 1.5 ** 2)))
    for maps in (point, tailed):
        score, _ = vet.through_line(_stack_seq(maps), 7, x8, y8, vx, vy)
        assert score < 0.5, score
    score, angle = vet.through_line(_stack_seq(streak), 7, x8, y8, vx, vy)
    assert score > 1.0 and angle in (0.0, 177.0, 3.0), (score, angle)


def test_line_like_needs_more_than_one_frame_to_call_a_line():
    from kreutz import vet
    H, W = 120, 120
    yy, xx = np.mgrid[0:H, 0:W]
    vx, vy = -5.0, -4.5
    x8, y8 = 60.0, 60.0

    def frames(bars):
        maps = []
        for i in range(8):
            k = 7 - i
            r, c = H - 1 - (y8 - vy * k), x8 - vx * k
            m = 5.0 * np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * 1.2 ** 2))
            if i in bars:  # the peak merged into a short bar, as the 2026-10-24 12:00 frame
                m = 5.0 * np.exp(-((yy - r) ** 2) / (2 * 1.0 ** 2) - (xx - c) ** 2 / (2 * 6.0 ** 2))
            maps.append(m.astype(np.float32))
        return _stack_seq(maps)

    assert vet.line_like(frames(()), 7, x8, y8, vx, vy) is None
    assert vet.line_like(frames((7,)), 7, x8, y8, vx, vy) is None  # one bar in eight: a comet
    why = vet.line_like(frames((5, 7)), 7, x8, y8, vx, vy)
    assert why.startswith("a line, not a point") and "in 2 of 8 frames" in why


def test_lasco_bleed_mask_takes_a_saturation_line_but_not_a_bright_ring():
    import importlib, os
    old = os.environ.get("KREUTZ_INSTRUMENT")
    os.environ["KREUTZ_INSTRUMENT"] = "c3"
    try:
        import kreutz.instrument, kreutz.lasco
        importlib.reload(kreutz.instrument); lasco = importlib.reload(kreutz.lasco)
        rng = np.random.default_rng(4)
        raw = (1000 + rng.normal(0, 20, (400, 400))).astype(np.float32)
        yy, xx = np.mgrid[0:400, 0:400]
        rr = np.hypot(yy - 200, xx - 200)
        raw[np.abs(rr - 60) < 3] = 13000          # a bright ring: short runs along rows
        raw[300, 20:380] = 13500                  # a bleed line, 360 px
        raw[100, 300] = 14000                     # a cosmic ray
        m = lasco.bleed_mask(raw)
        assert m[300, 200] and m[302, 30] and not m[296, 200]
        assert not m[140, 200] and not m[200, 140] and not m[100, 300]
        quiet = raw.copy(); quiet[300] = 1000
        assert not lasco.bleed_mask(quiet).any()
    finally:
        if old is None: os.environ.pop("KREUTZ_INSTRUMENT", None)
        else: os.environ["KREUTZ_INSTRUMENT"] = old
        import kreutz.instrument, kreutz.lasco
        importlib.reload(kreutz.instrument); importlib.reload(kreutz.lasco)


def test_notify_holds_an_orbit_only_candidate_until_confirmed():
    from kreutz import notify
    track = [{"time": "2026-10-24T11:30:14", "x": 1631.0, "y": 1600.0, "vx": -5.0, "vy": -4.5}]
    d = {"track": track, "other_instruments": [], "paged": False, "heading_deg": 3, "found_by": "orbit"}
    assert notify.compose(d, "a.json") is None
    title, lines, prio, _ = notify.compose(dict(d, kind="confirmed", frames=5, note="ok"), "a.confirmed.json")
    assert prio == "urgent" and "found by the orbit search only" in lines[0] and "off sunward" not in lines[0]


def test_notify_holds_a_track_on_no_kreutz_orbit_and_confirms_it_quietly():
    from kreutz import notify
    track = [{"time": "2026-10-28T16:00:25", "x": 1421.0, "y": 1441.0, "vx": -8.0, "vy": -5.0}]
    d = {"track": track, "other_instruments": [], "paged": False, "heading_deg": 12,
         "orbit_rms": 2540, "off_orbit": True}
    assert notify.compose(d, "a.json") is None
    assert notify.held_because(d) == 'fits no Kreutz orbit (2540")'
    title, lines, prio, _ = notify.compose(dict(d, kind="confirmed", frames=9, note="ok"), "a.confirmed.json")
    assert prio == "low" and 'fits no Kreutz orbit (2540")' in lines[0] and "off sunward" not in lines[0]


def test_orbit_gate_measures_the_track_and_fails_open(monkeypatch):
    track = [{"time": f"2026-10-28T16:{m:02d}:25", "x": 1421.0 - m, "y": 1441.0 - m, "vx": -8.0, "vy": -5.0}
             for m in (0, 15, 30)]
    seen = []
    monkeypatch.setattr(live, "orbit_of", lambda pts: seen.append(pts) or (None, 2540.4))
    assert live.orbit_rms(track) == 2540 and seen[0][1] == (datetime(2026, 10, 28, 16, 15, 25), 1406.0, 1426.0)
    monkeypatch.setattr(live, "orbit_of", lambda pts: None)  # too few points on frames on disk
    assert live.orbit_rms(track) is None

    def fails(pts):
        raise ValueError("date value out of range")
    monkeypatch.setattr(live, "orbit_of", fails)
    assert live.orbit_rms(track) is None  # a fit that can't run holds nothing
    assert live.ORBIT_GATE and live.OFF_ORBIT_RMS == 1000.0  # CCOR-1, the default instrument


def test_notify_holds_a_track_on_a_line_but_confirms_it_in_full(monkeypatch):
    from kreutz import notify
    track = [{"time": "2026-10-28T14:45:14", "x": 1567.0, "y": 1592.0, "vx": -9.5, "vy": -8.0}]
    d = {"track": track, "other_instruments": [], "paged": False, "heading_deg": 2,
         "line_score": 3.37, "on_line": True}
    assert notify.compose(d, "a.json") is None
    assert notify.held_because(d) == "a line runs through it (3.37)"
    title, lines, prio, _ = notify.compose(dict(d, kind="confirmed", frames=6, note="ok"), "a.confirmed.json")
    assert prio == "urgent" and "a line ran through it as a candidate (3.37)" in lines[0]
    monkeypatch.setattr(live, "through_line", lambda *a: (3.374, 0.0))
    assert live.line_score(None, 7, track[0]) == 3.37

    def fails(*a):
        raise IndexError("frame missing")
    monkeypatch.setattr(live, "through_line", fails)
    assert live.line_score(None, 7, track[0]) is None  # can't measure: the page isn't held


def test_a_point_on_the_rolled_lasco_frame_turns_to_ours(monkeypatch):
    # Fields' C2 comet of 2026-10-24 at 04:36: (873, 939) on the frame as it
    # came, rolled by CROTA1 -4.77; our north-up track had it at (837, 966)
    monkeypatch.setattr(evaluate, "_roll_at", lambda t: (-4.77293, 510.2, 516.5))
    x, y = evaluate.lasco_rolled_to_report(datetime(2026, 10, 24, 4, 36), 873.0, 939.0)
    assert np.hypot(x - 837.0, y - 966.0) < 3.0
    assert np.hypot(x - 510.2, y - 516.5) == pytest.approx(np.hypot(873.0 - 510.2, 939.0 - 516.5))
    monkeypatch.setattr(evaluate, "_roll_at", lambda t: None)  # no frame near the report time
    assert evaluate.lasco_rolled_to_report(datetime(2026, 10, 24, 4, 36), 873.0, 939.0) is None


def _rep(rid, kind, posted, reporter="Eryk Banach", telescope="CCOR2"):
    return {"id": rid, "kind": kind, "reporter": reporter, "telescope": telescope,
            "posted_utc": posted, "group": "Kreutz", "points": []}


def test_a_retraction_withdraws_the_reporters_latest_report_only():
    reps = [_rep("a", "Potential Comet", "2026-10-22T07:00:00Z"),
            _rep("b", "Potential Comet", "2026-10-22T08:36:02Z"),
            _rep("x", "Retracting Report", "2026-10-22T08:48:42Z"),
            _rep("c", "Potential Comet", "2026-10-22T08:40:00Z", reporter="Jiangao Ruan"),
            _rep("d", "Potential Comet", "2026-10-22T08:40:00Z", telescope="CCOR1"),
            _rep("e", "Potential Comet", "2026-10-22T09:00:00Z")]  # after the retraction
    assert evaluate.retracted(reps) == {"b"}
    assert evaluate.retracted([_rep("x", "Retracting Report", "2026-10-22T08:48:42Z")]) == set()


def test_a_retraction_naming_no_telescope_withdraws_the_reporters_latest_on_any():
    # report_jr_20261030112229: "Retracting Report", no group, no telescope
    reps = [_rep("a", "Potential Comet", "2026-10-30T14:00:00Z", telescope="C2"),
            _rep("b", "Potential Comet", "2026-10-30T15:09:12Z", telescope="CCOR1"),
            _rep("x", "Retracting Report", "2026-10-30T15:22:29Z", telescope=None)]
    reps[2]["group"] = None
    assert evaluate.retracted(reps) == {"b"}
    # cached by the old parse: the telescope read as the group
    reps[2]["group"] = "C2"
    assert evaluate.retracted(reps) == {"a"}


def test_a_report_without_a_group_keeps_its_telescope():
    lines = ["report_jr_20261030112229", "Jiangao Ruan", "2026-10-30", "Retracting Report", "C2",
             "report_jr_20261030110912", "Jiangao Ruan", "2026-10-30", "Potential Comet", "Kreutz", "CCOR1",
             "14:15 1436 1289", "14:30 1427 1284"]
    r, q = reports.parse(lines)
    assert (r["group"], r["telescope"]) == (None, "C2")
    assert (q["group"], q["telescope"]) == ("Kreutz", "CCOR1")
    assert q["points"] == [["1415", 1436.0, 1289.0], ["1430", 1427.0, 1284.0]]


def test_a_correction_joins_the_report_it_amends_by_its_path():
    t = lambda h, m: datetime(2026, 10, 6, h, m)  # noqa: E731
    orig = [(t(11, 45), 1700.0, 1650.0), (t(12, 0), 1692.0, 1643.0), (t(12, 15), 1684.0, 1636.0)]
    corr = [(t(11, 45), 1688.0, 1640.0), (t(12, 0), 1680.0, 1633.0), (t(12, 15), 1672.0, 1626.0)]  # ~15 px off
    more = [(t(13, 0), 1660.0, 1615.0), (t(13, 15), 1652.0, 1608.0)]  # the same path, later
    other = [(t(12, 0), 1300.0, 1200.0), (t(12, 15), 1292.0, 1193.0)]  # another comet
    rp = {"o": (_rep("o", "Potential Comet", "2026-10-06T13:02:00Z"), orig),
          "c": (_rep("c", "Correction", "2026-10-06T13:02:22Z"), corr),
          "m": (_rep("m", "Additional Observations", "2026-10-06T14:00:00Z"), more),
          "z": (_rep("z", "Additional Observations", "2026-10-06T14:30:00Z"), other)}
    assert evaluate.amended(rp) == {"c": "o", "m": "o"}  # z continues nothing
    rp["k"] = (_rep("k", "Correction", "2026-10-06T13:10:00Z", reporter="Jiangao Ruan"), corr)
    assert "k" not in evaluate.amended(rp)  # someone else's report isn't amended


def test_orbit_fit_keeps_q_positive(monkeypatch):
    # A free q fitted to a track it doesn't suit went negative and NaN'd the
    # motion (our Marsden track with Ruan's fragment: 3 RuntimeWarnings).
    import scipy.optimize
    from kreutz import orbit
    seen = {}
    real = scipy.optimize.least_squares

    def spy(fun, x0, **kw):
        seen.update(kw)
        return real(fun, x0, **kw)
    monkeypatch.setattr(scipy.optimize, "least_squares", spy)
    e0 = orbit.Elements(T=datetime(2026, 10, 28, 12), q=0.047, i=26.6, node=79.5, peri=24.3)
    ts = [datetime(2026, 10, 28, 21, 45) + timedelta(minutes=15 * k) for k in range(4)]
    orbit.fit(ts, [1.5, 1.51, 1.52, 1.53], [2.5, 2.2, 1.9, 1.6], e0, free=("T", "q", "i"))
    lo, hi = seen["bounds"]
    assert lo == [-orbit.T_RANGE_H, orbit.Q_MIN, -np.inf] and hi[1] == np.inf


def test_a_track_on_no_kreutz_orbit_is_matched_to_other_groups_reports(monkeypatch):
    t0 = datetime(2026, 10, 28, 21, 45)
    track = [{"t": t0 + timedelta(minutes=15 * k), "x": 1357.0 - 3.4 * k, "y": 527.0 - 3.4 * k} for k in range(4)]
    pts = [(datetime(2026, 10, 28, 5, 30), 1350.0, 962.0), (datetime(2026, 10, 28, 6, 0), 1357.0, 945.0)]
    marsden = {"id": "wb", "group": "Marsden", "posted_utc": "2026-10-28T06:44:39Z"}
    fragment = {"id": "jr", "group": "Marsden", "posted_utc": "2026-10-28T12:47:44Z"}
    kreutz = {"id": "cw", "group": "Kreutz", "posted_utc": "2026-10-28T12:00:00Z"}
    monkeypatch.setattr(live, "orbit_of", lambda own: (None, 6480.0))  # no Kreutz orbit
    tried = []

    def rms(group, own, other, inst=None):
        tried.append(group)
        return 7.0 if other is pts else 829.0
    monkeypatch.setattr(live, "family_rms", rms)
    got = live.by_orbit(track, [(marsden, pts, None), (fragment, pts[:1] * 2, None), (kreutz, pts, None)])
    assert got == [marsden] and "Kreutz" not in tried
    monkeypatch.setattr(live, "family_rms", lambda *a, **k: float("inf"))
    assert live.by_orbit(track, [(marsden, pts, None)]) is None  # nothing matched: position decides


def test_notify_logs_why_a_candidate_was_held():
    from kreutz import notify
    d = {"track": [], "paged": False, "heading_deg": 7, "lane_px": 74}
    assert notify.held_because(dict(d, found_by="orbit")) == "found by the orbit search only"
    assert "off sunward" in notify.held_because(d) and "lane" not in notify.held_because(d)
    assert notify.held_because(dict(d, known={"reporter": "x"})) == "known comet"


def test_orbit_fit_recovers_a_kreutz_track():
    from kreutz import orbit
    e = orbit.Elements(T=datetime(2026, 10, 23, 18, 0), node=12.0, peri=89.0, i=143.9)
    times = [datetime(2026, 10, 23, 8, 0) + timedelta(minutes=15 * k) for k in range(12)]
    tx, ty = orbit.helioprojective(e, times, "earth")
    assert np.all(np.hypot(tx, ty) < 20) and np.all(np.diff(np.hypot(tx, ty)) < 0)  # in view, falling sunward
    start = orbit.Elements(T=datetime(2026, 10, 23, 16, 0), node=9.0, peri=86.6, i=143.9)
    fitted, rms = orbit.fit(times, tx, ty, start, "earth")
    assert rms < 1 and abs((fitted.T - e.T).total_seconds()) < 120 and abs(fitted.node - 12.0) < 0.2


def test_orbit_search_finds_a_comet_along_its_orbit(monkeypatch):
    import types
    from kreutz import orbit, orbitsearch
    from astropy.wcs import WCS
    # a CCOR-1-like plate: 2048 px wide, ~0.0103 deg/px, Sun at the center
    w = WCS(naxis=2)
    w.wcs.ctype = ["HPLN-TAN", "HPLT-TAN"]
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.cdelt = [0.0103, 0.0103]
    w.wcs.crpix = [1024.5, 960.5]
    w.wcs.crval = [0.0, 0.0]
    H, W = 1920, 2048
    t_last = datetime(2026, 10, 23, 10, 0)
    times = [t_last - timedelta(minutes=15 * k) for k in range(7, -1, -1)]
    e = orbit.Elements(T=t_last + timedelta(hours=8), node=10.0, peri=orbitsearch.PERI_A + orbitsearch.PERI_B * 10.0,
                       i=orbitsearch.INCLINATION)
    tx, ty = orbit.helioprojective(e, times, "earth")
    col, row = w.wcs_world2pix(tx, ty, 0)
    rng = np.random.default_rng(5)
    maps = []
    for c, r in zip(col, row):
        m = rng.normal(0, 1, (H, W)).astype(np.float32)
        m[int(round(r)), int(round(c))] += 3.0  # 3 sigma a frame: 8.5 stacked
        maps.append(m)
    paths = [f"f{k}" for k in range(8)]
    monkeypatch.setattr(orbitsearch, "_wcs", lambda p: (w, H, W))
    monkeypatch.setattr(orbitsearch, "INSTRUMENT", types.SimpleNamespace(key="ccor1"))
    seq = types.SimpleNamespace(times=times, paths=paths, shape=(H, W), significance=lambda i: maps[i],
                                rc_to_report=lambda r, c: (c, H - 1 - r))
    cands = orbitsearch.orbit_search(seq, 7, 8, 6.0)
    best = cands[0]
    assert best.score > 7 and np.hypot(best.x - col[-1], best.y - (H - 1 - row[-1])) < 3, best


def test_orbit_match_ties_a_report_to_a_track_a_day_later(monkeypatch):
    from kreutz import orbit, orbitsearch
    # positions in degrees stand in for report coordinates: no WCS needed
    monkeypatch.setattr(orbitsearch, "sky_points",
                        lambda pts, inst=None: ([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]))
    monkeypatch.setattr(orbitsearch, "_where", lambda inst=None: "l1")
    e = orbit.Elements(T=datetime(2026, 10, 25, 16, 0), node=12.0,
                       peri=orbitsearch.PERI_A + orbitsearch.PERI_B * 12.0, i=orbitsearch.INCLINATION)

    def path(e, start, n):
        times = [start + timedelta(minutes=15 * k) for k in range(n)]
        tx, ty = orbit.helioprojective(e, times, "l1")
        return list(zip(times, tx, ty))
    report = path(e, datetime(2026, 10, 24, 7, 45), 6)  # reported in the morning
    track = path(e, datetime(2026, 10, 25, 0, 15), 12)  # followed from after midnight
    other = path(replace(e, T=e.T + timedelta(hours=6)), datetime(2026, 10, 24, 20, 15), 5)  # the next comet along
    best, rms = orbitsearch.orbit_of(track)
    assert rms < 60
    assert orbitsearch.joint_rms(best, track, report) < orbitsearch.SAME_RMS
    assert orbitsearch.joint_rms(best, track, other) > orbitsearch.SAME_RMS


def test_orbit_match_across_instruments(monkeypatch):
    from kreutz import orbit, orbitsearch
    from kreutz.instrument import INSTRUMENTS
    monkeypatch.setattr(orbitsearch, "sky_points",
                        lambda pts, inst=None: ([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]))
    monkeypatch.setattr(orbitsearch, "INSTRUMENT", INSTRUMENTS["ccor1"])
    e = orbit.Elements(T=datetime(2026, 10, 24, 6, 0), node=13.0,
                       peri=orbitsearch.PERI_A + orbitsearch.PERI_B * 13.0, i=orbitsearch.INCLINATION)

    def path(e, start, n, where):
        times = [start + timedelta(minutes=15 * k) for k in range(n)]
        tx, ty = orbit.helioprojective(e, times, where)
        return list(zip(times, tx, ty))
    track = path(e, datetime(2026, 10, 23, 8, 0), 12, "earth")        # CCOR-1, from Earth
    report = path(e, datetime(2026, 10, 22, 23, 45), 6, "l1")         # the CCOR-2 report, from L1
    best, rms = orbitsearch.orbit_of(track)
    assert rms < 60
    assert orbitsearch.joint_rms(best, track, report, INSTRUMENTS["ccor2"]) < 60
    later = path(replace(e, T=e.T + timedelta(hours=6)), datetime(2026, 10, 23, 5, 45), 6, "l1")
    assert orbitsearch.joint_rms(best, track, later, INSTRUMENTS["ccor2"]) > orbitsearch.SAME_RMS


def test_pull_keeps_only_fits_files(tmp_path):
    from kreutz import sync
    good, empty = tmp_path / "a.fts", tmp_path / "b.fts"
    bodies = {"u/a": b"SIMPLE  =                    T" + b" " * 50, "u/b": b""}
    assert sync.pull([(None, "u/a", good), (None, "u/b", empty)], get=bodies.__getitem__) == 1
    assert good.exists() and not empty.exists() and not empty.with_suffix(".part").exists()


def test_pull_writes_through_a_part_file_of_its_own(tmp_path, monkeypatch):
    from kreutz import sync
    dest = tmp_path / "a.fts"
    shared = dest.with_suffix(".part")
    shared.write_bytes(b"another process's half-written frame")
    body = b"SIMPLE  =                    T" + b" " * 50
    assert sync.pull([(None, "u/a", dest)], get=lambda url: body) == 1
    assert dest.read_bytes() == body and shared.exists()  # the other process's part file untouched
    assert not list(tmp_path.glob(".*.part"))


def test_status_carries_the_wait_after_a_data_gap():
    t = datetime(2026, 10, 26, 19, 30)
    s = live.status([], [], t, [], t, waiting="waiting for 4 more frames after a data gap")
    assert s["waiting"].startswith("waiting for 4") and s["updated_utc"] == "2026-10-26T19:30:00"
    assert live.status([], [], t, [], t)["waiting"] is None


def test_a_search_waits_for_fresh_frames_after_a_long_gap():
    t0 = datetime(2026, 10, 26, 0, 0)
    times = [t0 + timedelta(minutes=12 * k) for k in range(20)]                    # until 03:48
    times += [t0 + timedelta(hours=8, minutes=12 * k) for k in range(5)]          # 4 h gap, then 5 frames
    starts = live.run_starts(times)
    assert starts[19] == 0 and starts[20:] == [20] * 5
    short = [t0 + timedelta(minutes=12 * k) for k in range(10)] + [t0 + timedelta(hours=3, minutes=12 * k) for k in range(5)]
    assert live.run_starts(short) == [0] * 15  # a gap under BG_MAX_GAP doesn't restart the run


def test_orbit_fit_keeps_the_perihelion_time_in_range():
    from kreutz import orbit
    # a point 42 deg from the Sun: unbounded, the fit moved T 14.8 days
    times = [datetime(2026, 10, 27, 4, 0) + timedelta(minutes=15 * k) for k in range(4)]
    start = orbit.Elements(T=datetime(2026, 10, 27, 12, 0), node=12.0, peri=89.0, i=143.9)
    e, rms = orbit.fit(times, [30.0] * 4, [-30.0] * 4, start, "l1", free=("T",))
    assert abs((e.T - start.T).total_seconds()) <= orbit.T_RANGE_H * 3600 + 1


def test_page_writes_from_two_containers_use_their_own_temp_files(tmp_path, monkeypatch):
    # every watcher is pid 1 in its own container; after a reboot four
    # started together, and two died renaming a temp file another had taken
    from kreutz import page
    monkeypatch.setattr(page.os, "getpid", lambda: 1)
    seen = []
    real = page.Path.write_text

    def write_text(self, text, *a, **k):
        seen.append(self.name)
        return real(self, text, *a, **k)
    monkeypatch.setattr(page.Path, "write_text", write_text)
    for host in ("a1b2c3", "d4e5f6"):
        monkeypatch.setattr(page.socket, "gethostname", lambda h=host: h)
        page._write(tmp_path / "instruments.json", host)
    assert len(set(seen)) == 2
    assert (tmp_path / "instruments.json").read_text() == "d4e5f6"


def test_frame_downloads_from_two_containers_use_their_own_part_files(tmp_path, monkeypatch):
    # the CCOR-2 watcher and the sync-ccor2 mirror, both pid 1, pull the same new frame
    from kreutz import sync
    monkeypatch.setattr(sync.os, "getpid", lambda: 1)
    seen = []
    real = type(tmp_path).write_bytes

    def write_bytes(self, data):
        seen.append(self.name)
        return real(self, data)
    monkeypatch.setattr(type(tmp_path), "write_bytes", write_bytes)
    dest = tmp_path / "CCOR2_1A_20261105T004514_V00_NC.fits"
    for host in ("a1b2c3", "d4e5f6"):
        monkeypatch.setattr(sync.socket, "gethostname", lambda h=host: h)
        dest.unlink(missing_ok=True)
        assert sync.pull([(None, "u", dest)], get=lambda url: b"SIMPLE  = T") == 1
    assert len(set(seen)) == 2
    assert dest.read_bytes() == b"SIMPLE  = T"


def test_orbit_long_takes_frames_and_threshold():
    import argparse
    assert live.orbit_long("16:7.0") == (16, 7.0)
    assert live.orbit_long("24:7") == (24, 7.0)
    for bad in ("16", "x:7", "16:x", "1:7.0"):
        with pytest.raises(argparse.ArgumentTypeError):
            live.orbit_long(bad)


def test_a_long_chain_where_only_a_faded_alert_was_is_alerted_again(monkeypatch):
    # the comet of 2026-11-02: alerted 10:15, faded 12:01, chained on all afternoon
    t0 = datetime(2026, 11, 2, 10, 15, 25)
    alert = {"time": t0.isoformat(), "x": 1775.0, "y": 1536.0, "vx": -7.0, "vy": -3.5,
             "why": "alert: 20261102T1015_1775_1536"}
    later = t0 + timedelta(hours=3)  # 12 frames on
    c = {"x": 1775.0 - 7.0 * 12, "y": 1536.0 - 3.5 * 12}
    monkeypatch.setattr(live, "known", lambda: [alert])
    assert live.is_known(c, later)
    assert live.revivable(c, later) == ["20261102T1015_1775_1536"]
    # a veto on the same path keeps it quiet
    star = dict(alert, why="auto: moves with the stars")
    monkeypatch.setattr(live, "known", lambda: [alert, star])
    assert live.revivable(c, later) is None
    # so does a report, and nothing remembered there at all is no revival
    monkeypatch.setattr(live, "known", lambda: [dict(alert, why="reported: Zesheng Yang")])
    assert live.revivable(c, later) is None
    monkeypatch.setattr(live, "known", lambda: [])
    assert live.revivable(c, later) is None


def test_a_revived_alert_says_so():
    from kreutz import notify
    track = [{"time": f"2026-11-02T{13 + k // 4:02d}:{15 * (k % 4):02d}:25", "x": 1700.0 - 7 * k, "y": 1500.0,
              "vx": -7.0, "vy": -3.5} for k in range(6)]
    title, lines, _, _ = notify.compose({"track": track, "paged": True, "revived": ["20261102T1015_1775_1536"]},
                                        "x")
    assert "back after 20261102T1015_1775_1536 faded, held 6 windows" in lines[0]


def test_audit_counts_a_paged_alert_as_a_catch_even_if_it_faded():
    # the comet of 2026-11-02: paged 10:32, faded 12:01, reported from that page 18:29
    from kreutz import audit
    r, pts = _audit_report()
    later = pts[-1][0] + timedelta(hours=3)
    alert = _audit_record("alert", "a")
    cands = audit.candidates([alert], {"chains": []})
    assert audit.judge(r, pts, cands, {"a": "faded"}, later)[0] == "LOST"
    cands = audit.candidates([dict(alert, paged=True)], {"chains": []})
    assert audit.judge(r, pts, cands, {"a": "faded"}, later)[0] == "OK"


def test_audit_takes_back_a_false_alert_a_report_later_matched():
    from kreutz import audit
    flagged = ["a", "b", "c"]
    assert audit.unflag(flagged, {"b": "report_xx_1", "z": "report_xx_2"}) == ["NOT FALSE b: matched report_xx_1"]
    assert flagged == ["a", "c"]
    assert audit.unflag(flagged, {}) == []


def test_audit_says_when_we_had_it_against_the_reporter():
    from kreutz import audit
    r, pts = _audit_report()
    r = dict(r, posted_utc="2026-10-16T12:00:00Z")
    rec = _audit_record("alert", "a")  # first frame 10:00:25, decided 10:45
    s = audit.timing([("alert", rec)], r, pts)
    assert "ours from 10:00, theirs 10:00 (+0.0 h)" in s
    assert "decided 10:45, posted 12:00 (+1.2 h)" in s  # they posted 1.25 h after we knew
    assert audit.timing([("star", _audit_record("star"))], r, pts).startswith("ours from")
    assert audit.timing([], r, pts) == ""


def test_audit_digest_collects_the_day(tmp_path):
    from kreutz import audit
    log = tmp_path / "audit.log"
    log.write_text("2026-11-06T12:00 [CCOR-1] MISS report_a X Kreutz\n"
                   "2026-11-06T13:30 [CCOR-2] FALSE? 20261106T1300_1_2: alert 11-06 13:00, then faded 13:20\n"
                   "2026-11-06T14:00 [CCOR-1] FALSE? 20261106T1330_3_4: alert 11-06 13:30, paged, then faded 14:00\n"
                   "2026-11-06T18:00 [LASCO C2] OK report_b Y Kreutz\n"
                   "garbage line\n")
    live = tmp_path / "live"
    live.mkdir()
    (live / "log.jsonl").write_text(
        json.dumps({"kind": "alert", "decided_utc": "2026-11-06T13:35:00", "paged": True}) + "\n"
        + json.dumps({"kind": "alert", "decided_utc": "2026-11-06T11:00:00", "paged": True}) + "\n"
        + json.dumps({"kind": "alert", "decided_utc": "2026-11-06T16:00:00", "paged": False}) + "\n")
    since, now = datetime(2026, 11, 6, 13, 0), datetime(2026, 11, 7, 13, 0)
    assert audit.logged_since(since, log) == [
        "[CCOR-2] FALSE? 20261106T1300_1_2: alert 11-06 13:00, then faded 13:20",
        "[CCOR-1] FALSE? 20261106T1330_3_4: alert 11-06 13:30, paged, then faded 14:00",
        "[LASCO C2] OK report_b Y Kreutz"]
    # the log's "paged" means the page gates passed; it's a page only where the watcher notifies
    assert audit.pages_since(since, [("ccor1", live)]) == {"CCOR-1": (1, None)}  # no status.json yet
    assert audit.digest(since, now, log, [("ccor1", live)])[1][0].endswith("1 passed the gates CCOR-1 (no pager)")
    (live / "status.json").write_text(json.dumps({"notify": True}))
    assert audit.pages_since(since, [("ccor1", live)]) == {"CCOR-1": (1, True)}
    title, lines = audit.digest(since, now, log, [("ccor1", live)])
    assert lines[0] == "11-06 13:00 to 11-07 13:00 UTC: 1 page CCOR-1"
    assert lines[1:] == ["[CCOR-1] FALSE? 20261106T1330_3_4: alert 11-06 13:30, paged, then faded 14:00",
                         "[LASCO C2] OK report_b Y Kreutz"]  # the unpaged FALSE? cost nothing: left out
    _, lines = audit.digest(datetime(2026, 11, 8), datetime(2026, 11, 9), log, [("ccor1", live)])
    assert lines == ["11-08 00:00 to 11-09 00:00 UTC: no pages", "no reports to judge, nothing taken back"]


def test_animation_marks_only_where_the_search_looked(tmp_path):
    # issue #89: the earlier frames' marks are the line run back, and were
    # drawn off the frame too; now dim, and none off the frame, stepped by time
    import types
    from PIL import Image
    from kreutz.frames import CADENCE
    t0 = datetime(2026, 11, 5, 12, 0, 22)
    times = [t0 + CADENCE * i for i in range(7)] + [t0 + CADENCE * i for i in (8, 9, 10)]  # a frame missing
    H, W = 200, 200
    last = {"frame": 9, "x": 100.0, "y": 100.0, "vx": -13.0, "vy": 0.0, "per_frame": [3.0] * 4}
    track = [dict(last, frame=7), dict(last, frame=8), last]  # three 4-frame windows: frames 4..9 seen
    by_time = 100.0 - 13.0 * (times[6] - times[9]) / CADENCE  # 152: by frame count it would be 139

    def raw(i):
        a = np.zeros((H, W), np.float32)
        xi = 100.0 - 13.0 * (times[i] - times[9]) / CADENCE
        if 0 <= xi < W:
            a[H - 1 - 100 - 30 + i, int(xi)] = 50.0  # something different in every crop: GIF frames that match are merged
        if i == 6:
            a[H - 1 - 100, int(by_time)] = 50.0
        return a
    seq = types.SimpleNamespace(times=times, shape=(H, W), raw=raw, significance=lambda i: np.zeros((H, W), np.float32),
                                report_to_rc=lambda x, y: (H - 1 - y, x))
    out = tmp_path / "a.gif"
    live.animation(seq, track, out)
    im = Image.open(out)
    assert im.n_frames == 8  # frames 2..9

    def px(k, x, y):
        im.seek(k)
        return im.convert("L").getpixel((x, y))
    tick = (34 * 3 + 1, 40 * 3 + 1)
    assert px(7, *tick) == live.TICK_SEEN       # frame 9, in the last window
    assert px(2, *tick) == live.TICK_SEEN       # frame 4, the first window's first frame
    assert px(1, *tick) == live.TICK_GUESS      # frame 3: before the search looked
    assert px(0, *tick) == 0                    # frame 2: x = 204, off the frame; no mark
    assert px(4, 40 * 3 + 1, 40 * 3 + 1) == 255  # frame 6: the crop is centred by time, on the bright pixel


def test_audit_follows_the_watcher_s_revival_links(tmp_path):
    # the comet of 2026-11-02 in CCOR-2: alerted 16:45, faded, alerted again 21:15 as the same
    # object (live.py "revived"), and the report matched the 21:15 one by position only
    from kreutz import audit
    (tmp_path / "b.json").write_text(json.dumps({"revived": ["a"]}))
    (tmp_path / "c.json").write_text(json.dumps({"revived": None}))
    m = audit.revived_too({"b": "report_xx_1", "c": "report_xx_2", "d": "report_xx_3"}, tmp_path)
    assert m == {"b": "report_xx_1", "c": "report_xx_2", "d": "report_xx_3", "a": "report_xx_1"}
    assert audit.pager(tmp_path) is None
    (tmp_path / "status.json").write_text(json.dumps({"notify": False}))
    assert audit.pager(tmp_path) is False


def test_lane_match_only_while_the_other_comet_can_still_arrive():
    # Lecci's C2 points of 2026-11-08 18:36-19:12; a C3 report 21 h old is not
    # this track, a C3 report 5 h old could be, a CCOR-2 report 2 h old could be
    first = datetime(2026, 11, 9, 11, 48, 22)
    c3_old = {"reporter": "Chris Wage", "id": "report_cw_20261108134440", "telescope": "C3", "group": "Kreutz",
              "kind": "Potential Comet", "posted_utc": "2026-11-08T18:44:40Z",
              "points": [["1330", 591.0, 652.0], ["1454", 582.0, 639.0]]}
    c3_new = dict(c3_old, id="r2", posted_utc="2026-11-09T08:00:00Z", points=[["0630", 600.0, 660.0]])
    ccor2 = dict(c3_old, id="r3", telescope="CCOR2", posted_utc="2026-11-09T11:00:00Z", points=[["0945", 1500.0, 1600.0]])
    got = {o["id"]: o for o in evaluate.other_kreutz([c3_old, c3_new, ccor2], first, first + timedelta(minutes=24))}
    assert got["report_cw_20261108134440"]["last_utc"] == "2026-11-08T14:54:00"
    assert not live.still_arriving(got["report_cw_20261108134440"], first)
    assert live.still_arriving(got["r2"], first) and live.still_arriving(got["r3"], first)
    assert live.still_arriving({"telescope": "C3"}, first)  # an old record without the field: as before


def test_lane_samples_topped_up_from_reports_and_sector_follows(monkeypatch, tmp_path):
    from kreutz import lane, lanesamples
    f = tmp_path / "lane-samples-x.json"
    old = [[1800.0, 1700.0, -3.0, -3.0, 12, "2026-10-01T00:00:00"], [1700.0, 1600.0, -3.0, -3.0, 12, "2026-10-16T00:00:00"]]
    f.write_text(json.dumps(old))
    monkeypatch.setattr(type(lane.INSTRUMENT), "lane_samples", property(lambda self: f))
    fresh = [[1400.0, 1780.0, -4.0, -5.0, 0, "2026-11-05T02:30:00"], [1300.0, 1500.0, -4.0, -5.0, 0, "2026-10-10T00:00:00"],
             [1200.0, 1400.0, -4.0, -5.0, 0, "2025-12-10T00:00:00"]]
    monkeypatch.setattr(lanesamples, "from_reports", lambda quiet=False: fresh)
    monkeypatch.setattr(lane, "_SAMPLES", {})
    monkeypatch.setattr(lane, "_REPORT_SAMPLES", {})
    got = lane.lane_samples(datetime(2026, 11, 9))
    assert got == old + [fresh[0], fresh[2]]  # report points from outside the file's span: after it, and earlier years
    assert lane.lane_samples(datetime(2026, 11, 9)) is got  # cached for the day
    other = lane.lane_samples(datetime(2026, 11, 8))  # another date: another entry, the first kept
    assert other is not got and lane.lane_samples(datetime(2026, 11, 9)) is got
    calls = []
    monkeypatch.setattr(lanesamples, "from_reports", lambda quiet=False: calls.append(1) or fresh)
    lane.lane_samples(datetime(2026, 11, 7))
    assert calls == []  # the converted reports are cached per archive state, not per date
    # the sector from samples: centred on them, SECTOR_MARGIN either side of their spread
    monkeypatch.setattr(lane, "INSTRUMENT", replace(lane.INSTRUMENT, season_days=None))
    monkeypatch.setattr(lane, "_SAMPLES", {})
    monkeypatch.setattr(lane, "_REPORT_SAMPLES", {})
    monkeypatch.setattr(lanesamples, "from_reports", lambda quiet=False: [])
    sun = (1006.3, 985.4)
    ring = [[sun[0] + 800 * np.cos(np.radians(a)), sun[1] - 800 * np.sin(np.radians(a)), -4.0, -4.0, 0, "2026-11-01T00:00:00"]
            for a in np.linspace(-70, -50, 41)]
    f.write_text(json.dumps(ring))
    center, half = lane.season_sector(datetime(2026, 11, 9), sun)
    assert center == pytest.approx(-60, abs=1.0) and half == pytest.approx(10 + lane.SECTOR_MARGIN, abs=1.5)


def test_reports_refresh_appends_only_new(monkeypatch, tmp_path):
    from kreutz import reports
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "old.jsonl").write_text(json.dumps({"id": "report_aa_20261101000000"}) + "\n")
    monkeypatch.setattr(reports, "DATA", tmp_path)
    monkeypatch.setattr(reports, "RECENT", tmp_path / "reports" / "recent.jsonl")
    pages = {0: [{"id": "report_bb_20261109000000", "points": []}, {"id": "report_aa_20261101000000", "points": []}],
             1: [{"id": "report_cc_20261108000000", "points": []}]}
    monkeypatch.setattr(reports, "_fetch", lambda page: page)
    monkeypatch.setattr(reports, "parse", lambda page: pages[page])
    assert reports.refresh(pages=2) == 2
    assert reports.refresh(pages=2) == 0  # idempotent
    ids = [json.loads(ln)["id"] for ln in (tmp_path / "reports" / "recent.jsonl").read_text().splitlines()]
    assert ids == ["report_bb_20261109000000", "report_cc_20261108000000"]


def test_persistence_with_a_real_stack_lifts_an_orbit_hold(monkeypatch, tmp_path):
    from kreutz import stackcheck
    t0 = datetime(2026, 11, 9, 1, 0, 25)
    track = [{"time": (t0 + timedelta(minutes=15) * k).isoformat(), "x": 200.0 - 2 * k, "y": 160.0 - 1.5 * k,
              "vx": -2.0, "vy": -1.5, "per_frame": [1] * 4} for k in (3, 4, 5)]
    where = stackcheck.line_through(track)
    seq = _noisy_seq(t0, 16, source=where)
    st = live.track_stack(seq, track)
    monkeypatch.setattr(live, "LIVE", tmp_path)
    (tmp_path / "alerts").mkdir()
    alert = tmp_path / "alerts" / "held.json"
    alert.write_text("{}")
    sent = []
    monkeypatch.setattr(live.subprocess, "Popen", lambda cmd: sent.append(cmd))
    f = {"alert": str(alert), "track": track, "others": [], "known": None, "points": [], "t": datetime.fromisoformat(track[-1]["time"]),
         "x": 190.0, "y": 152.5, "vx": -2.0, "vy": -1.5, "misses": 0, "paged": False, "motion_ok": True,
         "held_why": 'fits no Kreutz orbit (2369"), not paged', "stack": st.to_dict(), "report_paged": False, "held_windows": 0}
    for k, i in enumerate(range(6, 16)):
        x, y = where(seq.times[i])
        f["held_windows"] = k + 1
        live.extend_stack(f, seq, i, x, y, "python -m kreutz.notify")
        assert f["report_paged"] is (k + 1 >= live.HELD_WINDOWS)  # the stack reads real from the start here
    assert len(sent) == 1
    rep = json.loads(alert.with_name("held.report.json").read_text())
    assert "the alert was held" in rep["why"] and "no Kreutz orbit" in rep["why"]
    # a held candidate whose stack never reads real stays silent however long it's held
    g = dict(f, alert=str(alert.with_name("weak.json")), report_paged=False, stack=stackcheck.Stack().to_dict())
    g["stack"]["offs"] = st.to_dict()["offs"]
    monkeypatch.setattr(live, "REAL_Z", 1000.0)
    for i in range(6, 16):
        g["held_windows"] = 20
        live.extend_stack(g, _noisy_seq(t0, 16), i, 50.0, 50.0, "python -m kreutz.notify")
    assert g["report_paged"] is False and len(sent) == 1


def test_other_instrument_report_must_be_where_the_track_is(monkeypatch):
    # our CCOR-1 track at 19 Rsun, PA -79 (2026-11-11 20:30); Ruan's C3 comet of that afternoon
    # at 26 Rsun, PA -65 is not it; a C3 report at 20 Rsun, PA -75 could be
    track = [{"time": "2026-11-11T20:00:25", "x": 1538.0, "y": 1765.0, "vx": -6.0, "vy": -5.0},
             {"time": "2026-11-11T20:30:25", "x": 1526.0, "y": 1755.0, "vx": -6.0, "vy": -5.0}]
    def place(tel, x, y, t):
        if tel == "CCOR1":
            return (19.2, -79.0)
        return (np.hypot((x - 518.2) * 56, (y - 490.5) * 56) / 960, np.degrees(np.arctan2(-(y - 490.5), x - 518.2)))
    real_place = live.sky_place
    monkeypatch.setattr(live, "sky_place", place)
    far = {"telescope": "C3", "last_utc": "2026-11-11T14:42:00", "last_xy": [727.0, 934.0]}   # 29 Rsun, PA -65
    assert not live.plausible(far, track)
    near = {"telescope": "C3", "last_utc": "2026-11-11T19:30:00", "last_xy": [600.0, 820.0]}  # ~20 Rsun, PA -76
    assert live.plausible(near, track)
    assert live.plausible({"telescope": "C3", "last_utc": None, "last_xy": None}, track)  # no place: by time, as before
    # the real LASCO arithmetic: C3 (727, 934) is 29 Rsun out at PA -65
    r, pa = real_place("C3", 727.0, 934.0, datetime(2026, 11, 11, 14, 42))
    assert r == pytest.approx(29.0, abs=0.5) and pa == pytest.approx(-65, abs=1.5)
