"""Page an ntfy topic about a live alert: what, where, and the command to run.

    docker compose run --rm kreutz kreutz.notify /data/live/alerts/<alert>.json   (live-ccor2 with KREUTZ_INSTRUMENT=ccor2)

Meant as the live watcher's --notify command:

    kreutz.live ... --notify "python -m kreutz.notify"

The server and topic come from NTFY_SERVER (default https://ntfy.sh) and
NTFY_TOPIC (docker-compose.yml passes both through from the host). The
message is terse on purpose: nothing can be done from a phone, so it says
what fired and gives the kreutz.report command to run at a computer,
which measures the object and prints the block for Sungrazer's form.
A comet somebody has already reported in this instrument (the file's
"known") gets nothing at any stage: there's nothing to do about it, and
the status page has it. (It had one low-priority message at confirmation,
which was noise.) Nor does a candidate that passed the gates with a weak
stack (the file's "page": "watching"): it pages REPORT if the stack gets
there or the follow-up confirms it, and the low "watching, no action"
message it had was noise, eleven in five days (2026-11-10..14) and every
one faded unreported. Nor does a candidate that doesn't move like a sungrazer (the
file's "paged" false: pointed more than live.PAGE_HEADING off the Sun, or
slower than the instrument's page_speed, on CCOR-1 fitting no Kreutz
orbit, "off_orbit", or on CCOR-2 with a line through it, "on_line"; older
files say "near_lane"): the watcher keeps
following it, and if it's confirmed it gets one low-priority message
saying why it wasn't paged. Such confirmations had paged urgently, and
the ones that came were all non-Kreutz movers (the Marsden comet receding,
94 deg off sunward, 2026-10-29 01:23). A track only the orbit search
found ("found_by": "orbit", live.py --orbit-threshold) is held back the
same way unless the watcher runs with --orbit-pages, but it's on a Kreutz
orbit by construction, so its confirmation pages in full. So does one
held for a line through it: it moved like a sungrazer, and the CCOR-2
streak that the line gate holds has never lasted to confirmation, while
a comet crossing it would.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

from .instrument import CURRENT as INSTRUMENT


def _send(server, topic, title, lines, priority, tags, click=None):
    headers = {"Title": title, "Priority": priority, "Tags": tags,
               # a header can't hold newlines; ntfy turns a literal \n into one
               "Message": "\\n".join(lines)}
    if click:
        headers["Click"] = click  # tapping the page opens the watcher's status page
    req = urllib.request.Request(f"{server}/{topic}", data=b"", headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        print(f"notified {server}: HTTP {r.status}")  # the topic is the secret; keep it out of the log


def compose(d, name):
    """(title, lines, priority, tags) for an alert or confirmed file's
    contents, or None when nothing should be sent: a comet somebody has
    already reported in this instrument gets nothing, nor does one that
    passed the gates with a weak stack ("watching") until the stack reads
    real or it's confirmed, and a candidate the watcher didn't page (off
    sunward, too slow, off any Kreutz orbit, or found by the orbit search
    alone without --orbit-pages) none until it's confirmed, and then a
    quiet one unless the orbit search found it."""
    track = d["track"] if isinstance(d, dict) else d
    c = track[-1]
    others = d.get("other_instruments", []) if isinstance(d, dict) else []
    known = d.get("known") if isinstance(d, dict) else None
    confirmed = isinstance(d, dict) and d.get("kind") == "confirmed"
    cmd = INSTRUMENT.measure_cmd(f"alerts/{name}")
    who = f"kreutz {INSTRUMENT.label}"
    if known:
        return None
    far = isinstance(d, dict) and (d.get("paged") is False if "paged" in d else d.get("near_lane") is False)
    if far and not confirmed:
        return None
    # Files from a watcher that stacks (live.track_stack): the verdict is
    # the stack's, and the page carries the evidence and the form block.
    # A confirmation of a candidate a gate held goes the old way below: the
    # stack's REPORT is for what passed the gates (or what persistence or a
    # real stack lifted, which pages as kind "report"). Ruan's comet back in
    # C3 on 2026-11-12, held at 20 deg off sunward with no report attached,
    # would otherwise have paged "REPORT, confirmed" urgently on confirmation.
    if isinstance(d, dict) and (d.get("kind") == "report" or d.get("page") or (confirmed and "stack" in d and not far)):
        return verdict_page(d, who, others, confirmed)
    hint = f"likely {others[0]['reporter']}'s ({others[0]['telescope']})" if others else "no report anywhere"
    if isinstance(d, dict) and d.get("revived"):
        hint += f"; back after {', '.join(d['revived'])} faded, held {len(track)} windows by the search"
    if far and d.get("found_by") == "orbit":
        hint += "; found by the orbit search only"
    elif far and d.get("on_line"):
        hint += f"; a line ran through it as a candidate ({d['line_score']})"
    elif far and d.get("off_orbit"):
        hint += f"; fits no Kreutz orbit ({d['orbit_rms']}\")"
    elif far:
        hint += (f"; {d['lane_px']} px off the Kreutz lane" if d.get("page_gate") == "lane" or "heading_deg" not in d
                 else f"; {d['flow_deg']} deg off the Kreutz lane's flow" if d.get("page_gate") == "flow"
                 else f"; {d['heading_deg']} deg off sunward")
    if confirmed and far and d.get("found_by") != "orbit" and not d.get("on_line"):
        return (f"{who}: confirmed, unlikely Kreutz",
                [f"Held {d['frames']} frames, {d['note']}; {hint}. Wasn't paged as a candidate.",
                 "Measure it anyway:", cmd], "low", "comet")
    if confirmed:
        return (f"{who}: CONFIRMED, report ready" if not others else f"{who}: confirmed, likely known",
                [f"Held {d['frames']} frames, {d['note']}; {hint}.", "Measure and get the form block:", cmd],
                "urgent" if not others else "default", "comet,clipboard")
    speed = (c["vx"] ** 2 + c["vy"] ** 2) ** 0.5
    return (f"{who}: candidate, UNREPORTED" if not others else f"{who}: candidate, likely known",
            [f"Mover at ({c['x']:.0f},{c['y']:.0f}) {speed:.1f} px/frame, seen since "
             f"{track[0]['time'][11:16]} UTC; {hint}.", "Vet and measure:", cmd],
            "high" if not others else "default", "comet")


def evidence(d, others):
    """One line of what the watcher has on the object, for a page: the
    stack, the gates, the frames held, and whether anyone has reported
    anything nearby."""
    bits = []
    st = d.get("stack")
    if st and st.get("z") is not None:
        bits.append(f"stack z {st['z']} over {st['n']} frames (offsets {st['null_mean']}+-{st['null_std']})")
    else:
        bits.append("no stack yet")
    if d.get("frames"):
        bits.append(f"held {d['frames']} frames" + (f", {d['note']}" if d.get("note") else ""))
    gates = []
    if d.get("lane_px") is not None:
        gates.append(f"lane {d['lane_px']} px")
    if d.get("heading_deg") is not None:
        gates.append(f"heading {d['heading_deg']} deg")
    if d.get("orbit_rms") is not None:
        gates.append(f"orbit rms {d['orbit_rms']}\"")
    if gates:
        bits.append(", ".join(gates))
    bits.append(f"likely {others[0]['reporter']}'s ({others[0]['telescope']})" if others else "no report anywhere")
    return "; ".join(bits) + "."


def verdict_page(d, who, others, confirmed):
    """(title, lines, priority, tags) for a file from a watcher that
    stacks. REPORT (urgent) when the stack reads real or the follow-up
    confirmed the object, with the form block in the page; nothing for a
    candidate that passed the gates with a weak stack ("watching": it's
    on the status page, and pages REPORT if it gets there), nor for a
    confirmation after a REPORT page already went (it's on the status
    page) except the measured block, quietly."""
    where = f"{INSTRUMENT.label} {d['track'][0]['time'][11:16]}-{d['track'][-1]['time'][11:16]} UTC"
    block = ["Form (" + INSTRUMENT.telescope + ", Kreutz, " + d['track'][-1]['time'][:10] + "):"] + \
        (d.get("report") or [])
    if d.get("kind") == "report":
        why = [d["why"]] if d.get("why") else []
        return (f"{who}: REPORT, stack says real" if not others else f"{who}: stack says real, likely known",
                [f"{where}. {evidence(d, others)}"] + why + block, "urgent" if not others else "default", "comet,clipboard")
    if confirmed:
        if d.get("report_paged"):
            return (f"{who}: confirmed, measured block",
                    [f"{where}. {evidence(d, others)}"] + block, "low", "comet,clipboard")
        return (f"{who}: REPORT, confirmed" if not others else f"{who}: confirmed, likely known",
                [f"{where}. {evidence(d, others)}"] + block, "urgent" if not others else "default", "comet,clipboard")
    if d.get("page") == "report":
        return (f"{who}: REPORT, stack says real" if not others else f"{who}: stack says real, likely known",
                [f"{where}. {evidence(d, others)}"] + block, "urgent" if not others else "default", "comet,clipboard")
    return None


def held_because(d):
    """Why compose() sent nothing, for the log: the watcher's own reason."""
    if not isinstance(d, dict):
        return "not paged"
    if d.get("known"):
        return f"known comet ({d['known'].get('reporter')}'s)"
    if d.get("page") == "watching":
        z = (d.get("stack") or {}).get("z")
        return f"watching (stack z {z}), no action" if z is not None else "watching (no stack yet), no action"
    if d.get("found_by") == "orbit":
        return "found by the orbit search only"
    if d.get("on_line"):
        return f"a line runs through it ({d['line_score']})"
    if d.get("off_orbit"):
        return f"fits no Kreutz orbit ({d['orbit_rms']}\")"
    if d.get("page_gate") == "lane":
        return f"not paged ({d.get('lane_px')} px off the Kreutz lane or too slow)"
    if d.get("page_gate") == "flow":
        return f"not paged ({d.get('flow_deg')} deg off the Kreutz lane's flow or too slow)"
    if "heading_deg" in d:
        return f"not paged ({d['heading_deg']} deg off sunward or too slow)"
    return f"{d.get('lane_px')} px off the Kreutz lane"


def post(alert_path):
    server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        print("NTFY_TOPIC not set; not notifying", file=sys.stderr)
        return
    p = Path(alert_path)
    d = json.load(open(p))
    msg = compose(d, p.name)
    if msg is None:
        when = "" if isinstance(d, dict) and d.get("known") else " at the candidate stage"
        print(f"{p.name}: {held_because(d)}{when}; not notifying")
        return
    _send(server, topic, *msg, click=status_url())


def status_url():
    """The watcher's status page, when KREUTZ_URL names where data/ is
    served (the web service, or a tunnel to it); else None."""
    base = os.environ.get("KREUTZ_URL", "").rstrip("/")
    return f"{base}/live{INSTRUMENT.tag}/status.html" if base else None


def main():
    if len(sys.argv) != 2:
        print(__doc__.splitlines()[0], file=sys.stderr)
        sys.exit(2)
    post(sys.argv[1])


if __name__ == "__main__":
    main()
