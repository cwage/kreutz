"""Page an ntfy topic about a live alert: what, where, and the command to run.

    docker compose run --rm kreutz kreutz.notify /data/live/alerts/<alert>.json   (live-ccor2 with KREUTZ_INSTRUMENT=ccor2)

Meant as the live watcher's --notify command:

    kreutz.live ... --notify "python -m kreutz.notify"

The server and topic come from NTFY_SERVER (default https://ntfy.sh) and
NTFY_TOPIC (docker-compose.yml passes both through from the host). The
message is terse on purpose: nothing can be done from a phone, so it says
what fired and gives the kreutz.report command to run at a computer,
which measures the object and prints the block for Sungrazer's form.
A comet somebody has already reported (the file's "known") gets a single
low-priority message when it's confirmed, and nothing at the candidate
stage. Nor does a candidate that doesn't move like a sungrazer (the
file's "paged" false: pointed more than live.PAGE_HEADING off the Sun, or
slower than the instrument's page_speed; older files say "near_lane");
the watcher keeps following it, and it pages if it's confirmed, with
the distance in the message.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

from .instrument import CURRENT as INSTRUMENT


def _send(server, topic, title, lines, priority, tags):
    headers = {"Title": title, "Priority": priority, "Tags": tags,
               # a header can't hold newlines; ntfy turns a literal \n into one
               "Message": "\\n".join(lines)}
    req = urllib.request.Request(f"{server}/{topic}", data=b"", headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        print(f"notified {server}: HTTP {r.status}")  # the topic is the secret; keep it out of the log


def compose(d, name):
    """(title, lines, priority, tags) for an alert or confirmed file's
    contents, or None when nothing should be sent: a comet somebody has
    already reported gets one quiet message, at the confirmed stage, and
    a candidate off the lane none until it's confirmed."""
    track = d["track"] if isinstance(d, dict) else d
    c = track[-1]
    others = d.get("other_instruments", []) if isinstance(d, dict) else []
    known = d.get("known") if isinstance(d, dict) else None
    confirmed = isinstance(d, dict) and d.get("kind") == "confirmed"
    cmd = INSTRUMENT.measure_cmd(f"alerts/{name}")
    who = f"kreutz {INSTRUMENT.label}"
    if known:
        if not confirmed:
            return None
        posted = known["posted_utc"].replace("T", " ")[:16]
        return (f"{who}: known comet, {known['reporter']}'s ({known['telescope']})",
                [f"Reported {posted} UTC, matched by {known['how']}; held {d['frames']} frames, {d['note']}.",
                 "Measure it anyway:", cmd], "low", "comet")
    far = isinstance(d, dict) and (d.get("paged") is False if "paged" in d else d.get("near_lane") is False)
    if far and not confirmed:
        return None
    hint = f"likely {others[0]['reporter']}'s ({others[0]['telescope']})" if others else "no report anywhere"
    if far:
        hint += (f"; {d['heading_deg']} deg off sunward" if "heading_deg" in d
                 else f"; {d['lane_px']} px off the Kreutz lane")
    if confirmed:
        return (f"{who}: CONFIRMED, report ready" if not others else f"{who}: confirmed, likely known",
                [f"Held {d['frames']} frames, {d['note']}; {hint}.", "Measure and get the form block:", cmd],
                "urgent" if not others else "default", "comet,clipboard")
    speed = (c["vx"] ** 2 + c["vy"] ** 2) ** 0.5
    return (f"{who}: candidate, UNREPORTED" if not others else f"{who}: candidate, likely known",
            [f"Mover at ({c['x']:.0f},{c['y']:.0f}) {speed:.1f} px/frame, seen since "
             f"{track[0]['time'][11:16]} UTC; {hint}.", "Vet and measure:", cmd],
            "high" if not others else "default", "comet")


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
        why = "known comet" if isinstance(d, dict) and d.get("known") else "off the lane"
        print(f"{p.name}: {why} at the candidate stage; not notifying")
        return
    _send(server, topic, *msg)


def main():
    if len(sys.argv) != 2:
        print(__doc__.splitlines()[0], file=sys.stderr)
        sys.exit(2)
    post(sys.argv[1])


if __name__ == "__main__":
    main()
