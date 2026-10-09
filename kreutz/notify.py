"""Page an ntfy topic about a live alert: what, where, and the command to run.

    docker compose run --rm kreutz kreutz.notify /data/live/alerts/<alert>.json

Meant as the live watcher's --notify command:

    kreutz.live ... --notify "python -m kreutz.notify"

The server and topic come from NTFY_SERVER (default https://ntfy.sh) and
NTFY_TOPIC (docker-compose.yml passes both through from the host). The
message is terse on purpose: nothing can be done from a phone, so it says
what fired and gives the kreutz.report command to run at a computer,
which measures the object and prints the block for Sungrazer's form.
A comet somebody has already reported (the file's "known") gets a single
low-priority message when it's confirmed, and nothing at the candidate
stage.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path


def _send(server, topic, title, lines, priority, tags):
    headers = {"Title": title, "Priority": priority, "Tags": tags,
               # a header can't hold newlines; ntfy turns a literal \n into one
               "Message": "\\n".join(lines)}
    req = urllib.request.Request(f"{server}/{topic}", data=b"", headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        print(f"notified {server}/{topic}: HTTP {r.status}")


def compose(d, name):
    """(title, lines, priority, tags) for an alert or confirmed file's
    contents, or None when nothing should be sent: a comet somebody has
    already reported gets one quiet message, at the confirmed stage."""
    track = d["track"] if isinstance(d, dict) else d
    c = track[-1]
    others = d.get("other_instruments", []) if isinstance(d, dict) else []
    known = d.get("known") if isinstance(d, dict) else None
    confirmed = isinstance(d, dict) and d.get("kind") == "confirmed"
    cmd = f"docker compose run --rm kreutz kreutz.report /data/live/alerts/{name}"
    if known:
        if not confirmed:
            return None
        posted = known["posted_utc"].replace("T", " ")[:16]
        return (f"kreutz: known comet, {known['reporter']}'s ({known['telescope']})",
                [f"Reported {posted} UTC, matched by {known['how']}; held {d['frames']} frames, {d['note']}.",
                 "Measure it anyway:", cmd], "low", "comet")
    hint = f"likely {others[0]['reporter']}'s ({others[0]['telescope']})" if others else "no report anywhere"
    if confirmed:
        return ("kreutz: CONFIRMED, report ready" if not others else "kreutz: confirmed, likely known",
                [f"Held {d['frames']} frames, {d['note']}; {hint}.", "Measure and get the form block:", cmd],
                "urgent" if not others else "default", "comet,clipboard")
    speed = (c["vx"] ** 2 + c["vy"] ** 2) ** 0.5
    return ("kreutz: candidate, UNREPORTED" if not others else "kreutz: candidate, likely known",
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
    msg = compose(json.load(open(p)), p.name)
    if msg is None:
        print(f"{p.name}: known comet at the candidate stage; not notifying")
        return
    _send(server, topic, *msg)


def main():
    if len(sys.argv) != 2:
        print(__doc__.splitlines()[0], file=sys.stderr)
        sys.exit(2)
    post(sys.argv[1])


if __name__ == "__main__":
    main()
