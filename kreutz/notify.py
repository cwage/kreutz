"""Page an ntfy topic about a live alert: what, where, and the command to run.

    docker compose run --rm kreutz kreutz.notify /data/live/alerts/<alert>.json

Meant as the live watcher's --notify command:

    kreutz.live ... --notify "python -m kreutz.notify"

The server and topic come from NTFY_SERVER (default https://ntfy.sh) and
NTFY_TOPIC (docker-compose.yml passes both through from the host). The
message is terse on purpose: nothing can be done from a phone, so it says
what fired and gives the kreutz.report command to run at a computer,
which measures the object and prints the block for Sungrazer's form.
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


def post(alert_path):
    server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        print("NTFY_TOPIC not set; not notifying", file=sys.stderr)
        return
    p = Path(alert_path)
    d = json.load(open(p))
    track = d["track"] if isinstance(d, dict) else d
    c = track[-1]
    others = d.get("other_instruments", []) if isinstance(d, dict) else []
    known = f"likely {others[0]['reporter']}'s ({others[0]['telescope']})" if others else "no report anywhere"
    cmd = f"docker compose run --rm kreutz kreutz.report /data/live/alerts/{p.name}"
    if isinstance(d, dict) and d.get("kind") == "confirmed":
        lines = [f"Held {d['frames']} frames, {d['note']}; {known}.", "Measure and get the form block:", cmd]
        title = "kreutz: CONFIRMED, report ready" if not others else "kreutz: confirmed, likely known"
        _send(server, topic, title, lines, "urgent" if not others else "default", "comet,clipboard")
        return
    speed = (c["vx"] ** 2 + c["vy"] ** 2) ** 0.5
    lines = [f"Mover at ({c['x']:.0f},{c['y']:.0f}) {speed:.1f} px/frame, seen since "
             f"{track[0]['time'][11:16]} UTC; {known}.",
             "Vet and measure:", cmd]
    title = "kreutz: candidate, UNREPORTED" if not others else "kreutz: candidate, likely known"
    _send(server, topic, title, lines, "high" if not others else "default", "comet")


def main():
    if len(sys.argv) != 2:
        print(__doc__.splitlines()[0], file=sys.stderr)
        sys.exit(2)
    post(sys.argv[1])


if __name__ == "__main__":
    main()
