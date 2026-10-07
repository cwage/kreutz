"""Post a live alert to an ntfy topic.

    docker compose run --rm kreutz kreutz.notify /data/live/alerts/<alert>.json

Meant as the live watcher's --notify command:

    kreutz.live ... --notify "python -m kreutz.notify"

The server and topic come from NTFY_SERVER (default https://ntfy.sh) and
NTFY_TOPIC (docker-compose.yml passes both through from the host). The
message has the time the object was seen, where and how fast, and any
Kreutz reports from other instruments in the previous day; the alert's
vetting GIF is attached when it exists.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path


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
    speed = (c["vx"] ** 2 + c["vy"] ** 2) ** 0.5
    lines = [f"Tracked {track[0]['time'][11:16]}-{c['time'][11:16]} UTC {c['time'][:10]} at "
             f"({c['x']:.0f},{c['y']:.0f}), {speed:.1f} px/frame, peak {max(x['score'] for x in track):.1f} sigma. "
             f"Moves, not a star, no CCOR-1 report matches."]
    if others:
        o = others[0]
        more = f" (+{len(others) - 1} more)" if len(others) > 1 else ""
        lines.append(f"Likely already known: {o['reporter']} reported a Kreutz comet from {o['telescope']} "
                     f"at {o['posted_utc'][11:16]}Z{more}; other instruments can't be matched by position.")
        title, priority, tags = "kreutz: comet candidate, likely known", "default", "comet"
    else:
        lines.append("No Kreutz report from CCOR-1, CCOR-2 or LASCO in the last 24 h: an unreported comet "
                     "or an artifact. Check the GIF: real if it holds still while the stars drift.")
        title, priority, tags = "kreutz: comet candidate, UNREPORTED", "high", "comet,eyes"
    lines.append(f"File: {p.name}")
    headers = {"Title": title, "Priority": priority, "Tags": tags,
               # a header can't hold newlines; ntfy turns a literal \n into one
               "Message": "\\n".join(lines)}
    gif = p.with_suffix(".gif")
    if gif.exists():
        headers["Filename"] = gif.name
        body = gif.read_bytes()
    else:
        body = "\n".join(lines).encode()
        headers.pop("Message")
    req = urllib.request.Request(f"{server}/{topic}", data=body, headers=headers, method="PUT" if gif.exists() else "POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        print(f"notified {server}/{topic}: HTTP {r.status}")


def main():
    if len(sys.argv) != 2:
        print(__doc__.splitlines()[0], file=sys.stderr)
        sys.exit(2)
    post(sys.argv[1])


if __name__ == "__main__":
    main()
