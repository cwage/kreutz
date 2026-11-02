"""One status page over every instrument's watcher.

    docker compose run --rm kreutz kreutz.page

Writes data/index.html (the page, kreutz/status.html) and
data/instruments.json (every instrument in instrument.py: its label, its
live directory and the prefix of its measure command), and replaces each
existing live directory's status.html with a redirect to the page. The
page reads instruments.json, then each live directory's status.json, so
an instrument whose watcher isn't running just shows as such. Every
watcher does this when it starts; run it by hand after changing the page
without restarting them.
"""

import json
import os
import socket
from pathlib import Path

from . import DATA
from .instrument import INSTRUMENTS

REDIRECT = ('<!doctype html>\n<meta charset="utf-8">\n'
            '<meta http-equiv="refresh" content="0; url=../index.html">\n'
            '<title>kreutz live</title>\n<a href="../index.html">kreutz live</a>\n')


def _write(path, text):
    # Two watchers may write the same files at once; each writes its own
    # temporary file and renames it, so a reader sees one whole version.
    # Each watcher is pid 1 in its own container, so the hostname (the
    # container's) tells them apart: four started together after a reboot
    # shared one temporary file, and two died renaming it.
    tmp = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def instruments():
    return [{"key": i.key, "label": i.label, "live": f"live{i.tag}", "measure_prefix": i.measure_cmd("")}
            for i in INSTRUMENTS.values()]


def write(root=None):
    root = root or DATA
    root.mkdir(parents=True, exist_ok=True)
    insts = instruments()
    _write(root / "instruments.json", json.dumps(insts, indent=1) + "\n")
    _write(root / "index.html", Path(__file__).with_name("status.html").read_text())
    # a frame player for circling things in a render, http://localhost:8000/mark.html?dir=<frames dir>
    _write(root / "mark.html", Path(__file__).with_name("mark.html").read_text())
    for i in insts:
        if (root / i["live"]).is_dir():
            _write(root / i["live"] / "status.html", REDIRECT)


def main():
    write()


if __name__ == "__main__":
    main()
