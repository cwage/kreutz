"""Sungrazer comet reports: who reported what, when, and where in the frame.

    docker compose run --rm kreutz kreutz.reports 2025-07-01 2025-08-31

Scrapes the plain-text report listing (newest first, paginated) for every
report whose report id falls in the date range, and writes them to
data/reports/<start>_<end>.jsonl. The listing has no date filter, so the
pages covering the range are found by binary search over page numbers.

Each report:
    id        report_jr_20250717052526  (the stamp is when it was posted)
    posted    2025-07-17T05:25:26  as stamped, which is US Eastern time:
              reports carry observations from after their own stamp
              unless it's read as EDT/EST
    posted_utc
    reporter, date, kind (Potential Comet, Additional Observations, ...),
    group (Kreutz, Meyer, ...), telescope (C2, C3, CCOR1, CCOR2, ...)
    points    [[HHMM, x, y], ...]  as reported, in the telescope's image
"""

import argparse
import html
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from . import DATA

LISTING = "https://sungrazer.nrl.navy.mil/index.php/plain-text-reports?page={}"
REPORT = "https://sungrazer.nrl.navy.mil/report/{}"  # a report's own page: its id without underscores
POLITE_S = 1.0

EASTERN = ZoneInfo("America/New_York")

_ID = re.compile(r"^report_[a-z]+_(\d{14})$")
_POINT = re.compile(r"^(\d{1,2}):?(\d{2})\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)$")


def report_url(report_id):
    """Where a report (e.g. report_jr_20261009074834) is on Sungrazer's site."""
    return REPORT.format(report_id.replace("_", ""))


def _fetch(page):
    req = urllib.request.Request(LISTING.format(page), headers={"User-Agent": "kreutz"})
    with urllib.request.urlopen(req, timeout=120) as r:
        text = r.read().decode("utf-8", "replace")
    time.sleep(POLITE_S)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = html.unescape(re.sub(r"<[^>]+>", "\n", text))
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def parse(lines):
    """Reports in one page's text lines."""
    reports, cur = [], None
    for ln in lines:
        m = _ID.match(ln)
        if m:
            posted = datetime.strptime(m[1], "%Y%m%d%H%M%S")
            utc = posted.replace(tzinfo=EASTERN).astimezone(timezone.utc)
            cur = {"id": ln, "posted": posted.isoformat(),
                   "posted_utc": utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "fields": [], "points": [], "endorsements": []}
            reports.append(cur)
            continue
        if cur is None or cur.get("done"):
            continue
        p = _POINT.match(ln)
        if p and len(cur["fields"]) >= 5:
            cur["points"].append([f"{int(p[1]):02d}{p[2]}", float(p[3]), float(p[4])])
        elif ln in ("published", "unpublished"):
            cur["status"] = ln
        elif ln.startswith("endorsements"):
            cur["in_endorsements"] = True
        elif cur.get("in_endorsements") and not ln.startswith("report_"):
            if ln in ("Pagination",) or ln.startswith("(Page"):
                cur["done"] = True
            else:
                cur["endorsements"] += [n.strip() for n in ln.split(",") if n.strip()]
        elif len(cur["fields"]) < 5 and not p:
            cur["fields"].append(ln)
        elif ln in ("Pagination",) or ln.startswith("(Page"):
            cur["done"] = True
    out = []
    for r in reports:
        f = r.pop("fields") + [None] * 5
        r.pop("done", None)
        r.pop("in_endorsements", None)
        r.update(reporter=f[0], date=f[1], kind=f[2], group=f[3], telescope=f[4])
        out.append(r)
    return out


def _page_span(page):
    """(newest, oldest) posted times on a page, or None past the end."""
    reps = parse(_fetch(page))
    if not reps:
        return None
    return reps[0]["posted"], reps[-1]["posted"]


def first_page_older_than(stamp):
    """Smallest page number whose newest report is older than `stamp`
    (ISO string), by binary search; pages are newest first."""
    lo, hi = 0, 1
    while True:  # gallop to bracket the end
        span = _page_span(hi)
        if span is None or span[0] < stamp:
            break
        lo, hi = hi, hi * 2
    while lo < hi:
        mid = (lo + hi) // 2
        span = _page_span(mid)
        if span is None or span[0] < stamp:
            hi = mid
        else:
            lo = mid + 1
    return lo


def collect(start, end):
    """Every report posted in [start, end] (dates, inclusive)."""
    end_stamp = end + "T23:59:59"
    page = max(first_page_older_than(end_stamp) - 1, 0)
    out = []
    while True:
        reps = parse(_fetch(page))
        if not reps:
            break
        out += [r for r in reps if start <= r["posted"][:10] <= end]
        print(f"  page {page}: {reps[0]['posted'][:10]} .. {reps[-1]['posted'][:10]}",
              file=sys.stderr)
        if reps[-1]["posted"][:10] < start:
            break
        page += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("start", help="YYYY-MM-DD")
    ap.add_argument("end", help="YYYY-MM-DD")
    args = ap.parse_args()
    reps = collect(args.start, args.end)
    out = DATA / "reports" / f"{args.start}_{args.end}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in reps))
    print(f"{out}  {len(reps)} reports")


if __name__ == "__main__":
    main()
