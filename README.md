# kreutz

Finds faint moving objects, mostly Kreutz-group sungrazing comets, in images
from CCOR-1, the coronagraph on NOAA's GOES-19 weather satellite. It searches
the NOAA archive, watches the real-time feed, and checks what it finds against
the reports on NASA/NRL's [Sungrazer Project](https://sungrazer.nrl.navy.mil/).

It is a candidate finder, not a reporter. Everything it flags is meant to be
looked at by a person before anyone tells Sungrazer about it.

## How it works

CCOR-1 publishes a 2048x1920 calibrated frame every 15 minutes. For each
window of the last N frames:

1. **Background.** Subtract the median of the previous 16 frames (past only,
   as a real-time system would have), skipping the most recent so a slow
   object doesn't subtract itself out.
2. **Stars.** Stars drift about 2 px/frame, non-uniformly (plate scale and the
   satellite's orbit). Each pair of frames is aligned by matching a few hundred
   stars, a star model is built from 8 aligned past frames and subtracted, and
   confirmed stars are masked.
3. **Point filter.** A difference-of-Gaussians filter keeps point sources and
   cancels diffuse coronal structure; the result is divided by a local noise
   estimate to give a significance map per frame.
4. **Shift-and-stack.** For each of ~1,900 trial velocities, shift the N maps
   back along that motion and sum. An object moving at that velocity adds up
   (signal x N, noise x sqrt N). Each frame is clipped first so one hot pixel
   can't make a track. Position and velocity priors come from reported Kreutz
   tracks.
5. **Linking and vetting.** A detection has to recur in consecutive windows
   where its velocity predicts, show up in most frames of each window, and
   move. It is then followed through the raw frames to make sure it isn't a
   star, and in the archive checked for persistence: does it keep going in the
   frames after, compared with parallel control paths in the same frames?

`kreutz/lane.py` is a faster, deeper variant: Kreutz comets all enter the
field along one lane, so a flow field built from observed tracks gives the
expected velocity at each pixel, and the stack can follow the comet's
acceleration over 16 frames.

Coordinates throughout are Sungrazer report coordinates (x right, y down from
the top of the displayed image, pixels per 15-minute frame) so results compare
directly with reports.

## Running it

Everything runs in Docker. `data/` is bind-mounted into the container as
`/data` and holds frames, reports, run output and vetting images; it is not
checked in. Frames are about 880 MB per day.

```
docker compose build
docker compose run --rm kreutz kreutz.fetch 2025-07-16 2025-07-24     # frames from NOAA NCEI
docker compose run --rm kreutz kreutz.reports 2025-07-01 2025-08-31   # Sungrazer reports
docker compose run --rm kreutz kreutz.run 2025-07-17 --n 8            # search one day
docker compose run --rm kreutz kreutz.evaluate /data/runs/n8/20250717*.jsonl
```

Modules, each with usage in its docstring:

| module | what it does |
|---|---|
| `kreutz.fetch` | download archived frames from NCEI |
| `kreutz.reports` | scrape Sungrazer reports into `data/reports/` |
| `kreutz.run` | search every window in a span; `--lane` for the lane search |
| `kreutz.evaluate` | score a run against reports: what would have been found, and how early |
| `kreutz.shortlist` | archive run to a persistence-ranked list of unexplained tracks |
| `kreutz.look` | strip of crops along a hypothesized track, for vetting by eye |
| `kreutz.sheet` | contact sheets of those strips for a candidate list |
| `kreutz.live` | watch the SWPC real-time feed and stop on the first unreported candidate |

`scripts/` has the wrappers for longer jobs: `day.sh` (one day in three
parallel chunks), `archive.sh` and `archive-gpu.sh` (a span of days),
`lane.sh`, `benchmark.sh` (the July 2025 week below) and `stop.sh`.

Containers are CPU-capped (`KREUTZ_CPUS`, default 1.5) with single-threaded
numeric libraries, so several can run side by side at a predictable load. The
GPU path is opt-in and much faster for the full search:

```
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm kreutz kreutz.run 2025-07-17 --n 8
```

## Results so far

On a benchmark week (July 17-24, 2025, eight confirmed CCOR-1 comets) the
detector found six in time to have beaten the human report, by one minute to
over four hours, found one late, and missed one that entered at the field edge
just before a data gap. Over a two-month archive search it found every bright
Kreutz comet that hunters reported, plus a planet, and no unreported comet.
On the live feed it has flagged a real comet 30 minutes after it entered the
field; a hunter had already reported it from another coronagraph.

Sungrazer's rules are at https://sungrazer.nrl.navy.mil/. Reports there must
be made by the person who found and measured the object.

## License

MIT, see `LICENSE`.
