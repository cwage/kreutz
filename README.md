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
| `kreutz.weekly` | collate a span's runs into shortlist, sheets and summary |
| `kreutz.render` | follow one detection through the frames; write the aligned cutouts as FITS cubes, the track, and a quick-look preview |
| `kreutz.style` | pictures from a render's FITS cubes in a named preset (reference styles, copy and change) |
| `kreutz.live` | watch the SWPC real-time feed; alert on unreported candidates and keep going |
| `kreutz.score` | table of the live alerts against the reports that appeared: our lead or lag |

### Pictures

```
docker compose run --rm kreutz kreutz.render 2026-09-16T01:45 1953 1126 -9 -1.5 --name sep16
docker compose run --rm kreutz kreutz.style /data/render/sep16 --preset tintype
```

`kreutz.render` follows the object forward and back through the frames until
it is lost, across data gaps, and writes `data/render/<name>/`: `track.json`,
`cutouts.fits` (raw and background-subtracted crops aligned on the object, the
field region it crosses, a shift-and-stack along its motion, and the track as
a table) and a plain `preview.gif`. The FITS data is unfiltered and
unstretched; it is the product.

`kreutz.style` turns those cubes into a crop animation (object held still,
stars sliding past), a field animation, PNG frames and a still, in one of a
few reference presets. Styling is a matter of taste, so the presets are
starting points to copy, not the project's opinion.

### Live

```
docker compose run --rm -d kreutz kreutz.live --n 8 --threshold 6.5 [--notify /srv/kreutz/../notify.sh]
```

polls NOAA's real-time feed, searches each new window, and when a track
fires that isn't a star and isn't already reported from CCOR-1, writes
`data/live/alerts/<time>_<x>_<y>.json` and `.gif`, appends to
`data/live/log.jsonl`, and keeps watching. Kreutz reports from other
coronagraphs can't be matched by position, so the alert lists any from the
previous day as a hint. `kreutz.score` later compares each alert with the
reports that appeared and prints the lead or lag in minutes.

### A week at a time

```
scripts/weekly.sh            # the seven complete UTC days through yesterday
scripts/weekly.sh 2026-10-04 # the week ending before that date
```

fetches the frames and reports, runs the full search day by day and the lane
search over the span, and collates (`kreutz.weekly`) into
`data/weekly/<start>_<end>/`: a shortlist of every fired track, contact sheets
of the unexplained candidates and the brightest reported comets, and a
`summary.txt`. Add `--gpu` to use the GPU for the full search. Days already
searched are skipped, so it can be rerun.

`scripts/` also has the wrappers for longer jobs: `day.sh` (one day in three
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

## Tests

`scripts/test.sh` runs lint and the unit tests in the container; CI runs
the same on every push. Those cover the bookkeeping (coordinates, window
and linking rules, report parsing and conversion) on synthetic inputs. The
detector itself is tested by the July 2025 benchmark, `scripts/benchmark.sh`,
run by hand whenever it changes, and that's where the numbers above come
from.

## License

MIT, see `LICENSE`.
