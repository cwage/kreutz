"""Which coronagraph the package is working on: CCOR-1 (the default),
CCOR-2 or LASCO C2/C3, chosen with KREUTZ_INSTRUMENT, as KREUTZ_DATA
chooses the data directory. Everything that names the instrument (the frame directory and
file names, the SWPC feed, the NCEI archive prefix, Sungrazer's telescope
label, the lane samples, the live watcher's state directory and the
significance cache) comes from here, so the two can share one checkout
and one data directory without touching each other's files.

CCOR-1 is on GOES-19 in geostationary orbit; CCOR-2 is on SWFO-L1 at the
Sun-Earth L1 point. Both frames are 2048x1920 at a 15-minute cadence with
the same calibration units and WCS, but CCOR-2's plate scale is 25%
coarser and its roll differs, so the Kreutz lane and the search priors
measured on CCOR-1 (search.py, lane-samples.json) don't carry over; each
instrument has its own lane-samples file.

LASCO C2 and C3 are on SOHO, also at L1: 1024x1024 frames about every 12
minutes, at 11.9 and 56 arcsec/px. They come as level 0.5 (raw counts,
SOHO's roll, files named by sequence number) and lasco.py turns them into
what the rest of the package expects: frames stored under their
observation time, and loaded bias-subtracted, per second of exposure,
rotated to solar north up, with everything off the sky set to NaN.
"""

import os
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from . import DATA


@dataclass(frozen=True)
class Instrument:
    key: str             # ccor1: data directory and cache/state tags
    telescope: str       # CCOR1: the name in frame files and Sungrazer reports
    label: str           # CCOR-1: for people and FITS headers
    level: str           # 1B: the product on the SWPC feed and in the archive
    feed: str            # real-time directory of frames (SWPC; LASCO: NASA's lastimage)
    archive_prefix: str  # NCEI archive prefix (LASCO: the archive's root URL), or None if not archived yet
    cosmic_rays: bool    # take particle hits out of each frame (frames.cosmic_rays)
    dark_spots: bool     # fill sharp dark spots in each raw frame (frames.fill_dark)
    # Where Kreutz comets appear, for search._roi: distance from the Sun
    # in px, and the sector's center and half-width in degrees from
    # straight right of the Sun, counterclockwise with north up.
    r_min: int
    r_max: int
    angle_center: float
    angle_max: float
    sun_xy: tuple        # the Sun in report coordinates (frames.sun_pixel), for filters without a frame
    swpc_canvas: bool    # reports measured on SWPC's derotated image can be converted (evaluate.py)
    second: int          # frames are stamped this many seconds past the quarter hour (None: they aren't on a grid)
    star_mask_r: int     # radius masked around confirmed stars in the significance map
    page_speed: float    # px/frame: a candidate slower than this isn't paged (live.pageworthy)
    passage: int         # hours a Kreutz comet stays in the field after it's found (live.PASSAGE)
    source: str = "ccor"         # file layout and product: "ccor" (NOAA L1a/L1b) or "lasco" (lasco.py)
    cadence_min: int = 15        # minutes from one standard frame to the next
    shape: tuple = (1920, 2048)  # frame (rows, cols)
    speed_max: float = 12.5      # px/frame: the fastest motion the search tries (search.SPEED_MAX)
    edge_px: int = 30            # the search stays this far inside the frame (search.EDGE_PX)
    star_min_r: int = 350        # px from the Sun: stars inside aren't used or masked (frames.STAR_MIN_R)
    max_missing: int = 1         # frames a window may lack (search.MAX_MISSING)
    # Days of year either side of a date whose lane samples (any year)
    # make that date's lane, and its search sector when angle_center is
    # None (lane.season_samples); None: one lane all year.
    season_days: int = None

    @property
    def cadence(self):
        return timedelta(minutes=self.cadence_min)

    @property
    def frames_dir(self):
        return DATA / self.key / f"l{self.level.lower()}"

    @property
    def name_re(self):
        if self.source == "lasco":  # C2_05_20261022T031233_23903880.fts: time, then the original name
            return re.compile(rf"{self.telescope}_{self.level}_(\d{{8}}T\d{{6}})_\d+\.fts$")
        return re.compile(rf"{self.telescope}_{self.level}_(\d{{8}}T\d{{6}})_V\d+_NC\.fits$")

    def glob(self, stamp="*"):
        if self.source == "lasco":
            return f"{self.telescope}_{self.level}_{stamp}_*.fts"
        return f"{self.telescope}_{self.level}_{stamp}_NC.fits"

    @property
    def tag(self):
        """Suffix for per-instrument directories; empty for CCOR-1 so its
        existing cache, live state and runs keep their names."""
        return "" if self.key == "ccor1" else f"-{self.key}"

    @property
    def env(self):
        """What a command needs in front of it to run on this instrument."""
        return "" if self.key == "ccor1" else f"KREUTZ_INSTRUMENT={self.key} "

    def measure_cmd(self, path):
        """The kreutz.report command for an alert file, as the status page
        and a push show it; `path` is relative to the live directory."""
        return f"{self.env}docker compose run --rm kreutz kreutz.report /data/live{self.tag}/{path}"

    @property
    def lane_samples(self):
        return Path(__file__).with_name(f"lane-samples{self.tag}.json")


# LASCO's outer rows and columns are as quiet as the middle of the frame
# (C2 2026-09-16: 68th percentile of |significance| 0.97 within 30 px of
# the edges and in the middle), and C2's Kreutz comets come in across the
# right edge at up to 20 px/frame, so CCOR's 30 px margin would cost them
# two frames; the margin is just inside frames.BORDER instead. SOHO skips
# two frames in a row several times a day (36 minutes between frames
# 05:48-06:24, 09:48-10:24, 12:48-13:25 most days in Aug 2025).
# Where Kreutz comets come in swings through the year as SOHO goes round
# the Sun: in C3 the reports' median position angle is -40 deg (lower
# right) in Aug-Oct and -141 in Feb-Apr (lower left), within ~15 deg in
# any one month (2024-2026 reports), and C2 follows. So the lane and the
# search sector are the season's, from the samples within a month of the
# date in any year (lane.season_samples).
_LASCO = dict(edge_px=14, max_missing=2, season_days=15)
# Stars down to just outside the search's inner edge: CCOR's 350 px would
# leave out most of C3's field (it ends ~540 px out). Bright stars leave
# residue past CCOR's 4 px mask, which stacks into slow tracks drifting
# with the stars (C3 2026-09-15: 12 of 20 unexplained tracks); the mask is
# 8 px (2.4% of C3's field).
_STAR_MIN_R = {"c2": 200, "c3": 100}

INSTRUMENTS = {
    "ccor1": Instrument("ccor1", "CCOR1", "CCOR-1", "1B",
                        "https://services.swpc.noaa.gov/products/ccor1/fits/",
                        "SWFO/GOES-19/CCOR-1/ccor1-l1b", False, False,
                        # see search.py: all 81 Kreutz tracks of Jul-Aug 2025
                        460, 1250, 0, 75, (1005.7, 957.1), True, 25, 4,
                        # Reported Kreutz comets: 5th percentile 4.3 px/frame; the
                        # slowest real alert so far 4.1.
                        4.0, 12),
    # The feed carries CCOR-2 at level 1A only, and NCEI's archive prefix
    # (SWFO/SWFO-L1/CCOR-2/) is empty as of 2026-10-20; the feed's ~30
    # days of frames are the only source. Outside the magnetosphere at L1
    # its frames are peppered with particle hits (five times CCOR-1's
    # count of >5 sigma blobs in a frame, most of them gone the next
    # frame, sharper than the PSF); 1A isn't cleaned of them, so we are.
    # Its dark detector defects (clusters 10-14 sigma below their
    # surroundings) also wander a few pixels from frame to frame on the
    # 1A grid, so they don't cancel against the background: the
    # background and star model carry displaced copies, and the
    # residual is a bright spot beside the defect that stacks as a slow
    # mover (8 of 22 unexplained tracks on 2026-10-16). They're filled
    # in every raw frame instead.
    "ccor2": Instrument("ccor2", "CCOR2", "CCOR-2", "1A",
                        "https://services.swpc.noaa.gov/products/ccor2/fits/",
                        None, True, True,
                        # The 255 points of the 73 CCOR-2 Kreutz reports of
                        # 2026-07-28..10-16 lie 412-1127 px from the Sun, at
                        # -25..-49 deg (lower right), moving 3-10 px/frame.
                        # The sector is 20 deg either side: the extremes are 12
                        # off center, and at -66 deg (below the lane, near the
                        # bottom edge) stray-light striping stacked into tracks.
                        400, 1250, -37, 20, (1006.3, 985.4), False, 14,
                        # CCOR-2's bright stars leave a ring of +2.5 sigma at 4-8 px
                        # after the star model (CCOR-1's leave nothing there):
                        # 14 of 22 unexplained tracks on 2026-10-16 were 7-12 px
                        # from a star.
                        10,
                        # Reported Kreutz comets: slowest 2.8 px/frame, 5th percentile 3.7.
                        2.5,
                        # The lane runs ~700 px at 3-10 px/frame (CCOR-1: ~700 at
                        # 9-12), so about twice CCOR-1's passage.
                        20),
    # sun_xy is CRPIX, which the level 0.5 headers hold fixed and which
    # lasco.load rotates about. Measured on the 2024-01..2026-10 Kreutz
    # reports (less those their reporter retracted within a day): C2's 479
    # lie 239-663 px from the Sun (2nd-98th percentile), first seen at
    # 520 px (median), moving 4.2-31 px/frame, median 10; C3's 532 lie
    # 141-539 px out, moving 1.0-4.0 px/frame (5th-95th), median 2.2.
    # The sector comes from the season's lane (angle_center None).
    "c2": Instrument("c2", "C2", "LASCO C2", "05",
                     "https://umbra.nascom.nasa.gov/pub/lasco/lastimage/level_05/",
                     "https://umbra.nascom.nasa.gov/pub/lasco_level05/", True, False,
                     200, 700, None, None, (510.2, 516.5), False, None, 8, 4.0, 12,
                     source="lasco", cadence_min=12, shape=(1024, 1024), speed_max=24.0, star_min_r=_STAR_MIN_R["c2"], **_LASCO),
    "c3": Instrument("c3", "C3", "LASCO C3", "05",
                     "https://umbra.nascom.nasa.gov/pub/lasco/lastimage/level_05/",
                     "https://umbra.nascom.nasa.gov/pub/lasco_level05/", True, False,
                     120, 560, None, None, (518.2, 490.5), False, None, 8, 1.0, 48,
                     source="lasco", cadence_min=12, shape=(1024, 1024), star_min_r=_STAR_MIN_R["c3"], **_LASCO),
}

_name = os.environ.get("KREUTZ_INSTRUMENT", "ccor1").lower()
if _name not in INSTRUMENTS:
    raise SystemExit(f"KREUTZ_INSTRUMENT={_name!r}: known instruments are {', '.join(INSTRUMENTS)}")
CURRENT = INSTRUMENTS[_name]
