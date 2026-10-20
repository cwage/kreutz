"""Which coronagraph the package is working on: CCOR-1 (the default) or
CCOR-2, chosen with KREUTZ_INSTRUMENT, as KREUTZ_DATA chooses the data
directory. Everything that names the instrument (the frame directory and
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
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

from . import DATA


@dataclass(frozen=True)
class Instrument:
    key: str             # ccor1: data directory and cache/state tags
    telescope: str       # CCOR1: the name in frame files and Sungrazer reports
    label: str           # CCOR-1: for people and FITS headers
    level: str           # 1B: the product on the SWPC feed and in the archive
    feed: str            # SWPC real-time directory of frames
    archive_prefix: str  # NCEI archive prefix, or None if not archived yet
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
    second: int          # frames are stamped this many seconds past the quarter hour
    star_mask_r: int     # radius masked around confirmed stars in the significance map
    passage: int         # hours a Kreutz comet stays in the field after it's found (live.PASSAGE)

    @property
    def frames_dir(self):
        return DATA / self.key / f"l{self.level.lower()}"

    @property
    def name_re(self):
        return re.compile(rf"{self.telescope}_{self.level}_(\d{{8}}T\d{{6}})_V\d+_NC\.fits$")

    def glob(self, stamp="*"):
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


INSTRUMENTS = {
    "ccor1": Instrument("ccor1", "CCOR1", "CCOR-1", "1B",
                        "https://services.swpc.noaa.gov/products/ccor1/fits/",
                        "SWFO/GOES-19/CCOR-1/ccor1-l1b", False, False,
                        # see search.py: all 81 Kreutz tracks of Jul-Aug 2025
                        460, 1250, 0, 75, (1005.7, 957.1), True, 25, 4, 12),
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
                        # The lane runs ~700 px at 3-10 px/frame (CCOR-1: ~700 at
                        # 9-12), so about twice CCOR-1's passage.
                        20),
}

_name = os.environ.get("KREUTZ_INSTRUMENT", "ccor1").lower()
if _name not in INSTRUMENTS:
    raise SystemExit(f"KREUTZ_INSTRUMENT={_name!r}: known instruments are {', '.join(INSTRUMENTS)}")
CURRENT = INSTRUMENTS[_name]
