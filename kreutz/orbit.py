"""Where a Kreutz sungrazer appears in a coronagraph frame.

All Kreutz comets are fragments of one parent on nearly the same orbit, so
where one is in the frame almost fixes how it moves: what's free is when
it reaches perihelion and small differences in the orbit. This module
puts a parabolic orbit through that geometry: heliocentric position from
the orbital elements (ecliptic J2000), seen from the instrument (GOES-19
near Earth for CCOR-1, SWFO-L1 and SOHO at L1 for CCOR-2 and LASCO), as
helioprojective longitude and latitude (west and north of Sun center,
solar north up), which each frame's WCS turns into pixels.

The elements default to the Kreutz group's (Marsden's subgroup I); fit()
adjusts them to a measured track, which is how the family's spread is
found from reported comets.
"""

from dataclasses import dataclass, replace
from datetime import datetime

import numpy as np
from astropy.coordinates import get_body_barycentric
from astropy.time import Time

K2 = 2.959122082855911e-4        # GM of the Sun, AU^3/day^2
OBLIQUITY = np.radians(23.4392911)  # J2000
# The Sun's north rotation pole, ICRS (IAU): RA 286.13, Dec 63.87 deg.
_RA, _DEC = np.radians(286.13), np.radians(63.87)
SOLAR_POLE = np.array([np.cos(_DEC) * np.cos(_RA), np.cos(_DEC) * np.sin(_RA), np.sin(_DEC)])
L1_FRACTION = 0.01  # SWFO-L1 and SOHO sit about 1% of the way from Earth to the Sun


@dataclass(frozen=True)
class Elements:
    """A parabolic orbit: perihelion time, perihelion distance (AU), and
    the angles (deg, ecliptic and equinox J2000)."""
    T: datetime
    q: float = 0.0055
    i: float = 144.4
    node: float = 3.7     # longitude of the ascending node
    peri: float = 80.4    # argument of perihelion


def _orbit_to_icrs(e: Elements):
    """Rotation from the orbital plane (x to perihelion) to ICRS."""
    O, w, i = np.radians([e.node, e.peri, e.i])
    cO, sO, cw, sw, ci, si = np.cos(O), np.sin(O), np.cos(w), np.sin(w), np.cos(i), np.sin(i)
    # perifocal -> ecliptic
    P = np.array([[cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
                  [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
                  [sw * si, cw * si, ci]])
    ce, se = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
    ecl_to_eq = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
    return ecl_to_eq @ P


def heliocentric(e: Elements, times):
    """Heliocentric ICRS positions (AU), (n, 3), at datetimes `times`."""
    dt = np.array([(t - e.T).total_seconds() / 86400 for t in times])
    W = 3 * np.sqrt(K2 / (2 * e.q ** 3)) * dt
    # Barker's equation s^3 + 3s = W, s = tan(nu/2)
    y = np.cbrt(W / 2 + np.sqrt(W ** 2 / 4 + 1))
    s = y - 1 / y
    r = e.q * (1 + s ** 2)
    nu = 2 * np.arctan(s)
    xy = np.stack([r * np.cos(nu), r * np.sin(nu), np.zeros_like(r)], axis=1)
    return xy @ _orbit_to_icrs(e).T


def observer(times, where="earth"):
    """Heliocentric ICRS positions (AU) of the instrument: Earth's (GOES-19
    is a sixth of a lunar distance out, under 0.02 deg of parallax at a
    sungrazer), or L1's."""
    t = Time(times)
    earth = get_body_barycentric("earth", t).xyz.to("au").value.T
    sun = get_body_barycentric("sun", t).xyz.to("au").value.T
    o = earth - sun
    return o * (1 - L1_FRACTION) if where == "l1" else o


def helioprojective(e: Elements, times, where="earth"):
    """(Tx, Ty) in degrees, west and north of Sun center as seen from the
    instrument, solar north up; arrays over `times`."""
    c = heliocentric(e, times)
    o = observer(times, where)
    u = -o / np.linalg.norm(o, axis=1, keepdims=True)          # observer -> Sun
    v = (c - o) / np.linalg.norm(c - o, axis=1, keepdims=True)  # observer -> comet
    n = SOLAR_POLE - (u @ SOLAR_POLE)[:, None] * u
    n /= np.linalg.norm(n, axis=1, keepdims=True)               # solar north on the sky
    w = np.cross(u, n)                                          # west: right, north up
    tx = np.degrees(np.arctan2(np.sum(v * w, 1), np.sum(v * u, 1)))
    ty = np.degrees(np.arcsin(np.clip(np.sum(v * n, 1), -1, 1)))
    return tx, ty


def fit(times, tx, ty, start: Elements, where="earth", free=("T", "node", "peri")):
    """Elements fitted to measured helioprojective positions (deg), varying
    the `free` ones from `start`, and the rms residual in arcsec."""
    from scipy.optimize import least_squares
    from datetime import timedelta
    tx, ty = np.asarray(tx), np.asarray(ty)

    def make(p):
        kw = {}
        for name, val in zip(free, p):
            kw[name] = start.T + timedelta(hours=val) if name == "T" else val
        return replace(start, **kw)

    def resid(p):
        mx, my = helioprojective(make(p), times, where)
        return np.concatenate([mx - tx, my - ty]) * 3600

    p0 = [0.0 if name == "T" else getattr(start, name) for name in free]
    sol = least_squares(resid, p0, x_scale="jac")
    e = make(sol.x)
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    return e, rms
