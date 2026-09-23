"""Element colours and radii, for every element the rest of the app may meet.

ASE names 119 elements (up to oganesson) and gives covalent radii for all of
them, but its Jmol colour table stops at meitnerium — 110 rows, Z 0 to 109.
Indexing it with anything heavier raises ``IndexError``, and the atom colours
are read deep inside a redraw: adding a single atom of, say, roentgenium took
the whole viewport down, which is not a thing a periodic table should be able
to do. The lookups here answer for any atomic number instead.

Jmol draws an element it has no colour for in pink, and so do we: a heavy
element then stands out as "not a colour anyone chose" rather than quietly
borrowing a neighbour's.
"""

from __future__ import annotations

import numpy as np
from ase.data import covalent_radii
from ase.data.colors import jmol_colors

#: Jmol's own colour for an element it has no entry for (deep pink).
UNKNOWN_COLOUR = np.array([1.0, 0.078, 0.576])

#: Radius drawn for an element ASE has no covalent radius for, in ångström.
#: ASE's own filler for the heaviest elements it names, so nothing jumps.
UNKNOWN_RADIUS = 0.2


def colours(numbers) -> np.ndarray:
    """Jmol colours as floats in ``[0, 1]``, one ``(r, g, b)`` row per atom."""
    z = np.atleast_1d(np.asarray(numbers, dtype=int))
    rgb = np.repeat(UNKNOWN_COLOUR[None, :], len(z), axis=0)
    known = (z >= 0) & (z < len(jmol_colors))
    rgb[known] = jmol_colors[z[known]]
    return rgb


def colour(z: int) -> np.ndarray:
    """The Jmol colour of element ``z`` as one ``(r, g, b)`` row of floats."""
    return colours([z])[0]


def hex_colour(z: int) -> str:
    """The Jmol colour of element ``z`` as ``"#rrggbb"``."""
    r, g, b = (int(round(c * 255)) for c in colour(z))
    return f"#{r:02x}{g:02x}{b:02x}"


def radii(numbers) -> np.ndarray:
    """Covalent radii in ångström, one per atom (array in, array out)."""
    z = np.atleast_1d(np.asarray(numbers, dtype=int))
    out = np.full(len(z), UNKNOWN_RADIUS, dtype=float)
    known = (z >= 0) & (z < len(covalent_radii))
    out[known] = covalent_radii[z[known]]
    return out


def radius(z: int) -> float:
    """The covalent radius of element ``z``, in ångström."""
    return float(radii([z])[0])


__all__ = ["UNKNOWN_COLOUR", "UNKNOWN_RADIUS", "colours", "colour", "hex_colour",
           "radii", "radius"]
