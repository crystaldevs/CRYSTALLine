"""Element colours and radii answer for every element, not just ASE's tables.

ASE's Jmol colour table stops at meitnerium (Z 109) while the periodic-table
picker offers every element up to oganesson (Z 118). Adding one of the nine in
between used to raise IndexError inside the redraw and take the viewport with
it.
"""

import numpy as np
import pytest

pytest.importorskip("ase")

from ase.data import chemical_symbols, covalent_radii  # noqa: E402
from ase.data.colors import jmol_colors  # noqa: E402

from crystalline.core import elements  # noqa: E402

# The elements the picker offers that ASE has no colour for: Ds .. Og.
_UNCOLOURED = range(len(jmol_colors), 119)


def test_every_element_the_picker_offers_has_a_colour_and_a_radius():
    for z in range(1, len(chemical_symbols)):
        rgb = elements.colour(z)
        assert rgb.shape == (3,)
        assert np.all((rgb >= 0.0) & (rgb <= 1.0)), chemical_symbols[z]
        assert elements.radius(z) > 0.0, chemical_symbols[z]


def test_elements_ase_colours_keep_their_colour():
    for z in (1, 6, 8, 26, 42, 109):
        assert np.allclose(elements.colour(z), jmol_colors[z])
        assert elements.radius(z) == pytest.approx(covalent_radii[z])


def test_the_heaviest_elements_fall_back_to_the_unknown_colour():
    assert list(_UNCOLOURED), "ASE grew its colour table; this test is now moot"
    for z in _UNCOLOURED:
        assert np.allclose(elements.colour(z), elements.UNKNOWN_COLOUR), chemical_symbols[z]


def test_an_atomic_number_off_the_table_is_answered_not_raised():
    """CRYSTAL writes ghost atoms as Z + 200; nothing may raise on one."""
    for z in (0, 200, 1000, -1):
        assert elements.colour(z).shape == (3,)
        assert elements.radius(z) > 0.0


def test_colours_and_radii_map_over_arrays_in_order():
    numbers = [8, 111, 1]
    assert elements.colours(numbers).shape == (3, 3)
    assert np.allclose(elements.colours(numbers)[0], jmol_colors[8])
    assert np.allclose(elements.colours(numbers)[1], elements.UNKNOWN_COLOUR)
    assert np.allclose(elements.radii(numbers)[:1], covalent_radii[8])


def test_hex_colour_is_a_css_colour():
    assert elements.hex_colour(8) == "#ff0d0d"       # Jmol's oxygen red
    assert elements.hex_colour(111).startswith("#")  # roentgenium: the fallback
