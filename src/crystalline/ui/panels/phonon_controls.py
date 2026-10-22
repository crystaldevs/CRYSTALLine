"""How phonon bands or a phonon DOS are drawn, beside the plot.

The controls every band plot has (:mod:`~crystalline.ui.panels.bands_controls`),
for frequencies: the window in the plot's unit, and the zero-frequency line as
the reference line. The unit stays in the dialog, which has read the files a
change of it needs.
"""

from __future__ import annotations

from crystalline.crystalio import electronic as el
from crystalline.crystalio import phonon_bands as pb
from crystalline.ui.panels.bands_controls import (PATH_KEYS, BandsControls, aspect,
                                                  split_options)

#: What the dialog's tracks were, carried with a plot's options for these
#: controls and not handed to the plotter.
EXTENT_KEYS = ("frequency_bounds", "dos_bounds", "dos_handles", "dos_projections")


def plot_arguments(options: dict) -> tuple:
    """``(bands_path, dos_path, PhononOptions)`` from a plot's options."""
    return split_options(options, EXTENT_KEYS, pb.PhononOptions)


def title(options: dict) -> str:
    """A tab title naming what is drawn."""
    return pb.plot_title(options.get("mode", el.BANDS))


class PhononControls(BandsControls):
    """Frequency window, DOS axis and line styling."""

    WINDOW_SECTION = "Frequency"
    WINDOW_KEY, BOUNDS_KEY = "frequency_range", "frequency_bounds"
    WINDOW_TRACK = (-200.0, 1200.0, 0.0, 1000.0, 1, 10.0)
    DOS_DECIMALS, DOS_STEP = 4, 0.001
    LINE_KEYS = ("show_zero", "zero_colour")
    LINE_LABEL = "Zero frequency"
    LINE_COLOUR = pb.ZERO_COLOUR


__all__ = ["EXTENT_KEYS", "PATH_KEYS", "PhononControls", "aspect", "plot_arguments", "title"]
