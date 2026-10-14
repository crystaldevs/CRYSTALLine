"""Elastic properties from a run's elastic tensor, drawn so they can be tuned.

An elastic run holds one 6×6 tensor, and every plot of it — Young's modulus,
linear compressibility, the shear modulus and Poisson's ratio, as a surface
over all directions or as sections through the principal planes — is a view of
that tensor. So it is read once, and each view is drawn from it in memory: the
controls beside the plot switch between them as fast as the drawing goes.

The values are CRYSTALClear's, from its functions for each property; the
surface is drawn here (see :func:`elastic_surface`), the sections by
CRYSTALClear's ``plot_cry_ela_2D``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import numpy as np

#: What can be shown, as (key, label).
QUANTITIES = (
    ("young", "Young's modulus"),
    ("comp", "Linear compressibility"),
    ("shear", "Shear modulus"),
    ("poisson", "Poisson's ratio"),
)
#: Over the directions perpendicular to each one, for the two properties that
#: depend on a second direction: shear and Poisson's ratio.
STATISTICS = (("avg", "Average"), ("min", "Minimum"), ("max", "Maximum"))
VIEWS = (("surface", "3D surface"), ("sections", "Sections"))
PLANES = ("xy", "xz", "yz")
COLOUR_MAPS = ("viridis", "cividis", "plasma", "magma", "coolwarm")
#: Directions per angle: what a surface is evaluated on.
RESOLUTIONS = (("Coarse", 50), ("Normal", 100), ("Fine", 200))
DEFAULTS = {"quantity": "young", "statistic": "avg", "view": "surface",
            "planes": PLANES, "cmap": "viridis", "resolution": 100}

_SECOND_DIRECTION = {"shear", "poisson"}


def choose_key(quantity: str, statistic: str = "avg") -> str:
    """The name CRYSTALClear's functions know a property by."""
    return f"{quantity} {statistic}" if quantity in _SECOND_DIRECTION else quantity


def needs_statistic(quantity: str) -> bool:
    return quantity in _SECOND_DIRECTION


def read_tensor(path: str) -> np.ndarray:
    """The elastic tensor of the run in ``path``, in GPa."""
    import contextlib
    import io

    from CRYSTALClear.crystal_io import Crystal_output

    with contextlib.redirect_stdout(io.StringIO()):
        out = Crystal_output(path)
        out.get_elatensor()
    return np.asarray(out.elatensor, dtype=float)


def options_for(menu_key: str) -> dict:
    """The options a Plot-menu entry opens on: ``ela_young``, ``ela2d_comp``…"""
    view = "sections" if menu_key.startswith("ela2d_") else "surface"
    quantity = menu_key.split("_", 1)[1]
    return dict(DEFAULTS, quantity=quantity, view=view)


def title(options: dict) -> str:
    """The tab's name: the property, and how it is shown."""
    label = dict(QUANTITIES)[options.get("quantity", "young")]
    if needs_statistic(options.get("quantity", "young")):
        label += f", {dict(STATISTICS)[options.get('statistic', 'avg')].lower()}"
    return label if options.get("view", "surface") == "surface" else f"{label} (sections)"


def plot_elastic(tensor, *, quantity: str = "young", statistic: str = "avg",
                 view: str = "surface", planes: Sequence[str] = PLANES,
                 cmap: str = "viridis", resolution: int = 100):
    """One view of ``tensor``, as a Figure."""
    from crystalline.crystalio.plotting import _plot_module, _to_figure

    CCplt = _plot_module()
    key = choose_key(quantity, statistic)
    if view == "surface":
        return elastic_surface(tensor, key, int(resolution), module=CCplt, cmap=cmap)
    chosen = tuple(p for p in PLANES if p in set(planes)) or PLANES
    # CRYSTALClear wants an object holding the tensor, which is all it reads.
    holder = SimpleNamespace(elatensor=np.asarray(tensor, dtype=float))
    figure = _to_figure(CCplt.plot_cry_ela_2D(holder, key, max(int(resolution), 100),
                                              planes=chosen))
    _tidy_sections(figure.axes[0], key)
    return figure


def _tidy_sections(ax, key: str) -> None:
    """The polar sections in the surface's units, and out of their own way.

    CRYSTALClear draws them in its own units — compressibility in GPa⁻¹, where
    the surface beside them says TPa⁻¹ — under an abbreviated title with no
    unit, the legend on top of the 45° label and the radial labels crowding
    each other. Rescaled, untitled with the unit over the legend, the legend
    outside the circle, and four radial labels rather than a fence of them.
    """
    from matplotlib.ticker import MaxNLocator

    _label, unit, scale = _ELASTIC_QUANTITIES[key]
    if scale != 1.0:
        for line in ax.get_lines():
            line.set_ydata(np.asarray(line.get_ydata(), dtype=float) * scale)
        low, high = ax.get_ylim()
        ax.set_ylim(low * scale, high * scale)
    # No title (the tab names the quantity); the unit, which a polar plot has
    # no axis label to carry, heads the legend instead.
    ax.set_title("")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="y", labelsize="small")
    ax.set_rlabel_position(67.5)
    legend = ax.get_legend()
    if legend is not None:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.04, 1.0),
                  borderaxespad=0.0, title=unit or None, alignment="left")


# What each surface shows, its unit, and the factor from CRYSTALClear's value.
# Compressibility comes in GPa⁻¹ — 0.002 and the like — and reads as TPa⁻¹.
_ELASTIC_QUANTITIES = {
    "young": ("Young's modulus", "GPa", 1.0),
    # In maths rather than as the character ⁻, which STIX — the plot font —
    # does not have.
    "comp": ("Linear compressibility", "TPa$^{-1}$", 1000.0),
    "shear avg": ("Shear modulus, average", "GPa", 1.0),
    "shear min": ("Shear modulus, minimum", "GPa", 1.0),
    "shear max": ("Shear modulus, maximum", "GPa", 1.0),
    "poisson avg": ("Poisson's ratio, average", "", 1.0),
    "poisson min": ("Poisson's ratio, minimum", "", 1.0),
    "poisson max": ("Poisson's ratio, maximum", "", 1.0),
}


def elastic_surface(tensor, choose: str, ndeg: int = 100, module=None,
                    cmap: str = "viridis"):
    """A directional elastic property as a lit surface, coloured by its value.

    CRYSTALClear's own surface (``plot_cry_ela``) colours by jet — which puts
    false edges where the data has none, and none survive greyscale — keeps
    its scale on the far side with no unit, and leaves the shape small in a
    box of grey walls. The values are CRYSTALClear's still: its functions give
    the property in every direction; only the drawing is done here.

    viridis by default, lit from one side so the shape reads as a solid; the colour bar
    beside it with what it measures and in what; no walls, a light grid, three
    ticks to an axis. A plain :class:`Figure`, never pyplot, since builders run
    off the UI thread.
    """
    import matplotlib
    import numpy as np
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    from crystalline.crystalio.plotting import _plot_module

    CCplt = module if module is not None else _plot_module()
    label, unit, scale = _ELASTIC_QUANTITIES[choose]
    S = np.linalg.inv(np.asarray(tensor, dtype=float))
    theta_1d = np.linspace(0.0, np.pi, ndeg)
    phi_1d = np.linspace(0.0, 2.0 * np.pi, ndeg)
    theta, phi = np.meshgrid(theta_1d, phi_1d)
    if choose == "young":
        R = CCplt.plot_cry_young(theta, phi, S)
    elif choose == "comp":
        R = CCplt.plot_cry_comp(theta, phi, S)
    elif choose.startswith("shear"):
        R = CCplt.plot_cry_shear(theta_1d, phi_1d, S, ndeg, choose.split()[1])
    else:
        R = CCplt.plot_cry_poisson(theta_1d, phi_1d, S, ndeg, choose.split()[1])
    R = np.asarray(R, dtype=float) * scale
    X = R * np.sin(theta) * np.cos(phi)
    Y = R * np.sin(theta) * np.sin(phi)
    Z = R * np.cos(theta)

    colours = matplotlib.colormaps[cmap]
    norm = Normalize(vmin=float(R.min()), vmax=float(R.max()))
    with matplotlib.rc_context({"grid.color": "#e2e7eb", "grid.linewidth": 0.6,
                                "xtick.minor.visible": False,
                                "ytick.minor.visible": False}):
        figure = Figure(figsize=(6.4, 5.2))
        ax = figure.add_subplot(projection="3d")
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colours(norm(R)),
                        linewidth=0, antialiased=False, shade=True)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor("#cfd6dc")
            axis.set_major_locator(MaxNLocator(3, symmetric=True))
        reach = 1.02 * float(max(np.abs(X).max(), np.abs(Y).max(), np.abs(Z).max()))
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_zlim(-reach, reach)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.tick_params(labelsize="small", pad=1)
        ax.view_init(elev=22, azim=-58)
        bar = figure.colorbar(ScalarMappable(norm=norm, cmap=colours), ax=ax,
                              shrink=0.62, pad=0.10, aspect=22)
        bar.set_label(f"{label} ({unit})" if unit else label)
        bar.outline.set_visible(False)
        bar.ax.minorticks_off()
        figure.subplots_adjust(left=0.0, right=0.94, bottom=0.04, top=0.98)
    return figure


__all__ = ["COLOUR_MAPS", "DEFAULTS", "PLANES", "QUANTITIES", "RESOLUTIONS", "STATISTICS",
           "VIEWS", "choose_key", "elastic_surface", "needs_statistic", "options_for",
           "plot_elastic", "read_tensor", "title"]
