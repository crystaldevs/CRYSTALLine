"""How every figure the app draws looks: the CRYSTALLine house style.

A figure leaves the app — into a paper, a talk, a thesis — and is the part of
it most people ever see. So every plot is drawn in one style, whichever code
draws it: CRYSTALClear's plotters and this app's own alike. It is set through
matplotlib's rcParams, which every figure reads as it is created, and applied
where every plot passes (:func:`crystalline.crystalio.plotting._plot_module`).

The font is not here. It is the person's choice, made in Plot ▸ Plot font and
applied by :func:`crystalline.crystalio.plotting.apply_font`; this sets
everything else, and leaves the font keys alone so the two never fight.

The style is the one figures in the physical sciences usually take: thin
axes in the app's ink, ticks on all four sides pointing in, minor ticks, no
grid, a legend on a plain white backing with no border — it often sits over
a curve, and text drawn straight onto a line is hard to read. The colours lead with CRYSTAL's cyan (the
app's accent, taken deep enough to read as a line on white) and orange — a
pair told apart under every common colour-vision deficiency — then green,
purple, red and slate.
"""

from __future__ import annotations

INK = "#14212b"

#: The colours curves are drawn in, in order.
CYCLE = ("#0b7a96", "#d9822b", "#3f9e4d", "#8e5bb5", "#c2414b", "#5b6878",
         "#c9a227", "#1f4e79")

STYLE = {
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.9,
    "axes.grid": False,
    # Only 3D boxes and polar sections draw a grid; there it should recede.
    "grid.color": "#dde3e8",
    "grid.linewidth": 0.6,
    "axes.titlesize": "medium",
    "axes.titleweight": "semibold",
    "text.color": INK,
    "lines.linewidth": 1.6,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.minor.width": 0.7,
    "ytick.minor.width": 0.7,
    "legend.frameon": True,
    "legend.edgecolor": "none",
    "legend.facecolor": "white",
    "legend.framealpha": 0.85,
    "legend.fancybox": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    # An axis offset written −3.7088×10³, not −3.7088e3 as code would print it.
    "axes.formatter.use_mathtext": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


# matplotlib's stock colours, as CRYSTALClear's plotters name them outright —
# "tab:blue" for an equation of state, red and blue for a PES cut, green, blue
# and red for the elastic sections — mapped to the house colour in each role.
# Only these: black stays black, a colour map stays as it is, and a colour a
# person chose (a band, a curve scheme) is none of these and is left alone.
_STOCK = {
    "#1f77b4": CYCLE[0], "#0000ff": CYCLE[0],                    # blue
    "#ff7f0e": CYCLE[1],                                         # orange
    "#2ca02c": CYCLE[2], "#008000": CYCLE[2], "#00ff00": CYCLE[2],  # green
    "#9467bd": CYCLE[3], "#bf00bf": CYCLE[3],                    # purple, "m"
    "#d62728": CYCLE[4], "#ff0000": CYCLE[4],                    # red
}


def apply() -> None:
    """Set the house style for every figure built from now on."""
    import matplotlib
    from cycler import cycler

    matplotlib.rcParams.update(STYLE)
    matplotlib.rcParams["axes.prop_cycle"] = cycler(color=list(CYCLE))


def _house(colour):
    """The house colour standing in for a stock one, or ``None``."""
    from matplotlib.colors import to_hex, to_rgba

    try:
        rgba = to_rgba(colour)
    except (TypeError, ValueError):
        return None
    replacement = _STOCK.get(to_hex(rgba, keep_alpha=False))
    return None if replacement is None else to_rgba(replacement, rgba[3])


def _recolour(artist) -> None:
    """A line's or a legend swatch's stock colours, in the house ones."""
    from matplotlib.lines import Line2D

    if isinstance(artist, Line2D):
        for get, put in ((artist.get_color, artist.set_color),
                         (artist.get_markerfacecolor, artist.set_markerfacecolor),
                         (artist.get_markeredgecolor, artist.set_markeredgecolor)):
            house = _house(get())
            if house is not None:
                put(house)
    elif hasattr(artist, "get_facecolor") and hasattr(artist, "set_facecolor"):
        try:
            faces = artist.get_facecolor()
        except Exception:  # noqa: BLE001 - not every patch answers alike
            return
        if len(getattr(faces, "shape", ())) == 2 and len(faces) == 1:
            house = _house(faces[0])
            if house is not None:
                artist.set_facecolor(house)


def _finish_3d(ax) -> None:
    """A 3D box without the clutter: no walls, no grid line per minor tick.

    The style's minor ticks are meant for flat axes; in a 3D box matplotlib
    draws a grid line at each, which crowds the walls with lines. The walls'
    grey fill goes too, leaving their edges, and an axis still on matplotlib's
    automatic ticks gets fewer, smaller labels — a box shows three of them
    edge-on, and twelve numbers along each collide.
    """
    from matplotlib.ticker import AutoLocator, MaxNLocator

    ax.minorticks_off()
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("#cfd6dc")
        if type(axis.get_major_locator()) is AutoLocator:
            axis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(labelsize="small")


#: Set on a figure whose title was asked for by name — typed into a dialog —
#: and is to be kept (see :func:`keep_title`).
_KEEPS_TITLE = "_crystalline_keeps_title"


def keep_title(figure):
    """Mark ``figure``'s title as one the person asked for, which :func:`finish` keeps."""
    setattr(figure, _KEEPS_TITLE, True)
    return figure


def _drop_titles(figure) -> None:
    """No title over a figure, unless it was asked for.

    The tab it opens in names it already, and a title is the first thing to be
    cropped off when a figure goes into a paper — where the caption names it.
    The plotters add their own all the same: CRYSTALClear's elastic sections,
    its VCI levels ("VCI@…"), an XRD pattern's file title.
    """
    if getattr(figure, _KEEPS_TITLE, False):
        return
    for ax in figure.axes:
        for where in ("center", "left", "right"):
            if ax.get_title(loc=where):
                ax.set_title("", loc=where)
    suptitle = getattr(figure, "_suptitle", None)
    if suptitle is not None:
        suptitle.set_text("")


def finish(figure) -> None:
    """Bring a figure drawn elsewhere into the house style. Idempotent.

    rcParams reach what a figure is built with by default; a plotter that names
    its own colours, or its own ticks, is past their reach. So, once drawn:
    stock colours become the house ones, legends included (a legend keeps
    copies of its lines, not the lines); and a heatmap or a colour bar loses
    the minor and the top and right ticks the style gives every other axis —
    between the cells of a map they mark nothing.
    """
    from matplotlib.collections import QuadMesh
    from matplotlib.contour import ContourSet
    from matplotlib.ticker import MaxNLocator

    _drop_titles(figure)
    for ax in figure.axes:
        for artist in list(ax.get_lines()) + list(ax.collections):
            _recolour(artist)
        legend = ax.get_legend()
        if legend is not None:
            for handle in getattr(legend, "legend_handles", None) or legend.legendHandles:
                _recolour(handle)
        if getattr(ax, "name", "") == "3d":
            _finish_3d(ax)
            continue
        if getattr(ax, "name", "") == "polar":
            continue
        is_colourbar = ax.get_label() == "<colorbar>"
        is_map = bool(ax.images) or any(isinstance(c, (QuadMesh, ContourSet))
                                        for c in ax.collections)
        if is_colourbar or is_map:
            ax.minorticks_off()
            ax.tick_params(which="both", top=False, right=False)
        bar = getattr(ax, "_colorbar", None) if is_colourbar else None
        if bar is not None and isinstance(getattr(bar, "mappable", None), ContourSet):
            # A filled contour's bar is ticked at its level boundaries — numbers
            # like 4177 and 8355 that nobody chose. Round ones instead.
            bar.locator = MaxNLocator(nbins=7)
            bar.update_ticks()
    for legend in getattr(figure, "legends", ()):
        for handle in getattr(legend, "legend_handles", None) or legend.legendHandles:
            _recolour(handle)


__all__ = ["CYCLE", "INK", "STYLE", "apply", "finish", "keep_title"]
