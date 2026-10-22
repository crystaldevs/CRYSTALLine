"""One tab per open file: what each tab owns, and how the window reaches it.

Several files can be open at once, one shown at a time. A tab is everything the
window used to hold for its one file — the structure as loaded and as shown,
the cell view and supercell, the phonon modes, the undo history, the orbital or
density on screen — together with the widgets that show it: a 3D view of its
own, and its own Phonons, Info, Display, Geometry, Point-symmetry and Plots
panels. Switching tab is then a matter of showing the other tab's widgets; the
camera, the selection, a half-read plot and the display settings of each file
are simply still there, because nothing had to be torn down and rebuilt.

:class:`~crystalline.ui.main_window.MainWindow` keeps reading and writing these
as its own attributes (``self.structure``, ``self._source``, ``self.viewport``,
…): :func:`route_to_active_tab` turns each name listed in :data:`PER_TAB` into a
property that forwards to the tab being shown. So the window's code — two and a
half thousand lines written for one file — acts on "the file on screen" without
knowing there are others.

Qt-only glue: no VTK or crystallography here, so it can be tested alone.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence

# What belongs to one file. Everything else on the window — the docks, the
# menus, the editing mode, the busy overlay, the remembered plot-dialog
# settings — is shared by all of them.
PER_TAB_STATE = (
    "structure",        # the structure on screen (a view of _source)
    "_source",          # the structure as loaded
    "_cell_view",
    "_supercell",
    "_show_boundary",
    "_shown_edited",
    "_modes",
    "_qmodes",
    "_qindex",
    "_modes_natom",
    "_tile_restore",
    "_adps",
    "_adp_index",
    "_history",
    "_suppress_undo",
    "_output_path",
    "_output_props",
    "_orbital",
    "_density_shown",
    "_unit_cell",
    "_bond_structure",
)

# The widgets each tab has its own of. The panels sit in the window's docks, in
# a stack per dock that shows the current tab's; the viewport is the tab page.
PER_TAB_PANELS = (
    "phonon_panel",
    "info_panel",
    "display_panel",
    "geometry_panel",
    "symmetry_panel",
    "plot_panel",
)
PER_TAB_WIDGETS = ("viewport", "animator", "structure_panel") + PER_TAB_PANELS

PER_TAB = PER_TAB_STATE + PER_TAB_WIDGETS

# Shown on a tab with no file in it.
UNTITLED = "Untitled"


class FileTab:
    """The state and the widgets of one open file.

    The widgets are not necessarily there yet. A tab's page is a bare container
    from the moment the tab exists; the 3D view goes into it, and the panels are
    built, the first time the tab is shown — see
    :meth:`~crystalline.ui.main_window.MainWindow._realise_tab`. Opening ten
    files at once would otherwise build ten VTK render windows, nine of them for
    tabs nobody has looked at yet, at ~100 ms each before anything is drawn.
    """

    def __init__(self) -> None:
        for name in PER_TAB:
            setattr(self, name, None)
        self.path: Optional[str] = None        # the file it was opened from
        self.label = UNTITLED                  # what its tab says (see tab_labels)
        # The tab's page in the bar — always present, so the tab can be added,
        # named, moved and closed before it has a 3D view. The view is put
        # inside it when the tab is first shown.
        self.page = None
        # A file read but not yet put into widgets, for a tab that has not been
        # shown: ``(path, LoadedFile)``. The parse happens when the file is
        # opened (one that will not read still opens no tab); only the showing
        # of it waits.
        self.pending = None
        # The display settings the tab starts from — the look of the tab it was
        # opened from, kept until there is a renderer to give them to.
        self.start_settings = None
        # Whether this tab's Plots window was open when it was last shown, so
        # coming back to it brings its plots back — and a tab without plots does
        # not leave another file's figures floating over it.
        self.plots_open = False
        # What the open output offers (plots, spectra, VCI, ...), worked out
        # once: some of the probes read the whole .out, and switching tab must
        # not re-read every file each time.
        self.capabilities: dict = {}
        # What opening the file had to leave out (its modes, say), still to be
        # said: a tab read in behind the one being worked in says it when it is
        # first brought to the front.
        self.notice: Optional[str] = None

    def built(self) -> bool:
        """Whether this tab's widgets exist yet (it has been shown at least once)."""
        return self.viewport is not None

    def is_blank(self) -> bool:
        """No file and no atoms: a tab the next opened file can simply take over."""
        if self.path is not None or self.pending is not None:
            return False
        for structure in (self._source, self.structure):
            if structure is not None and len(structure):
                return False
        return True

    def widgets(self) -> List[object]:
        """Every widget the tab owns, for tearing it down when it is closed."""
        return [getattr(self, name) for name in PER_TAB_WIDGETS
                if getattr(self, name) is not None]


def _tab_attribute(name: str) -> property:
    def get(window):
        tab = window.__dict__.get("_tab")
        if tab is None:
            # Before the first tab exists: an AttributeError, as for any
            # attribute not yet set — which is what ``hasattr``/``getattr``
            # guards in the window rely on while it is being built.
            raise AttributeError(name)
        return getattr(tab, name)

    def set_(window, value) -> None:
        tab = window.__dict__.get("_tab")
        if tab is None:
            raise AttributeError(f"{name} belongs to a tab, and there is none yet")
        setattr(tab, name, value)

    return property(get, set_, doc=f"``{name}`` of the tab on screen.")


def route_to_active_tab(cls):
    """Class decorator: make every name in :data:`PER_TAB` the current tab's."""
    for name in PER_TAB:
        setattr(cls, name, _tab_attribute(name))
    return cls


def tab_labels(paths: Sequence[Optional[str]]) -> List[str]:
    """What each tab is called: its file's name, told apart where two share one.

    CRYSTAL runs are routinely named alike — a ``fort.34`` or a ``run.out`` in
    each folder of a series — so a name that appears twice is shown with the
    folder it is in, and with more of the path until the two differ.
    """
    labels = [UNTITLED if not path else os.path.basename(path) for path in paths]
    depth = 1
    while True:
        seen = {}
        for index, label in enumerate(labels):
            if paths[index]:
                seen.setdefault(label, []).append(index)
        clashes = [group for group in seen.values() if len(group) > 1]
        if not clashes or depth > 8:
            return labels
        depth += 1
        for group in clashes:
            for index in group:
                labels[index] = _tail(paths[index], depth)


def _tail(path: str, depth: int) -> str:
    """The last ``depth`` components of ``path``, joined with ``/``."""
    parts = [part for part in os.path.normpath(path).replace("\\", "/").split("/") if part]
    return "/".join(parts[-depth:])


def window_title(name: Optional[str], application: str) -> str:
    """``run.out — CRYSTALLine``, or the application's own title without a file."""
    if not name:
        return application
    return f"{name} — {application.split(' — ')[0]}"


def openable(paths: Iterable[str], action_of, can_import: bool = True) -> tuple:
    """Split dropped paths into ``(to_open, to_import, ignored)``.

    Every file that is a structure of its own opens, each in its own tab.
    Atoms to import (an .xyz, a .pdb) go into the structure on screen — but
    only when nothing is being opened alongside them, since the structure on
    screen is then about to be another one; and only the first, as before.

    ``can_import`` is whether there is a structure to add them to. An atom list
    is not a structure: it has no cell and no calculation behind it, only
    coordinates, so with nothing open there is nothing for it to join and it is
    left out rather than becoming the structure itself.
    """
    to_open, to_import, ignored = [], [], []
    for path in paths:
        action = action_of(path)
        if action == "open":
            to_open.append(path)
        elif action == "import" and can_import:
            to_import.append(path)
        else:
            ignored.append(path)
    if to_open:
        return to_open, [], ignored + to_import
    return [], to_import[:1], ignored + to_import[1:]


__all__ = [
    "FileTab",
    "PER_TAB",
    "PER_TAB_PANELS",
    "PER_TAB_STATE",
    "PER_TAB_WIDGETS",
    "UNTITLED",
    "openable",
    "route_to_active_tab",
    "tab_labels",
    "window_title",
]
