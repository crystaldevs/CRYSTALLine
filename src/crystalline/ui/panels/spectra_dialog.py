"""Choose which vibrational spectra to plot, and how to broaden them.

A FREQCALC output can hold sixty-odd distinct curves once Raman polarisations
and anharmonic levels are counted (see :mod:`crystalline.crystalio.spectra`),
which is far too many for a menu of one-click entries. So the Plot menu carries
a single "Vibrational spectra…" item that opens this: a tree of exactly what the
loaded file contains, with checkboxes, and the broadening controls.

Several curves at once is the normal case rather than the exception — xx/yy/zz
on shared axes show the Raman anisotropy, harmonic under VCI shows the
anharmonic shift — so selection is multiple and the figure gets a legend.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from crystalline.ui.panels.controls import chooser_size
from crystalline.ui.panels.spectra_options import SpectraOptions
from crystalline.crystalio.spectra import (
    BROADENING_PARAMETERS,
    DEFAULT_ETA,
    DEFAULT_HWHM,
    DEFAULT_STDEV,
    LINESHAPES,
    SpectrumKind,
)


# How little of a list is still a list: about eight rows, which is enough to
# see what is on offer without scrolling for it.
_LIST_MIN_HEIGHT = 180


class SpectraDialog(QDialog):
    """Pick spectra and broadening; returns them through :meth:`selection`."""

    def __init__(self, kinds: Sequence[SpectrumKind], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Vibrational spectra")
        # Tall enough that the list gets the room, not merely its floor: the
        # broadening controls below it have a minimum of their own near 390 px,
        # so a dialog sized to its own minimum is one where the curves — the
        # thing actually being chosen here — are the part that gives way.
        self.resize(470, 840)
        self._kinds = list(kinds)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("Curves found in the loaded output — tick any number to overlay:")
        )

        self.tree = QTreeWidget(self)
        self.tree.setHeaderHidden(True)
        # A floor for the thing being chosen from. The broadening controls
        # below have a minimum height of their own, and without this the tree
        # was squeezed to a few lines — a list of thirteen curves shown four
        # at a time, with the rest behind a scrollbar.
        self.tree.setMinimumHeight(_LIST_MIN_HEIGHT)
        self.tree.itemChanged.connect(self._on_item_changed)
        self._populate()
        layout.addWidget(self.tree, 1)

        buttons_row = QHBoxLayout()
        for text, slot in (("Select all", self._select_all), ("Clear", self._select_none)):
            button = QPushButton(text, self)
            button.clicked.connect(slot)
            buttons_row.addWidget(button)
        buttons_row.addStretch(1)
        layout.addLayout(buttons_row)

        # The same controls the plot itself carries, so that what is tuned here
        # and what is tuned there cannot drift apart (see SpectraOptions).
        self.options_widget = SpectraOptions(self)
        layout.addWidget(self.options_widget)
        # Named here as well, because dialog_state remembers a dialog by walking
        # the widgets it holds, and because these names are what the tests and
        # the window have always reached for.
        self.lineshape = self.options_widget.lineshape
        self.hwhm = self.options_widget.hwhm
        self.stdev = self.options_widget.stdev
        self.eta = self.options_widget.eta
        self.fmin = self.options_widget.fmin
        self.fmax = self.options_widget.fmax
        # Not named here before, so reopening the dialog — an Edit included —
        # put the stacking and the legend back to their defaults.
        self.offset = self.options_widget.offset
        self.legend = self.options_widget.legend
        self.linewidth = self.options_widget.linewidth

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        # A plain harmonic IR run offers a single curve; making the user tick it
        # to get the only thing there is would be silly. Done here rather than in
        # _populate because ticking updates the OK button, which exists by now.
        if len(self._kinds) == 1:
            self._select_all()
        # The same rule as the other chooser dialogs, so two of them open at
        # the same size unless their rows genuinely differ. Taken before the
        # Curves section is put away for want of a second curve, so that the
        # tick which brings it back finds its room already there, rather than
        # taking it out of the list.
        self.resize(*chooser_size(self, (self.tree,)))
        self._sync_ok()

    # ── tree ────────────────────────────────────────────────────────────
    def _populate(self) -> None:
        """One branch per (kind, level, temperature); the curves hang off it."""
        self.tree.blockSignals(True)
        groups: dict = {}
        for kind in self._kinds:
            parent = groups.get(kind.group)
            if parent is None:
                parent = QTreeWidgetItem(self.tree, [kind.group])
                parent.setFlags(parent.flags() | Qt.ItemIsAutoTristate | Qt.ItemIsUserCheckable)
                parent.setCheckState(0, Qt.Unchecked)
                parent.setExpanded(True)
                groups[kind.group] = parent
            leaf = QTreeWidgetItem(parent, [kind.leaf_label])
            leaf.setFlags(leaf.flags() | Qt.ItemIsUserCheckable)
            leaf.setCheckState(0, Qt.Unchecked)
            leaf.setData(0, Qt.UserRole, kind)
        self.tree.blockSignals(False)

    def _leaves(self):
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            for j in range(parent.childCount()):
                yield parent.child(j)

    def _on_item_changed(self, _item, _column) -> None:
        self._sync_ok()

    def _set_all(self, state) -> None:
        self.tree.blockSignals(True)
        for leaf in self._leaves():
            leaf.setCheckState(0, state)
        self.tree.blockSignals(False)
        self._sync_ok()

    def _select_all(self) -> None:
        self._set_all(Qt.Checked)

    def _select_none(self) -> None:
        self._set_all(Qt.Unchecked)

    # ── state ───────────────────────────────────────────────────────────
    def _sync_ok(self) -> None:
        """Every change of what is ticked comes through here."""
        chosen = self.selected_kinds()
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(bool(chosen))
        self.options_widget.set_curve_count(len(chosen))

    def selected_kinds(self) -> List[SpectrumKind]:
        """The ticked curves, in the order they were listed."""
        return [
            leaf.data(0, Qt.UserRole)
            for leaf in self._leaves()
            if leaf.checkState(0) == Qt.Checked
        ]

    def selection_state(self) -> List[str]:
        """The ticked curves, named — what :mod:`dialog_state` keeps for next time.

        By name (the attribute each curve lives under on the output) rather than
        by position, for the same reason a combo's choice is kept by its data: a
        different run holds a different set of curves, and the third row of one
        is not the third row of the next.
        """
        return [kind.attribute for kind in self.selected_kinds()]

    def set_selection_state(self, attributes) -> None:
        """Tick the named curves; anything this output does not hold is skipped."""
        wanted = set(attributes or ())
        self.tree.blockSignals(True)
        for leaf in self._leaves():
            kind = leaf.data(0, Qt.UserRole)
            leaf.setCheckState(
                0, Qt.Checked if kind.attribute in wanted else Qt.Unchecked)
        self.tree.blockSignals(False)
        self._sync_ok()

    def options(self) -> dict:
        """Broadening and range, as :func:`plot_spectra` keyword arguments."""
        return self.options_widget.options()

__all__ = ["SpectraDialog"]
