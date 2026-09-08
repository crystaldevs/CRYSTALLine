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

from crystalline.ui.panels.controls import Section, range_row, slider_row
from crystalline.crystalio.spectra import (
    BROADENING_PARAMETERS,
    DEFAULT_ETA,
    DEFAULT_HWHM,
    DEFAULT_STDEV,
    LINESHAPES,
    SpectrumKind,
)


class SpectraDialog(QDialog):
    """Pick spectra and broadening; returns them through :meth:`selection`."""

    def __init__(self, kinds: Sequence[SpectrumKind], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Vibrational spectra")
        self.resize(460, 560)
        self._kinds = list(kinds)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("Curves found in the loaded output — tick any number to overlay:")
        )

        self.tree = QTreeWidget(self)
        self.tree.setHeaderHidden(True)
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

        broadening = Section(layout, "Broadening")
        self.lineshape = QComboBox(self)
        for label, name in LINESHAPES:
            self.lineshape.addItem(label, name)
        self.lineshape.currentIndexChanged.connect(self._sync_broadening)
        broadening.add("Lineshape", self.lineshape)

        # One control per parameter rather than a single relabelled "width":
        # a pseudo-Voigt is eta*Lorentzian(HWHM) + (1-eta)*Gaussian(sigma), so
        # its two widths are independent and both have to be reachable.
        # One control per parameter rather than a single relabelled "width":
        # a pseudo-Voigt is eta*Lorentzian(HWHM) + (1-eta)*Gaussian(sigma), so
        # its two widths are independent and both have to be reachable.
        #
        # Widths are logarithmic: a broadening is judged by ratio, and on a
        # linear 0.1-200 scale every value anyone uses is squeezed into the first
        # few pixels. The mixing fraction is a plain fraction, so it stays linear.
        self._rows = {}
        for name, caption, value, low, high, step, decimals, log, tip in (
            ("hwhm", "Lorentzian HWHM (cm⁻¹)", DEFAULT_HWHM, 0.1, 200.0, 1.0, 1, True,
             "Half-width at half-maximum of the Lorentzian profile."),
            ("stdev", "Gaussian std. dev. (cm⁻¹)", DEFAULT_STDEV, 0.1, 200.0, 1.0, 1, True,
             "Standard deviation of the Gaussian profile."),
            ("eta", "Mixing η (Lorentzian fraction)", DEFAULT_ETA, 0.0, 1.0, 0.05, 2, False,
             "Lorentzian fraction of the pseudo-Voigt mixture:\n"
             "1.00 is pure Lorentzian, 0.00 is pure Gaussian."),
        ):
            box = slider_row(
                broadening, caption, value, low, high, step,
                decimals=decimals, logarithmic=log,
            )
            box.setToolTip(tip)
            setattr(self, name, box)
            self._rows[name] = broadening.row_widgets(caption)

        # One track with two handles, rather than a "from" and a "to" that know
        # nothing about each other: the span is the thing being chosen, and this
        # shows how much of the spectrum it covers.
        span = Section(layout, "Frequency range")
        _slider, self.fmin, self.fmax = range_row(
            span, "Range (cm⁻¹)", 0.0, 4000.0, 0.0, 5000.0, decimals=0, step=100.0,
        )

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._sync_broadening()
        # A plain harmonic IR run offers a single curve; making the user tick it
        # to get the only thing there is would be silly. Done here rather than in
        # _populate because ticking updates the OK button, which exists by now.
        if len(self._kinds) == 1:
            self._select_all()
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
    def _sync_broadening(self) -> None:
        """Show only the parameters the chosen lineshape actually uses.

        A Lorentzian ignores the Gaussian width, a stick spectrum ignores all of
        them, and only a pseudo-Voigt needs the mixing fraction. Greying the
        others out beats leaving controls that quietly do nothing.
        """
        used = BROADENING_PARAMETERS.get(self.lineshape.currentData(), ())
        for name, (label, widget) in self._rows.items():
            enabled = name in used
            label.setEnabled(enabled)
            widget.setEnabled(enabled)

    def _sync_ok(self) -> None:
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(bool(self.selected_kinds()))

    def selected_kinds(self) -> List[SpectrumKind]:
        """The ticked curves, in the order they were listed."""
        return [
            leaf.data(0, Qt.UserRole)
            for leaf in self._leaves()
            if leaf.checkState(0) == Qt.Checked
        ]

    def options(self) -> dict:
        """Broadening and range, as :func:`plot_spectra` keyword arguments."""
        low, high = sorted((self.fmin.value(), self.fmax.value()))
        return {
            "lineshape": self.lineshape.currentData(),
            "hwhm": self.hwhm.value(),
            "stdev": self.stdev.value(),
            "eta": self.eta.value(),
            # An inverted or zero-width range means "don't clip".
            "frequency_range": None if low == high else (low, high),
        }


__all__ = ["SpectraDialog"]
