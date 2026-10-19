"""Choose how to draw the VCI wavefunctions of the loaded output.

A VCI run holds one state per configuration — tens of thousands of them for a
crystal — so unlike the elastic surfaces there is no single figure to show: which
states, and how much mixing to keep, decide whether the plot says anything. Hence
a dialog rather than two one-click menu entries, the same reasoning as for
:mod:`crystalline.ui.panels.spectra_dialog`.

The two representations answer different questions, so picking one switches the
options: a coefficient map takes many states and can show the sign of
:math:`A_{n,s}`, a Sankey takes a handful and reads as weights.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QWidget

from crystalline.crystalio.vci import VCIRun
from crystalline.ui.panels import vci_options
from crystalline.ui.panels.vci_options import VCIOptions
from crystalline.ui.widgets.wrapped_label import WrappedLabel


class VCIDialog(QDialog):
    """Pick a representation and its options; read them back with :meth:`options`."""

    def __init__(self, run: VCIRun, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("VCI states")
        self._run = run

        layout = QVBoxLayout(self)
        summary = WrappedLabel(run.summary, self)
        layout.addWidget(summary)
        if not run.modes:
            # Without the PES scan's mode indices the labels fall back on a dense
            # 1..nmodes numbering, which is not what the Phonons dock shows.
            note = WrappedLabel(
                "The VCI-active modes could not be matched to the phonon "
                "numbering, so configurations are labelled from 1.", self
            )
            note.setEnabled(False)
            layout.addWidget(note)

        # The same controls the plot carries beside it (see VCIOptions), so
        # what is set here and what is pulled there cannot drift apart.
        self.options_widget = VCIOptions(run, self)
        layout.addWidget(self.options_widget)
        # Named here as well: dialog_state remembers a dialog by walking the
        # widgets it holds, and these are the names it has always found.
        for name in ("representation", "fmin", "fmax", "irrep", "threshold",
                     "signed", "annotate", "weight"):
            setattr(self, name, getattr(self.options_widget, name))
        self._count = self.options_widget.count

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        # An empty or overfull window cannot be accepted: there is nothing to
        # draw, or too much for a figure to say anything.
        self.options_widget.window_checked.connect(
            self._buttons.button(QDialogButtonBox.Ok).setEnabled)
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(
            self.options_widget.acceptable())
        # Each representation has its own section, so switching changes how
        # tall the dialog needs to be; nothing in it is sized by hand.
        self.representation.currentIndexChanged.connect(lambda _i: self.adjustSize())

    # ── state ───────────────────────────────────────────────────────────
    def window(self) -> tuple:
        """The chosen window as ``(min, max)``, in ascending order."""
        return self.options_widget.window()

    def options(self) -> dict:
        """The chosen settings, as :func:`plot_vci` keyword arguments."""
        return self.options_widget.options()

    def title(self) -> str:
        """A tab title naming the flavour and the representation."""
        return vci_options.title(self._run, self.options())


__all__ = ["VCIDialog"]
