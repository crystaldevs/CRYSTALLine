"""Choose what to draw on top of an anharmonic scan.

The scan itself — the points, the fitted potential, the levels — is the figure
whatever one asks for; what is optional is everything drawn *on* it. The
wavefunctions and their densities are the reason to open the plot at all, but
they are normalised curves being drawn on an energy axis, so what they are
multiplied by is a setting rather than a fact. It opens on the mean level
spacing of the run, which puts a state at roughly three quarters of the gap
above it — the doublet of a double well being the case that still overlaps.

A dialog rather than a menu entry for the same reason as
:mod:`crystalline.ui.panels.vci_dialog`: the useful figure is not the same for a
stiff single well and a shallow double one.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QWidget

from crystalline.crystalio.anscan import AnscanRun
from crystalline.ui.panels.anscan_options import AnscanOptions
from crystalline.ui.widgets.wrapped_label import WrappedLabel


class AnscanDialog(QDialog):
    """Pick what goes on the scan; read it back with :meth:`options`."""

    def __init__(self, run: AnscanRun, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Anharmonic scan")
        self._run = run

        layout = QVBoxLayout(self)
        summary = WrappedLabel(run.summary, self)
        layout.addWidget(summary)

        # The same controls the plot carries beside it (see AnscanOptions), so
        # what is set here and what is pulled there cannot drift apart.
        self.options_widget = AnscanOptions(run, self)
        layout.addWidget(self.options_widget)
        # Named here as well: dialog_state remembers a dialog by walking the
        # widgets it holds, and these are the names it has always found.
        for name in ("wavefunctions", "scale_wf", "densities", "scale_prob",
                     "nstates", "harmonic", "points", "legend"):
            setattr(self, name, getattr(self.options_widget, name))

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

    # ── state ───────────────────────────────────────────────────────────
    def options(self) -> dict:
        """The chosen settings, as :func:`plot_anscan` keyword arguments."""
        return self.options_widget.options()

    def title(self) -> str:
        """A tab title naming the mode that was scanned."""
        return f"ANSCAN mode {self._run.mode}" if self._run.mode else "ANSCAN"


__all__ = ["AnscanDialog"]
