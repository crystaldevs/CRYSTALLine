"""Remember what a dialog was last set to, so reopening it does not start over.

The property-plot dialogs (spectra, VCI, PES, anharmonic scan) are tuned rather
than answered: a broadening width, a frequency window, a state count. Each was
built fresh on every open and discarded on close, so every one of those had to be
typed again — and the values that matter are exactly the ones that took a moment
to settle on.

State is captured off the dialog's own attributes rather than declared per
dialog, so a control added to one of them is remembered without anything here
changing. Only the settings widgets are touched:

* spin boxes  → their value,
* check boxes → their checked state,
* combo boxes → the *data* (or text) of the current item, never its index, since
  what a file offers can differ between runs and index 2 of one output is not
  index 2 of the next.

Anything file-specific is deliberately left alone. A combo entry that is gone
when the state is restored is skipped, and the list of curves a spectra dialog
found in *this* output is not a setting at all — it is the file's contents, and
carrying a tick across to a different file would assert something untrue.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QAbstractButton, QAbstractSpinBox, QComboBox

# The value carried for a combo whose items have no user data attached.
_BY_TEXT = "__text__"


def capture(dialog) -> dict:
    """The settings currently shown in ``dialog``, ready to hand back to :func:`restore`."""
    state: dict = {}
    for name, widget in vars(dialog).items():
        if isinstance(widget, QComboBox):
            data = widget.currentData()
            state[name] = (_BY_TEXT, widget.currentText()) if data is None else ("data", data)
        elif isinstance(widget, QAbstractSpinBox):
            state[name] = ("value", widget.value())
        elif isinstance(widget, QAbstractButton) and widget.isCheckable():
            state[name] = ("checked", widget.isChecked())
    return state


def restore(dialog, state: Optional[dict]) -> None:
    """Put a previously captured ``state`` back onto ``dialog`` (``None`` = leave defaults).

    Best-effort by design: a control that has since disappeared, or an option
    this file does not offer, is skipped rather than raising. A dialog opened on
    a different output must still come up usable.
    """
    if not state:
        return
    for name, entry in state.items():
        widget = getattr(dialog, name, None)
        if widget is None:
            continue
        try:
            kind, value = entry
            if kind == "value" and isinstance(widget, QAbstractSpinBox):
                widget.setValue(value)
            elif kind == "checked" and isinstance(widget, QAbstractButton):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                index = (
                    widget.findText(value) if kind == _BY_TEXT else widget.findData(value)
                )
                if index >= 0:  # the option this file offers may differ
                    widget.setCurrentIndex(index)
        except Exception:  # noqa: BLE001 - a remembered value must never block a dialog
            continue


__all__ = ["capture", "restore"]
