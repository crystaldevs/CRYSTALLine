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
* colour swatches (anything with ``colour()``/``setColour()``) → the colour,
* combo boxes → the *data* (or text) of the current item, never its index, since
  what a file offers can differ between runs and index 2 of one output is not
  index 2 of the next.

What a dialog has *selected* is its own business, since only it knows what the
selection is made of: a dialog that answers ``selection_state()`` has the answer
kept, and is handed it back through ``set_selection_state()``. The spectra
dialog names its ticked curves, which is the same rule as for a combo — a name,
never a position — so a curve this output does not have is simply not ticked.
Ticks were left out of this entirely at first, on the grounds that the curves a
file holds are its contents rather than a setting; but what the user ticked is
a setting, and reopening the dialog to find it blank is what sent them back to
the menu.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QAbstractButton, QAbstractSpinBox, QComboBox, QWidget

# The value carried for a combo whose items have no user data attached.
_BY_TEXT = "__text__"
# Where a dialog's own answer to "what is selected here" is kept. Not a widget
# name, so it cannot collide with one.
_SELECTION = "__selection__"


def capture(dialog) -> dict:
    """The settings currently shown in ``dialog``, ready to hand back to :func:`restore`."""
    state: dict = {}
    for name, widget in vars(dialog).items():
        if _is_swatch(widget):
            state[name] = ("colour", widget.colour())
        elif isinstance(widget, QComboBox):
            data = widget.currentData()
            state[name] = (_BY_TEXT, widget.currentText()) if data is None else ("data", data)
        elif isinstance(widget, QAbstractSpinBox):
            state[name] = ("value", widget.value())
        elif isinstance(widget, QAbstractButton) and widget.isCheckable():
            state[name] = ("checked", widget.isChecked())
    chosen = getattr(dialog, "selection_state", None)
    if callable(chosen):
        state[_SELECTION] = chosen()
    return state


def restore(dialog, state: Optional[dict]) -> None:
    """Put a previously captured ``state`` back onto ``dialog`` (``None`` = leave defaults).

    Best-effort by design: a control that has since disappeared, or an option
    this file does not offer, is skipped rather than raising. A dialog opened on
    a different output must still come up usable.
    """
    if not state:
        return
    chosen = state.get(_SELECTION)
    setter = getattr(dialog, "set_selection_state", None)
    if chosen is not None and callable(setter):
        try:
            setter(chosen)
        except Exception:  # noqa: BLE001 - a remembered choice must never block a dialog
            pass
    for name, entry in state.items():
        if name == _SELECTION:
            continue
        widget = getattr(dialog, name, None)
        if widget is None:
            continue
        try:
            kind, value = entry
            if kind == "colour" and _is_swatch(widget):
                widget.setColour(value)
            elif kind == "value" and isinstance(widget, QAbstractSpinBox):
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


def _is_swatch(widget) -> bool:
    return (isinstance(widget, QWidget) and callable(getattr(widget, "colour", None))
            and callable(getattr(widget, "setColour", None)))


__all__ = ["capture", "restore"]
