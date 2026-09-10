"""Saving a 3D view as a picture — one dialog, for every 3D view.

The structure window offered format, resolution and transparency; the Brillouin
zone offered a bare PNG. Same job, two answers, and the difference is only
visible to someone who has used both — which is everyone who uses the app.
So the dialog lives here and both windows call it.
"""

from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QMessageBox,
    QSpinBox,
)

# (extension, menu label, is_vector, has_alpha)
FORMATS = [
    ("png", "PNG image", False, True),
    ("jpg", "JPEG image", False, False),
    ("tif", "TIFF image", False, True),
    ("svg", "SVG vector", True, False),
    ("pdf", "PDF vector", True, False),
    ("eps", "EPS vector", True, False),
]


def ask_image_options(parent) -> Optional[Tuple[str, str, int, bool]]:
    """Prompt for ``(ext, filter_label, scale, transparent)``; ``None`` if cancelled.

    Scale supersamples raster output (the 3D analogue of DPI); transparency
    needs an alpha channel, so both are greyed out for the formats that cannot
    use them (vector, and opaque rasters like JPEG/BMP).
    """
    dialog = QDialog(parent)
    dialog.setWindowTitle("Export image")
    form = QFormLayout(dialog)

    fmt_box = QComboBox(dialog)
    for ext, label, is_vector, has_alpha in FORMATS:
        fmt_box.addItem(label, (ext, label, is_vector, has_alpha))
    form.addRow("Format:", fmt_box)

    scale_box = QSpinBox(dialog)
    scale_box.setRange(1, 8)
    scale_box.setValue(2)
    scale_box.setPrefix("×")
    scale_box.setToolTip("Supersampling: ×2 renders at twice the on-screen pixels each way.")
    form.addRow("Resolution:", scale_box)

    transparent = QCheckBox("Transparent background", dialog)
    form.addRow("", transparent)

    def sync_enabled() -> None:
        _ext, _label, is_vector, has_alpha = fmt_box.currentData()
        scale_box.setEnabled(not is_vector)
        transparent.setEnabled(not is_vector and has_alpha)

    fmt_box.currentIndexChanged.connect(sync_enabled)
    sync_enabled()

    buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, dialog)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    form.addRow(buttons)

    if dialog.exec() != QDialog.Accepted:
        return None
    ext, label, is_vector, _has_alpha = fmt_box.currentData()
    scale = 1 if is_vector else scale_box.value()
    return ext, label, scale, transparent.isChecked() and transparent.isEnabled()


def export_view(parent, save, default_name: str) -> Optional[str]:
    """Ask for the options and a name, then hand both to ``save``.

    ``save(path, scale=…, transparent=…)`` does the writing, so a caller with a
    plotter and a caller with a whole viewport can share this without either
    knowing about the other.
    """
    options = ask_image_options(parent)
    if options is None:
        return None
    ext, label, scale, transparent = options
    path, _filter = QFileDialog.getSaveFileName(
        parent, "Export image", f"{default_name}.{ext}", f"{label} (*.{ext})")
    if not path:
        return None
    try:
        save(path, scale=scale, transparent=transparent)
    except Exception as exc:  # noqa: BLE001 - surface any render/write error
        QMessageBox.critical(parent, "Export failed",
                             f"Could not save the image:\n{exc}")
        return None
    return path


__all__ = ["FORMATS", "ask_image_options", "export_view"]
