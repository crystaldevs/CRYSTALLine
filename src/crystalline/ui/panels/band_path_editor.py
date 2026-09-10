"""One path editor, used by both builders.

A band path is the same object whether it carries electrons (``BAND`` in a
``.d3``) or phonons (``BANDS`` inside ``FREQCALC`` in a ``.d12``): a walk
between points of the Brillouin zone. Only the record it is written into
differs. So it is edited by one widget, and the two builders cannot drift into
offering different powers over the same thing.

The endpoints are kept as **fractional** coordinates, never as the integers
that reach the deck. That is what makes the shrinking factor safe to change:
CRYSTAL reads a path as whole numbers over ISS, so ``4 0 4`` means X only if
ISS is 8, and an editor that stored the integers would silently redefine every
point the moment ISS moved. Storing fractions and multiplying on the way out
means the integers are derived, every time, from something that does not
depend on ISS at all.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from crystalline.core.brillouin import display_label, special_points, zone_lattice
from crystalline.core.properties_input import PropertiesInputError, band_path, band_shrink
from crystalline.core.structure import Structure
from crystalline.ui.safety import guard

CONVENTIONAL_NOTE = "the conventional path for this lattice"

# The shrinking-factor box shows this instead of a number when it is 0: the
# smallest ISS that makes every coordinate on the path a whole number.
AUTO_SHRINK = 0


class BandPathEditor(QWidget):
    """Pick, edit and order the segments of a band path.

    ``changed`` is emitted whenever the path or the shrinking factor moves, so
    a builder can refresh its preview without knowing which control did it.
    """

    changed = Signal()

    def __init__(self, structure: Structure, parent: Optional[QWidget] = None,
                 *, conventional: bool = True) -> None:
        super().__init__(parent)
        self._structure = structure
        self._points = special_points(zone_lattice(structure))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.conventional = _tick(
            "Use the conventional path (Setyawan–Curtarolo)",
            "The standard path for this Bravais lattice, derived from the "
            "lattice itself — so it follows the cell if the structure changes, "
            "and is what a reader will expect to see.\n\n"
            "Untick to edit the segments by hand.",
            conventional,
        )
        self.conventional.toggled.connect(self._on_conventional_toggled)
        outer.addWidget(self.conventional)

        self._list = QListWidget()
        self._list.setToolTip(
            "The segments the bands are computed along, in order. Each carries "
            "its own endpoints, so a sub-path or a point the conventional walk "
            "never visits is as expressible as the default."
        )
        self._list.setMinimumHeight(96)
        outer.addWidget(self._list)

        add_row = QHBoxLayout()
        add_row.setContentsMargins(0, 0, 0, 0)
        self._from = _kpoint_combo(self._points)
        self._to = _kpoint_combo(self._points)
        add = QPushButton("Add")
        add.clicked.connect(lambda: self.add_typed_segment())
        add_row.addWidget(self._from, 1)
        add_row.addWidget(QLabel("→"))
        add_row.addWidget(self._to, 1)
        add_row.addWidget(add)
        outer.addWidget(_row_widget(add_row))

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        edit_buttons = []
        for label, slot in (("Remove", self.remove_segment),
                            ("Up", lambda: self.move_segment(-1)),
                            ("Down", lambda: self.move_segment(1)),
                            # a lambda, not the bound method: clicked(bool)
                            # would pass the checked state as its first argument
                            ("Reset", lambda: self.reset())):
            button = QPushButton(label)
            button.clicked.connect(slot)
            buttons.addWidget(button)
            edit_buttons.append(button)
        buttons.addStretch(1)
        pick = QPushButton("Path builder…")
        pick.setToolTip("Build the path by clicking points on the Brillouin zone.")
        pick.clicked.connect(self.pick_on_zone)
        buttons.addWidget(pick)
        outer.addWidget(_row_widget(buttons))

        shrink_row = QHBoxLayout()
        shrink_row.setContentsMargins(0, 0, 0, 0)
        shrink_row.addWidget(QLabel("Shrinking factor (ISS)"))
        self._shrink = QSpinBox()
        self._shrink.setRange(AUTO_SHRINK, 96)
        self._shrink.setValue(AUTO_SHRINK)
        self._shrink.setSpecialValueText("auto")
        self._shrink.setToolTip(
            "The denominator the path's whole numbers are written over: a point "
            "is ISS × its fractional coordinates, so 4 0 4 is X only when ISS "
            "is 8.\n\n"
            "'auto' uses the smallest factor that makes every coordinate whole. "
            "A larger multiple of it works too and rescales the numbers; a "
            "factor that would leave a fraction is refused rather than rounded."
        )
        self._shrink.valueChanged.connect(lambda _value: self.changed.emit())
        shrink_row.addWidget(self._shrink)
        shrink_row.addStretch(1)
        outer.addWidget(_row_widget(shrink_row))

        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: palette(mid);")
        outer.addWidget(self.note)

        # Everything the conventional tick greys out. The shrinking factor is
        # not among them: it rescales whatever path is in force, conventional
        # or not, and is a property of how the path is *written*.
        self._editors = [self._list, self._from, self._to, add, pick] + edit_buttons
        self.fill_conventional()
        self._sync_enabled()

    # ── what the builders read ──────────────────────────────────────────
    def rows(self) -> List[Tuple[Tuple[str, str], tuple]]:
        """Every segment as ``((start_label, end_label), (start, end))``."""
        return [self._list.item(index).data(Qt.UserRole)
                for index in range(self._list.count())]

    def segments(self) -> List[tuple]:
        """The fractional endpoint pairs, in order."""
        return [segment for _labels, segment in self.rows()]

    def labels(self) -> List[Tuple[str, str]]:
        return [labels for labels, _segment in self.rows()]

    def is_conventional(self) -> bool:
        return self.conventional.isChecked()

    def shrink(self) -> Optional[int]:
        """The ISS to write the path over, or ``None`` when it is automatic.

        ``None`` rather than a number so a caller can tell "let the core work
        it out" from "the user asked for this one".
        """
        value = self._shrink.value()
        return None if value == AUTO_SHRINK else int(value)

    def effective_shrink(self) -> Optional[int]:
        """The ISS the path will actually be written over, chosen or derived."""
        chosen = self.shrink()
        if chosen is not None:
            return chosen
        segments = self.segments()
        if not segments:
            return None
        try:
            return band_shrink(segments)
        except PropertiesInputError:
            return None

    def integer_rows(self) -> List[str]:
        """The path as CRYSTAL reads it: whole numbers over the shrinking factor.

        Raises :class:`PropertiesInputError` when the chosen factor would leave
        a coordinate fractional — better a complaint than a path silently
        rounded onto points the crystal does not have.
        """
        # An empty list means the path was deliberately emptied — the tick
        # populates it otherwise. Falling back to the conventional path here
        # would quietly write a path nobody asked for, in place of the
        # complaint that says the path is missing.
        segments = self.segments()
        if not segments:
            return []
        shrink = self.shrink()
        if shrink is None:
            shrink = band_shrink(segments)
        rows = []
        for start, end in segments:
            values = []
            for coordinate in (*start, *end):
                scaled = coordinate * shrink
                whole = int(round(scaled))
                if abs(scaled - whole) > 1e-6:
                    raise PropertiesInputError(
                        f"A shrinking factor of {shrink} cannot express "
                        f"{coordinate:g} as a whole number. Use 'auto', or a "
                        f"multiple of {band_shrink(segments)}."
                    )
                values.append(str(whole))
            rows.append(" ".join(values))
        return rows

    # ── editing ─────────────────────────────────────────────────────────
    def _conventional(self):
        return band_path(self._structure)

    def fill_conventional(self) -> None:
        """Populate the list with the lattice's conventional path."""
        self._list.clear()
        try:
            labels, segments = self._conventional()
        except PropertiesInputError as exc:
            self.note.setText(str(exc))
            return
        for pair, segment in zip(labels, segments):
            self._add_row(pair, segment)
        self.note.setText(CONVENTIONAL_NOTE)

    def reset(self) -> None:
        self.fill_conventional()
        self.changed.emit()

    def _add_row(self, labels, segment) -> None:
        item = QListWidgetItem(
            f"{display_label(labels[0])}  →  {display_label(labels[1])}")
        item.setData(Qt.UserRole, (tuple(labels), tuple(segment)))
        self._list.addItem(item)

    def add_typed_segment(self) -> None:
        if self.is_conventional():
            return
        try:
            start_label, start = read_kpoint(self._from.currentText(), self._points)
            end_label, end = read_kpoint(self._to.currentText(), self._points)
        except ValueError as exc:
            self.note.setText(str(exc))
            return
        self._add_row((start_label, end_label), (start, end))
        self.note.setText("edited")
        self.changed.emit()

    def remove_segment(self) -> None:
        if self.is_conventional():
            return
        row = self._list.currentRow()
        if row >= 0:
            self._list.takeItem(row)
            self.note.setText("edited")
            self.changed.emit()

    def move_segment(self, delta: int) -> None:
        if self.is_conventional():
            return
        row = self._list.currentRow()
        target = row + delta
        if row < 0 or not 0 <= target < self._list.count():
            return
        item = self._list.takeItem(row)
        self._list.insertItem(target, item)
        self._list.setCurrentRow(target)
        self.note.setText("edited")
        self.changed.emit()

    @guard()
    def pick_on_zone(self) -> None:
        from crystalline.ui.panels.zone_picker import ZonePickerDialog

        # Picking a path *is* asking for one of your own, so it unticks for you
        # rather than refusing.
        self.conventional.setChecked(False)
        picked = ZonePickerDialog.pick(self._structure, self)
        if not picked:
            return
        self._list.clear()
        for pair, segment in picked:
            self._add_row(pair, segment)
        self.note.setText("picked on the zone")
        self.changed.emit()

    @guard()
    def _on_conventional_toggled(self, conventional: bool) -> None:
        """Ticking it restores the standard path; unticking leaves it to edit.

        Unticking deliberately keeps whatever is on the list rather than
        clearing it — the conventional path is the natural thing to start
        editing from, and clearing it would make the tick destructive.
        """
        if conventional:
            self.fill_conventional()
        self._sync_enabled()
        self.changed.emit()

    def _sync_enabled(self) -> None:
        for widget in self._editors:
            widget.setEnabled(not self.is_conventional())


def _tick(text: str, tip: str, checked: bool):
    from PySide6.QtWidgets import QCheckBox

    box = QCheckBox(text)
    box.setToolTip(tip)
    box.setChecked(checked)
    return box


def _row_widget(layout) -> QWidget:
    widget = QWidget()
    widget.setLayout(layout)
    return widget


def _kpoint_combo(points) -> QComboBox:
    """A combo of the lattice's special points, editable for anything else.

    Editable because the conventional set is not everything anyone wants: a
    point part-way along a line, or one this lattice's classification does not
    name, has to be typeable or the path editor is only a reordering tool.
    """
    combo = QComboBox()
    combo.setEditable(True)
    for label in points:
        # Shown as Γ, and read back as Γ too — read_kpoint takes either.
        combo.addItem(display_label(label))
    combo.setToolTip(
        "A labelled special point, or three fractional coordinates — "
        "'0.5 0 0.5', or '1/2 0 1/2'."
    )
    return combo


def read_kpoint(text: str, points: dict):
    """``"X"`` or ``"1/2 0 1/2"`` -> ``(label, (x, y, z))``.

    Raises ValueError with something actionable: an unreadable endpoint has to
    stop the segment being added, not be quietly rounded to the origin.
    """
    from fractions import Fraction

    text = text.strip()
    if not text:
        raise ValueError("Give a point label, or three fractional coordinates.")
    if text == "Γ":       # what the combo shows for the point stored as G
        text = "G"
    if text in points:
        return text, tuple(points[text])
    for label, point in points.items():          # tolerate a different case
        if label.lower() == text.lower():
            return label, tuple(point)
    values = [t for t in text.replace(",", " ").split() if t]
    if len(values) != 3:
        raise ValueError(
            f"{text!r} is not a point on this lattice, and not three coordinates."
        )
    try:
        coords = tuple(float(Fraction(v)) for v in values)
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"Could not read {text!r} as three numbers.") from None
    label = " ".join(f"{c:g}" for c in coords)
    return f"({label})", coords


__all__ = ["AUTO_SHRINK", "CONVENTIONAL_NOTE", "BandPathEditor", "read_kpoint"]
