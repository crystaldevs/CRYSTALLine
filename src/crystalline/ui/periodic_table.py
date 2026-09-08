"""A visual periodic-table element picker.

A modal dialog laying every element out in the familiar 18-column table, each a
clickable button tinted by chemical category. Used by the Geometry panel so a
user can pick an element to add/set without hunting through a tiny combo box.

The table is pure Qt — no external data files — so it works wherever the rest of
the UI does. :meth:`PeriodicTableDialog.pick` is the one-liner entry point.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# The long-form table, 18 columns per period. "" is an empty cell; group 3 of
# periods 6–7 is blank because the f-block (below) holds those elements.
_MAIN_ROWS = [
    ["H", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "He"],
    ["Li", "Be", "", "", "", "", "", "", "", "", "", "", "B", "C", "N", "O", "F", "Ne"],
    ["Na", "Mg", "", "", "", "", "", "", "", "", "", "", "Al", "Si", "P", "S", "Cl", "Ar"],
    ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"],
    ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"],
    ["Cs", "Ba", "", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"],
    ["Fr", "Ra", "", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"],
]
# f-block: placed under the table, aligned from group 3 (column index 2).
_F_ROWS = [
    ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"],
    ["Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"],
]
_F_BLOCK_COLUMN = 2  # first f-block element sits under group 3

# Category → tint. Membership is by symbol so it needs no atomic-number tables.
_CATEGORY_COLORS = {
    "alkali": "#ff8a80",
    "alkaline": "#ffcc80",
    "transition": "#ffe082",
    "post_transition": "#c5e1a5",
    "metalloid": "#80cbc4",
    "nonmetal": "#81d4fa",
    "halogen": "#b39ddb",
    "noble": "#f48fb1",
    "lanthanide": "#e0e0a0",
    "actinide": "#e0b0a0",
}
_CATEGORIES = {
    "alkali": {"Li", "Na", "K", "Rb", "Cs", "Fr"},
    "alkaline": {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"},
    "metalloid": {"B", "Si", "Ge", "As", "Sb", "Te", "Po"},
    "nonmetal": {"H", "C", "N", "O", "P", "S", "Se"},
    "halogen": {"F", "Cl", "Br", "I", "At", "Ts"},
    "noble": {"He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"},
    "post_transition": {"Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "Nh", "Fl", "Mc", "Lv"},
    "lanthanide": set(_F_ROWS[0]),
    "actinide": set(_F_ROWS[1]),
}


def _category(symbol: str) -> str:
    for name, members in _CATEGORIES.items():
        if symbol in members:
            return name
    return "transition"  # everything else in the d-block region


# A cell has to hold a two-letter symbol at the app's font size, with the ring
# the current element is drawn with, and nothing else.
_CELL_WIDTH = 40
_CELL_HEIGHT = 34


class PeriodicTableDialog(QDialog):
    """Modal element picker. Emits :attr:`element_selected` and closes on a click."""

    element_selected = Signal(str)

    def __init__(self, current: Optional[str] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select element")
        self.selected: Optional[str] = None

        outer = QVBoxLayout(self)
        grid = QGridLayout()
        grid.setSpacing(2)
        outer.addLayout(grid)

        for r, row in enumerate(_MAIN_ROWS):
            for c, symbol in enumerate(row):
                if symbol:
                    grid.addWidget(self._element_button(symbol, symbol == current), r, c)
        gap = len(_MAIN_ROWS)  # a blank row separating the main table from the f-block
        grid.addWidget(QLabel(""), gap, 0)
        for r, row in enumerate(_F_ROWS):
            for c, symbol in enumerate(row):
                grid.addWidget(
                    self._element_button(symbol, symbol == current), gap + 1 + r, _F_BLOCK_COLUMN + c
                )

    def _element_button(self, symbol: str, is_current: bool) -> QPushButton:
        """One cell of the table: the symbol on its category colour.

        ``padding: 0`` is not cosmetic. The application stylesheet gives every
        push button comfortable padding, which is right for a button with a word
        on it and fatal in a 38-pixel cell: it left about a dozen pixels for the
        text and every two-letter symbol came out clipped ("Mg", "Zn", "Cd").
        A cell sizes itself, so it has to opt out.

        The selection ring is the app's accent rather than a fixed near-black,
        which was invisible against the dark theme.
        """
        from PySide6.QtWidgets import QApplication

        from crystalline.ui import theme

        accent = theme.active_palette(QApplication.instance()).accent
        button = QPushButton(symbol)
        button.setFixedSize(_CELL_WIDTH, _CELL_HEIGHT)
        button.setCursor(Qt.PointingHandCursor)
        color = _CATEGORY_COLORS[_category(symbol)]
        border = f"2px solid {accent}" if is_current else "1px solid rgba(0, 0, 0, 0.22)"
        button.setStyleSheet(
            f"QPushButton {{ background-color: {color}; color: #17202a;"
            f" font-weight: 700; font-size: 12px; padding: 0; margin: 0;"
            f" border: {border}; border-radius: 4px; }}"
            f"QPushButton:hover {{ border: 2px solid {accent}; }}"
        )
        button.setToolTip(symbol)
        button.clicked.connect(lambda _=False, s=symbol: self._choose(s))
        return button

    def _choose(self, symbol: str) -> None:
        self.selected = symbol
        self.element_selected.emit(symbol)
        self.accept()

    @classmethod
    def pick(cls, parent: Optional[QWidget] = None, current: Optional[str] = None) -> Optional[str]:
        """Show the table modally; return the chosen symbol, or ``None`` if cancelled."""
        dialog = cls(current=current, parent=parent)
        if dialog.exec() == QDialog.Accepted:
            return dialog.selected
        return None


__all__ = ["PeriodicTableDialog"]
