"""Completing molecules at the cell boundary draws the crystal; it must not undo edits."""

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from crystalline.ui.main_window import MainWindow  # noqa: E402


def _window(edited):
    calls = []
    return SimpleNamespace(
        _show_boundary=True,
        _shown_edited=edited,
        _apply_cell_view=lambda: calls.append("derived again"),
        _rebuild_keeping_edits=lambda: calls.append("rebuilt around the edits"),
        calls=calls,
    )


def test_with_edits_on_screen_the_copies_are_redrawn_around_them():
    window = _window(edited=True)
    MainWindow._on_boundary_toggled(window, False)
    assert window.calls == ["rebuilt around the edits"] and window._show_boundary is False


def test_without_edits_the_view_is_simply_derived_again():
    window = _window(edited=False)
    MainWindow._on_boundary_toggled(window, False)
    assert window.calls == ["derived again"]


def test_an_edit_is_drawn_with_the_bonds_and_polyhedra_of_the_edited_atoms():
    """The selected mode's equilibrium is what the bonds are worked out from, and
    the clean cell is what the polyhedra are: both move to the edited geometry
    before the redraw, or the edited atoms are linked as they were before."""
    import inspect

    source = inspect.getsource(MainWindow._on_structure_changed)
    redraw = source.index("self.viewport.renderer.refresh()")
    assert source.index("self.phonon_panel.invalidate_on_edit(") < redraw
    assert source.index("set_bond_structure(") < redraw
