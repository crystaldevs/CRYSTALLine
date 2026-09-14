"""The Symmetry panel: analysing on demand, ticking elements, and drawing them."""

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("spglib")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from ase.spacegroup import crystal  # noqa: E402

from crystalline.core import symmetry as S  # noqa: E402
from crystalline.core.structure import Structure  # noqa: E402
from crystalline.ui.panels.symmetry_panel import SymmetryPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _cubic() -> Structure:
    """A cubic perovskite: the full m3̄m — 13 axes, 9 planes and a centre."""
    return Structure.from_ase(
        crystal(
            ["Sr", "Ti", "O"], [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
            spacegroup=221, cellpar=[3.9] * 3 + [90] * 3,
        )
    )


def _monoclinic() -> Structure:
    """P2₁/c: a screw axis and a glide plane hold no point, so only a centre."""
    return Structure.from_ase(
        crystal("Si", [(0.1, 0.2, 0.3)], spacegroup=14, cellpar=[5, 6, 7, 90, 100, 90])
    )


def test_nothing_is_analysed_until_the_panel_is_asked(qapp):
    panel = SymmetryPanel(_cubic())
    assert panel.elements() == []
    assert panel.shown_elements() == []


def test_ticking_show_analyses_and_draws_every_element(qapp):
    panel = SymmetryPanel(_cubic())
    emitted = []
    panel.elements_changed.connect(lambda els, labels: emitted.append((list(els), labels)))

    panel._show.setChecked(True)

    assert len(panel.elements()) == 23
    assert len(panel.shown_elements()) == 23  # a point group is small enough to show whole
    assert emitted and len(emitted[-1][0]) == 23
    assert emitted[-1][1] is False  # labels off by default


def test_the_status_line_names_the_group_and_its_centre(qapp):
    panel = SymmetryPanel(_cubic())
    panel._show.setChecked(True)

    assert panel._status.text() == "Point symmetry m3̄m · 23 elements through (½, ½, ½)"


def test_a_structure_whose_symmetry_is_all_screws_shows_only_its_centre(qapp):
    panel = SymmetryPanel(_monoclinic())
    panel._show.setChecked(True)

    assert [e.kind for e in panel.elements()] == [S.POINT]
    assert panel._status.text().startswith("Point symmetry 1̄ ·")


def test_unticking_show_hides_them_without_losing_the_analysis(qapp):
    panel = SymmetryPanel(_cubic())
    emitted = []
    panel._show.setChecked(True)
    panel.elements_changed.connect(lambda els, labels: emitted.append(list(els)))

    panel._show.setChecked(False)

    assert emitted[-1] == []            # nothing drawn
    assert len(panel.elements()) == 23  # but the list is still there


def test_elements_are_grouped_by_kind_with_counts(qapp):
    panel = SymmetryPanel(_cubic())
    panel._show.setChecked(True)

    headings = [
        panel._tree.topLevelItem(i).text(0) for i in range(panel._tree.topLevelItemCount())
    ]
    assert headings == ["Symmetry axes (13)", "Mirror planes (9)", "Centre of inversion (1)"]


def test_ticking_one_group_draws_only_that_kind(qapp):
    panel = SymmetryPanel(_cubic())
    panel._show.setChecked(True)
    panel._set_all(False)
    assert panel.shown_elements() == []

    panel._tree.topLevelItem(1).setCheckState(0, Qt.Checked)  # auto-tristate ticks children

    shown = panel.shown_elements()
    assert shown and {e.kind for e in shown} == {S.PLANE}


def test_editing_the_structure_drops_what_is_drawn(qapp):
    structure = _cubic()
    panel = SymmetryPanel(structure)
    panel._show.setChecked(True)
    assert panel.shown_elements()

    panel.invalidate(structure)

    assert panel.shown_elements() == []
    assert not panel._show.isChecked()
    assert "edited" in panel._status.text()


def test_invalidating_rebinds_to_the_edited_structure(qapp):
    """What is re-analysed afterwards is the geometry as it now is, not as it was."""
    panel = SymmetryPanel(_cubic())
    panel._show.setChecked(True)

    panel.invalidate(_monoclinic())
    panel._show.setChecked(True)

    assert [e.kind for e in panel.elements()] == [S.POINT]


def test_a_tighter_tolerance_re_runs_the_search(qapp):
    atoms = crystal("Si", [(0.0, 0.0, 0.0)], spacegroup=221, cellpar=[4.0] * 3 + [90] * 3)
    atoms.set_cell([4.0, 4.0, 4.05, 90, 90, 90], scale_atoms=True)
    panel = SymmetryPanel(Structure.from_ase(atoms))

    panel._symprec.setValue(0.2)
    panel._show.setChecked(True)
    loose = len(panel.elements())

    panel._symprec.setValue(0.001)
    panel._reanalyse()

    assert len(panel.elements()) < loose


def test_a_new_structure_is_analysed_in_place_of_the_old(qapp):
    panel = SymmetryPanel(_cubic())
    panel._show.setChecked(True)
    assert len(panel.elements()) == 23

    panel.set_structure(_monoclinic())  # still ticked, so it re-analyses

    assert [e.kind for e in panel.elements()] == [S.POINT]


def test_shown_elements_reach_the_renderer(qapp):
    """The panel's output is what the renderer draws, end to end."""
    pyvista = pytest.importorskip("pyvista")
    from crystalline.viz.renderer import StructureRenderer

    structure = _cubic()
    panel = SymmetryPanel(structure)
    renderer = StructureRenderer(pyvista.Plotter(off_screen=True))
    renderer.set_structure(structure)
    panel.elements_changed.connect(
        lambda els, labels: renderer.set_symmetry_elements(els, labels=labels)
    )

    panel._show.setChecked(True)

    assert len(renderer._symmetry_actors) == 23  # one actor per element, no labels
    scene = np.asarray(renderer._scene_bounds(0.5)).reshape(3, 2)
    # Elements are clipped to the drawn scene. The allowance is the thickness they
    # are drawn with — a tube around a clipped line, a sphere on the centre, both
    # of which reach a little past the geometry they stand for.
    thickness = 0.15
    for actor in renderer._symmetry_actors:
        bounds = np.asarray(actor.GetBounds()).reshape(3, 2)
        assert np.all(bounds[:, 0] >= scene[:, 0] - thickness)
        assert np.all(bounds[:, 1] <= scene[:, 1] + thickness)


def test_the_cell_menu_entry_opens_the_panel_and_analyses(qapp):
    """Cell ▸ Point symmetry analysis: the panel is not on screen until it is
    asked for, and when it is, it opens with the answer rather than with a box
    still to tick."""
    from PySide6.QtWidgets import QDockWidget, QMainWindow

    from crystalline.ui.main_window import MainWindow

    class _StubWindow(QMainWindow):
        _show_symmetry_panel = MainWindow._show_symmetry_panel

    window = _StubWindow()
    window.symmetry_panel = SymmetryPanel(_cubic())
    window._symmetry_dock = QDockWidget("Point symmetry", window)
    window._symmetry_dock.setWidget(window.symmetry_panel)
    window.addDockWidget(Qt.RightDockWidgetArea, window._symmetry_dock)
    window._symmetry_dock.hide()  # as MainWindow leaves it

    assert window.symmetry_panel.elements() == []  # nothing analysed yet

    window._show_symmetry_panel()

    assert not window._symmetry_dock.isHidden()
    assert window.symmetry_panel.shown_elements()


def test_asking_for_the_analysis_again_leaves_the_ticks_alone(qapp):
    """Re-opening the panel must not undo what the user switched off in it."""
    panel = SymmetryPanel(_cubic())
    panel.show_analysis()
    panel._set_all(False)

    panel.show_analysis()

    assert panel.shown_elements() == []


def _label_hierarchy(renderer):
    """The label actor's text pipeline: its anchor points and text property."""
    algorithm = renderer._symmetry_actors[-1].GetMapper().GetInputAlgorithm()
    return np.asarray(algorithm.GetInput().points), algorithm.GetTextProperty()


def test_every_element_gets_its_own_label_spot(qapp):
    """Point symmetry puts every element through one centre, so labelling each at
    its own middle would stack them all on that spot — where VTK draws one and
    drops the rest. This is the bug that left the mirror planes unlabelled."""
    pyvista = pytest.importorskip("pyvista")
    from crystalline.viz.renderer import StructureRenderer

    structure = _cubic()
    renderer = StructureRenderer(pyvista.Plotter(off_screen=True))
    renderer.set_structure(structure)
    elements = S.analyse(structure).elements

    renderer.set_symmetry_elements(elements, labels=True)

    points, _property = _label_hierarchy(renderer)
    assert len(points) == len(elements)  # one anchor each, none dropped
    separations = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    separations += np.eye(len(points)) * 1e3  # ignore each point against itself
    assert separations.min() > 0.2  # and no two of them land on the same spot


def test_labels_use_a_font_that_has_the_symbols_they_need(qapp):
    """VTK's built-in fonts are Latin-1: σ comes out blank and the overbar of 1̄
    is dropped, which would label a centre of inversion "1" — the identity."""
    pytest.importorskip("matplotlib")
    pyvista = pytest.importorskip("pyvista")
    from crystalline.viz.renderer import StructureRenderer

    renderer = StructureRenderer(pyvista.Plotter(off_screen=True))
    renderer.set_structure(_cubic())

    renderer.set_symmetry_elements(S.analyse(_cubic()).elements, labels=True)

    _points, text = _label_hierarchy(renderer)
    assert text.GetFontFile().endswith(".ttf")


def test_labels_add_one_more_actor_for_the_text(qapp):
    pyvista = pytest.importorskip("pyvista")
    from crystalline.viz.renderer import StructureRenderer

    renderer = StructureRenderer(pyvista.Plotter(off_screen=True))
    renderer.set_structure(_cubic())
    elements = S.analyse(_cubic()).elements

    renderer.set_symmetry_elements(elements, labels=False)
    plain = len(renderer._symmetry_actors)
    renderer.set_symmetry_elements(elements, labels=True)

    assert len(renderer._symmetry_actors) == plain + 1


def test_a_row_is_tall_enough_for_its_subscripts(qapp):
    """The rows are drawn as rich text, and a subscript sits below the line.

    A QTextDocument also keeps a margin of its own unless told otherwise, and
    a row sized for a plain line has no room for either: the descenders of
    every entry were cut off.
    """
    from PySide6.QtGui import QTextDocument
    from PySide6.QtWidgets import QStyleOptionViewItem

    panel = SymmetryPanel(_cubic())
    panel._show.setChecked(True)
    tree = panel._tree
    delegate = tree.itemDelegate()
    axes = tree.topLevelItem(0)
    index = tree.indexFromItem(axes.child(0))

    option = QStyleOptionViewItem()
    option.initFrom(tree)
    document = QTextDocument()
    document.setDocumentMargin(0)
    document.setDefaultFont(option.font)
    document.setHtml(S.rich(axes.child(0).text(0)))

    assert delegate.sizeHint(option, index).height() >= document.size().height()


def test_a_row_too_narrow_for_its_text_is_elided(qapp):
    """Rather than cut mid-word, as clipping did."""
    from PySide6.QtGui import QFontMetrics

    panel = SymmetryPanel(_cubic())
    panel._show.setChecked(True)
    tree = panel._tree
    label = tree.topLevelItem(0).child(0).text(0)

    metrics = QFontMetrics(tree.font())
    short = metrics.elidedText(label, Qt.ElideRight, metrics.horizontalAdvance(label) // 2)

    assert short.endswith("…")
    assert S.rich(short).endswith("…")  # and the markup survives the ellipsis
