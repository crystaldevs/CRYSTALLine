"""The characterisation of normal modes (CNM) of a LOCMODES run, and its figures.

CRYSTALClear reads and draws it. The tests draw from a
stand-in for its ``Crystal_output`` — the decomposition as a table, the
frequencies — rather than from a run: the synthetic H₂O₂ output has a CNM but
no frequencies. A real run is checked, on the machine that has it, by a test
that stays out of the repository.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from crystalline.crystalio import cnm
from crystalline.crystalio.cnm import CNMRun



def _run(groups=("1 NO", "2 NO", "9 CH"), frequencies=(100.0, 205.0, 310.0, 1500.0),
         table=None, eigenvectors=True, path="/runs/a.out"):
    """A run of modes 4, 5, … at ``frequencies`` (cm⁻¹), its CNM ``table`` (%)."""
    import pandas as pd

    modes = tuple(range(4, 4 + len(frequencies)))
    if table is None:
        rows = [[60.0, 39.5, 0.5], [10.0, 0.0, 90.0], [33.0, 33.0, 34.0], [0.2, 99.6, 0.2]]
        table = rows[:len(frequencies)]
    to_thz = 1.0 / 33.35640951981521                 # cm⁻¹ → THz, as get_phonon reads them
    every = [0.0, 0.0, 0.0] + list(frequencies)
    output = SimpleNamespace(
        cnm_sym=pd.DataFrame(table, index=list(modes), columns=list(groups)),
        frequency=np.array([[f * to_thz for f in every]]),
    )
    return CNMRun(path=path, output=output, modes=modes, frequencies=tuple(frequencies),
                  coordinates=len(groups) * 4, groups=len(groups), spacegroup="P2_1/c (14)",
                  eigenvectors=eigenvectors)


# ── a run's modes ─────────────────────────────────────────────────────────
def test_a_span_of_frequencies_holds_the_modes_in_it():
    run = _run()
    assert run.modes_in(90, 210) == [4, 5]
    assert run.modes_in(210, 90) == [4, 5]                  # either way round
    assert run.modes_in(2000, 3000) == []
    assert run.label(5) == "Mode 5, 205.0 cm⁻¹"


def test_the_modes_ticked_first_are_round_the_mode_selected_if_there_is_one():
    frequencies = tuple(100.0 + 10 * i for i in range(30))
    run = _run(groups=("1 NO",), frequencies=frequencies, table=[[100.0]] * 30)
    first = run.default_modes()
    assert first[0] == 4 and len(first) == cnm.DEFAULT_WINDOW_MODES   # from the lowest
    first = run.default_modes(around=25)
    assert 25 in first and len(first) == cnm.DEFAULT_WINDOW_MODES


def test_the_groups_a_window_shows_are_those_over_the_threshold():
    """As plot_cry_cnm keeps them: a group reaching the threshold in any of the
    modes drawn gets a colour and a line of the legend."""
    run = _run()
    assert run.groups_in([4]) == 2                           # mode 4: 60 and 39.5, not 0.5
    assert run.groups_in([4], threshold=0.5) == 3
    assert run.groups_in([4, 5, 6], threshold=40.0) == 2     # 1 NO at 60, 9 CH at 90
    assert run.groups_in([]) == 0 and run.groups_in([99]) == 0


def test_the_legend_and_the_figure_widen_with_what_they_hold():
    rows = cnm.LEGEND_ROWS
    assert cnm.legend_columns(rows - 1) == 1                 # with the residues, one full column
    assert cnm.legend_columns(rows) == 2
    many = tuple(100.0 + i for i in range(30))
    run = _run(groups=("1 NO",), frequencies=many, table=[[100.0]] * 30)
    narrow = cnm.figure_aspect(run, {"modes": (4, 5, 6), "legend": False})
    wide = cnm.figure_aspect(run, {"modes": tuple(range(4, 34)), "legend": False})
    assert narrow == pytest.approx(4 / 3) and wide > narrow
    assert cnm.figure_aspect(run, {"modes": tuple(range(4, 34)), "legend": True}) > wide


def test_a_series_is_as_wide_as_its_runs_and_the_legend_it_drew():
    figure = SimpleNamespace(**{cnm.LEGEND_COLUMNS: 3})
    assert cnm.series_aspect(2) == pytest.approx(4 / 3)       # nothing drawn yet
    assert cnm.series_aspect(6, figure) > cnm.series_aspect(6)
    assert cnm.series_aspect(12) > cnm.series_aspect(2)      # six runs still fit in 4:3


def test_a_figure_says_how_many_columns_its_legend_has():
    assert getattr(cnm.plot_cnm(_run(), modes=(4, 5, 6)), cnm.LEGEND_COLUMNS) == 1
    assert getattr(cnm.plot_cnm(_run(), modes=(4, 5, 6), legend=False), cnm.LEGEND_COLUMNS) == 0


def test_the_summary_says_whether_the_coordinates_are_grouped():
    assert _run().summary == ("4 normal modes, 12 internal coordinates in 3 sets "
                              "equivalent by symmetry in P2₁/c (No. 14)")
    ungrouped = CNMRun("/runs/a.out", None, (4,), (100.0,), 12, 12,
                       note="Invalid file. Dimension information not found")
    assert "each on its own: not grouped by symmetry (Invalid file." in ungrouped.summary


# ── a series ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("names, labels", [
    (["sample_FREQ_LOCMODES_02GPa_bugfix.out", "sample_FREQ_LOCMODES_04GPa_bugfix.out",
      "sample_FREQ_LOCMODES_1GPa_bugfix.out"], ["02GPa", "04GPa", "1GPa"]),
    (["sample_FREQ_tight_LOCMODES.out", "sample_FREQ_LOCMODES_02GPa_bugfix.out",
      "sample_FREQ_LOCMODES_1GPa_bugfix.out"], ["tight", "02GPa", "1GPa"]),
    (["run_0.out", "run_10.out"], ["0", "10"]),
    (["run_a_1.out", "run_b_1.out"], ["a", "b"]),             # the number is shared
    (["alpha.out", "beta.out"], ["alpha", "beta"]),            # nothing shared
    (["only.out"], ["only"]),
])
def test_a_run_is_labelled_with_what_tells_it_apart(names, labels):
    assert cnm.series_labels(["/runs/" + n for n in names]) == labels


def test_a_series_needs_two_runs_labelled_apart_with_eigenvectors():
    pytest.importorskip("CRYSTALClear")
    with pytest.raises(ValueError, match="two runs"):
        cnm.plot_cnm_series([_run()], labels=["0"], mode=4)
    with pytest.raises(ValueError, match="label of its own"):
        cnm.plot_cnm_series([_run(), _run()], labels=["0", "0"], mode=4)
    with pytest.raises(ValueError, match="needs a label"):
        cnm.plot_cnm_series([_run(), _run()], labels=["0", " "], mode=4)


def test_without_frequencies_each_bar_carries_its_label_alone(monkeypatch):
    """CRYSTALClear prints "label\\nfrequency" under each bar and adds the
    frequency to the axis title; asked not to, the plot is left with the labels
    as written and the axis as named."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from crystalline.crystalio import plotting

    def pressure(cos, mode, labels, threshold, min_overlap, xlabel):
        figure, ax = plt.subplots()
        ax.bar(range(len(labels)), [100.0] * len(labels), label="1 NO")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([f"{label}\n{469.3 - i:.1f}" for i, label in enumerate(labels)])
        ax.set_xlabel(f"{xlabel} (frequency [cm$^{{-1}}$])", fontsize=14)
        return figure, ax, {}

    monkeypatch.setattr(plotting, "_plot_module",
                        lambda: SimpleNamespace(plot_cry_cnm_pressure=pressure))
    runs = [_run(path="/runs/a.out"), _run(path="/runs/b.out")]
    figure = cnm.plot_cnm_series(runs, labels=["ambient", "1 GPa"], mode=4,
                                 axis="Pressure", legend=False)
    ax = figure.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["ambient\n469.3", "1 GPa\n468.3"]
    figure = cnm.plot_cnm_series(runs, labels=["ambient", "1 GPa"], mode=4,
                                 axis="Pressure", legend=False, frequencies=False)
    ax = figure.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["ambient", "1 GPa"]
    assert ax.get_xlabel() == "Pressure" and ax.xaxis.label.get_fontsize() == 14
    with pytest.raises(ValueError, match="no eigenvectors printed in b.out"):
        cnm.plot_cnm_series([_run(), _run(eigenvectors=False, path="/runs/b.out")],
                            labels=["0", "1"], mode=4)


# ── colours ───────────────────────────────────────────────────────────────
def test_the_house_colours_have_names():
    pytest.importorskip("matplotlib")
    named = cnm.named_palette()
    assert [n for n, _ in named[:2]] == ["Cyan", "Orange"]
    assert named[8][0] == "Light cyan" and named[16][0] == "Dark cyan"
    assert [c for _, c in named] == list(cnm.palette())


def test_the_groups_lead_with_the_house_colours():
    pytest.importorskip("matplotlib")
    from crystalline.crystalio.figure_style import CYCLE

    colours = cnm.palette()
    assert colours[:len(CYCLE)] == tuple(CYCLE)
    assert len(set(colours)) == len(colours) == 3 * len(CYCLE)


def test_a_group_keeps_its_colour_from_one_draw_to_the_next():
    pytest.importorskip("matplotlib")
    colours = cnm.palette()
    memory = {}
    assert cnm.assign_colours(["1 NO", "2 NO"], memory) == {"1 NO": colours[0],
                                                            "2 NO": colours[1]}
    # The window moves: 1 NO goes, 9 CH comes, and 2 NO stays as it was.
    assert cnm.assign_colours(["2 NO", "9 CH"], memory) == {"2 NO": colours[1],
                                                            "9 CH": colours[0]}


def test_past_the_palette_the_colours_go_round_again():
    pytest.importorskip("matplotlib")
    colours = cnm.palette()
    groups = [f"{n} CH" for n in range(len(colours) + 3)]
    given = cnm.assign_colours(groups, {})
    assert [given[g] for g in groups[len(colours):]] == list(colours[:3])


# ── the figures ───────────────────────────────────────────────────────────
def test_the_bars_are_drawn_in_the_house_colours():
    from matplotlib.colors import to_hex

    from crystalline.crystalio.figure_style import INK

    memory = {}
    figure = cnm.plot_cnm(_run(), modes=(6, 4, 5), colours=memory)      # any order
    ax = figure.axes[0]
    faces = {c.get_label(): to_hex(c.patches[0].get_facecolor()) for c in ax.containers}
    colours = cnm.palette()
    assert faces == {"1 NO": colours[0], "2 NO": colours[1], "9 CH": colours[2],
                     cnm.RESIDUES: cnm.RESIDUES_COLOUR}
    assert memory == {"1 NO": colours[0], "2 NO": colours[1], "9 CH": colours[2]}
    edges = {to_hex(p.get_edgecolor()) for c in ax.containers for p in c.patches}
    assert edges == {INK}
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["1 NO", "2 NO", "9 CH",
                                                                  cnm.RESIDUES]


def test_the_empty_bars_are_hidden_and_every_group_keeps_its_swatch():
    """CRYSTALClear gives every group a bar on every mode, empty where it is
    under the threshold; hidden, they cost nothing to draw. 9 CH is under it
    on mode 4, its first: hidden before the legend was built, its swatch,
    a copy of that bar, would have gone with it."""
    from matplotlib.container import BarContainer

    figure = cnm.plot_cnm(_run(), modes=(4, 5, 6, 7), colours={})
    ax = figure.axes[0]
    bars = {c.get_label(): c.patches for c in ax.containers if isinstance(c, BarContainer)}
    assert bars["9 CH"][0].get_height() == 0                 # under 1 % on mode 4
    every = [p for patches in bars.values() for p in patches]
    assert all(p.get_visible() == (p.get_height() != 0) for p in every)
    assert any(not p.get_visible() for p in every)
    legend = ax.get_legend()
    handles = getattr(legend, "legend_handles", None) or legend.legendHandles
    assert len(handles) == len(bars) and all(h.get_visible() for h in handles)


def test_a_number_on_a_bar_is_in_the_colour_that_reads_on_it():
    from crystalline.crystalio.figure_style import INK

    figure = cnm.plot_cnm(_run(), modes=(4,), colours={})        # mode 4 alone
    texts = {t.get_text(): t.get_color() for t in figure.axes[0].texts}
    assert texts["60"] == "white"                            # on the deep cyan
    assert texts["39"] == INK                                # on the orange, light enough
    figure = cnm.plot_cnm(_run(), modes=(4,), colours={"1 NO": "#eaba8a"})
    assert {t.get_text(): t.get_color() for t in figure.axes[0].texts}["60"] == INK


def test_in_the_dark_window_a_number_keeps_the_colour_that_reads_on_its_bar():
    """The window draws a figure in its own colours, swapping the paper's ink
    and white (figure_style.to_screen): taken for those, the numbers turned
    light on the light bars and dark on the dark ones."""
    from crystalline.crystalio import figure_style
    from crystalline.ui.theme import DARK

    figure = cnm.plot_cnm(_run(), modes=(4,), colours={})
    drawn = {t.get_text(): t.get_color() for t in figure.axes[0].texts}
    assert {"white", figure_style.INK} <= set(drawn.values())          # both kinds there
    figure_style.to_screen(figure, background=DARK.scene, ink=DARK.text, rule=DARK.border)
    assert {t.get_text(): t.get_color() for t in figure.axes[0].texts} == drawn


def test_moving_the_threshold_keeps_every_group_in_its_colour():
    from matplotlib.colors import to_hex

    memory = {}
    cnm.plot_cnm(_run(), modes=(4, 5, 6), colours=memory)
    before = dict(memory)
    figure = cnm.plot_cnm(_run(), modes=(4, 5, 6), threshold=35.0, colours=memory)
    faces = {c.get_label(): to_hex(c.patches[0].get_facecolor())
             for c in figure.axes[0].containers if c.get_label() != cnm.RESIDUES}
    assert faces and all(before[group] == colour for group, colour in faces.items())


def test_a_colour_chosen_for_a_group_is_drawn_and_kept():
    """Chosen by name, the residues too; and once chosen it stays, through a
    draw that does not ask again, as the plot's memory holds it."""
    from matplotlib.colors import to_hex

    memory = {}
    chosen = {"2 NO": "#123456", cnm.RESIDUES: "#abcdef"}
    figure = cnm.plot_cnm(_run(), modes=(4, 5, 6), group_colours=chosen, colours=memory)
    assert dict(cnm.drawn_groups(figure))["2 NO"] == "#123456"
    assert dict(cnm.drawn_groups(figure))[cnm.RESIDUES] == "#abcdef"
    legend = {t.get_text(): to_hex(h.get_facecolor()) for t, h in
              zip(figure.axes[0].get_legend().get_texts(),
                  figure.axes[0].get_legend().legend_handles)}
    assert legend["2 NO"] == "#123456"                       # the legend as the bars
    again = cnm.plot_cnm(_run(), modes=(4, 5, 6), threshold=0.4, colours=memory)
    assert dict(cnm.drawn_groups(again))["2 NO"] == "#123456"


def test_the_groups_drawn_are_listed_in_the_legend_s_order_residues_last():
    figure = cnm.plot_cnm(_run(), modes=(4, 5, 6), colours={})
    assert [g for g, _ in cnm.drawn_groups(figure)] == ["1 NO", "2 NO", "9 CH", cnm.RESIDUES]


def test_without_a_legend_there_is_none():
    figure = cnm.plot_cnm(_run(), modes=(4, 5, 6), legend=False)
    assert figure.axes[0].get_legend() is None


def test_no_modes_too_many_or_ones_the_run_has_not_are_not_drawn():
    with pytest.raises(ValueError, match="no normal modes"):
        cnm.plot_cnm(_run(), modes=())
    with pytest.raises(ValueError, match="not characterised in this run: mode 2, 99"):
        cnm.plot_cnm(_run(), modes=(4, 2, 99))
    many = tuple(100.0 + i for i in range(cnm.MAX_MODES + 1))
    run = _run(groups=("1 NO",), frequencies=many, table=[[100.0]] * len(many))
    with pytest.raises(ValueError, match="more than"):
        cnm.plot_cnm(run, modes=run.modes)


def test_the_bars_are_labelled_with_frequencies_or_mode_numbers():
    figure = cnm.plot_cnm(_run(), modes=(4, 5, 6), ticks="freq")
    assert [t.get_text() for t in figure.axes[0].get_xticklabels()] == ["100.0", "205.0", "310.0"]
    figure = cnm.plot_cnm(_run(), modes=(4, 5, 6), ticks="mode")
    assert [t.get_text() for t in figure.axes[0].get_xticklabels()] == ["4", "5", "6"]


# ── the controls ──────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _ticked_items(widget):
    from PySide6.QtCore import Qt

    return [widget.modes.item(i) for i in range(widget.modes.count())
            if widget.modes.item(i).checkState() == Qt.Checked]


def test_every_mode_is_listed_by_number_and_frequency(qapp):
    from crystalline.ui.panels.cnm_options import CNMOptions

    widget = CNMOptions(_run())
    assert [widget.modes.item(i).text() for i in range(widget.modes.count())] == [
        "Mode 4, 100.0 cm⁻¹", "Mode 5, 205.0 cm⁻¹", "Mode 6, 310.0 cm⁻¹",
        "Mode 7, 1500.0 cm⁻¹"]


def test_the_window_ticks_a_span_and_single_modes_are_ticked_after(qapp):
    """Filippo's checklist: a mode asked about is known by its number, and the
    ones wanted are not always next to each other."""
    from PySide6.QtCore import Qt

    from crystalline.ui.panels.cnm_options import CNMOptions

    many = tuple(100.0 + 5 * i for i in range(60))           # modes 4–63
    widget = CNMOptions(_run(groups=("1 NO",), frequencies=many, table=[[100.0]] * 60))
    checked, changed = [], []
    widget.selection_checked.connect(checked.append)
    widget.changed.connect(lambda: changed.append(True))

    widget.fmax.setValue(1000.0)                             # all 60
    assert "60 modes ticked, too many" in widget.count.text() and checked[-1] is False
    assert not changed                                       # nothing to draw
    widget.fmin.setValue(150.0)
    widget.fmax.setValue(200.0)
    assert widget.ticked() == list(range(14, 25))
    assert widget.count.text() == "11 modes ticked: 14–24" and checked[-1] is True
    assert changed and widget.groups.text() == "1 group above the threshold"

    changed.clear()
    widget.modes.item(40 - 4).setCheckState(Qt.Checked)      # mode 40, far off the span
    widget.modes.item(20 - 4).setCheckState(Qt.Unchecked)
    assert widget.ticked() == [14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 40]
    assert widget.count.text() == "11 modes ticked: 14–19, 21–24, 40"
    assert len(changed) == 2
    # The window's handles follow what is ticked, and ticked nothing by it.
    assert widget.fmin.value() <= 150.0 and widget.fmax.value() >= many[40 - 4]
    assert len(_ticked_items(widget)) == 11

    for item in _ticked_items(widget):
        item.setCheckState(Qt.Unchecked)
    assert widget.count.text() == "no modes ticked" and checked[-1] is False


def test_a_threshold_that_leaves_too_many_groups_is_said(qapp):
    from crystalline.ui.panels.cnm_options import CNMOptions

    groups = [f"{n} CH" for n in range(cnm.READABLE_GROUPS + 5)]
    table = [[100.0 / len(groups)] * len(groups)]
    widget = CNMOptions(_run(groups=groups, frequencies=(100.0,), table=table))
    assert widget.groups.text().endswith("raise it to read the legend")
    widget.threshold.setValue(2.0)                           # every group is under 1.5 %
    assert widget.groups.text() == "0 groups above the threshold"


def test_the_controls_give_back_what_they_were_set_to(qapp):
    from crystalline.ui.panels.cnm_options import CNMOptions

    widget = CNMOptions(_run())
    options = {"modes": (4, 6), "threshold": 2.5, "ticks": "mode", "legend": False,
               "group_colours": {"1 NO": "#ff0000"}}
    announced = []
    widget.changed.connect(lambda: announced.append(True))
    widget.set_options(options)
    assert widget.options() == options and not announced


def test_each_group_drawn_gets_a_colour_dropdown(qapp, monkeypatch):
    """Filippo's request: the colour of every group, chosen from a dropdown
    beside it — the house colours by name, or any other through Other…"""
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QColorDialog

    from crystalline.ui.panels.cnm_options import CNMOptions

    widget = CNMOptions(_run())
    assert widget._colour_section.header.isHidden()          # nothing drawn yet
    drawn = [("1 NO", "#0b7a96"), ("2 NO", "#d9822b"), (cnm.RESIDUES, cnm.RESIDUES_COLOUR)]
    widget.figure_drawn(SimpleNamespace(axes=[]))            # an empty figure: still none
    monkeypatch.setattr(cnm_options_module(), "drawn_groups", lambda _figure: drawn)
    widget.figure_drawn(object())
    assert not widget._colour_section.header.isHidden()
    rows = widget.colours.groups
    assert [rows.topLevelItem(i).text(0) for i in range(rows.topLevelItemCount())] == [
        "1 NO", "2 NO", cnm.RESIDUES]

    changed = []
    widget.changed.connect(lambda: changed.append(True))
    box = widget.colours.box("2 NO")
    assert box.currentText() == "Orange"
    box.activated.emit(box.findText("Dark navy"))
    assert widget.options()["group_colours"] == {"2 NO": box.colour()} and changed
    assert box.colour() == dict(cnm.named_palette())["Dark navy"]

    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor("#ff00ff")))
    residues = widget.colours.box(cnm.RESIDUES)
    assert residues.currentText() == "Pale grey"
    residues.activated.emit(residues.findText("Other…"))
    assert widget.options()["group_colours"][cnm.RESIDUES] == "#ff00ff"
    assert residues.currentText() == "#ff00ff"               # listed, and shown


def cnm_options_module():
    from crystalline.ui.panels import cnm_options

    return cnm_options


def test_a_series_is_followed_from_a_mode_of_its_first_run(qapp):
    from crystalline.ui.panels.cnm_options import CNMSeriesOptions

    widget = CNMSeriesOptions(_run())
    assert [widget.mode.itemText(i) for i in range(widget.mode.count())][:2] == [
        "Mode 4, 100.0 cm⁻¹", "Mode 5, 205.0 cm⁻¹"]
    widget.set_options({"mode": 6, "threshold": 3.0, "min_overlap": 0.7, "legend": False,
                        "frequencies": False, "group_colours": {"1 NO (0.2)": "#00ff00"}})
    assert widget.options() == {"mode": 6, "threshold": 3.0, "min_overlap": 0.7,
                                "legend": False, "frequencies": False,
                                "group_colours": {"1 NO (0.2)": "#00ff00"}}


def test_each_bar_of_a_series_is_labelled_beside_the_plot(qapp):
    """Filippo: "the labels for each of the column". A field per run, named by
    what tells its file apart, taken when left; a label empty or used twice is
    said, and nothing is drawn with it."""
    from PySide6.QtWidgets import QFormLayout

    from crystalline.ui.panels.cnm_options import CNMSeriesOptions

    widget = CNMSeriesOptions(_run())
    paths = ("/runs/p_tight.out", "/runs/p_02GPa.out", "/runs/p_1GPa.out")
    widget.set_options({"paths": paths, "labels": ("tight", "02GPa", "1GPa"), "mode": 4})
    rows = widget._label_form
    names = [rows.itemAt(i, QFormLayout.LabelRole).widget().text() for i in range(rows.rowCount())]
    assert names == ["tight", "02GPa", "1GPa"]
    assert [e.text() for e in widget.label_edits] == ["tight", "02GPa", "1GPa"]
    assert widget.options()["labels"] == ("tight", "02GPa", "1GPa")

    changed = []
    widget.changed.connect(lambda: changed.append(True))
    for edit, label in zip(widget.label_edits, ("0", "0.2", "1")):
        edit.setText(label)
        edit.editingFinished.emit()
    assert widget.options()["labels"] == ("0", "0.2", "1") and changed

    changed.clear()
    widget.label_edits[2].setText("0.2")                    # twice
    widget.label_edits[2].editingFinished.emit()
    assert not changed and widget.label_problem.text() == "Give each run a label of its own."
    assert not widget.label_problem.isHidden()
    assert widget.options()["labels"] == ("0", "0.2", "1")  # what was drawn stands
    widget.label_edits[2].setText("")
    widget.label_edits[2].editingFinished.emit()
    assert not changed and widget.label_problem.text() == "Give every run a label."
    widget.label_edits[2].setText("1.0")
    widget.label_edits[2].editingFinished.emit()
    assert widget.options()["labels"] == ("0", "0.2", "1.0") and changed
    assert widget.label_problem.isHidden()

    changed.clear()
    widget.frequencies.setChecked(False)
    assert widget.options()["frequencies"] is False and changed


def test_the_dialog_opens_on_the_mode_selected(qapp):
    from crystalline.ui.panels.cnm_dialog import CNMDialog

    frequencies = tuple(100.0 + 10 * i for i in range(30))
    dialog = CNMDialog(_run(groups=("1 NO",), frequencies=frequencies, table=[[100.0]] * 30))
    dialog.fit_selection(around=None)
    assert dialog.options_widget.acceptable()
    dialog.fit_selection(around=30)                          # not ticked: ticked round it
    assert 30 in dialog.options_widget.ticked()
    assert len(dialog.options_widget.ticked()) == cnm.DEFAULT_WINDOW_MODES


def test_the_dialog_keeps_the_modes_ticked(qapp):
    """Through dialog_state, as the plot is edited or the next one opened: the
    modes ticked, not the window, which would tick its span over them."""
    from crystalline.ui.panels import dialog_state
    from crystalline.ui.panels.cnm_dialog import CNMDialog

    dialog = CNMDialog(_run())
    dialog.set_selection_state([4, 7])
    dialog.threshold.setValue(3.0)
    state = dialog_state.capture(dialog)
    assert "fmin" not in state and "fmax" not in state

    again = CNMDialog(_run())
    dialog_state.restore(again, state)
    assert again.options() == {"modes": (4, 7), "threshold": 3.0, "ticks": "freq",
                               "legend": True, "group_colours": {}}
    again.fit_selection(around=None)                         # it suits: kept
    assert again.options_widget.ticked() == [4, 7]


@pytest.mark.parametrize("modes, name", [
    ((95, 96, 97, 98, 150), "CNM, modes 95–98, 150"),
    ((100,), "CNM, mode 100"),
    (tuple(range(4, 40, 2)), "CNM, 18 modes, 4–38"),          # too long to spell out
])
def test_a_tab_is_named_after_its_modes(modes, name):
    from crystalline.ui.panels.cnm_options import title

    assert title(_run(), {"modes": modes}) == name


def _series_dialog(paths):
    from crystalline.ui.panels.cnm_dialog import CNMSeriesDialog

    return CNMSeriesDialog(paths)


def test_a_series_is_labelled_ordered_and_checked(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialogButtonBox

    paths = ["/runs/p_02GPa.out", "/runs/p_04GPa.out", "/runs/p_1GPa.out"]
    dialog = _series_dialog(paths)
    ok = dialog._buttons.button(QDialogButtonBox.Ok)
    assert dialog.chosen() == [(paths[0], "02GPa"), (paths[1], "04GPa"), (paths[2], "1GPa")]
    assert ok.isEnabled()

    dialog.runs.setCurrentItem(dialog.runs.topLevelItem(2))
    dialog._move(-1)                                         # 1 GPa before 0.4
    assert [p for p, _ in dialog.chosen()] == [paths[0], paths[2], paths[1]]

    dialog.runs.topLevelItem(0).setText(1, "1GPa")           # the same label twice
    assert not ok.isEnabled() and "of its own" in dialog.problem()
    dialog.runs.topLevelItem(0).setText(1, "0.2")
    dialog.runs.topLevelItem(1).setCheckState(0, Qt.Unchecked)
    dialog.runs.topLevelItem(2).setCheckState(0, Qt.Unchecked)
    assert not ok.isEnabled() and dialog.problem() == "Tick two runs or more."


def test_a_series_drawn_before_is_shown_as_it_was(qapp, tmp_path):
    files = []
    for name in ("p_02GPa.out", "p_04GPa.out", "p_1GPa.out"):
        (tmp_path / name).write_text("")
        files.append(str(tmp_path / name))
    dialog = _series_dialog(files)
    dialog.set_selection_state({"paths": (files[2], files[0]), "labels": ("1", "0.2"),
                                "axis": "Pressure (kbar)"})
    assert dialog.chosen() == [(files[2], "1"), (files[0], "0.2")]
    assert dialog.options()["axis"] == "Pressure (kbar)"
    assert dialog.selection_state() == {"paths": (files[2], files[0]), "labels": ("1", "0.2"),
                                        "axis": "Pressure (kbar)"}
    # A run whose file has gone is left out.
    dialog.set_selection_state({"paths": ("/gone/x.out", files[1]), "labels": ("0", "0.4"),
                                "axis": "Pressure (GPa)"})
    assert dialog.chosen() == [(files[1], "0.4")]


# ── the window ────────────────────────────────────────────────────────────
@pytest.fixture
def window(qapp, monkeypatch, tmp_path):
    from _window import close_window, make_window

    made = make_window(monkeypatch, tmp_path)
    yield made
    close_window(made)


def test_the_entry_is_offered_for_an_output_with_local_modes(window, monkeypatch):
    """Tab by tab: the synthetic H₂O₂ output has no geometry to open, so the
    second sample stands in for a LOCMODES run, found as the probe finds one."""
    import sample_data as sample
    from _window import open_file, show_tab

    from crystalline import crystalio

    plain = sample.path("thiourea/thiourea_pbed3_Ahlrichs-pVTZ_freqcalc_80K.out")
    run = sample.path("coesite/coesite_ela.out")
    assert window._cnm_action is not None and window._cnm_series_action.isEnabled()
    first = open_file(window, plain)
    assert not window._cnm_action.isEnabled()
    monkeypatch.setattr(crystalio, "has_local_modes", lambda path: path == run)
    monkeypatch.setattr(crystalio, "read_local_modes", lambda path: None)
    open_file(window, run)
    assert window._cnm_action.isEnabled()
    show_tab(window, first)
    assert not window._cnm_action.isEnabled() and window._cnm_series_action.isEnabled()


def test_a_bar_clicked_selects_its_mode(window, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from crystalline.ui.panels.cnm_dialog import CNMDialog

    monkeypatch.setattr(CNMDialog, "exec", lambda self: QDialog.Accepted)
    picked = []
    monkeypatch.setattr(window, "_select_mode_near", picked.append)
    window._show_cnm(_run())

    tab = window.plot_panel.current_tab()
    assert tab is not None and tab.recipe.title(tab.recipe.options).startswith("CNM, mode")
    drawn = sorted(tab.recipe.options["modes"])
    tab._on_pick(1.1)                                        # the second bar
    assert picked == [_run().frequency(drawn[1])]



def test_a_colour_chosen_beside_the_plot_redraws_it(window, monkeypatch, qtbot):
    from matplotlib.colors import to_hex
    from PySide6.QtWidgets import QDialog

    from crystalline.ui.panels.cnm_dialog import CNMDialog

    monkeypatch.setattr(CNMDialog, "exec", lambda self: QDialog.Accepted)
    window._show_cnm(_run())
    tab = window.plot_panel.current_tab()
    box = tab.controls.colours.box("1 NO")
    assert box is not None and box.currentText() == "Cyan"
    figure = tab.figure
    box.activated.emit(box.findText("Red"))
    qtbot.waitUntil(lambda: tab.figure is not figure, timeout=10000)

    bars = {c.get_label(): to_hex(c.patches[0].get_facecolor()) for c in tab.figure.axes[0].containers}
    assert bars["1 NO"] == dict(cnm.named_palette())["Red"]
    assert tab.controls.colours.box("1 NO").currentText() == "Red"   # the list follows
