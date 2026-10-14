"""Elastic plots: one tensor, read once, and every view of it beside the plot."""

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from PySide6.QtWidgets import QApplication  # noqa: E402

from crystalline.crystalio import elastic as el  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _isotropic(E=200.0, nu=0.25):
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] = lam + 2 * mu
    C[np.arange(3, 6), np.arange(3, 6)] = mu
    return C


def test_a_menu_entry_opens_on_its_own_property_and_view():
    assert el.options_for("ela_young") == dict(el.DEFAULTS, quantity="young", view="surface")
    assert el.options_for("ela2d_poisson")["view"] == "sections"
    assert el.title(el.options_for("ela_young")) == "Young's modulus"
    assert el.title(el.options_for("ela2d_shear")) == "Shear modulus, average (sections)"


def test_only_a_property_with_a_second_direction_takes_a_statistic():
    assert el.choose_key("young", "max") == "young"
    assert el.choose_key("shear", "min") == "shear min"
    assert el.choose_key("poisson", "avg") == "poisson avg"


def test_every_view_of_one_tensor_is_drawn_from_memory():
    pytest.importorskip("CRYSTALClear")
    C = _isotropic()

    surface = el.plot_elastic(C, quantity="comp", resolution=50, cmap="cividis")
    assert surface.axes[0].name == "3d"
    assert "Linear compressibility" in surface.axes[1].get_ylabel()

    sections = el.plot_elastic(C, quantity="young", view="sections", planes=("xz",),
                               resolution=50)
    polar = sections.axes[0]
    assert polar.name == "polar"
    assert {line.get_label() for line in polar.get_lines()} == {"xz"}


# ── the controls ──────────────────────────────────────────────────────────
def _controls():
    from crystalline.ui.panels.elastic_controls import ElasticControls

    knobs = ElasticControls()
    moves = []
    knobs.changed.connect(lambda: moves.append(knobs.options()))
    return knobs, moves


def test_a_plots_options_come_back_quietly(qapp):
    knobs, moves = _controls()
    wanted = dict(el.DEFAULTS, quantity="shear", statistic="max", view="sections",
                  planes=("xy", "yz"), cmap="plasma", resolution=200)

    knobs.set_options(wanted)

    assert moves == [] and knobs.options() == wanted


def test_only_what_applies_is_offered(qapp):
    knobs, _moves = _controls()
    knobs.set_options(dict(el.DEFAULTS, quantity="young", view="surface"))
    assert knobs.statistic.isHidden() and knobs.planes["xy"].parent().isHidden()
    assert not knobs.cmap.isHidden()

    knobs.set_options(dict(el.DEFAULTS, quantity="poisson", view="sections"))
    assert not knobs.statistic.isHidden() and not knobs.planes["xy"].parent().isHidden()
    assert knobs.cmap.isHidden()                       # sections are drawn in lines


def test_turning_a_control_redraws_and_the_last_plane_stays(qapp):
    knobs, moves = _controls()
    knobs.set_options(dict(el.DEFAULTS, view="sections"))

    knobs.quantity.setCurrentIndex(knobs.quantity.findData("comp"))
    assert moves and moves[-1]["quantity"] == "comp"

    knobs.planes["xy"].setChecked(False)
    knobs.planes["xz"].setChecked(False)
    knobs.planes["yz"].setChecked(False)               # the last one: refused
    assert knobs.options()["planes"] == ("yz",)


def test_the_elastic_entries_open_with_their_controls():
    import inspect

    from crystalline.ui.main_window import MainWindow

    assert "_open_elastic(kind, path)" in inspect.getsource(MainWindow._open_plot)
    assert "controls=ElasticControls" in inspect.getsource(MainWindow._open_elastic)
