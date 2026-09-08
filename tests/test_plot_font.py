"""The plot font setting.

Figures go into papers, so the family and size have to be settable — Computer
Modern in particular, to match a LaTeX document. Two things need pinning: that
each named choice sets the maths font along with the text font (mixed typography
in one figure is the failure this prevents), and that switching *away* from
Computer Modern undoes its Unicode-minus workaround rather than leaving it on
every later plot.
"""

import os

import pytest

matplotlib = pytest.importorskip("matplotlib")

from crystalline.crystalio.plotting import (  # noqa: E402
    DEFAULT_FONT_FAMILY,
    DEFAULT_FONT_SIZE,
    FONT_FAMILIES,
    apply_font,
    installed_font_names,
)


@pytest.fixture(autouse=True)
def restore_rcparams():
    """Every test here mutates global rcParams; put them back afterwards."""
    saved = matplotlib.rcParams.copy()
    yield
    matplotlib.rcParams.update(saved)


def test_every_named_family_applies_and_sets_the_maths_font_too():
    """A label like $\\Phi^\\mathbf{n}$ has to be typeset in the same face as the
    text around it, so no recipe may leave mathtext at the default."""
    for _label, key in FONT_FAMILIES:
        applied = apply_font(family=key, size=11)

        assert "font.family" in applied
        assert "mathtext.fontset" in applied
        assert matplotlib.rcParams["font.size"] == 11


def test_computer_modern_asks_for_the_cm_maths_set():
    applied = apply_font(family="cm")

    assert applied["mathtext.fontset"] == "cm"
    assert matplotlib.rcParams["font.family"] == ["serif"]
    # cmr10 ships with matplotlib; CMU Serif is the fuller cut when installed
    assert "cmr10" in matplotlib.rcParams["font.serif"]


def test_leaving_computer_modern_undoes_its_minus_sign_workaround():
    """cmr10 has no U+2212, so 'cm' turns unicode_minus off. Carrying that into
    every later figure would silently change how negatives are drawn."""
    apply_font(family="cm")
    assert matplotlib.rcParams["axes.unicode_minus"] is False

    apply_font(family="default")

    assert matplotlib.rcParams["axes.unicode_minus"] is True
    assert matplotlib.rcParams["mathtext.fontset"] == "dejavusans"


def test_an_installed_font_can_be_named_directly():
    """The named recipes cover the common cases; anything else installed is
    passed to matplotlib as-is."""
    applied = apply_font(family="Times New Roman", size=9)

    assert applied["font.family"] == "Times New Roman"
    assert matplotlib.rcParams["font.size"] == 9


def test_the_size_alone_can_be_changed():
    apply_font(family="serif", size=14)
    assert matplotlib.rcParams["font.size"] == 14

    apply_font(family="serif", size=8)
    assert matplotlib.rcParams["font.size"] == 8


def test_the_defaults_are_applicable():
    applied = apply_font(DEFAULT_FONT_FAMILY, DEFAULT_FONT_SIZE)

    assert applied["font.size"] == DEFAULT_FONT_SIZE


def test_installed_fonts_are_listed_without_duplicates():
    names = installed_font_names()

    assert names == sorted(set(names))
    assert "DejaVu Sans" in names  # matplotlib always ships it


def test_a_figure_drawn_after_the_change_uses_the_new_font():
    """rcParams are read when a figure is created, which is the whole reason the
    dialog says it applies to plots opened from now on."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def title_size(size):
        apply_font(family="cm", size=size)
        figure, axes = plt.subplots()
        axes.set_title("test")
        try:
            assert axes.title.get_fontfamily() == ["serif"]
            return axes.title.get_fontsize()
        finally:
            plt.close(figure)

    # Titles are sized relative to font.size ('large' is 1.2x it), so what has
    # to hold is that they scale with it, not that they equal it.
    assert title_size(26) == pytest.approx(2 * title_size(13))


# ── the dialog ──────────────────────────────────────────────────────────
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from crystalline.ui.panels.font_dialog import PlotFontDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_the_dialog_returns_apply_font_keyword_arguments(qapp):
    import inspect

    dialog = PlotFontDialog()
    accepted = set(inspect.signature(apply_font).parameters)

    assert set(dialog.options()) <= accepted


def test_the_dialog_opens_on_the_font_it_was_given(qapp):
    dialog = PlotFontDialog("cm", 12.5)

    assert dialog.options() == {"family": "cm", "size": 12.5}


def test_a_font_outside_the_named_list_round_trips(qapp):
    """Reopening the dialog has to show the current choice, including one typed
    into the "other" box."""
    dialog = PlotFontDialog("Times New Roman", 10.0)

    assert dialog.options()["family"] == "Times New Roman"


def test_the_installed_font_box_shows_only_for_the_other_choice(qapp):
    dialog = PlotFontDialog("cm", 10.0)
    assert not dialog.installed.isVisibleTo(dialog)

    dialog = PlotFontDialog("Times New Roman", 10.0)
    assert dialog.installed.isVisibleTo(dialog)


def test_an_empty_other_box_falls_back_rather_than_setting_nothing(qapp):
    dialog = PlotFontDialog("Times New Roman", 10.0)
    dialog.installed.setCurrentText("   ")

    assert dialog.options()["family"] == DEFAULT_FONT_FAMILY
