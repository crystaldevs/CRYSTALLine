"""The application's visual style: it builds, it applies, and it stays coherent."""

import os
import re
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from crystalline.resources import asset_path  # noqa: E402
from crystalline.ui import theme  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("palette", [theme.LIGHT, theme.DARK])
def test_the_sheet_builds_and_leaves_no_placeholder_behind(palette):
    """An f-string that lost a brace shows up as a literal ``{p.`` in the output
    and silently drops the rule it was in."""
    sheet = theme.stylesheet(palette)

    assert "{p." not in sheet and "{_" not in sheet
    assert sheet.count("{") == sheet.count("}")


@pytest.mark.parametrize("palette", [theme.LIGHT, theme.DARK])
def test_nothing_styles_a_bare_widget(palette):
    """A rule on ``QWidget`` inherits into every child, the VTK render window
    included — painting a background behind an OpenGL surface is wasted at best
    and a flicker at worst. Every rule has to name the class it means."""
    sheet = theme.stylesheet(palette)

    selectors = re.findall(r"^\s*([A-Za-z][^{]*?)\s*\{", sheet, re.M)
    bare = [s for s in selectors if re.fullmatch(r"QWidget", s.strip())]
    assert bare == []


def test_the_glyphs_the_sheet_points_at_actually_exist():
    """Qt's ``url()`` goes through the file system, not data URIs — a missing
    file is not an error, the indicator just comes out blank, so a ticked box
    would be a plain coloured square."""
    for name in ("check.svg", "radio.svg", "mixed.svg"):
        path = asset_path(name)
        assert os.path.isfile(path), f"{name} is referenced by the theme but missing"

    sheet = theme.stylesheet(theme.LIGHT)
    for name in ("check.svg", "radio.svg", "mixed.svg"):
        assert asset_path(name) in sheet


def test_both_variants_keep_text_readable_against_their_ground():
    """Muted text is the one most likely to fall below contrast, and it carries
    every group title and hint in the app."""

    def contrast(a, b):
        def channel(value):
            value /= 255.0
            return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

        def luminance(colour):
            c = QColor(colour)
            r, g, bl = (channel(v) for v in (c.red(), c.green(), c.blue()))
            return 0.2126 * r + 0.7152 * g + 0.0722 * bl

        first, second = luminance(a), luminance(b)
        light, dark = max(first, second), min(first, second)
        return (light + 0.05) / (dark + 0.05)

    for palette in (theme.LIGHT, theme.DARK):
        assert contrast(palette.text, palette.window) > 7.0      # body text
        assert contrast(palette.text_muted, palette.window) > 3.5  # titles and hints
        assert contrast("#ffffff", palette.accent) > 3.0         # text on a selection


def test_applying_the_theme_sets_both_the_palette_and_the_sheet(qapp):
    """They have to come from one table: a dark sheet over a light palette gives
    dark cards on a light ground, which is worse than either alone."""
    before = qapp.styleSheet()
    try:
        theme._paint(qapp, theme.DARK)

        assert qapp.styleSheet() == theme.stylesheet(theme.DARK)
        window = qapp.palette().color(qapp.palette().ColorRole.Window)
        assert window.name() == QColor(theme.DARK.window).name()
    finally:
        qapp.setStyleSheet(before)


def test_the_variant_follows_the_desktop_while_the_choice_is_to_follow_it(qapp, monkeypatch):
    """Only in "system" mode. Asserted with the mode passed explicitly, because
    the stored choice is a real user preference on this machine — a test that
    read it would pass or fail depending on what the app was last set to."""
    monkeypatch.setattr(theme, "is_dark", lambda _app: True)
    assert theme.palette_for(qapp, "system") is theme.DARK
    monkeypatch.setattr(theme, "is_dark", lambda _app: False)
    assert theme.palette_for(qapp, "system") is theme.LIGHT


# ── the appearance choice ─────────────────────────────────────────────────
def test_the_three_modes_pick_the_right_table(qapp, monkeypatch):
    """"System" follows the desktop; the other two override it, which is the
    point — a dark desktop and a light figure is a normal pairing for a viewer."""
    monkeypatch.setattr(theme, "is_dark", lambda _app: True)

    assert theme.palette_for(qapp, "light") is theme.LIGHT   # overrides a dark desktop
    assert theme.palette_for(qapp, "dark") is theme.DARK
    assert theme.palette_for(qapp, "system") is theme.DARK

    monkeypatch.setattr(theme, "is_dark", lambda _app: False)
    assert theme.palette_for(qapp, "dark") is theme.DARK     # overrides a light one
    assert theme.palette_for(qapp, "system") is theme.LIGHT


def test_an_unknown_mode_is_refused_rather_than_guessed(qapp):
    with pytest.raises(ValueError):
        theme.set_mode(qapp, "sepia")


def test_the_choice_survives_a_restart(qapp):
    """It is a preference, not a session setting."""
    before = theme.current_mode()
    try:
        theme.set_mode(qapp, "dark")
        assert theme.current_mode() == "dark"
        theme.set_mode(qapp, "light")
        assert theme.current_mode() == "light"
    finally:
        theme.set_mode(qapp, before)


def test_unreadable_settings_fall_back_to_following_the_system(qapp, monkeypatch):
    def broken():
        raise OSError("no settings store")

    monkeypatch.setattr(theme, "_settings", broken)
    assert theme.current_mode() == theme.DEFAULT_MODE


def test_a_theme_change_redraws_what_baked_a_colour_in(qapp):
    """The undo/redo glyphs are rendered once in the text colour, so a switch has
    to redraw them or they stay dark on a dark toolbar."""
    from PySide6.QtWidgets import QWidget

    refreshed = []

    class _Window(QWidget):
        def refresh_theme(self):
            refreshed.append(True)

    window = _Window()
    window.show()
    try:
        theme._paint(qapp, theme.DARK)
        assert refreshed, "windows were not asked to refresh after a theme change"
    finally:
        window.close()
        theme._paint(qapp, theme.palette_for(qapp))


def test_the_glyphs_are_drawn_in_the_colour_asked_for(qapp):
    """One file serves both themes — a glyph baked dark would vanish into the
    dark toolbar."""
    light = theme.monochrome_icon("undo.svg", "#ffffff")
    dark = theme.monochrome_icon("undo.svg", "#000000")

    assert not light.isNull() and not dark.isNull()
    a = light.pixmap(18, 18).toImage()
    b = dark.pixmap(18, 18).toImage()
    assert a != b, "the colour substitution did nothing"


def test_the_chips_are_styled_by_the_theme_not_by_themselves(qapp):
    """A chip that painted itself would keep its light colours on a dark toolbar,
    so the rules live in the sheet and key off a property."""
    for palette in (theme.LIGHT, theme.DARK):
        sheet = theme.stylesheet(palette)
        for kind in ("axis", "ghost", "toggle"):
            assert f'[chip="{kind}"]' in sheet
        for axis in ("a", "b", "c"):
            assert f'[axis="{axis}"]' in sheet


def test_dock_tabs_and_widget_tabs_are_styled_apart(qapp):
    """Dock tabs sit *below* their dock and a QTabWidget's sit *above* their
    pane; one style for both comes out upside down on the docks, which is what
    made them look broken."""
    sheet = theme.stylesheet(theme.LIGHT)

    assert "QMainWindow > QTabBar::tab" in sheet
    assert "QTabWidget > QTabBar::tab" in sheet


# ── the sliding switch ────────────────────────────────────────────────────
def test_a_blocked_set_still_moves_the_knob(qapp):
    """Regression: a signal-blocked ``setChecked`` — how a view syncs a control
    to the state it displays — left the knob at the wrong end while the switch
    reported itself on.

    The guard read ``not self._slide.state()``, and a PySide6 enum is an object:
    that is False even for ``Stopped``, so the placement never ran.
    """
    from crystalline.ui.widgets import ToggleSwitch

    switch = ToggleSwitch()
    blocked = switch.blockSignals(True)
    switch.setChecked(True)
    switch.blockSignals(blocked)

    assert switch.isChecked()
    assert switch.offset == 1.0, "the knob did not follow the state"

    blocked = switch.blockSignals(True)
    switch.setChecked(False)
    switch.blockSignals(blocked)
    assert switch.offset == 0.0


def test_a_switch_does_not_animate_into_its_initial_state(qapp):
    """Built on, it should *be* on — not slide there. Otherwise the first paint
    catches the knob mid-travel and the control draws as off."""
    from crystalline.ui.widgets import ToggleSwitch

    switch = ToggleSwitch()
    switch.setChecked(True)  # never shown, so nothing to animate into

    assert switch.offset == 1.0


def test_the_switch_takes_its_colours_from_the_palette(qapp):
    """Painted, not styled — so it has to read the palette or it would keep its
    light colours on a dark toolbar.

    The event pump is not incidental. Qt hands an application palette change to
    widgets as a posted ``ApplicationPaletteChange``, so a widget read in the
    same breath as ``setPalette`` still holds the *previous* palette. PySide6
    6.9 happened to resolve it lazily and answered correctly without one; 6.11
    does not, and the test failed there for a reason that was never about the
    widget. The app returns to its event loop after every theme change, so it
    only ever sees the settled value.
    """
    from crystalline.ui.widgets import ToggleSwitch

    switch = ToggleSwitch()
    theme._paint(qapp, theme.DARK)
    qapp.processEvents()
    dark_highlight = switch.palette().highlight().color().name()
    theme._paint(qapp, theme.LIGHT)
    qapp.processEvents()
    light_highlight = switch.palette().highlight().color().name()

    assert dark_highlight == QColor(theme.DARK.accent).name()
    assert light_highlight == QColor(theme.LIGHT.accent).name()


def test_the_toolbar_chip_colours_match_the_3d_gizmo():
    """The chips claim to be the same axes as the arrows in the view. They are
    copied rather than imported (so styling the app does not drag VTK in at
    startup), which is exactly the sort of copy that drifts."""
    pytest.importorskip("pyvista")
    from crystalline.viz import renderer

    assert (theme._AXIS_A, theme._AXIS_B, theme._AXIS_C) == tuple(renderer._LATTICE_COLORS)


# ── the lamp, and the 3D ground ───────────────────────────────────────────
def test_the_lamp_is_on_for_the_light_theme(qapp):
    """A lamp that lit up for the *dark* theme would be backwards however it is
    explained."""
    from PySide6.QtWidgets import QMainWindow

    from crystalline.ui import menus

    window = QMainWindow()
    window._appearance_lamp = None
    window._appearance_switch = __import__(
        "crystalline.ui.widgets", fromlist=["ToggleSwitch"]
    ).ToggleSwitch()

    theme._paint(qapp, theme.LIGHT)
    menus.refresh_appearance_button(window)
    assert window._appearance_switch.isChecked(), "light theme: the lamp should be on"

    theme._paint(qapp, theme.DARK)
    menus.refresh_appearance_button(window)
    assert not window._appearance_switch.isChecked(), "dark theme: the lamp should be off"

    theme._paint(qapp, theme.palette_for(qapp))


def test_the_3d_ground_follows_the_theme_but_not_over_a_chosen_colour(qapp):
    """A dark chrome around a white viewport reads as a bug. But the background
    is a real setting, so a colour someone picked has to survive a theme change.
    """
    from crystalline.ui.main_window import MainWindow
    from crystalline.ui.panels.display_settings import DisplayPanel
    from crystalline.viz.render_settings import RenderSettings

    class _Stub:
        _follow_theme_background = MainWindow._follow_theme_background

    window = _Stub()
    window.display_panel = DisplayPanel(RenderSettings(), lambda s: None)

    # the default is spelled "white", a theme's is "#ffffff" — the same colour,
    # and comparing the text would treat a fresh app as already customised
    assert window.display_panel.background() == "white"

    theme._paint(qapp, theme.DARK)
    window._follow_theme_background()
    assert window.display_panel.background() == theme.DARK.scene

    theme._paint(qapp, theme.LIGHT)
    window._follow_theme_background()
    assert window.display_panel.background() == theme.LIGHT.scene

    window.display_panel.set_background("#204020")   # deliberately chosen
    theme._paint(qapp, theme.DARK)
    window._follow_theme_background()
    assert window.display_panel.background() == "#204020"

    theme._paint(qapp, theme.palette_for(qapp))


def test_the_window_takes_the_theme_ground_when_it_is_built():
    """The regression: launching in dark mode gave a white viewport.

    The theme is painted on the application *before* the window exists, so the
    theme-change signal that normally carries the 3D ground along with it has
    already fired by the time there is a viewport to carry it to. Construction
    has to ask for the ground itself.

    Checked on ``__init__``'s own names rather than by building a window: a real
    ``MainWindow`` needs a VTK render window, which is not available here.
    """
    from crystalline.ui.main_window import MainWindow

    assert "_follow_theme_background" in MainWindow.__init__.__code__.co_names


def test_the_dark_ground_is_not_pure_black():
    """A structure lit against #000 loses its darkest faces entirely, and the
    cell wireframe with them."""
    assert theme.DARK.scene != "#000000"
    assert QColor(theme.DARK.scene).lightness() < 60   # still unmistakably dark


# ── the chrome the layout depends on ──────────────────────────────────────
def test_the_default_appearance_is_light(monkeypatch):
    """Dark is a click away — the toolbar lamp, or View ▸ Appearance."""
    def unreadable():
        raise OSError("no settings store")

    monkeypatch.setattr(theme, "_settings", unreadable)
    assert theme.current_mode() == "light"
    assert theme.DEFAULT_MODE == "light"


def test_a_logarithmic_slider_puts_a_multiplicative_default_in_the_middle(qapp):
    """A 0.1-10x speed on a linear scale puts 1x nine percent along, crushing the
    whole useful range against the left stop."""
    from PySide6.QtWidgets import QSlider, QVBoxLayout, QWidget

    from crystalline.ui.panels.controls import Section, slider_row

    host = QWidget()
    section = Section(QVBoxLayout(host), "Playback")
    slider_row(section, "Speed", 1.0, 0.1, 10.0, 0.1, decimals=1, logarithmic=True)
    slider_row(section, "Linear", 1.0, 0.1, 10.0, 0.1, decimals=1)

    log_slider, linear_slider = host.findChildren(QSlider)
    assert log_slider.value() == pytest.approx(500, abs=2)     # dead centre
    assert linear_slider.value() < 120                          # against the stop


def test_a_logarithmic_slider_refuses_bounds_it_cannot_take_a_log_of(qapp):
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    from crystalline.ui.panels.controls import Section, slider_row

    host = QWidget()   # held: a dropped parent takes its layout with it
    section = Section(QVBoxLayout(host), "x")
    with pytest.raises(ValueError):
        slider_row(section, "bad", 1.0, 0.0, 10.0, 0.1, logarithmic=True)


def test_the_range_slider_keeps_its_handles_in_order(qapp):
    """Two independent boxes can be put the wrong way round; a span cannot."""
    from crystalline.ui.widgets import RangeSlider

    slider = RangeSlider(0.0, 5000.0)
    slider.setValues(4000.0, 1000.0)          # handed over backwards
    assert slider.values() == (1000.0, 4000.0)

    slider.setValues(-500.0, 9999.0)          # outside the bounds
    assert slider.values() == (0.0, 5000.0)


def test_the_range_slider_reports_while_dragging(qapp):
    from crystalline.ui.widgets import RangeSlider

    slider = RangeSlider(0.0, 100.0)
    slider.resize(200, 20)
    slider.setValues(0.0, 100.0)
    seen = []
    slider.valuesChanged.connect(lambda a, b: seen.append((a, b)))

    slider._dragging = 1
    slider._drag_to(slider.width() / 2.0)

    assert seen, "a drag should report as it happens, not only on release"
    low, high = seen[-1]
    assert 0.0 < low < high == 100.0


def test_the_spectra_dialog_offers_a_span_not_two_loose_ends(qapp):
    from crystalline.ui.panels.spectra_dialog import SpectraDialog
    from crystalline.ui.widgets import RangeSlider

    dialog = SpectraDialog([])

    assert dialog.findChild(RangeSlider) is not None
    # and the options it produces are unchanged in shape
    assert dialog.options()["frequency_range"] == (0.0, 4000.0)


def test_glyphs_are_drawn_at_the_screen_s_pixel_density(qapp, monkeypatch):
    """A pixmap made 18x18 on a 2x display is 18 physical pixels blown up to 36 —
    which is exactly the soft, low-resolution look the toolbar glyphs had."""
    icon = theme.monochrome_icon("undo.svg", "#ffffff", size=18)
    pixmap = icon.pixmap(18, 18)

    ratio = float(qapp.devicePixelRatio()) or 1.0
    # the pixmap carries the ratio, so Qt places it at 18 points and paints every
    # physical pixel rather than scaling one up
    assert pixmap.devicePixelRatio() == pytest.approx(ratio)


def test_the_slider_row_is_tall_enough_for_its_handle(qapp):
    """The handle is drawn taller than the groove; without the row being that
    tall, Qt clips its top against the widget rectangle."""
    sheet = theme.stylesheet(theme.LIGHT)
    assert "QSlider:horizontal" in sheet and "min-height" in sheet


def test_ghost_chips_are_readable_rather_than_whispered(qapp):
    """The rotate arrows were drawn in the muted colour on the dark toolbar and
    could barely be made out."""
    for palette in (theme.LIGHT, theme.DARK):
        sheet = theme.stylesheet(palette)
        block = sheet.split('QToolBar QToolButton[chip="ghost"] {')[1].split("}")[0]
        assert palette.text in block
        assert palette.text_muted not in block


# ── the busy overlay, and not freezing behind it ──────────────────────────
def test_work_runs_off_the_ui_thread_and_reports_back_on_it(qapp):
    """An animated indicator over work that blocks the main thread is worse than
    a cursor: the animation freezes too, and a stopped spinner reads as a crash.
    So the work has to actually leave the UI thread."""
    import threading

    from crystalline.ui.widgets import Worker

    ui_thread = threading.get_ident()
    seen = {}

    def work():
        seen["ran_on"] = threading.get_ident()
        return 6 * 7

    worker = Worker(work)
    worker.finished.connect(lambda value: seen.update(result=value,
                                                      reported_on=threading.get_ident()))
    worker.start()

    deadline = time.time() + 5.0
    while "result" not in seen and time.time() < deadline:
        qapp.processEvents()

    assert seen.get("result") == 42
    assert seen["ran_on"] != ui_thread, "the work never left the UI thread"
    assert seen["reported_on"] == ui_thread, "the result must come back on the UI thread"


def test_a_failure_in_the_work_is_reported_not_swallowed(qapp):
    import time as _time

    from crystalline.ui.widgets import Worker

    def work():
        raise ValueError("no good")

    caught = {}
    worker = Worker(work)
    worker.failed.connect(lambda exc: caught.update(error=exc))
    worker.start()

    deadline = _time.time() + 5.0
    while "error" not in caught and _time.time() < deadline:
        qapp.processEvents()

    assert isinstance(caught.get("error"), ValueError)


def test_the_overlay_animates_and_covers_its_parent(qapp):
    from PySide6.QtWidgets import QWidget

    from crystalline.ui.widgets import BusyOverlay

    host = QWidget()
    host.resize(400, 300)
    overlay = BusyOverlay(host)
    host.show()
    qapp.processEvents()

    overlay.start("Working…")
    assert overlay.isVisible()
    assert overlay.size() == host.size()      # it covers the whole parent
    first = overlay._angle
    for _ in range(6):
        overlay._advance()
    assert overlay._angle != first             # the arc actually turns

    overlay.stop()
    assert not overlay.isVisible()


def test_the_spin_arrows_are_drawn_and_themed(qapp):
    """Fusion's own are two or three pixels of near-invisible triangle, smaller
    still once the field has comfortable padding.

    A stylesheet cannot recolour an image, so each variant names its own pair —
    which is exactly the sort of thing that silently points at a missing file.
    """
    import os

    from crystalline.resources import asset_path

    for tone in ("light", "dark"):
        for direction in ("up", "down"):
            assert os.path.isfile(asset_path(f"chevron-{direction}-{tone}.svg"))

    light_sheet = theme.stylesheet(theme.LIGHT)
    dark_sheet = theme.stylesheet(theme.DARK)
    assert "chevron-up-dark.svg" in light_sheet    # a dark glyph on a light field
    assert "chevron-up-light.svg" in dark_sheet    # and the reverse
    assert "QSpinBox::up-arrow" in light_sheet
