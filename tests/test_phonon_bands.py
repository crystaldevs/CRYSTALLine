"""Phonon bands and DOS: telling the files apart, naming the path, the plot, the dialog.

Checked against a real ZnO dispersion run (tests/data/zno) and against what the
numbers mean: a phonon fort.25 opens like an electronic one and must not be
offered as a band structure of electrons, its corners carry no denominator and
are named from the path the output states, and a frequency in THz is 33.36
times smaller than in cm⁻¹.
"""

import os
import shutil

import numpy as np
import pytest
from CRYSTALClear.units import thz_to_cm

pytest.importorskip("PySide6")

import sample_data as sample  # noqa: E402
from crystalline.crystalio import electronic as el  # noqa: E402
from crystalline.crystalio import phonon_bands as pb  # noqa: E402

_ZNO = sample.path("zno/ZnO_scelphono_333_shrink_22_tight_bands.f25")
needs_zno = sample.needs("zno/ZnO_scelphono_333_shrink_22_tight_bands.f25")

# The path of the 5×5×5 ZnO run, Γ-K-M-Γ-A-H-K, as its output states it.
_CORNERS = ((0, 0, 0), (4, 4, 0), (6, 0, 0), (0, 0, 0), (0, 0, 6), (4, 4, 6), (4, 4, 0))
_NAMES = ["Γ", "K", "M", "Γ", "A", "H", "K"]


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module", autouse=True)
def _agg():
    import matplotlib

    matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _no_session_folder():
    from crystalline.ui.panels.phonon_dialog import PhononDialog

    PhononDialog.last_folder = ""
    yield
    PhononDialog.last_folder = ""


def _zno():
    from ase.build import bulk

    from crystalline.core.structure import Structure

    return Structure.from_ase(bulk("ZnO", "wurtzite", a=3.205, c=5.140))


def _output(path, corners=_CORNERS, shrink=12, first=(-0.2914, 0.0203, 0.0203, 93.4181)):
    """The part of a dispersion output that states the path and its first point."""
    lines = [" *                              PHONON BANDS                                   *"]
    for start, end in zip(corners[:-1], corners[1:]):
        lines.append("  FROM K  ({:4d}{:4d}{:4d} ) TO K  ({:4d}{:4d}{:4d} ) WITH DENOMINATOR{:5d}"
                     .format(*start, *end, shrink))
    lines += ["", "  DISPERSION K POINT NUMBER     1 COORD:  R(   0   0   0 )    WEIGHT:    1.", ""]
    for index, value in enumerate(first, start=1):
        lines.append(f"  {index:3d}-{index:4d}    0.1E-11 {value:11.4f} {value / 33.356:9.4f}  (  1)")
    lines.append("")
    lines.append("  DISPERSION K POINT NUMBER     2 COORD:  C(   4   4   0 )    WEIGHT:    1.")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _pdos(path, projections=2, points=50):
    """A phonon DOS fort.25 (``-%-0PDOS``), as a dispersion run with PDOS writes it."""
    step = 10.0
    blocks = []
    grid = np.arange(points) * step
    for p in range(projections):
        values = np.exp(-((grid - 200.0 - 100 * p) / 40.0) ** 2) * (p + 1)
        head = "-%-0PDOS{:5d}{:5d}{:12.5E}{:12.5E}{:12.5E}".format(1, points, 0.0, step, 0.0)
        body = [head, "{:12.5E}{:12.5E}".format(0.0, 0.0), "    0    0    0    0    0    0"]
        for start in range(0, points, 6):
            body.append("".join(f"{v:12.5E}" for v in values[start:start + 6]))
        blocks.append("\n".join(body))
    path.write_text("\n".join(blocks) + "\n")
    return str(path)


# ── telling the files apart ───────────────────────────────────────────────
@pytest.mark.parametrize("text, kind", [
    ("-%-0BAND   12  100 0.00000E+00 6.91656E-01 0.00000E+00\n", (el.PHONON, "band")),
    ("-%-0BAND    8   20 0.00000E+00 4.15667E-02-1.76552E-01\n", (el.ELECTRON, "band")),
    ("-%-0PDOS    1   50 0.00000E+00 1.00000E+01 0.00000E+00\n", (el.PHONON, "dos")),
    ("-%-0DOSS    1  202 0.00000E+00 1.00000E+01-1.76552E-01\n", (el.ELECTRON, "dos")),
    ("# NKPT   300 NBND    12 NSPIN     1\n 0.0 1.0\n", (el.PHONON, "band")),
    ("# NKPT   120 NBND     8 NSPIN     1\n 0.0 1.0\n# EFERMI (HARTREE)   -0.15411\n",
     (el.ELECTRON, "band")),
    ("# NEPTS   102 NPROJ     3 NSPIN     1\n 0.0 1.0\n", (el.PHONON, "dos")),
])
def test_a_phonon_file_is_told_from_an_electronic_one_by_what_it_holds(tmp_path, text, kind):
    """The same formats; a phonon file records no Fermi level."""
    path = tmp_path / "BAND.DAT"           # the name says nothing either way
    path.write_text(text)
    assert el.data_kind(path) == kind
    assert el.file_kind(path) == (kind[1] if kind[0] == el.ELECTRON else None)


@needs_zno
def test_the_electronic_dialog_does_not_offer_phonon_bands(tmp_path):
    shutil.copy(_ZNO, tmp_path / "fort.25")
    assert el.find_files(tmp_path) == ([], [])
    bands, doss = pb.find_files(tmp_path)
    assert [os.path.basename(p) for p in bands] == ["fort.25"] and doss == []


@needs_zno
def test_the_open_runs_band_file_comes_first_by_its_first_point_not_its_name(tmp_path):
    """Two runs in one folder: the output's own is the one starting where the
    output says its path starts, whatever either is called and however old."""
    ours = tmp_path / "other_run.f25"
    shutil.copy(_ZNO, ours)
    lines = open(_ZNO).read().splitlines()
    fields = [lines[3][i:i + 12] for i in range(0, len(lines[3]), 12)]   # fixed width
    lines[3] = "".join(f"{float(v) + 5.0:12.5E}" for v in fields)
    theirs = tmp_path / "zno_bands.f25"                 # another run's, its Γ elsewhere
    theirs.write_text("\n".join(lines) + "\n")
    os.utime(ours, (1e9 - 100, 1e9 - 100))
    os.utime(theirs, (1e9, 1e9))                    # and the newer
    first = pb.describe_bands(str(ours)).first_point
    output = _output(tmp_path / "run.out", first=first)

    bands, _doss = pb.find_files(tmp_path, output=output)

    assert os.path.basename(bands[0]) == "other_run.f25"
    assert os.path.basename(pb.find_files(tmp_path)[0][0]) == "zno_bands.f25"


# ── naming the path ───────────────────────────────────────────────────────
def _info(ticks):
    return pb.BandsInfo(path="/nowhere/fort.25", n_bands=12, n_kpoints=600, ticks=tuple(ticks))


def test_the_output_states_the_path_with_its_denominator(tmp_path):
    stated = pb.output_path(_output(tmp_path / "zno.out"))
    assert stated.corners == _CORNERS and stated.shrink == 12
    assert stated.first_point == pytest.approx((-0.2914, 0.0203, 0.0203, 93.4181))


def test_fort25_corners_are_named_from_the_path_the_output_states(tmp_path):
    """(4,4,0) means nothing without its 12; with it, it is K."""
    ticks = ["({},{},{})".format(*c) for c in _CORNERS]
    output = _output(tmp_path / "zno.out")

    assert pb.path_labels(_info(ticks), _zno(), output=output) == _NAMES
    assert pb.path_labels(_info(ticks), _zno()) == ticks       # nothing to go on


def test_a_path_of_another_run_names_nothing(tmp_path):
    ticks = ["({},{},{})".format(*c) for c in _CORNERS]
    output = _output(tmp_path / "zno.out", corners=_CORNERS[:-1] + ((0, 0, 0),))
    assert pb.path_labels(_info(ticks), _zno(), output=output) == ticks


def test_the_input_deck_states_the_path_too(tmp_path):
    (tmp_path / "zno.d12").write_text(
        "ZnO\nCRYSTAL\n0 0 0\n186\n3.2 5.1\n2\n30 0.3333 -0.3333 0.0\n8 0.3333 -0.3333 0.38\n"
        "SCELPHONO\n5 0 0\n0 5 0\n0 0 5\nFREQCALC\nDISPERSI\nBANDS\n12 100\n6\n"
        + "".join("{} {} {}  {} {} {}\n".format(*a, *b) for a, b in zip(_CORNERS, _CORNERS[1:]))
        + "END\nEND\n")
    info = pb.BandsInfo(path=str(tmp_path / "fort.25"), n_bands=12, n_kpoints=600,
                        ticks=tuple("({},{},{})".format(*c) for c in _CORNERS))

    assert pb.path_labels(info, _zno()) == _NAMES


def test_a_text_file_carries_its_own_denominator(tmp_path):
    ticks = ["({},{},{})/12".format(*c) for c in _CORNERS]
    assert pb.path_labels(_info(ticks), _zno()) == _NAMES


# ── the window ────────────────────────────────────────────────────────────
def test_the_window_starts_at_zero_through_the_noise_at_gamma():
    assert pb.suggest_window([-1.3, 0.0, 300.0, 619.6]) == (0.0, 650.0)


def test_an_imaginary_branch_deeper_than_the_noise_is_shown():
    low, high = pb.suggest_window([-48.0, 0.0, 300.0, 619.6])
    assert low < -48.0 and high == 650.0


# ── the plot ──────────────────────────────────────────────────────────────
def _curves(ax):
    return [line for line in ax.get_lines() if len(line.get_xdata()) > 2]


@needs_zno
def test_the_bands_are_drawn_in_the_unit_asked_for():
    cm = pb.plot_phonons(_ZNO, None, pb.PhononOptions(unit=pb.CM))
    thz = pb.plot_phonons(_ZNO, None, pb.PhononOptions(unit=pb.THZ))

    top_cm = max(np.max(line.get_ydata()) for line in _curves(cm.axes[0]))
    top_thz = max(np.max(line.get_ydata()) for line in _curves(thz.axes[0]))
    assert top_cm == pytest.approx(top_thz * thz_to_cm(1.0), rel=1e-6)
    assert 550 < top_cm < 650                      # ZnO's highest branch
    assert cm._supylabel.get_text() == "Frequency (cm$^{-1}$)"
    assert thz._supylabel.get_text() == "Frequency (THz)"


@needs_zno
def test_the_window_and_the_names_reach_the_axes():
    names = ["Γ", "K", "M", "Γ", "A"]
    figure = pb.plot_phonons(_ZNO, None, pb.PhononOptions(frequency_range=(0.0, 650.0),
                                                          k_labels=names))
    ax = figure.axes[0]
    assert ax.get_ylim() == pytest.approx((0.0, 650.0))
    assert [label.get_text() for label in ax.get_xticklabels()] == names


@needs_zno
def test_the_zero_line_takes_its_style_or_is_hidden():
    from matplotlib.colors import to_rgba

    def zero_lines(figure):
        return [line for line in figure.axes[0].get_lines()
                if len(line.get_xdata()) == 2
                and np.allclose(to_rgba(line.get_color()), to_rgba("#123456"))]

    shown = pb.plot_phonons(_ZNO, None, pb.PhononOptions(zero_colour="#123456", zero_width=2.0))
    (line,) = zero_lines(shown)
    assert line.get_visible() and line.get_linewidth() == 2.0
    assert np.allclose(line.get_ydata(), 0.0)
    hidden = pb.plot_phonons(_ZNO, None, pb.PhononOptions(zero_colour="#123456", show_zero=False))
    assert not any(line.get_visible() for line in zero_lines(hidden))


def test_a_phonon_dos_is_drawn_per_unit_of_frequency(tmp_path):
    path = _pdos(tmp_path / "fort.25")
    cm = pb.plot_phonons(None, path, pb.PhononOptions(mode=pb.DOS, unit=pb.CM))
    thz = pb.plot_phonons(None, path, pb.PhononOptions(mode=pb.DOS, unit=pb.THZ))

    peak = lambda fig: max(np.max(line.get_ydata()) for line in _curves(fig.axes[0]))  # noqa: E731
    where = lambda fig: max(np.max(line.get_xdata()) for line in _curves(fig.axes[0]))  # noqa: E731
    assert peak(thz) == pytest.approx(peak(cm) * thz_to_cm(1.0), rel=1e-6)
    assert where(cm) == pytest.approx(where(thz) * thz_to_cm(1.0), rel=1e-6)
    assert cm._supylabel.get_text() == "DOS (states/cm$^{-1}$)"
    assert cm._supxlabel.get_text() == "Frequency (cm$^{-1}$)"


@needs_zno
def test_bands_and_dos_share_one_frequency_axis(tmp_path):
    figure = pb.plot_phonons(_ZNO, _pdos(tmp_path / "pdos.f25"),
                             pb.PhononOptions(mode=pb.BANDS_AND_DOS, frequency_range=(0, 650)))
    assert len(figure.axes) == 2
    assert figure.axes[0].get_ylim() == pytest.approx(figure.axes[1].get_ylim())


def test_a_missing_file_is_refused_with_what_is_missing():
    with pytest.raises(ValueError, match="band file"):
        pb.plot_phonons(None, None, pb.PhononOptions(mode=pb.BANDS))
    with pytest.raises(ValueError, match="DOS file"):
        pb.plot_phonons("x", None, pb.PhononOptions(mode=pb.BANDS_AND_DOS))


# ── the dialog ────────────────────────────────────────────────────────────
def _dialog(qapp, folder="", structure=None, output=None, state=None):
    from crystalline.ui.panels.dialog_state import restore
    from crystalline.ui.panels.phonon_dialog import PhononDialog

    dialog = PhononDialog(structure=structure, folder=str(folder), output=output)
    restore(dialog, state)
    dialog.show()
    qapp.processEvents()
    return dialog


def _ok(dialog):
    from PySide6.QtWidgets import QDialogButtonBox

    return dialog._buttons.button(QDialogButtonBox.Ok).isEnabled()


def test_nothing_can_be_plotted_until_a_file_is_chosen(qapp):
    dialog = _dialog(qapp)
    assert not _ok(dialog)
    assert dialog.width() > dialog.height()


@needs_zno
def test_the_runs_files_are_found_and_its_path_named(qapp, tmp_path):
    shutil.copy(_ZNO, tmp_path / "fort.25")
    ticks = pb.describe_bands(str(tmp_path / "fort.25")).ticks
    corners = [tuple(int(n) for n in t.strip("()").split(",")) for t in ticks]
    output = _output(tmp_path / "zno.out", corners=tuple(corners))

    dialog = _dialog(qapp, folder=tmp_path, structure=_zno(), output=output)
    band, dos, options = dialog.request()

    assert band.endswith("fort.25") and dos is None and _ok(dialog)
    assert options.k_labels[0] == "Γ" and len(options.k_labels) == len(ticks)
    assert options.unit == pb.CM


@needs_zno
def test_the_window_is_fitted_to_the_data_and_carried_across_a_change_of_unit(qapp, tmp_path):
    shutil.copy(_ZNO, tmp_path / "fort.25")
    dialog = _dialog(qapp, folder=tmp_path)
    info = pb.describe_bands(str(tmp_path / "fort.25"))
    assert dialog.window_values() == pytest.approx(info.window)

    dialog.unit.setCurrentIndex(dialog.unit.findData(pb.THZ))
    dialog.unit.activated.emit(dialog.unit.currentIndex())
    low, high = dialog.window_values()
    assert high == pytest.approx(info.window[1] / thz_to_cm(1.0), abs=1e-3)

    dialog.unit.setCurrentIndex(dialog.unit.findData(pb.CM))
    dialog.unit.activated.emit(dialog.unit.currentIndex())
    assert dialog.window_values()[1] == pytest.approx(info.window[1], abs=0.1)


def test_a_folder_with_only_a_dos_shows_the_dos(qapp, tmp_path):
    _pdos(tmp_path / "PHONDOS.f25", projections=3)
    dialog = _dialog(qapp, folder=tmp_path)
    band, dos, options = dialog.request()
    assert options.mode == pb.DOS and band is None and dos.endswith("PHONDOS.f25")
    assert dialog._table.rowCount() == 3 and _ok(dialog)


@needs_zno
def test_the_dialog_output_feeds_the_plotter_and_comes_back(qapp, tmp_path):
    shutil.copy(_ZNO, tmp_path / "fort.25")
    _pdos(tmp_path / "pdos.f25")
    dialog = _dialog(qapp, folder=tmp_path)
    dialog.mode.setCurrentIndex(dialog.mode.findData(pb.BANDS_AND_DOS))
    dialog._title.setText("ZnO")
    band, dos, options = dialog.request()
    figure = pb.plot_phonons(band, dos, options)
    assert len(figure.axes) == 2

    again = _dialog(qapp)
    again.set_request(band, dos, options)
    again.show()
    assert again.request()[2] == options


def test_settings_are_remembered_but_files_and_windows_are_not(qapp):
    from crystalline.ui.panels.dialog_state import capture

    state = capture(_dialog(qapp))
    for remembered in ("mode", "unit", "band_colour", "zero_colour", "zero_style",
                       "zero_width", "overlay", "dos_auto", "linewidth", "show_zero", "legend"):
        assert remembered in state, remembered
    assert "_files" not in state and "_k_edit" not in state and "_title" not in state


# ── the controls beside the plot ──────────────────────────────────────────
def test_the_controls_hand_back_what_the_plotter_takes(qapp):
    from crystalline.ui.panels import phonon_controls as controls

    panel = controls.PhononControls()
    options = {"mode": pb.BANDS, "bands_path": "a", "dos_path": None, "unit": pb.CM,
               "frequency_range": (0.0, 650.0), "frequency_bounds": (-50.0, 700.0),
               "dos_bounds": (0.0, 1.0), "dos_handles": (0.0, 0.5), "dos_projections": 0}
    panel.set_options(options)
    merged = {**options, **panel.options()}

    bands, dos, plot_options = controls.plot_arguments(merged)
    assert bands == "a" and dos is None
    assert plot_options.frequency_range == pytest.approx((0.0, 650.0))
    assert not panel.dos_auto.isVisibleTo(panel)        # no DOS in a band plot


# ── the menu and the output ───────────────────────────────────────────────
def test_one_menu_entry_opens_the_dialog():
    import inspect

    from crystalline.crystalio import plotting
    from crystalline.ui.main_window import MainWindow

    keys = {kind.key for kind in plotting.available_plots()}
    assert not {"phonon_band", "phonon_dos"} & keys
    source = inspect.getsource(MainWindow._open_phonon_bands)
    assert "PhononDialog" in source and "plot_phonons" in source


def test_a_bands_only_run_is_recognised_from_its_output(tmp_path):
    from crystalline.crystalio.loader import _dispersion_without_modes

    path = tmp_path / "run.out"
    path.write_text("SCELPHONO\n5 0 0\n INFORMATION **** INPFREQ **** BANDS: MODES PRINTING "
                    "DISABLED. USE KEYWORD MODES TO ENABLE.\n")
    assert _dispersion_without_modes(str(path)) == (True, True)
    path.write_text("FREQCALC\n")
    assert _dispersion_without_modes(str(path)) == (False, False)


_RUN = sample.path("scelphono555_bands/ZnO_LDA_scelphono555_bands.out")


@pytest.mark.skipif(not _RUN, reason="the 5×5×5 ZnO dispersion run is not available")
def test_a_bands_only_scelphono_run_opens_on_the_cell_it_expanded():
    from crystalline.crystalio import load

    loaded = load(_RUN)
    assert len(loaded.structure) == 4 and loaded.modes is None
    assert "MODES" in loaded.note and "Phonon bands & DOS" in loaded.note
