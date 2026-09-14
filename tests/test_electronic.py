"""Electronic band structures and DOS: the plotter, the dialog, the wiring.

Checked against real PROPERTIES runs from the MSSC2025 school where they exist
(silicon, beryllium, MgO, and spin-polarised LiF and ITO), and against what the
numbers mean rather than what the code happens to draw: silicon's path is
Γ-X-W-L-Γ, its Fermi level is at -4.19 eV, beryllium's 1s band sits 104 eV
below its valence band, and a spin-down band has to be distinguishable from a
spin-up one.
"""

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")

from crystalline.crystalio import electronic as el  # noqa: E402

_SCHOOL = os.path.expanduser("~/MSSC2025/Basic/Day4/OneElectronProperties")
_SI = os.path.join(_SCHOOL, "Silicon")
_BE = os.path.join(_SCHOOL, "Berillium")
_SI_BAND, _SI_DOS = os.path.join(_SI, "BAND.DAT"), os.path.join(_SI, "DOSS.DAT")
_MGO = os.path.expanduser("~/MSSC2025/Advanced/Day1/one-electron-properties")
_LIF_DIR = os.path.expanduser("~/MSSC2025/Basic/Day5/Defects/4_LiF_2_2_2_F_center_band")
_LIF = os.path.join(_LIF_DIR, "BAND.DAT")
_ITO = os.path.expanduser(
    "~/Desktop/PyCrystal/CRYSTALpytools/examples/data/doss_ito-cu.DOSS")
_SI_TICKS = ("(0,0,0)/8", "(4,0,4)/8", "(4,2,6)/8", "(4,4,4)/8", "(0,0,0)/8")

needs_silicon = pytest.mark.skipif(
    not (os.path.isfile(_SI_BAND) and os.path.isfile(_SI_DOS)),
    reason="the MSSC2025 silicon PROPERTIES run is not on this machine")
needs_lif = pytest.mark.skipif(not os.path.isfile(_LIF), reason="no spin-polarised band file")
needs_ito = pytest.mark.skipif(not os.path.isfile(_ITO), reason="no spin-polarised DOS file")


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
    """The dialog's session memory of a folder must not leak between tests."""
    from crystalline.ui.panels.electronic_dialog import ElectronicDialog

    ElectronicDialog.last_folder = ""
    yield
    ElectronicDialog.last_folder = ""


def _silicon():
    from ase.build import bulk

    from crystalline.core.structure import Structure

    return Structure.from_ase(bulk("Si", "diamond", a=5.43))


def _fermi_lines(figure, colour=el.FERMI_COLOUR):
    from matplotlib.colors import to_rgba

    found = []
    for index, ax in enumerate(figure.axes):
        for line in ax.get_lines():
            if len(line.get_xdata()) == 2 and np.allclose(
                    to_rgba(line.get_color()), to_rgba(colour)):
                found.append((index, line))
    return found


def _curves(ax):
    return [line for line in ax.get_lines() if len(line.get_xdata()) > 2]


# ── naming the path's corners ─────────────────────────────────────────────
def test_a_tick_is_read_as_the_fraction_it_stands_for():
    assert el.tick_coordinates('"(4,0,4)/8"') == (0.5, 0.0, 0.5)
    assert el.tick_coordinates("(4,2,6)/8") == (0.5, 0.25, 0.75)
    assert el.clean_tick('"(0,0,0)/8"') == "(0,0,0)/8"


def test_a_tick_without_its_denominator_is_not_guessed_at():
    """A fort.25 writes the integers but not the shrinking factor they are
    over; a guessed factor would name a point the path never visited."""
    assert el.tick_coordinates("(4,0,4)") is None


def test_a_fortran_overflow_is_not_read_as_a_number():
    """CRYSTAL writes ``*`` when an integer does not fit its field."""
    assert el.tick_coordinates("(*,4,3)/6") is None


def test_the_corners_are_named_from_the_structure():
    """Every band file labels its corners with coordinates, and without names
    CRYSTALClear marks them with their k-distance instead — so the x axis of
    every band plot was 0.000, 0.613, 0.920…"""
    assert el.name_ticks(_SI_TICKS, _silicon()) == ["Γ", "X", "W", "L", "Γ"]


def test_without_a_structure_the_coordinates_are_kept():
    """A wrong name is worse than an honest coordinate."""
    assert el.name_ticks(_SI_TICKS, None) == list(_SI_TICKS)


def _beryllium():
    from ase.build import bulk

    from crystalline.core.structure import Structure

    return Structure.from_ase(bulk("Be", "hcp", a=2.29, c=3.59))


def test_a_corner_is_named_however_the_axes_were_chosen():
    """CRYSTAL's own tutorial writes beryllium's path M=(0,1/2,0), K=(-1/3,2/3,0);
    the standard tables say (1/2,0,0) and (1/3,1/3,0). Same points, other axes —
    and compared number for number, none of them was named at all."""
    ticks = ("(0,0,0)/6", "(0,3,0)/6", "(0,3,3)/6", "(-2,4,3)/6", "(-2,4,0)/6")
    assert el.name_ticks(ticks, _beryllium()) == ["Γ", "M", "L", "H", "K"]


def test_labels_a_lower_symmetry_keeps_apart_are_not_run_together():
    """X, Y and Z of an orthorhombic lattice are three different points: no
    rotation of that lattice carries one into another."""
    from ase import Atoms

    from crystalline.core.structure import Structure

    structure = Structure.from_ase(Atoms("Cu", cell=[3.0, 4.0, 5.0, 90, 90, 90], pbc=True))
    assert el.name_ticks(("(1,0,0)/2", "(0,1,0)/2", "(0,0,1)/2"), structure) == ["X", "Y", "Z"]


def test_a_corner_that_matches_nothing_keeps_its_coordinates():
    named = el.name_ticks(("(0,0,0)/8", "(1,2,3)/8"), _silicon())
    assert named == ["Γ", "(1,2,3)/8"]


# ── the path the deck asked for ───────────────────────────────────────────
_BE_DECK = """BAND
BE  HCP Path
5 6 120 1 9 1 0
0 0 0   0 3 0   G M
0 3 0   0 3 3   M L
0 3 3  -2 4 3   L H
-2 4 3 -2 4 0   H K
-2 4 0  0 0 0   K G
END
"""


def test_a_deck_is_read_for_its_path_and_its_letters(tmp_path):
    (tmp_path / "be_band.d3").write_text("NEWK\n6 6\n1 0\n" + _BE_DECK + "BWID\n1 152\nEND\n")
    deck, = el.read_band_decks(tmp_path)
    assert deck.shrink == 6
    assert deck.corners == ((0, 0, 0), (0, 3, 0), (0, 3, 3), (-2, 4, 3), (-2, 4, 0), (0, 0, 0))
    assert deck.labels == ("G", "M", "L", "H", "K", "G")


@pytest.mark.skipif(not os.path.isdir(_BE), reason="no beryllium run")
def test_the_deck_names_corners_the_file_could_not_write_down():
    """Beryllium's (-2,4,3) overflows its field in BAND.DAT and is written
    (*,4,3) — the coordinates are gone, so only the deck can name it. CRYSTAL's
    tutorial gives this path as G-M-L-H-K-G."""
    info = el.describe_bands(os.path.join(_BE, "BAND.DAT"))
    assert "(*,4,3)/6" in info.ticks
    assert el.path_labels(info, None) == ["Γ", "M", "L", "H", "K", "Γ"]


@needs_silicon
def test_a_deck_with_no_letters_leaves_the_naming_to_the_structure():
    """Silicon's deck writes coordinates only."""
    info = el.describe_bands(_SI_BAND)
    assert el.path_labels(info, _silicon()) == ["Γ", "X", "W", "L", "Γ"]


def test_a_path_written_by_label_alone_is_read(tmp_path):
    """CRYSTAL takes the path as letters, with the shrinking factor written 0
    and no coordinates at all — the tutorial gives this as the equivalent of
    MgO's numeric deck."""
    (tmp_path / "mgo_band.d3").write_text(
        "BAND\nMgO\n4  0  60  1  18  1  0\nG X\nX W\nW L\nL G\nEND\n")
    deck, = el.read_band_decks(tmp_path)
    assert deck.labels == ("G", "X", "W", "L", "G") and deck.corners == ()
    ticks = ("(0,0,0)/8", "(4,0,4)/8", "(4,2,6)/8", "(4,4,4)/8", "(0,0,0)/8")
    assert el.deck_labels(ticks, tmp_path) == ["G", "X", "W", "L", "G"]


def test_a_path_by_label_is_left_alone_when_another_deck_could_fit(tmp_path):
    """Nothing in it can be checked against the file, so it is taken only when
    it is the one deck that could describe this path."""
    (tmp_path / "a_band.d3").write_text(
        "BAND\nMgO\n4  0  60  1  18  1  0\nG X\nX W\nW L\nL G\nEND\n")
    (tmp_path / "b_band.d3").write_text(
        "BAND\nOther\n4  8  60  1  18  1  0\n0 0 0 4 0 0\n4 0 0 4 4 0\n"
        "4 4 0 0 4 0\n0 4 0 0 0 0\nEND\n")
    ticks = ("(0,0,0)/8", "(4,0,4)/8", "(4,2,6)/8", "(4,4,4)/8", "(0,0,0)/8")
    assert el.deck_labels(ticks, tmp_path) is None


def test_a_deck_for_a_different_path_is_not_applied(tmp_path):
    """A folder can hold decks for several runs; one that does not describe
    this file's corners must not rename them."""
    (tmp_path / "other.d3").write_text(_BE_DECK)
    ticks = ("(0,0,0)/8", "(4,0,4)/8", "(4,2,6)/8", "(4,4,4)/8", "(0,0,0)/8")
    assert el.deck_labels(ticks, tmp_path) is None
    assert el.deck_labels(("(0,0,0)/6", "(0,3,0)/6", "(0,3,3)/6",
                           "(*,4,3)/6", "(*,4,0)/6", "(0,0,0)/6"), tmp_path) == [
        "G", "M", "L", "H", "K", "G"]


# ── the suggested window ──────────────────────────────────────────────────
def _bands(*intervals, nk=10):
    """A fake band array, one band per (low, high) interval."""
    return np.array([np.linspace(low, high, nk) for low, high in intervals])[:, :, None]


def test_the_window_stops_where_the_core_levels_start():
    """A 1s level 100 eV down flattens the valence bands into a line."""
    window = el.suggest_window(_bands((-121, -120), (-15, -8), (-7, 0), (5, 30)))
    assert window == (-16.0, 21.0)


def test_a_state_in_the_gap_does_not_end_the_valence_region():
    """LiF's F-centre sits 8.8 eV above the valence band — a defect state, not
    the top of the core."""
    window = el.suggest_window(_bands((-14, -10), (-1, 0), (4, 20)))
    assert window[0] == -15.0


def test_a_metal_window_is_as_tall_above_the_fermi_level_as_below():
    window = el.suggest_window(_bands((-60, -59.5), (-8, 6)))
    assert window == (-9.0, 7.0)


@pytest.mark.skipif(not os.path.isdir(_BE), reason="no beryllium run")
def test_beryllium_is_shown_without_its_1s_band():
    info = el.describe_bands(os.path.join(_BE, "BAND.DAT"))
    assert info.energy_span[0] < -120, "the 1s band is in the file"
    assert -25 < info.window[0] < -15, "and left out of the window"


# ── finding the files ─────────────────────────────────────────────────────
@pytest.mark.parametrize("name, first_line, kind", [
    ("BAND.DAT", "# NKPT   120 NBND     8 NSPIN     1", "band"),
    ("DOSS.DAT", "# NEPTS   102 NPROJ     3 NSPIN     1", "dos"),
    ("mgo_band.BAND", "# NKPT    61 NBND     8 NSPIN     1", "band"),
    ("mgo.DOSS", "# NEPTS   202 NPROJ     3 NSPIN     1", "dos"),
    ("si_band.f25", "-%-0BAND    8   20 0.00000E+00", "band"),
    ("fort.25", "-%-0DOSS    1  202 0.00000E+00", "dos"),
    ("mgo_echg.f25", "-%-0MAPN  100  100 0.11111E+00", None),
    ("PPAN.DAT", "# Mulliken Populations:", None),
    ("BAND.DAT", "# NKPT", None),                   # the marker, but not the record
    # Stems say nothing: these are what their first line says they are.
    ("DOSS.DAT", "# NKPT    61 NBND     8 NSPIN     1", "band"),
    ("mgo_band.BAND", "# NEPTS   202 NPROJ     3 NSPIN     1", "dos"),
    ("mgo_doss.f25", "-%-0BAND    8   20 0.00000E+00", "band"),
])
def test_a_file_is_known_by_what_it_says_it_is(tmp_path, name, first_line, kind):
    path = tmp_path / name
    path.write_text(first_line + "\n1 2 3\n")
    assert el.file_kind(path) == kind


def test_text_files_rank_above_fort25_then_the_newest(tmp_path):
    """A file named after the run is not preferred any more: names are not
    evidence. And a fort.25 is known by its first line, not by a .f25 suffix."""
    for name, header, age in (("si_band.f25", "-%-0BAND    8   20 0.0", 0),
                              ("BAND.DAT", "# NKPT   120 NBND 8", 50),
                              ("si_band.BAND", "# NKPT   120 NBND 8", 100),
                              ("si_old.BAND", "# NKPT   120 NBND 8", 200),
                              ("other.BAND", "# NKPT   120 NBND 8", 0),
                              ("looks_like_text.BAND", "-%-0BAND    8   20 0.0", 10),
                              ("notes.txt", "# NKPT   120 NBND 8", 0)):
        path = tmp_path / name
        path.write_text(header + "\n")
        os.utime(path, (1e9 - age, 1e9 - age))
    bands, doss = el.find_files(tmp_path, stem="si")
    assert [os.path.basename(p) for p in bands] == [
        "other.BAND", "BAND.DAT", "si_band.BAND", "si_old.BAND",
        "si_band.f25", "looks_like_text.BAND"]
    assert doss == []


def test_a_coop_file_is_told_from_a_dos_by_its_axis_label_not_its_name(tmp_path):
    """Both open with # NEPTS; a DOSS.DAT's y axis is "DENSITY OF STATES"."""
    header = "# NEPTS   202 NPROJ     2 NSPIN     1\n#\n@ XAXIS LABEL \"E-EFERMI (HARTREE)\"\n"
    (tmp_path / "DOSS.DAT").write_text(header + '@ YAXIS LABEL "COOP"\n')
    (tmp_path / "COOP.DAT").write_text(
        header + '@ YAXIS LABEL "DENSITY OF STATES (STATES/HARTREE/CELL)"\n')

    assert el.file_kind(tmp_path / "DOSS.DAT") is None
    assert el.file_kind(tmp_path / "COOP.DAT") == "dos"


def test_a_properties_deck_is_found_by_extension_and_content_whatever_its_stem(tmp_path):
    deck = "NEWK\n8 8\n1 0\nBAND\nPATH\n2 8 100 1 10 1 0\n0 0 0 4 0 4\n4 0 4 4 2 5\nEND\n"
    (tmp_path / "renamed_anything.d3").write_text(deck)
    (tmp_path / "notes.txt").write_text(deck)                              # not a deck extension
    (tmp_path / "mgo.d3").write_text(" " * 20 + "CRYSTAL23\n BAND\nEND\n")   # an output's banner

    decks = el.read_band_decks(tmp_path)

    assert {os.path.basename(d.path) for d in decks} == {"renamed_anything.d3"}


@pytest.mark.skipif(not os.path.isdir(_MGO), reason="no MgO properties folder")
def test_a_folder_with_several_runs_offers_every_one():
    """Two band structures and two DOS here; the old finder, wanting exactly
    one of each, filled in neither."""
    bands, doss = el.find_files(_MGO, stem="mgo")
    names = {os.path.basename(p) for p in bands + doss}
    assert {"mgo_band.BAND", "mgo_band_b3lyp.BAND",
            "mgo_doss_totalao.DOSS", "mgo_doss_partialao.DOSS"} <= names
    assert not bands[0].endswith(".f25") and not doss[0].endswith(".f25")
    assert not any("echg" in p or "pban" in p for p in bands + doss)


# ── the energy reference ──────────────────────────────────────────────────
@needs_silicon
@pytest.mark.parametrize("mode, axis", [
    (el.BANDS, "y"), (el.DOS, "x"), (el.BANDS_AND_DOS, "y"),
])
def test_relative_energies_put_the_fermi_level_at_zero(mode, axis):
    figure = el.plot_electronic(_SI_BAND, _SI_DOS, el.ElectronicOptions(mode=mode))
    lines = _fermi_lines(figure)
    assert lines
    for _index, line in lines:
        data = line.get_ydata() if axis == "y" else line.get_xdata()
        assert np.allclose(data, 0.0)


@needs_silicon
@pytest.mark.parametrize("mode, axis", [
    (el.BANDS, "y"), (el.DOS, "x"), (el.BANDS_AND_DOS, "y"),
])
def test_absolute_energies_move_the_fermi_line_with_the_data(mode, axis):
    """CRYSTALClear draws its Fermi line at zero whatever the energy scale, so
    shifting the data alone would leave the line marking nothing at all."""
    efermi = el.describe_bands(_SI_BAND).efermi
    figure = el.plot_electronic(_SI_BAND, _SI_DOS, el.ElectronicOptions(
        mode=mode, reference=el.ABSOLUTE))
    lines = _fermi_lines(figure)
    assert lines
    for _index, line in lines:
        data = line.get_ydata() if axis == "y" else line.get_xdata()
        assert np.allclose(data, efermi, atol=1e-6)


@needs_silicon
def test_the_fermi_level_of_silicon_is_where_the_school_run_puts_it():
    assert el.describe_bands(_SI_BAND).efermi == pytest.approx(-4.1935, abs=1e-3)


@needs_silicon
def test_an_absolute_plot_in_hartree_converts_the_shift_too():
    efermi = el.describe_bands(_SI_BAND).efermi
    figure = el.plot_electronic(_SI_BAND, None, el.ElectronicOptions(
        mode=el.BANDS, unit="Hartree", reference=el.ABSOLUTE))
    (_index, line), = _fermi_lines(figure)
    assert line.get_ydata()[0] == pytest.approx(efermi / 27.211386, abs=1e-5)


@needs_silicon
def test_the_axis_says_which_energy_it_carries():
    """CRYSTALClear's combined plot calls a relative axis plain "Energy"."""
    relative = el.plot_electronic(_SI_BAND, _SI_DOS, el.ElectronicOptions(
        mode=el.BANDS_AND_DOS))
    absolute = el.plot_electronic(_SI_BAND, _SI_DOS, el.ElectronicOptions(
        mode=el.BANDS_AND_DOS, reference=el.ABSOLUTE))
    assert any("E_" in text.get_text() for text in relative.texts)
    assert not any("E_" in text.get_text() for text in absolute.texts)


@needs_silicon
def test_the_energy_range_clips_the_plot():
    figure = el.plot_electronic(_SI_BAND, None, el.ElectronicOptions(
        mode=el.BANDS, energy_range=(-6.0, 6.0)))
    assert figure.axes[0].get_ylim() == pytest.approx((-6.0, 6.0))


@needs_silicon
def test_named_corners_reach_the_axis():
    figure = el.plot_electronic(_SI_BAND, None, el.ElectronicOptions(
        mode=el.BANDS, k_labels=["Γ", "X", "W", "L", "Γ"]))
    ticks = [label.get_text() for label in figure.axes[0].get_xticklabels()]
    assert ticks == ["Γ", "X", "W", "L", "Γ"]


@needs_silicon
def test_bands_and_dos_share_one_energy_axis():
    figure = el.plot_electronic(_SI_BAND, _SI_DOS, el.ElectronicOptions(
        mode=el.BANDS_AND_DOS, energy_range=(-5.0, 5.0)))
    assert len(figure.axes) == 2
    assert figure.axes[0].get_ylim() == pytest.approx(figure.axes[1].get_ylim())


def test_a_missing_file_is_refused_with_what_is_missing():
    with pytest.raises(ValueError, match="band file"):
        el.plot_electronic(None, None, el.ElectronicOptions(mode=el.BANDS))
    with pytest.raises(ValueError, match="DOS file"):
        el.plot_electronic(None, None, el.ElectronicOptions(mode=el.DOS))


# ── appearance ────────────────────────────────────────────────────────────
@needs_silicon
def test_the_fermi_line_takes_the_colour_style_and_width_asked_for():
    """Black included: the line is found before it is restyled, so it cannot be
    confused with the black zero lines beside it."""
    figure = el.plot_electronic(_SI_BAND, _SI_DOS, el.ElectronicOptions(
        mode=el.BANDS_AND_DOS, fermi_colour="#000000", fermi_style=":", fermi_width=2.5))
    assert not _fermi_lines(figure), "no line is left in the marker colour"
    # Horizontal, at E_F: the black k-point separators are vertical.
    level = [line for _index, line in _fermi_lines(figure, "#000000")
             if np.allclose(line.get_ydata(), 0.0)]
    assert len(level) == 2, "one across the bands, one across the DOS"
    for line in level:
        assert line.get_linestyle() == ":" and line.get_linewidth() == 2.5


@needs_silicon
def test_the_fermi_line_can_be_hidden():
    figure = el.plot_electronic(_SI_BAND, None, el.ElectronicOptions(
        mode=el.BANDS, show_fermi=False))
    assert all(not line.get_visible() for _index, line in _fermi_lines(figure))


@needs_silicon
def test_the_bands_take_their_colour():
    from matplotlib.colors import to_hex

    figure = el.plot_electronic(_SI_BAND, None, el.ElectronicOptions(
        mode=el.BANDS, band_colour="#123456"))
    assert {to_hex(line.get_color()) for line in _curves(figure.axes[0])} == {"#123456"}


@needs_silicon
def test_projections_carry_their_own_names_and_colours():
    from matplotlib.colors import to_hex

    figure = el.plot_electronic(None, _SI_DOS, el.ElectronicOptions(
        mode=el.DOS, projections=[2, 3], projection_labels=["Si s", "Si p"],
        projection_colours=["#aa0000", "#0000aa"]))
    ax = figure.axes[0]
    assert [to_hex(line.get_color()) for line in _curves(ax)] == ["#aa0000", "#0000aa"]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["Si s", "Si p"]


@needs_silicon
@pytest.mark.filterwarnings("error:When overlap is false")
def test_projections_can_be_drawn_one_panel_each():
    """CRYSTALClear takes one colour and no names for separate panels, and
    swaps anything else for blue and nothing; each panel is done here."""
    from matplotlib.colors import to_hex

    figure = el.plot_electronic(None, _SI_DOS, el.ElectronicOptions(
        mode=el.DOS, overlay_projections=False, projections=[1, 3],
        projection_colours=["#aa0000", "#0000aa"]))
    assert len(figure.axes) == 2
    names = [[text.get_text() for text in ax.texts] for ax in figure.axes]
    assert names == [["Projection 1"], ["Projection 3"]]
    assert [to_hex(_curves(ax)[0].get_color()) for ax in figure.axes] == ["#aa0000", "#0000aa"]


@needs_silicon
def test_a_title_is_set_and_none_is_none():
    titled = el.plot_electronic(_SI_BAND, None, el.ElectronicOptions(mode=el.BANDS,
                                                                     title="Silicon"))
    assert "Silicon" in [titled._suptitle.get_text() if titled._suptitle else "",
                         titled.axes[0].get_title()]


# ── spin ──────────────────────────────────────────────────────────────────
@needs_lif
def test_spin_down_bands_are_told_apart_from_spin_up():
    """CRYSTALClear's standalone band plot draws both spin channels in one
    colour and one style. Here β is laid over α in a colour of its own."""
    from matplotlib.colors import to_hex

    info = el.describe_bands(_LIF)
    assert info.spin == 2
    figure = el.plot_electronic(_LIF, None, el.ElectronicOptions(
        mode=el.BANDS, beta_colour="#00aa00", beta_style=":"))
    curves = _curves(figure.axes[0])
    alpha = [line for line in curves if line.get_linestyle() == "-"]
    beta = [line for line in curves if line.get_linestyle() == ":"]
    assert len(alpha) == len(beta) == info.n_bands
    assert {to_hex(line.get_color()) for line in beta} == {"#00aa00"}
    legend = figure.axes[0].get_legend()
    assert legend is not None and len(legend.get_texts()) == 2


@needs_lif
@needs_ito
def test_the_combined_view_restyles_its_spin_down_bands_too():
    from matplotlib.colors import to_hex

    figure = el.plot_electronic(_LIF, _ITO, el.ElectronicOptions(
        mode=el.BANDS_AND_DOS, beta_colour="#00aa00", beta_style="-."))
    beta = [line for line in _curves(figure.axes[0]) if line.get_linestyle() == "-."]
    assert beta and {to_hex(line.get_color()) for line in beta} == {"#00aa00"}


@needs_ito
def test_a_spin_down_dos_can_be_mirrored_or_drawn_alongside():
    def lowest(spin):
        figure = el.plot_electronic(None, _ITO, el.ElectronicOptions(
            mode=el.DOS, spin_down=spin))
        return min(np.asarray(line.get_ydata(), dtype=float).min()
                   for ax in figure.axes for line in _curves(ax))

    assert lowest(el.SPIN_MIRRORED) < -1.0          # β below the axis
    assert lowest(el.SPIN_ALONGSIDE) > lowest(el.SPIN_MIRRORED)


# ── the dialog ────────────────────────────────────────────────────────────
def _dialog(qapp, folder="", structure=None, stem="", state=None):
    from crystalline.ui.panels.dialog_state import restore
    from crystalline.ui.panels.electronic_dialog import ElectronicDialog

    dialog = ElectronicDialog(structure=structure, folder=folder, stem=stem)
    restore(dialog, state)
    dialog.show()
    qapp.processEvents()
    return dialog


def _ok(dialog):
    from PySide6.QtWidgets import QDialogButtonBox

    return dialog._buttons.button(QDialogButtonBox.Ok).isEnabled()


def test_the_dialog_is_wider_than_it_is_tall(qapp):
    """One column of five sections came out 452 × 866."""
    dialog = _dialog(qapp)
    assert dialog.width() > dialog.height()


def test_nothing_can_be_plotted_until_its_file_is_chosen(qapp):
    dialog = _dialog(qapp)
    assert not _ok(dialog)


@needs_silicon
def test_the_files_beside_the_output_are_filled_in(qapp):
    """A .d3 run is written beside its SCF output; that is where to look."""
    dialog = _dialog(qapp, folder=_SI, stem="si")
    dialog.mode.setCurrentIndex(dialog.mode.findData(el.BANDS_AND_DOS))
    band, dos, _options = dialog.request()
    assert band.endswith("BAND.DAT") and dos.endswith("DOSS.DAT")
    assert _ok(dialog)


@pytest.mark.skipif(not os.path.isdir(_MGO), reason="no MgO properties folder")
def test_several_runs_in_one_folder_are_all_offered_and_one_is_chosen(qapp):
    dialog = _dialog(qapp, folder=_MGO, stem="mgo")
    for kind in ("band", "dos"):
        combo = dialog._files[kind]
        assert combo.count() >= 2, kind
        assert dialog._path(kind), f"no {kind} file chosen"
    dialog.mode.setCurrentIndex(dialog.mode.findData(el.BANDS_AND_DOS))
    assert _ok(dialog)


@pytest.mark.skipif(not os.path.isdir(_MGO), reason="no MgO properties folder")
def test_the_bands_and_dos_chosen_come_from_one_calculation(qapp):
    """Ranked separately the newest of each was a B3LYP band structure beside a
    PBE DOS, drawn 0.75 eV apart. The Fermi level each carries says which SCF
    it came from."""
    band, dos = el.pair_files(*el.find_files(_MGO, stem="mgo"))
    assert el.describe_bands(band).efermi == pytest.approx(
        el.describe_dos(dos).efermi, abs=1e-3)
    dialog = _dialog(qapp, folder=_MGO, stem="mgo")
    assert dialog._bands_info.efermi == pytest.approx(dialog._dos_info.efermi, abs=1e-3)


def test_with_nothing_agreeing_the_best_of_each_is_kept():
    assert el.pair_files([], []) == (None, None)
    assert el.pair_files(["a.BAND"], []) == ("a.BAND", None)
    assert el.pair_files(["missing.BAND"], ["missing.DOSS"]) == ("missing.BAND", "missing.DOSS")


@pytest.mark.skipif(not os.path.isdir(_BE), reason="no beryllium run")
def test_fermi_levels_that_disagree_are_pointed_out(qapp):
    """Beryllium's BAND and DOSS put E_F 0.13 eV apart; side by side the DOS is
    shifted by exactly that, and the dialog should say so."""
    dialog = _dialog(qapp, folder=_BE)
    dialog.mode.setCurrentIndex(dialog.mode.findData(el.BANDS_AND_DOS))
    assert "differ by" in dialog._summary.text()
    dialog.mode.setCurrentIndex(dialog.mode.findData(el.BANDS))
    assert "differ by" not in dialog._summary.text()


@needs_silicon
def test_the_combined_legend_sits_beside_the_dos_not_on_it():
    figure = el.plot_electronic(_SI_BAND, _SI_DOS, el.ElectronicOptions(mode=el.BANDS_AND_DOS))
    legend = figure.axes[-1].get_legend()
    assert legend is not None
    anchor = legend.get_bbox_to_anchor().transformed(figure.axes[-1].transAxes.inverted())
    assert anchor.x0 >= 1.0, "outside the panel, to its right"


@needs_silicon
def test_choosing_another_file_from_the_list_loads_it(qapp):
    dialog = _dialog(qapp, folder=_SI, stem="si")
    combo = dialog._files["band"]
    other = next(i for i in range(combo.count()) if combo.itemData(i).endswith(".f25"))
    combo.setCurrentIndex(other)
    assert dialog._bands_info.path.endswith(".f25")


@needs_silicon
def test_the_last_folder_is_offered_when_the_output_has_none(qapp, tmp_path):
    _dialog(qapp, folder=_SI)
    dialog = _dialog(qapp, folder=str(tmp_path))
    assert dialog._path("band").endswith("BAND.DAT")


@needs_silicon
def test_the_path_labels_arrive_named(qapp):
    dialog = _dialog(qapp, folder=_SI, structure=_silicon())
    assert dialog.k_labels() == ["Γ", "X", "W", "L", "Γ"]


@needs_silicon
def test_typed_labels_of_the_wrong_length_fall_back_to_the_automatic_ones(qapp):
    """CRYSTALClear refuses a list of the wrong length, and marks the corners
    with k-distances when given none — neither is acceptable."""
    dialog = _dialog(qapp, folder=_SI, structure=_silicon())
    dialog._k_edit.setText("G X L")
    assert dialog.k_labels() == ["Γ", "X", "W", "L", "Γ"]
    assert "automatic" in dialog._summary.text()
    dialog._k_edit.setText("G X W L G")
    assert dialog.k_labels() == ["Γ", "X", "W", "L", "Γ"], "G is read as Γ"


@needs_silicon
def test_only_the_controls_a_plot_uses_are_live(qapp):
    dialog = _dialog(qapp, folder=_SI)
    dialog.mode.setCurrentIndex(dialog.mode.findData(el.DOS))
    assert not dialog._k_edit.isEnabled()
    assert dialog._table.isEnabled()
    assert dialog.overlay.isEnabled()
    assert not dialog.spin_down.isEnabled(), "silicon is spin-restricted"
    assert not dialog.beta_colour.isEnabled(), "and so has no spin-down bands"

    dialog.mode.setCurrentIndex(dialog.mode.findData(el.BANDS))
    assert dialog._k_edit.isEnabled()
    assert not dialog._table.isEnabled()

    dialog.mode.setCurrentIndex(dialog.mode.findData(el.BANDS_AND_DOS))
    assert not dialog.overlay.isEnabled(), "the combined view always overlays"


@needs_lif
def test_spin_down_colours_are_live_for_a_spin_polarised_run(qapp):
    dialog = _dialog(qapp, folder=_LIF_DIR)
    dialog.mode.setCurrentIndex(dialog.mode.findData(el.BANDS))
    assert dialog.beta_colour.isEnabled() and dialog.beta_style.isEnabled()


# ── the window fits the data ──────────────────────────────────────────────
@needs_silicon
def test_the_window_is_fitted_to_the_data(qapp):
    """It was −10…10 eV on a ±200 eV track for every file, whatever was in it."""
    dialog = _dialog(qapp, folder=_SI)
    info = dialog._bands_info
    low, high = dialog.energy_window()
    assert (low, high) == info.window
    bound_low, bound_high = dialog.energy_bounds()
    assert bound_low <= info.energy_span[0] and bound_high >= info.energy_span[1]
    assert bound_high - bound_low < 60, "the track spans the data, not ±200 eV"


@pytest.mark.skipif(not os.path.isdir(_BE), reason="no beryllium run")
def test_the_track_reaches_the_core_the_window_leaves_out(qapp):
    dialog = _dialog(qapp, folder=_BE)
    assert dialog.energy_bounds()[0] < -120
    assert dialog.energy_window()[0] > -25


@needs_silicon
def test_a_dos_plot_takes_the_window_its_d3_asked_for(qapp):
    dialog = _dialog(qapp, folder=_SI)
    dialog.mode.setCurrentIndex(dialog.mode.findData(el.DOS))
    low, high = dialog.energy_window()
    span = dialog._dos_info.energy_span
    assert low <= span[0] and high >= span[1]


@needs_silicon
def test_the_suggested_window_button_puts_it_back(qapp):
    dialog = _dialog(qapp, folder=_SI)
    dialog._energy[1].setValue(-2.0)
    dialog._fit_energy()
    assert dialog.energy_window() == dialog._bands_info.window


@needs_silicon
def test_switching_to_absolute_energies_carries_the_window_across(qapp):
    """-5..5 relative to E_F is E_F-5..E_F+5 in absolute terms; the boxes
    should say so rather than keep numbers that now mean something else."""
    dialog = _dialog(qapp, folder=_SI)
    efermi = dialog._bands_info.efermi
    bounds = dialog.energy_bounds()
    dialog._energy[1].setValue(-5.0)
    dialog._energy[2].setValue(5.0)
    dialog.reference.setCurrentIndex(dialog.reference.findData(el.ABSOLUTE))
    dialog._on_frame_chosen()
    # to the boxes' two decimals
    assert dialog.energy_window() == pytest.approx((-5.0 + efermi, 5.0 + efermi), abs=0.006)
    assert dialog.energy_bounds() == pytest.approx(
        (bounds[0] + efermi, bounds[1] + efermi), abs=1e-3), "and the track with it"


@needs_silicon
def test_switching_unit_converts_the_window_and_the_dos_range(qapp):
    dialog = _dialog(qapp, folder=_SI)
    dialog.mode.setCurrentIndex(dialog.mode.findData(el.BANDS_AND_DOS))
    dialog._energy[1].setValue(-10.0)
    dialog._energy[2].setValue(10.0)
    dos_high = dialog._dos_span[2].value()
    dialog.unit.setCurrentIndex(dialog.unit.findData("Hartree"))
    dialog._on_frame_chosen()
    assert dialog.energy_window()[1] == pytest.approx(10.0 / 27.211386, abs=1e-3)
    assert dialog._dos_span[2].value() == pytest.approx(dos_high * 27.211386, rel=1e-3)


@needs_silicon
def test_a_remembered_unit_gets_a_window_in_that_unit(qapp):
    """The window is fitted when the dialog is shown — after the remembered
    unit is back — so a Hartree user does not get eV numbers read as Hartree."""
    from crystalline.ui.panels.dialog_state import capture

    first = _dialog(qapp, folder=_SI)
    first.unit.setCurrentIndex(first.unit.findData("Hartree"))
    state = capture(first)
    dialog = _dialog(qapp, folder=_SI, state=state)
    window = dialog._bands_info.window
    assert dialog.energy_window() == pytest.approx(
        (window[0] / 27.211386, window[1] / 27.211386), abs=1e-3)


@needs_silicon
def test_the_dos_range_is_fitted_and_automatic_by_default(qapp):
    dialog = _dialog(qapp, folder=_SI)
    dialog.mode.setCurrentIndex(dialog.mode.findData(el.DOS))
    low, high = dialog._dos_span[1].value(), dialog._dos_span[2].value()
    assert low == 0.0 and dialog._dos_info.dos_max < high < 1.2 * dialog._dos_info.dos_max
    assert dialog.request()[2].dos_range is None
    assert not dialog._dos_span[2].isEnabled()
    dialog.dos_auto.setChecked(False)
    assert dialog._dos_span[2].isEnabled()
    assert dialog.request()[2].dos_range == (low, high)


@needs_ito
def test_a_mirrored_dos_range_runs_both_ways(qapp, tmp_path):
    import shutil

    shutil.copy(_ITO, tmp_path / "ito.DOSS")
    dialog = _dialog(qapp, folder=str(tmp_path))
    dialog.mode.setCurrentIndex(dialog.mode.findData(el.DOS))
    assert dialog._dos_span[1].value() < 0 < dialog._dos_span[2].value()


# ── what the dialog hands over ────────────────────────────────────────────
@needs_silicon
def test_the_request_carries_what_was_chosen(qapp):
    from PySide6.QtCore import Qt

    dialog = _dialog(qapp, folder=_SI, structure=_silicon())
    dialog.mode.setCurrentIndex(dialog.mode.findData(el.BANDS_AND_DOS))
    dialog._table.item(1, 0).setCheckState(Qt.Unchecked)
    dialog._table.item(0, 0).setText("Total")
    dialog._projection_colours[2].setColour("#112233")
    dialog._energy[1].setValue(-4.0)
    dialog._energy[2].setValue(4.0)
    dialog.band_colour.setColour("#445566")
    dialog.fermi_style.setCurrentIndex(dialog.fermi_style.findData("--"))
    dialog._title.setText("Silicon")

    band_path, dos_path, options = dialog.request()
    assert band_path.endswith("BAND.DAT") and dos_path.endswith("DOSS.DAT")
    assert options.mode == el.BANDS_AND_DOS
    assert options.energy_range == (-4.0, 4.0)
    assert options.projections == [1, 3]
    assert options.projection_labels == ["Total", "Projection 3"]
    assert options.projection_colours[1] == "#112233"
    assert options.k_labels == ["Γ", "X", "W", "L", "Γ"]
    assert options.band_colour == "#445566"
    assert options.fermi_style == "--"
    assert options.title == "Silicon"


@needs_silicon
def test_a_dos_plot_does_not_ask_for_the_band_file(qapp):
    dialog = _dialog(qapp, folder=_SI)
    dialog.mode.setCurrentIndex(dialog.mode.findData(el.DOS))
    band_path, dos_path, options = dialog.request()
    assert band_path is None and dos_path.endswith("DOSS.DAT")
    assert options.k_labels is None


@needs_silicon
def test_the_dialog_output_feeds_the_plotter(qapp):
    """The seam between dialog and figure, as in the spectra dialog."""
    dialog = _dialog(qapp, folder=_SI, structure=_silicon())
    dialog.mode.setCurrentIndex(dialog.mode.findData(el.BANDS_AND_DOS))
    figure = el.plot_electronic(*dialog.request())
    assert len(figure.axes) == 2
    ticks = [label.get_text() for label in figure.axes[0].get_xticklabels()]
    assert ticks == ["Γ", "X", "W", "L", "Γ"]
    assert figure.axes[0].get_ylim() == pytest.approx(dialog.energy_window())


def test_settings_are_remembered_but_files_and_windows_are_not(qapp):
    """A unit, a reference and the colours are tuned; which file was open, and
    the window that suited it, belong to that run."""
    from crystalline.ui.panels.dialog_state import capture

    state = capture(_dialog(qapp))
    for remembered in ("mode", "unit", "reference", "band_colour", "beta_colour",
                       "beta_style", "fermi_colour", "fermi_style", "fermi_width",
                       "overlay", "spin_down", "dos_auto", "linewidth",
                       "show_fermi", "legend"):
        assert remembered in state, remembered
    assert state["band_colour"] == ("colour", el.BAND_COLOUR)
    assert not any(isinstance(v, tuple) and k.startswith("_") and "energy" in k
                   for k, v in state.items())
    assert "_files" not in state and "_k_edit" not in state and "_title" not in state


def test_a_colour_comes_back_the_way_it_was_left(qapp):
    from crystalline.ui.panels.dialog_state import capture

    first = _dialog(qapp)
    first.fermi_colour.setColour("#abcdef")
    dialog = _dialog(qapp, state=capture(first))
    assert dialog.fermi_colour.colour() == "#abcdef"


# ── the menu ──────────────────────────────────────────────────────────────
def test_one_menu_entry_opens_the_dialog():
    """Bands and DOS left the one-click registry for the dialog, the way IR and
    Raman did for the vibrational spectra."""
    import inspect

    from crystalline.crystalio import plotting
    from crystalline.ui.main_window import MainWindow

    keys = {kind.key for kind in plotting.available_plots()}
    assert not {"electron_band", "electron_dos"} & keys
    source = inspect.getsource(MainWindow._open_electronic)
    assert "ElectronicDialog" in source and "plot_electronic" in source


def test_the_open_runs_files_come_first_by_their_fermi_level_not_their_names(tmp_path):
    """A folder shared by several systems. Which run a file came from is read
    from the Fermi level it records, never from being named after the output."""
    source = "/Users/davidemitoli/QMMC2026/OneElectronProperties/output/mgo_band_dat.BAND"
    if not os.path.isfile(source):
        pytest.skip("no MgO band file to build the fixture from")
    text = open(source).read()
    assert "# EFERMI (HARTREE)   -0.12156" in text

    ours = tmp_path / "urea_band.BAND"            # named after another run …
    ours.write_text(text)
    other = tmp_path / "mgo_band.BAND"            # … and this one after ours
    other.write_text(text.replace("# EFERMI (HARTREE)   -0.12156", "# EFERMI (HARTREE)   -0.38437"))
    os.utime(ours, (1e9 - 100, 1e9 - 100))
    os.utime(other, (1e9, 1e9))                   # and the newer

    bands, _doss = el.find_files(tmp_path, efermi=-0.12156 * 27.211386245988)

    assert os.path.basename(bands[0]) == "urea_band.BAND"
