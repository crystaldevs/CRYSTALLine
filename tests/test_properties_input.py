"""The PROPERTIES (.d3) builder, and the spin and functional additions to .d12.

Pinned against real, working decks from the user's own CRYSTAL runs and the
CRYSTAL23 manual's record definitions (§14.3 BAND, §14.6 DOSS, §14.12 ORBITALS)
rather than against what the code happens to emit.
"""

import pytest
from ase.build import bulk

from crystalline.core.properties_input import (
    BandOptions,
    DossOptions,
    CoopOptions,
    EmdOptions,
    Grid3DOptions,
    OrbitalsOptions,
    PropertiesInputError,
    PropertiesSpec,
    XrdOptions,
    band_path,
    band_shrink,
    build_properties_input,
)
from crystalline.core.structure import Structure
from crystalline.ui.panels.band_path_editor import CONVENTIONAL_NOTE


def _mgo() -> Structure:
    return Structure.from_ase(bulk("MgO", "rocksalt", a=4.21))


def _lines(spec, structure=None) -> list:
    return build_properties_input(structure or _mgo(), spec).splitlines()


# ── ordering, which is a hard constraint rather than a preference ─────────
def test_band_comes_before_newk_and_doss_after_it():
    """The manual is explicit: "to compute density of states and bands, the
    sequence must be: BAND - NEWK - DOSS", and NEWK BAND DOSS stops the run with
    "NEWK MUST BE CALLED BEFORE DOSS". Emitting them in form order would be a
    deck that parses and then fails."""
    lines = _lines(PropertiesSpec(band=BandOptions(enabled=True),
                                  doss=DossOptions(enabled=True)))
    assert lines.index("BAND") < lines.index("NEWK") < lines.index("DOSS")


def test_newk_is_always_written():
    """It computes the eigenvectors the rest of the run reads and is the first
    step of essentially every properties deck. Writing it only when something
    was known to need it left it out of the common case — a band structure —
    where its absence is silent: the run just uses whatever the SCF left."""
    for spec in (PropertiesSpec(band=BandOptions(enabled=True)),
                 PropertiesSpec(ppan=True),
                 PropertiesSpec(doss=DossOptions(enabled=True)),
                 PropertiesSpec(orbitals=OrbitalsOptions(enabled=True)),
                 PropertiesSpec(xrd=XrdOptions(enabled=True))):
        lines = _lines(spec)
        assert "NEWK" in lines
        assert lines[lines.index("NEWK") + 2] == "1 0"   # IFE, IPRINT


def test_a_deck_that_asks_for_nothing_is_refused():
    with pytest.raises(PropertiesInputError, match="at least one"):
        build_properties_input(_mgo(), PropertiesSpec())


# ── BAND ──────────────────────────────────────────────────────────────────
def test_the_band_record_matches_the_manual():
    """NLINE ISS NSUB INZB IFNB IPLO LPR66, then NLINE coordinate records."""
    lines = _lines(PropertiesSpec(
        band=BandOptions(enabled=True, title="MgO", points=200,
                         first_band=1, last_band=26)))
    assert lines[0] == "BAND"
    assert lines[1].startswith("MgO (")            # title, max 72 chars
    nline, iss, nsub, inzb, ifnb, iplo, lpr = lines[2].split()
    assert (nsub, inzb, ifnb, iplo, lpr) == ("200", "1", "26", "1", "0")
    assert len(lines[3:3 + int(nline)]) == int(nline)
    for record in lines[3:3 + int(nline)]:
        numbers = record.split()[:6]               # two 3-vectors of integers
        assert all(part.lstrip("-").isdigit() for part in numbers)
        # ...then the two point names, as CRYSTAL's own tutorial decks write
        # them. The six integers are what it reads: beryllium's BAND.DAT ticks
        # are exactly the coordinates of its labelled deck, so a record carrying
        # names was read for its numbers.
        names = record.split()[6:]
        assert names == [] or (len(names) == 2 and all(n.isalnum() for n in names))


def test_a_deck_says_which_corners_its_path_visits():
    """Written into the deck, the names survive into the plot: a band file
    records coordinates only, and for some paths not even those — beryllium's
    (-2,4,3) overflows its field and comes out (*,4,3)."""
    from crystalline.crystalio.electronic import _band_blocks

    # A connected walk: a conventional path can jump (fcc runs ... L K | U X),
    # and a deck whose corners are not one chain is not read back for labels —
    # its records cannot be lined up with the ticks in the data file.
    route = [("G", "X"), ("X", "W"), ("W", "L"), ("L", "G")]
    points = {"G": (0, 0, 0), "X": (0.5, 0, 0.5),      # MgO is face-centred cubic
              "W": (0.5, 0.25, 0.75), "L": (0.5, 0.5, 0.5)}
    lines = _lines(PropertiesSpec(band=BandOptions(
        enabled=True, labels=route,
        segments=[(points[a], points[b]) for a, b in route])))
    deck, = _band_blocks("\n".join(lines), "test.d3")
    assert deck.labels == ("G", "X", "W", "L", "G")
    assert len(deck.corners) == len(deck.labels)


def test_the_path_is_whole_integers_over_the_shrinking_factor():
    """CRYSTAL reads each endpoint as I/ISS along a reciprocal vector, so a
    coordinate that is not a whole number of 1/ISS silently lands somewhere
    else in the zone."""
    _labels, segments = band_path(_mgo())
    shrink = band_shrink(segments)
    for start, end in segments:
        for value in (*start, *end):
            assert abs(value * shrink - round(value * shrink)) < 1e-9


def test_a_path_that_is_not_made_of_special_points_is_refused():
    """Fraction.limit_denominator rounds, so an arbitrary coordinate comes back
    looking like a special point — 0.137 becomes 3/22. Without a check on the
    rounding, the deck would be written with a k-point somewhere else in the
    zone and nothing would say so."""
    odd = [((0.0, 0.0, 0.0), (0.137, 0.0, 0.0))]
    with pytest.raises(PropertiesInputError, match="special points"):
        band_shrink(odd)

    # ...while a genuine special point on a fine denominator is still fine
    assert band_shrink([((0.0, 0.0, 0.0), (0.375, 0.375, 0.75))]) == 8


def test_a_band_range_the_wrong_way_round_is_refused():
    with pytest.raises(PropertiesInputError, match="below the first"):
        build_properties_input(_mgo(), PropertiesSpec(
            band=BandOptions(enabled=True, first_band=10, last_band=2)))


def test_a_band_structure_needs_a_lattice():
    from ase.build import molecule

    with pytest.raises(PropertiesInputError, match="periodic"):
        band_path(Structure.from_ase(molecule("H2O")))


def test_the_title_says_where_the_path_goes_including_its_breaks():
    """A conventional path is not always one connected walk; a plain join of the
    segment ends loses where it jumped."""
    lines = _lines(PropertiesSpec(band=BandOptions(enabled=True, title="MgO")))
    assert "|" in lines[1], "a discontinuous path must show its break"
    assert lines[1].startswith("MgO (G ")


# ── DOSS ──────────────────────────────────────────────────────────────────
def test_the_doss_record_matches_a_working_deck():
    """Against mgo_DOSS.d3: '2 300 -1 -1 2 12 0', the window, then one record
    per projection with a negative count meaning "all AOs of these atoms"."""
    lines = _lines(PropertiesSpec(doss=DossOptions(
        enabled=True, points=300, window=(-0.7, 0.8), projections=((1,), (2,)))))
    start = lines.index("DOSS")
    assert lines[start + 1] == "2 300 -1 -1 2 12 0"
    assert lines[start + 2] == "-0.7 0.8"
    assert lines[start + 3:start + 5] == ["-1 1", "-1 2"]


def test_a_doss_projection_over_several_atoms_is_one_record():
    lines = _lines(PropertiesSpec(doss=DossOptions(enabled=True,
                                                   projections=((1, 2, 3),))))
    assert "-3 1 2 3" in lines


def test_an_empty_doss_window_is_refused():
    with pytest.raises(PropertiesInputError, match="window is empty"):
        build_properties_input(_mgo(), PropertiesSpec(
            doss=DossOptions(enabled=True, window=(0.5, 0.5))))


def test_doss_rejects_more_legendre_polynomials_than_crystal_allows():
    with pytest.raises(PropertiesInputError, match="25"):
        build_properties_input(_mgo(), PropertiesSpec(
            doss=DossOptions(enabled=True, npol=30)))


# ── ORBITALS ──────────────────────────────────────────────────────────────
def test_the_orbitals_block_matches_the_manual():
    """ORBITALS / filename / ICAR / ILOC / END — ICAR=1 for a periodic system,
    ILOC=0 for Bloch functions. Checked against the manual's own corundum
    example and the .d3 files beside the reference run."""
    lines = _lines(PropertiesSpec(orbitals=OrbitalsOptions(
        enabled=True, name="MgO-COs")))
    start = lines.index("ORBITALS")
    assert lines[start:start + 5] == ["ORBITALS", "MgO-COs", "1", "0", "END"]


def test_a_molecule_gets_cartesian_orbitals():
    from ase.build import molecule

    water = Structure.from_ase(molecule("H2O"))
    lines = build_properties_input(water, PropertiesSpec(
        orbitals=OrbitalsOptions(enabled=True, name="water"))).splitlines()
    start = lines.index("ORBITALS")
    assert lines[start + 2] == "0", "ICAR=0 is the default for molecules"


def test_an_orbital_name_that_cannot_be_a_file_name_is_refused():
    """It becomes the stem of the files CRYSTAL writes, and the app finds them
    by that stem."""
    for bad in ("", "my orbitals", "runs/orbitals"):
        with pytest.raises(PropertiesInputError):
            build_properties_input(_mgo(), PropertiesSpec(
                orbitals=OrbitalsOptions(enabled=True, name=bad)))


def test_the_deck_ends_with_end():
    text = build_properties_input(_mgo(), PropertiesSpec(ppan=True))
    assert text.splitlines()[-1] == "END"
    assert text.endswith("\n")


def test_extra_keywords_are_passed_through():
    lines = _lines(PropertiesSpec(ppan=True, extra_keywords="ECHG\n0\n95"))
    assert lines[-4:] == ["ECHG", "0", "95", "END"]


# ── the .d12 additions ────────────────────────────────────────────────────
def test_spinlock_and_atomspin_match_a_working_input():
    """Against the user's own brownmillerite deck: both sit after SHRINK and
    outside the DFT block, and ATOMSPIN is a count then one pair a line."""
    from crystalline.core.crystal_input import (
        CrystalInputSpec, MethodOptions, ScfOptions, build_input,
    )

    text = build_input(_mgo(), CrystalInputSpec(
        method=MethodOptions(kind="DFT", functional="PBEXC"),
        scf=ScfOptions(spin_polarized=True, spinlock=(0, 50),
                       atomspin=((1, 1), (2, -1)))))
    lines = text.splitlines()
    assert lines.index("SPIN") < lines.index("END") < lines.index("SPINLOCK")
    start = lines.index("SPINLOCK")
    assert lines[start + 1] == "0 50"
    assert lines[start + 2:start + 6] == ["ATOMSPIN", "2", "1 1", "2 -1"]
    assert lines.index("SHRINK") < start


@pytest.mark.parametrize("scf_kwargs, match", [
    (dict(spin_polarized=False, spinlock=(0, 50)), "spin-polarised"),
    (dict(spin_polarized=False, atomspin=((1, 1),)), "spin-polarised"),
    (dict(spin_polarized=True, atomspin=((99, 1),)), "has 2 atoms"),
    (dict(spin_polarized=True, atomspin=((1, 3),)), r"\+1 or -1"),
    (dict(spin_polarized=True, atomspin=((1, 1), (1, -1))), "more than once"),
    (dict(spin_polarized=True, spinlock=(0, 0)), "positive number"),
])
def test_spin_settings_that_crystal_would_reject_are_caught_here(scf_kwargs, match):
    """An ATOMSPIN label outside the structure is the dangerous one: CRYSTAL
    numbers atoms from 1, so an off-by-one puts the moment on the wrong atom and
    converges to the wrong magnetic state without complaining."""
    from crystalline.core.crystal_input import (
        CrystalInputError, CrystalInputSpec, ScfOptions, build_input,
    )

    with pytest.raises(CrystalInputError, match=match):
        build_input(_mgo(), CrystalInputSpec(scf=ScfOptions(**scf_kwargs)))


def test_spin_settings_are_only_written_when_spin_is_on():
    from crystalline.core.crystal_input import CrystalInputSpec, ScfOptions, build_input

    text = build_input(_mgo(), CrystalInputSpec(scf=ScfOptions(spin_polarized=False)))
    assert "SPINLOCK" not in text and "ATOMSPIN" not in text and "SPIN\n" not in text


def test_the_functional_groups_cover_the_flat_list_and_name_pbe():
    """The reason for grouping: PBE's stand-alone keyword is PBEXC, so a flat
    list of keywords hid the most common functional behind a spelling nobody
    searches for."""
    from crystalline.core.crystal_input import (
        COMMON_FUNCTIONALS, FUNCTIONAL_ALIASES, FUNCTIONAL_GROUPS,
    )

    flat = [k for _group, entries in FUNCTIONAL_GROUPS for k, _why in entries]
    assert flat == list(COMMON_FUNCTIONALS)
    assert len(set(flat)) == len(flat), "a functional is listed twice"
    assert "PBE" in flat
    for alias, keyword in FUNCTIONAL_ALIASES.items():
        assert keyword in flat, f"{alias} points at {keyword}, which is not offered"


# ── the supercell bound ───────────────────────────────────────────────────
def test_the_supercell_dialog_is_not_capped_at_twelve():
    """The old per-axis cap of 12 was arbitrary. What costs anything is the
    total atom count — rotation is display-locked and independent of size — and
    a slab or a polymer legitimately wants a large number down one axis."""
    from crystalline.ui import main_window as mw

    assert mw._MAX_SUPERCELL_REPEAT > 12
    assert mw._SLOW_SUPERCELL_ATOMS > 1000

    import inspect

    source = inspect.getsource(mw.MainWindow._open_supercell_dialog)
    assert "_MAX_SUPERCELL_REPEAT" in source, "the dialog must use the constant"
    assert "setRange(1, 12)" not in source
    # ...and it asks rather than refuses when the result is large
    assert "_SLOW_SUPERCELL_ATOMS" in source and "question" in source


# ── the properties added after the first cut ─────────────────────────────
def test_ech3_and_pot3_match_the_manual():
    """ECH3 takes the point count alone; POT3 adds the penetration tolerance
    (manual §14.8 and §14.14). Checked against mgo_density.d3, which is
    NEWK then 'ECH3 / 5'."""
    lines = _lines(PropertiesSpec(
        charge_density=Grid3DOptions(enabled=True, points=100),
        potential=Grid3DOptions(enabled=True, points=80, tolerance=5)))
    ech3 = lines.index("ECH3")
    assert lines[ech3 + 1] == "100"
    assert lines[ech3 + 2] == "POT3", "ECH3 takes no tolerance record"
    pot3 = lines.index("POT3")
    assert lines[pot3 + 1:pot3 + 3] == ["80", "5"]
def test_pato_is_written_with_its_record():
    lines = _lines(PropertiesSpec(pato=True))
    start = lines.index("PATO")
    assert lines[start + 1] == "0 0"


def test_the_functional_menu_shows_keywords_alone():
    """The grouping stays — it is what turns fifty keywords into five short
    lists — but spelling out what each functional is made the menu long and
    hard to scan."""
    import inspect

    from crystalline.ui.panels import input_builder

    source = inspect.getsource(input_builder._fill_functionals)
    assert "combo.addItem(keyword, keyword)" in source
    assert "{description}" not in source, "the row must be the keyword alone"


# ── COOP/COHP, EMDL, XRDSPEC, LOCALI ─────────────────────────────────────
def test_coop_writes_two_records_per_interaction():
    """An interaction is between two groups of atoms, so it takes two records —
    the same negative-count convention DOSS uses for its projections."""
    lines = _lines(PropertiesSpec(coop=CoopOptions(
        enabled=True, interactions=(((1,), (2, 3)),))))
    start = lines.index("COOP")
    assert lines[start + 1].startswith("1 300 ")
    assert lines[start + 2:start + 4] == ["-1 1", "-2 2 3"]


def test_cohp_is_the_same_record_under_another_keyword():
    lines = _lines(PropertiesSpec(coop=CoopOptions(
        enabled=True, hamiltonian=True, interactions=(((1,), (2,)),))))
    assert "COHP" in lines and "COOP" not in lines


def test_coop_needs_an_interaction_and_two_sides():
    with pytest.raises(PropertiesInputError, match="at least one interaction"):
        build_properties_input(_mgo(), PropertiesSpec(coop=CoopOptions(enabled=True)))
    with pytest.raises(PropertiesInputError, match="Both sides"):
        build_properties_input(_mgo(), PropertiesSpec(coop=CoopOptions(
            enabled=True, interactions=(((1,), ()),))))


def test_emdl_lists_its_directions_and_closes_with_no_projections():
    lines = _lines(PropertiesSpec(emd=EmdOptions(
        enabled=True, directions=((1, 0, 0), (1, 1, 0)), pmax=3.0, step=0.1)))
    start = lines.index("EMDL")
    assert lines[start + 1] == "2 3 0.1 2 0"
    assert lines[start + 2:start + 4] == ["1 0 0", "1 1 0"]
    assert lines[start + 4] == "0 0"   # no orbital and no band projections


def test_emdl_takes_at_most_ten_directions():
    with pytest.raises(PropertiesInputError, match="at most 10"):
        build_properties_input(_mgo(), PropertiesSpec(emd=EmdOptions(
            enabled=True, directions=tuple((i, 0, 0) for i in range(11)))))


def test_xrdspec_writes_its_single_record():
    lines = _lines(PropertiesSpec(xrd=XrdOptions(
        enabled=True, max_index=6, wavelength=1.5406, debye_waller=1.0)))
    assert lines[lines.index("XRDSPEC") + 1] == "6 1.5406 1"


def test_wannier_orbitals_get_localised_first():
    """ILOC=1 needs LOCALI to have run, and the manual's own example puts it
    between NEWK and ORBITALS. Asking for Wannier orbitals without it is a deck
    that runs and gives canonical orbitals instead."""
    lines = _lines(PropertiesSpec(orbitals=OrbitalsOptions(
        enabled=True, name="w", wannier=True)))
    assert lines.index("LOCALI") < lines.index("ORBITALS")
    assert lines[lines.index("ORBITALS") + 3] == "1"    # ILOC


def test_localise_is_not_written_twice():
    """The core still takes ``localise`` on its own — LOCALI without plotting is
    a legitimate run — but asking for it alongside Wannier orbitals must not
    emit the keyword twice."""
    lines = _lines(PropertiesSpec(
        localise=True,
        orbitals=OrbitalsOptions(enabled=True, name="w", wannier=True)))
    assert lines.count("LOCALI") == 1


def test_the_dialog_has_one_control_for_localise(qapp=None):
    """Two ticks that write the same keyword is two ways to be inconsistent.
    The Wannier one does strictly more — it also sets ILOC=1 — so it is the one
    that stays."""
    from PySide6.QtWidgets import QApplication, QCheckBox

    from crystalline.ui.panels.properties_builder import PropertiesBuilderDialog

    QApplication.instance() or QApplication([])
    dialog = PropertiesBuilderDialog(_mgo())
    localise = [c for c in dialog.findChildren(QCheckBox)
                if "LOCALI" in c.text() or "Wannier" in c.text()]
    assert len(localise) == 1, [c.text() for c in localise]


def test_the_two_builders_open_at_the_same_size():
    """They are the same kind of dialog doing the same kind of job; opening at
    different shapes made them look unrelated."""
    from PySide6.QtWidgets import QApplication

    from crystalline.ui.panels.input_builder import InputBuilderDialog
    from crystalline.ui.panels.properties_builder import PropertiesBuilderDialog

    QApplication.instance() or QApplication([])
    structure = _mgo()
    assert (PropertiesBuilderDialog(structure).size()
            == InputBuilderDialog(structure).size())


def test_pbe_is_the_keyword_offered():
    """The manual lists PBEXC as the stand-alone keyword, but PBE is what the
    code takes and what everyone writes. PBEXC still resolves, for anyone
    copying from an older deck."""
    from crystalline.core.crystal_input import COMMON_FUNCTIONALS, FUNCTIONAL_ALIASES

    assert "PBE" in COMMON_FUNCTIONALS
    assert "PBEXC" not in COMMON_FUNCTIONALS
    assert FUNCTIONAL_ALIASES["PBEXC"] == "PBE"


def test_a_checkable_group_box_has_a_visible_indicator():
    """The .d3 builder is built from checkable group boxes, and the theme styled
    QCheckBox::indicator without QGroupBox::indicator — so every one of them had
    no box at all when unchecked and a bare floating tick when checked."""
    from crystalline.ui import theme

    for palette in (theme.LIGHT, theme.DARK):
        sheet = theme.stylesheet(palette)
        assert "QGroupBox::indicator" in sheet
        assert "QGroupBox::indicator:checked" in sheet
        # and a disabled-but-checked box keeps a filled ground, or the white
        # tick is drawn on the light theme's pale one and disappears
        assert "QGroupBox::indicator:checked:disabled" in sheet


# ── the editable band path ────────────────────────────────────────────────
def _builder():
    from PySide6.QtWidgets import QApplication

    from crystalline.ui.panels.properties_builder import PropertiesBuilderDialog

    QApplication.instance() or QApplication([])
    return PropertiesBuilderDialog(_mgo())


def _editable_builder():
    """A builder holding the conventional walk, with the tick off to edit it.

    The editor starts empty — a path is chosen, not assumed — and every editor
    refuses while the tick is set. Ticking it fills the list, and unticking
    leaves those segments there to be edited, which is how someone starts from
    the conventional walk and changes it.
    """
    dialog = _builder()
    dialog._path_editor.conventional.setChecked(True)
    dialog._path_editor.conventional.setChecked(False)
    return dialog


def test_the_path_starts_as_the_conventional_walk():
    """The default answer, ready to edit or to replace in the path builder."""
    dialog = _builder()
    assert dialog._path_editor.conventional.isChecked()
    assert dialog._path_editor._list.count() > 1
    assert dialog._path_editor.note.text() == CONVENTIONAL_NOTE


def test_an_emptied_path_is_refused_rather_than_quietly_refilled():
    """Once the tick is off the path is the user's, so an empty one is a
    mistake to report — not licence to write the conventional walk instead."""
    with pytest.raises(PropertiesInputError, match="No band path chosen"):
        build_properties_input(_mgo(), PropertiesSpec(
            band=BandOptions(enabled=True, conventional=False)))


def test_the_path_can_be_reset_to_the_conventional_one():
    dialog = _editable_builder()
    conventional = dialog._path_editor._list.count()
    assert conventional > 1
    dialog._path_editor._list.setCurrentRow(0)
    dialog._path_editor.remove_segment()
    assert dialog._path_editor._list.count() == conventional - 1
    dialog._path_editor.reset()
    assert dialog._path_editor._list.count() == conventional


def test_a_sub_path_is_expressible():
    """The whole point of A: the conventional walk is a starting point, not the
    only thing you can ask for."""
    dialog = _editable_builder()
    # Bounded, not `while count > 1`: a removal that silently does nothing would
    # spin that loop forever instead of failing the test.
    for _ in range(dialog._path_editor._list.count() - 1):
        dialog._path_editor._list.setCurrentRow(1)
        dialog._path_editor.remove_segment()
    assert dialog._path_editor._list.count() == 1
    lines = dialog._preview.toPlainText().splitlines()
    assert lines[2].split()[0] == "1", "one segment should mean NLINE = 1"


def test_a_custom_point_the_conventional_path_never_visits_can_be_added():
    dialog = _editable_builder()
    dialog._path_editor._from.setEditText("X")
    dialog._path_editor._to.setEditText("1/2 1/4 3/4")
    dialog._path_editor.add_typed_segment()
    labels = [dialog._path_editor._list.item(i).text() for i in range(dialog._path_editor._list.count())]
    assert any("(0.5 0.25 0.75)" in text for text in labels)
    segments = dialog._path_editor.segments()
    assert (0.5, 0.25, 0.75) in [end for _start, end in segments]


def test_an_unreadable_endpoint_is_refused_with_a_reason():
    """Not quietly rounded to the origin, which would be a band structure of a
    path nobody asked for."""
    dialog = _editable_builder()
    before = dialog._path_editor._list.count()
    dialog._path_editor._from.setEditText("nonsense")
    dialog._path_editor._to.setEditText("G")
    dialog._path_editor.add_typed_segment()
    assert dialog._path_editor._list.count() == before
    assert "not a point on this lattice" in dialog._path_editor.note.text()


def test_segments_can_be_reordered():
    dialog = _editable_builder()
    first = dialog._path_editor._list.item(0).text()
    dialog._path_editor._list.setCurrentRow(0)
    dialog._path_editor.move_segment(1)
    assert dialog._path_editor._list.item(1).text() == first
    assert dialog._path_editor._list.currentRow() == 1


def test_moving_past_the_ends_does_nothing():
    dialog = _editable_builder()
    rows = [dialog._path_editor._list.item(i).text() for i in range(dialog._path_editor._list.count())]
    dialog._path_editor._list.setCurrentRow(0)
    dialog._path_editor.move_segment(-1)
    dialog._path_editor._list.setCurrentRow(dialog._path_editor._list.count() - 1)
    dialog._path_editor.move_segment(1)
    assert [dialog._path_editor._list.item(i).text()
            for i in range(dialog._path_editor._list.count())] == rows


def test_a_typed_point_may_be_a_label_in_any_case_or_three_numbers():
    from crystalline.ui.panels.band_path_editor import read_kpoint as _read_kpoint

    points = {"G": (0.0, 0.0, 0.0), "X": (0.5, 0.0, 0.5)}
    assert _read_kpoint("X", points) == ("X", (0.5, 0.0, 0.5))
    assert _read_kpoint("  x ", points)[0] == "X"
    assert _read_kpoint("1/2 0 1/2", points)[1] == (0.5, 0.0, 0.5)
    assert _read_kpoint("0.5, 0, 0.5", points)[1] == (0.5, 0.0, 0.5)
    for bad in ("", "Z", "0.5 0.5", "a b c"):
        with pytest.raises(ValueError):
            _read_kpoint(bad, points)


def test_the_tick_makes_every_path_editor_refuse():
    """The tick is the authority, not just a greying-out.

    The editors are disabled while it is set, so this is unreachable by mouse —
    but an edit that slipped through used to be undone by the next refresh,
    which turned "remove until one segment is left" into an endless loop.
    """
    dialog = _builder()
    dialog._path_editor.conventional.setChecked(True)
    before = [dialog._path_editor._list.item(i).text() for i in range(dialog._path_editor._list.count())]
    dialog._path_editor._list.setCurrentRow(0)
    dialog._path_editor.remove_segment()
    dialog._path_editor.move_segment(1)
    dialog._path_editor.add_typed_segment()
    after = [dialog._path_editor._list.item(i).text() for i in range(dialog._path_editor._list.count())]
    assert after == before


def test_an_edited_path_survives_a_refresh():
    """What the removed self-repair used to eat."""
    dialog = _editable_builder()
    dialog._path_editor._list.setCurrentRow(0)
    dialog._path_editor.remove_segment()
    shortened = dialog._path_editor._list.count()
    dialog._refresh()
    assert dialog._path_editor._list.count() == shortened
