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
    OrbitalsOptions,
    PropertiesInputError,
    PropertiesSpec,
    band_path,
    band_shrink,
    build_properties_input,
)
from crystalline.core.structure import Structure


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


def test_newk_is_written_only_when_something_reads_it():
    """BAND and PPAN do not need it; DOSS and ORBITALS do."""
    assert "NEWK" not in _lines(PropertiesSpec(band=BandOptions(enabled=True)))
    assert "NEWK" not in _lines(PropertiesSpec(ppan=True))
    assert "NEWK" in _lines(PropertiesSpec(doss=DossOptions(enabled=True)))
    assert "NEWK" in _lines(PropertiesSpec(orbitals=OrbitalsOptions(enabled=True)))


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
        assert len(record.split()) == 6            # two 3-vectors of integers
        assert all(part.lstrip("-").isdigit() for part in record.split())


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
    described = {k: why for _g, entries in FUNCTIONAL_GROUPS for k, why in entries}
    assert "PBE" in described["PBEXC"]
    assert FUNCTIONAL_ALIASES["PBE"] == "PBEXC"
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
