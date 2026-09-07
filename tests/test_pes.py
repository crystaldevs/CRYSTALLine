"""Reading the anharmonic PES constants, and routing a cut to the right plot.

Three things are worth pinning. The cheap probe that enables the menu entry has
to agree with the real parse. The pairs come back ordered by how strongly the
two modes couple, because a run couples every pair it is given and a 36-mode
run gives 630 of them — the ordering *is* the picker. And a mode with no
harmonic frequency in the same file has no quadratic term, so it cannot be
drawn at all and must not be offered.
"""

import os

import pytest

from crystalline.crystalio.pes import (
    DIMENSIONS,
    PLOT_FUNCTIONS,
    QUANTITIES,
    REPRESENTATIONS,
    PESMode,
    PESPair,
    PESRun,
    has_pes,
    pes_run,
    representations,
)

_BANNER = "CALCULATION OF CUBIC AND QUARTIC TERMS"


def _pair(modei=7, modej=8, **kwargs):
    defaults = dict(frequencyi=1340.7, frequencyj=1340.7,
                    iij=0.0, ijj=0.0, iiij=0.0, ijjj=0.0, iijj=0.0)
    defaults.update(kwargs)
    return PESPair(modei=modei, modej=modej, **defaults)


# ── the cheap probe ─────────────────────────────────────────────────────
def test_a_file_with_the_banner_and_a_derivative_carries_a_pes(tmp_path):
    path = tmp_path / "run.out"
    path.write_text(f"  {_BANNER} OF THE POTENTIAL-ENERGY SURFACE (PES)\n"
                    " ETA(   7,   7,   7)      =     -4.6507E-08      -0.0003\n")

    assert has_pes(str(path))


def test_a_run_that_started_the_step_but_printed_nothing_does_not_count(tmp_path):
    """A calculation can reach ANHAPES and be cut short; offering the plot for
    it would open a dialog with no constants behind it."""
    path = tmp_path / "run.out"
    path.write_text(f"  {_BANNER} OF THE POTENTIAL-ENERGY SURFACE (PES)\n"
                    " MODE:     7 (FREQ:   1340.70 CM-1)\n")

    assert not has_pes(str(path))


def test_a_harmonic_run_does_not_count(tmp_path):
    path = tmp_path / "run.out"
    path.write_text(" FREQUENCIES COMPUTED ON A FRAGMENT\n MODES  1-  3\n")

    assert not has_pes(str(path))


@pytest.mark.parametrize("path", [None, "", "/nonexistent/run.out"])
def test_no_path_and_no_file_are_not_errors(path):
    assert not has_pes(path)


def test_a_missing_file_yields_no_run():
    assert pes_run("/nonexistent/run.out") == (None, None)


# ── the registry the dialog is built from ───────────────────────────────
def test_the_cuts_and_the_maps_are_named_by_a_key_each():
    assert {key for _label, key in DIMENSIONS} == {"1D", "2D"}
    assert {key for _label, key in QUANTITIES} == {"anharmonic", "coupling", "total"}


def test_every_representation_names_a_plot_function():
    assert {key for _label, key in REPRESENTATIONS} == set(PLOT_FUNCTIONS)


def test_only_the_representations_crystalclear_can_draw_are_offered(monkeypatch):
    """The surface arrived after the map; an older build has to offer one
    choice rather than an entry that fails once it is picked."""
    import CRYSTALClear.plot as CCplt

    monkeypatch.delattr(CCplt, "plot_cry_pes_3D", raising=False)
    assert {key for _label, key in representations()} == {"map"}


# ── how strongly a pair couples ─────────────────────────────────────────
def test_coupling_strength_is_what_the_terms_are_worth_at_unit_displacement():
    """Each derivative divided by its Taylor factor, which is what it
    contributes to the surface at ξ = (1, 1)."""
    pair = _pair(iij=2.0, ijj=4.0, iiij=6.0, ijjj=12.0, iijj=8.0)

    assert pair.strength == pytest.approx(1.0 + 2.0 + 1.0 + 2.0 + 2.0)


def test_terms_of_opposite_sign_do_not_cancel_into_an_uncoupled_pair():
    """They cancel at one corner of the map and add at the next; a pair that
    strong belongs at the top of the list either way."""
    assert _pair(iij=100.0, ijj=-100.0).strength == pytest.approx(100.0)


def test_a_pair_knows_which_modes_it_involves():
    pair = _pair(modei=7, modej=12)

    assert pair.involves(7) and pair.involves(12)
    assert not pair.involves(8)


# ── the description the dialog is built from ────────────────────────────
def _run(pairs=(), ntriplets=0):
    modes = tuple(PESMode(mode=m, frequency=1000.0 + m, eta3=-1.0, eta4=2.0)
                  for m in (7, 8, 12))
    return PESRun(modes=modes, pairs=pairs, ntriplets=ntriplets)


def test_the_summary_reads_as_a_description_of_the_run():
    summary = _run(pairs=(_pair(),), ntriplets=560).summary

    assert "3 modes scanned" in summary
    assert "1 coupled pairs" in summary
    assert "560 triplets (not plotted)" in summary
    assert "1007–1012 cm⁻¹" in summary


def test_a_run_with_no_couples_does_not_mention_pairs():
    assert "pair" not in _run().summary


def test_pairs_can_be_narrowed_to_one_mode():
    """The whole point of the filter: 630 pairs, four of which involve the mode
    being looked at."""
    run = _run(pairs=(_pair(7, 8), _pair(7, 12), _pair(8, 12)))

    assert len(run.pairs_with()) == 3
    assert {(p.modei, p.modej) for p in run.pairs_with(12)} == {(7, 12), (8, 12)}
    assert run.pairs_with(99) == ()


def test_a_mode_can_be_looked_up_by_its_crystal_index():
    run = _run()

    assert run.mode(12).frequency == pytest.approx(1012.0)
    assert run.mode(99) is None


# ── against a real run, when one is at hand ─────────────────────────────
_REAL = os.path.expanduser(
    "~/Desktop/PyCrystal/anharmonic_freq/CH4_anarmonic.out")


@pytest.mark.skipif(not os.path.isfile(_REAL),
                    reason="the reference ANHAPES run is not on this machine")
def test_a_real_run_is_described_and_drawn():
    pytest.importorskip("CRYSTALClear")
    from crystalline.crystalio.pes import plot_pes

    assert has_pes(_REAL)
    run, out = pes_run(_REAL)

    # CH4: nine modes, every pair of them coupled, no triplets.
    assert len(run.modes) == 9
    assert len(run.pairs) == 36
    assert run.ntriplets == 0
    assert run.mode(7).frequency == pytest.approx(1340.70, abs=0.05)

    # Strongest first, and the strongest of these is a stretch-stretch pair.
    assert run.pairs[0].strength >= run.pairs[-1].strength
    assert run.pairs[0].strength > 10 * run.pairs[-1].strength

    curve = plot_pes(out, dimension="1D", mode=12, levels=True, nstates=4)
    assert curve.axes and curve.axes[0].get_xlabel()

    surface = plot_pes(out, dimension="2D", modei=13, modej=15)
    assert surface.axes


@pytest.mark.skipif(not os.path.isfile(_REAL),
                    reason="the reference ANHAPES run is not on this machine")
def test_a_pair_can_be_drawn_as_a_surface_instead_of_a_map():
    """Same numbers, different axes: the 3D one has to come back with a real
    Axes3D, or the figure lands in the dock as an empty flat plot."""
    pytest.importorskip("CRYSTALClear")
    from crystalline.crystalio.pes import plot_pes

    _run_, out = pes_run(_REAL)

    flat = plot_pes(out, dimension="2D", representation="map",
                    modei=13, modej=15)
    solid = plot_pes(out, dimension="2D", representation="surface",
                     modei=13, modej=15)

    assert not hasattr(flat.axes[0], "plot_surface")
    assert hasattr(solid.axes[0], "plot_surface")


@pytest.mark.skipif(not os.path.isfile(_REAL),
                    reason="the reference ANHAPES run is not on this machine")
def test_a_one_mode_cut_ignores_the_representation():
    """It is a curve either way; picking a surface for it must not reach for
    the two-mode plot function with one mode."""
    pytest.importorskip("CRYSTALClear")
    from crystalline.crystalio.pes import plot_pes

    _run_, out = pes_run(_REAL)
    figure = plot_pes(out, dimension="1D", representation="surface", mode=7)

    assert not hasattr(figure.axes[0], "plot_surface")


def test_an_unknown_representation_is_refused():
    with pytest.raises(ValueError, match="representation"):
        pes_run  # keep the import used
        from crystalline.crystalio.pes import plot_pes

        plot_pes(object(), dimension="2D", representation="hologram",
                 modei=1, modej=2)


@pytest.mark.skipif(not os.path.isfile(_REAL),
                    reason="the reference ANHAPES run is not on this machine")
def test_a_pair_reads_the_same_either_way_round():
    """CRYSTAL writes each pair once, lower index first; asking for it the other
    way round has to relabel the constants, not fail or transpose the surface."""
    pytest.importorskip("CRYSTALClear")
    import numpy as np
    from CRYSTALClear.plot import _pes_couple

    _run_, out = pes_run(_REAL)
    forward = _pes_couple(out, 7, 10)
    backward = _pes_couple(out, 10, 7)

    # IIJ <-> IJJ and IIIJ <-> IJJJ swap, IIJJ is symmetric.
    assert np.allclose(forward, (backward[1], backward[0],
                                 backward[3], backward[2], backward[4]))
