"""The CRYSTAL input-builder dialog: form → live preview wiring."""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pymatgen")

from PySide6.QtWidgets import QApplication  # noqa: E402
from ase.build import bulk  # noqa: E402

from crystalline.core.structure import Structure  # noqa: E402
from crystalline.ui.panels.input_builder import _TASKS, InputBuilderDialog  # noqa: E402


def _select_task(dlg, kind: str) -> None:
    """Pick a task by its CRYSTAL kind, so renaming a label cannot break a test.

    ``setCurrentText`` silently does nothing when the label does not match, which
    would leave the dialog on the previous task and assert against the wrong deck.
    """
    dlg._task.setCurrentIndex([k for _, k in _TASKS].index(kind))


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _nacl() -> Structure:
    return Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))


def _co() -> Structure:
    mol = Structure.empty()
    mol.add_atom("C", [0.0, 0.0, 0.0])
    mol.add_atom("O", [0.0, 0.0, 1.128])
    return mol


def test_preview_shows_a_crystal_deck_on_open(qapp):
    dlg = InputBuilderDialog(_nacl())
    text = dlg._preview.toPlainText()
    assert text.splitlines()[1] == "CRYSTAL"
    assert "225" in text  # symmetry-reduced by default
    assert dlg._save_btn.isEnabled()


def test_switching_to_hartree_fock_drops_the_dft_block(qapp):
    dlg = InputBuilderDialog(_nacl())
    dlg._method.setCurrentText("Hartree–Fock")
    text = dlg._preview.toPlainText()
    assert "DFT" not in text.splitlines()
    # functional/grid rows disable when there is no DFT block
    assert not dlg._functional.isEnabled()


def test_geometry_optimisation_defaults_to_full_relaxation(qapp):
    dlg = InputBuilderDialog(_nacl())
    dlg._task.setCurrentIndex(1)  # Geometry optimisation
    lines = dlg._preview.toPlainText().splitlines()
    assert lines.index("OPTGEOM") < lines.index("BASISSET")
    # cell + atoms is CRYSTAL's own default: no extra keyword is written
    assert "FULLOPTG" not in lines and "ATOMONLY" not in lines


def test_switching_to_atoms_only_emits_atomonly(qapp):
    dlg = InputBuilderDialog(_nacl())
    dlg._task.setCurrentIndex(1)  # Geometry optimisation
    dlg._opt_scope.setCurrentIndex(0)  # Atoms only
    assert "ATOMONLY" in dlg._preview.toPlainText().splitlines()


def test_freqcalc_enables_ir_option_and_writes_the_block(qapp):
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "FREQCALC")
    assert dlg._preopt.isEnabled()
    dlg._freq_ir.setChecked(True)
    lines = dlg._preview.toPlainText().splitlines()
    assert lines[lines.index("FREQCALC") + 1] == "INTENS"


def test_elastcon_allows_preoptimisation(qapp):
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "ELASTCON")
    dlg._preopt.setChecked(True)
    lines = dlg._preview.toPlainText().splitlines()
    i = lines.index("ELASTCON")
    assert lines[i + 1 : i + 4] == ["PREOPTGEOM", "FULLOPTG", "END"]


# ── Raman / CPHF wiring ────────────────────────────────────────────────────
def test_raman_forces_intens_and_intcphf_and_locks_those_controls(qapp):
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "FREQCALC")
    dlg._freq_raman.setChecked(True)
    lines = dlg._preview.toPlainText().splitlines()
    block = lines[lines.index("FREQCALC") : lines.index("BASISSET")]
    assert block == ["FREQCALC", "INTENS", "INTRAMAN", "INTCPHF", "END", "END"]
    # IR and its technique stop being free choices once Raman is on
    assert not dlg._freq_ir.isEnabled()
    assert not dlg._freq_technique.isEnabled()
    # the CPHF block and its second-order knobs become relevant
    assert dlg._freq_cphf.isEnabled()
    assert dlg._freq_cphf.fmixing2.isEnabled()


def test_raman_experimental_conditions_reach_the_deck(qapp):
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "FREQCALC")
    dlg._freq_raman.setChecked(True)
    dlg._freq_ramanexp.setChecked(True)
    dlg._freq_temp.setValue(300.0)
    dlg._freq_laser.setValue(514.5)
    lines = dlg._preview.toPlainText().splitlines()
    assert lines[lines.index("RAMANEXP") + 1] == "300 514.5"


def test_intcphf_group_is_disabled_for_berry_phase_ir(qapp):
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "FREQCALC")
    dlg._freq_ir.setChecked(True)  # default technique is INTPOL
    assert not dlg._freq_cphf.isEnabled()
    dlg._freq_technique.setCurrentIndex(2)  # INTCPHF
    dlg._sync_enabled()
    assert dlg._freq_cphf.isEnabled()


def test_cphf_task_second_order_knobs_need_fourth_order(qapp):
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "CPHF")
    assert not dlg._cphf.fmixing2.isEnabled()  # second order by default
    dlg._cphf.order.setCurrentIndex(2)  # Fourth order
    assert dlg._cphf.fmixing2.isEnabled()
    lines = dlg._preview.toPlainText().splitlines()
    assert lines[lines.index("CPHF") + 1] == "FOURTH"


# ── optional per-block parameters ──────────────────────────────────────────
def test_unset_optional_fields_are_omitted_from_the_deck(qapp):
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "FREQCALC")
    text = dlg._preview.toPlainText()
    # nothing was set, so no optional keyword leaks into the block
    assert "NUMDERIV" not in text and "STEPSIZE" not in text


def test_numderiv_reaches_the_deck(qapp):
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "FREQCALC")
    dlg._freq_numderiv.setCurrentIndex(2)  # "2 — central difference"
    lines = dlg._preview.toPlainText().splitlines()
    assert lines[lines.index("NUMDERIV") + 1] == "2"


def test_eos_ranges_are_opt_in(qapp):
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "EOS")
    assert "RANGE" not in dlg._preview.toPlainText()
    dlg._eos_range.setChecked(True)
    lines = dlg._preview.toPlainText().splitlines()
    assert lines[lines.index("RANGE") + 1] == "0.9 1.1 10"


def test_cphf_group_deactivates_again_when_its_trigger_is_removed(qapp):
    # Regression: the group stayed enabled after Raman/IR were switched back off,
    # because the technique combo never re-ran the enable pass.
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "FREQCALC")
    assert not dlg._freq_cphf.isEnabled()

    dlg._freq_raman.setChecked(True)
    assert dlg._freq_cphf.isEnabled()
    dlg._freq_raman.setChecked(False)
    assert not dlg._freq_cphf.isEnabled()
    # the technique the user had chosen before Raman forced INTCPHF is restored
    assert dlg._freq_technique.currentIndex() == 0

    dlg._freq_ir.setChecked(True)
    dlg._freq_technique.setCurrentIndex(2)  # INTCPHF
    assert dlg._freq_cphf.isEnabled()
    dlg._freq_ir.setChecked(False)  # no intensities at all -> no INTCPHF block
    assert not dlg._freq_cphf.isEnabled()


# ── phonon dispersion / QHA ────────────────────────────────────────────────
def _task_block(dlg, first: str) -> list:
    lines = dlg._preview.toPlainText().splitlines()
    return lines[lines.index(first) : lines.index("BASISSET")]


def test_dispersion_task_writes_freqcalc_with_dispersi(qapp):
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "DISPERSION")
    assert _task_block(dlg, "FREQCALC") == ["FREQCALC", "DISPERSI", "END"]


def test_dispersion_supercell_is_on_by_default(qapp):
    # Dispersion without a supercell only repeats Gamma, so SCELPHONO starts on
    # with a real 2x2x2 expansion rather than the no-op identity.
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "DISPERSION")
    assert dlg._disp_cell.isChecked()
    assert _task_block(dlg, "SCELPHONO")[:4] == ["SCELPHONO", "2 0 0", "0 2 0", "0 0 2"]
    dlg._disp_cell.setChecked(False)  # still possible to opt out
    assert "SCELPHONO" not in dlg._preview.toPlainText()


def test_qha_supercell_stays_opt_in(qapp):
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "QHA")
    assert not dlg._qha_cell.isChecked()
    assert "SCELPHONO" not in dlg._preview.toPlainText()


def test_dispersion_bands_and_pdos(qapp):
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "DISPERSION")
    dlg._disp_bands.setChecked(True)
    dlg._disp_bands_path.setPlainText("0 0 0  8 0 0\n8 0 0  8 8 0")
    dlg._disp_pdos.setChecked(True)
    block = _task_block(dlg, "FREQCALC")
    # ISS and NPOINTS on one record, then NLINE on its own — as in a working
    # deck (~/Desktop/phonon_dispersion_forClaude/ZnO_LDA_scelphono444_bands.out).
    assert block[block.index("BANDS") + 1] == "16 30"
    assert block[block.index("BANDS") + 2] == "2"
    assert block[block.index("PDOS") + 1] == "2500 250"
    # BANDS implies NOKSYMDISP, so its own switch stops applying
    assert not dlg._disp_noksym.isEnabled()


def test_dispersion_bands_without_a_path_blocks_saving(qapp):
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "DISPERSION")
    dlg._disp_bands.setChecked(True)  # no path segments given
    assert not dlg._save_btn.isEnabled()
    assert "path segment" in dlg._preview.toPlainText()


def test_qha_temperature_range_reaches_the_deck(qapp):
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "QHA")
    assert _task_block(dlg, "QHA") == ["QHA", "END"]
    dlg._qha_temp.setChecked(True)
    block = _task_block(dlg, "QHA")
    assert block[block.index("TEMPERAT") + 1] == "100 10 1200"


def test_qha_step_and_points_are_optional(qapp):
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "QHA")
    assert "STEP" not in dlg._preview.toPlainText()
    dlg._qha_step.setValue(3.5)
    dlg._qha_points.setCurrentText("7")
    block = _task_block(dlg, "QHA")
    assert block[block.index("STEP") + 1] == "3.5"
    assert block[block.index("POINTS") + 1] == "7"


# ── anharmonic ─────────────────────────────────────────────────────────────
def test_anhapes_needs_modes_before_it_can_be_saved(qapp):
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "ANHAPES")
    assert not dlg._save_btn.isEnabled()
    assert "mode number" in dlg._preview.toPlainText()
    dlg._anh_modes.setPlainText("4 5 6 7")
    assert dlg._save_btn.isEnabled()


def test_anhapes_rejects_translational_modes_in_the_preview(qapp):
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "ANHAPES")
    dlg._anh_modes.setPlainText("1 2 3 7")
    assert "translations" in dlg._preview.toPlainText()
    assert not dlg._save_btn.isEnabled()


def test_anhapes_vci_block_and_scheme(qapp):
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "ANHAPES")
    dlg._anh_modes.setPlainText("4 5 6 7 8 9 10 11 12 13 14 15")
    dlg._anh_vci.setChecked(True)
    block = _task_block(dlg, "FREQCALC")
    assert block == [
        "FREQCALC", "ANHAPES", "12", "4 5 6 7 8 9 10 11 12 13 14 15",
        "3 0.9", "VCI", "6 3", "1", "END",
    ]
    # the VCI parameters only apply once VCI is requested
    assert dlg._anh_vci_quanta.isEnabled()
    dlg._anh_vci.setChecked(False)
    assert not dlg._anh_vci_quanta.isEnabled()


def test_vscf_tuning_enabled_by_either_vscf_or_vci(qapp):
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "ANHAPES")
    assert not dlg._anh_vscftol.isEnabled()
    dlg._anh_vci.setChecked(True)  # VCI runs a VSCF of its own
    assert dlg._anh_vscftol.isEnabled()


def test_anharm_writes_a_standalone_block(qapp):
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "ANHARM")
    dlg._anharm_atom.setValue(5)
    assert _task_block(dlg, "ANHARM") == ["ANHARM", "5", "END"]
    assert "FREQCALC" not in dlg._preview.toPlainText()


def test_anharm_isotope_row_is_opt_in(qapp):
    dlg = InputBuilderDialog(_nacl())
    _select_task(dlg, "ANHARM")
    assert not dlg._anharm_iso_mass.isEnabled()
    dlg._anharm_iso.setChecked(True)
    dlg._anharm_iso_atom.setValue(5)
    dlg._anharm_iso_mass.setValue(2.0)
    block = _task_block(dlg, "ANHARM")
    assert block[block.index("ISOTOPES") + 1 : block.index("ISOTOPES") + 3] == ["1", "5 2"]


# ── functional: combined vs split ──────────────────────────────────────────
def test_split_functional_mode_swaps_which_controls_apply(qapp):
    dlg = InputBuilderDialog(_nacl())
    assert dlg._functional.isEnabled() and not dlg._exchange.isEnabled()
    dlg._functional_mode.setCurrentIndex(1)  # separate exchange + correlation
    assert not dlg._functional.isEnabled()
    assert dlg._exchange.isEnabled() and dlg._correlation.isEnabled()
    dlg._exchange.setCurrentText("BECKE")
    dlg._correlation.setCurrentText("LYP")
    lines = dlg._preview.toPlainText().splitlines()
    i = lines.index("DFT")
    assert lines[i + 1 : i + 5] == ["EXCHANGE", "BECKE", "CORRELAT", "LYP"]


def test_split_defaults_are_the_documented_unset_states(qapp):
    # Hartree-Fock exchange / no correlation are CRYSTAL defaults, so neither
    # keyword is written; but with both unset there is nothing to build.
    dlg = InputBuilderDialog(_nacl())
    dlg._functional_mode.setCurrentIndex(1)
    assert not dlg._save_btn.isEnabled()
    assert "exchange or a correlation" in dlg._preview.toPlainText()
    dlg._exchange.setCurrentText("PBE")
    assert dlg._save_btn.isEnabled()
    assert "CORRELAT" not in dlg._preview.toPlainText()


def test_hybrid_and_nonlocal_rebuild_b3lyp(qapp):
    dlg = InputBuilderDialog(_nacl())
    dlg._functional_mode.setCurrentIndex(1)
    dlg._exchange.setCurrentText("BECKE")
    dlg._correlation.setCurrentText("LYP")
    dlg._hybrid.setValue(20)
    dlg._nonlocal.setChecked(True)
    lines = dlg._preview.toPlainText().splitlines()
    i = lines.index("DFT")
    assert lines[i + 1 : i + 10] == [
        "EXCHANGE", "BECKE", "CORRELAT", "LYP", "HYBRID", "20", "NONLOCAL", "0.9 0.81", "END",
    ]


# ── spin–orbit coupling ────────────────────────────────────────────────────
def test_two_component_group_is_opt_in(qapp):
    dlg = InputBuilderDialog(_nacl())
    assert "TWOCOMPON" not in dlg._preview.toPlainText()
    dlg._twocompon.setChecked(True)
    lines = dlg._preview.toPlainText().splitlines()
    assert lines[lines.index("TWOCOMPON") + 1] == "SOC"  # SOC on by default
    assert lines.index("TWOCOMPON") < lines.index("DFT")


def test_noncollinear_choice_lands_in_the_dft_block(qapp):
    dlg = InputBuilderDialog(_nacl())
    dlg._twocompon.setChecked(True)
    dlg._noncollinear.setCurrentText("NONCOLC")
    lines = dlg._preview.toPlainText().splitlines()
    assert lines[lines.index("DFT") + 1] == "NONCOLC"


def test_soc_angle_rows_follow_their_trigger(qapp):
    dlg = InputBuilderDialog(_nacl())
    dlg._twocompon.setChecked(True)
    assert not dlg._soc_theta.isEnabled()  # default guess needs no angles
    dlg._soc_guess.setCurrentText("GCOREROT")
    assert dlg._soc_theta.isEnabled()

    assert not dlg._soc_rot_theta.isEnabled()
    dlg._soc_2nd.setChecked(True)
    assert dlg._soc_rot.isEnabled()
    assert not dlg._soc_rot_theta.isEnabled()  # IROT is 0, so no rotation angles
    dlg._soc_rot.setValue(1)
    assert dlg._soc_rot_theta.isEnabled()


def test_two_component_blocks_an_unsupported_task(qapp):
    dlg = InputBuilderDialog(_nacl())
    dlg._twocompon.setChecked(True)
    _select_task(dlg, "OPTGEOM")
    assert not dlg._save_btn.isEnabled()
    assert "single-point" in dlg._preview.toPlainText()


def test_two_component_blocks_a_range_separated_functional(qapp):
    dlg = InputBuilderDialog(_nacl())
    dlg._twocompon.setChecked(True)
    dlg._functional.setCurrentText("HSE06")
    assert not dlg._save_btn.isEnabled()
    assert "meta-GGA or range-separated" in dlg._preview.toPlainText()


# ── SUPERCEL ───────────────────────────────────────────────────────────────
def test_supercel_group_is_opt_in_and_writes_noshift(qapp):
    dlg = InputBuilderDialog(_nacl())
    assert "SUPERCEL" not in dlg._preview.toPlainText()
    dlg._supercel.setChecked(True)
    lines = dlg._preview.toPlainText().splitlines()
    i = lines.index("SUPERCEL")
    assert lines[i + 1 : i + 4] == ["2 0 0", "0 2 0", "0 0 2"]
    dlg._supercel_noshift.setChecked(True)
    lines = dlg._preview.toPlainText().splitlines()
    assert lines[lines.index("SUPERCEL") - 1] == "NOSHIFT"


def test_supercel_conflicts_with_a_dispersion_supercell(qapp):
    dlg = InputBuilderDialog(_nacl())
    dlg._supercel.setChecked(True)
    _select_task(dlg, "DISPERSION")  # its SCELPHONO is on by default
    assert not dlg._save_btn.isEnabled()
    assert "cannot be combined" in dlg._preview.toPlainText()


# ── TOLINTEG ───────────────────────────────────────────────────────────────
def test_tolinteg_defaults_and_edits_reach_the_deck(qapp):
    dlg = InputBuilderDialog(_nacl())
    lines = dlg._preview.toPlainText().splitlines()
    assert lines[lines.index("TOLINTEG") + 1] == "7 7 7 7 14"
    assert dlg._tolinteg_warning.text() == ""
    dlg._tolinteg[0].setValue(8)
    lines = dlg._preview.toPlainText().splitlines()
    assert lines[lines.index("TOLINTEG") + 1] == "8 7 7 7 14"


def test_tolinteg_warns_when_the_exchange_gap_is_too_small(qapp):
    dlg = InputBuilderDialog(_nacl())
    dlg._tolinteg[4].setValue(8)  # ITOL5 only 1 above ITOL4
    assert "ITOL5" in dlg._tolinteg_warning.text()
    # still a valid deck — the warning is advisory, saving stays enabled
    assert dlg._save_btn.isEnabled()


def test_dispersion_checkbox_adds_d3(qapp):
    dlg = InputBuilderDialog(_nacl())
    dlg._functional.setCurrentText("PBE0")
    dlg._d3.setChecked(True)
    assert "PBE0-D3" in dlg._preview.toPlainText()


def test_molecule_disables_shrink_and_hides_k_mesh(qapp):
    dlg = InputBuilderDialog(_co())
    assert not dlg._shrink.isEnabled()
    text = dlg._preview.toPlainText()
    assert text.splitlines()[1] == "MOLECULE"
    assert "SHRINK" not in text


def test_write_saves_the_previewed_deck(qapp, tmp_path, monkeypatch):
    from crystalline.ui.panels import input_builder as mod

    dlg = InputBuilderDialog(_nacl())
    path = tmp_path / "nacl.d12"
    monkeypatch.setattr(
        mod.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), ""))
    )
    dlg._save()
    assert path.read_text() == dlg._preview.toPlainText()


def _stub_window(source, shown, unit_cell):
    """``MainWindow._analysis_cell`` bound to only the attributes it reads.

    ``MainWindow`` can't be constructed headless (VTK interactor), so the method
    is exercised unbound against a stub, as in ``test_compose_view``.
    """
    from crystalline.ui.main_window import MainWindow

    class _StubWindow:
        _analysis_cell = MainWindow._analysis_cell

    window = _StubWindow()
    window._source = source
    window.structure = shown
    window._unit_cell = unit_cell
    return window


def test_analysis_cell_folds_a_supercell_view_back_to_one_clean_cell():
    """What passing the pristine ``_source`` used to buy: symmetry reduction sees
    a single cell, not a tiling or the duplicate atoms boundary completion adds."""
    from crystalline.core.cells import (
        CellView,
        as_view,
        complete_boundary,
        tile_supercell,
    )

    source = _nacl()
    base = as_view(source, CellView.CRYSTALLOGRAPHIC)
    analysis, _ = tile_supercell(base, (2, 1, 1), None)
    shown, _ = complete_boundary(analysis, None)
    assert len(shown) > len(base)  # the view really is tiled and packed

    folded = _stub_window(source, shown, base.cell.copy())._analysis_cell()

    assert len(folded) == len(base)


def test_build_crystal_input_hands_the_dialog_the_edited_geometry(monkeypatch):
    """Regression: Build CRYSTAL input wrote the geometry as loaded, not as edited.

    ``_build_crystal_input`` passed ``_source`` — the pristine loaded cell — so a
    dragged atom, an added or deleted one, or a changed element never reached the
    deck, and nothing on screen said so. It now goes through ``_analysis_cell``,
    which keeps the edits while still folding the view down to one clean cell.

    Asserted on the real ``_build_crystal_input``, against the structure the
    dialog is actually constructed with: testing ``_analysis_cell`` alone would
    still pass if the builder went back to reading ``_source``.
    """
    from crystalline.core.cells import CellView, as_view
    from crystalline.core.crystal_input import CrystalInputSpec, build_input
    from crystalline.ui import main_window as mw
    from crystalline.ui.panels import input_builder

    handed = []

    class _FakeDialog:
        def __init__(self, structure, parent=None):
            handed.append(structure)

        def exec(self):
            return 0

    monkeypatch.setattr(input_builder, "InputBuilderDialog", _FakeDialog)

    class _StubWindow:
        _analysis_cell = mw.MainWindow._analysis_cell
        _build_crystal_input = mw.MainWindow._build_crystal_input

    source = _nacl()
    shown = as_view(source, CellView.CRYSTALLOGRAPHIC)
    window = _StubWindow()
    window._source = source
    window.structure = shown
    window._unit_cell = shown.cell.copy()

    shown.set_symbols([0], "K")  # the user re-elements an atom in the 3D view

    window._build_crystal_input()

    assert len(handed) == 1
    assert "K" in handed[0].symbols  # not the pristine _source, which has none

    deck = build_input(handed[0], CrystalInputSpec())
    assert " 19 " in deck.replace("\n", " ")                 # potassium reached the deck
    assert deck != build_input(source, CrystalInputSpec())   # and it differs from pristine
