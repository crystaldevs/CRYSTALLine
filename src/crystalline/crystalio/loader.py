"""Load CRYSTAL files into CRYSTALLine domain objects via CRYSTALClear.

Implementation note — CRYSTALClear's ``convert`` convenience wrappers
(``cry_out2ase``, ``cry_gui2ase``, ``cry_ase2gui``) are unusable in a normal
install of CRYSTALClear 0.2.15: internally they do unqualified imports like
``from convert import cry_out2pmg`` which raise ``ModuleNotFoundError: No module
named 'convert'`` unless the package's own directory happens to be on
``sys.path``. We therefore go through the *properly-qualified* class API
(``Crystal_output.get_geometry``, ``Crystal_gui.read_gui``, ``cry_pmg2gui``) and
do the ase conversion ourselves. See ``docs/CRYSTALClear_notes.md``.

CRYSTALClear is imported lazily (inside function bodies) so ``core`` and the
test suite stay importable without it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from ase import Atoms

from crystalline.core.adp import ADPSet
from crystalline.core.phonons import PhononMode, PhononModes
from crystalline.core.structure import Structure

# CRYSTALClear reports phonon frequencies in THz; the domain model stores cm^-1.
# 1 THz = 1e12 Hz -> nu[cm^-1] = 1e12 / c[cm/s] = 1e12 / 2.99792458e10.
THZ_TO_CM = 33.356409519815204

# Lines that mark a CRYSTAL frequency (FREQCALC) run — the same header
# CRYSTALClear itself keys on when parsing phonons.
_FREQ_MARKERS = (
    re.compile(r"\+\+\+\s*SYMMETRY ADAPTION OF VIBRATIONAL MODES"),
    re.compile(r"MODES\s+EIGV\s+FREQUENCIES\s+IRREP"),
)

# One sampled q, as a dispersion (SCELPHONO) run heads each block:
#   DISPERSION K POINT NUMBER     2 COORD:  C(  0  0  1 )    WEIGHT:    1.
# The letter is R for a q that is its own opposite (real eigenvectors) and C
# otherwise; the triple is in units of the shrinking factors below.
_QPOINT_HEADER = re.compile(
    r"DISPERSION K POINT NUMBER\s+(\d+)\s+COORD:\s*[A-Z]?\(\s*"
    r"(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\)"
)
# The denominators of that triple, in the two spellings CRYSTAL uses: a
# supercell run states one factor per axis, a plain dispersion run one for all.
_SHRINK_FACTORS = re.compile(
    r"WITH SHRINKING FACTORS:\s*IS1\s*=\s*(\d+)\s+IS2\s*=\s*(\d+)\s+IS3\s*=\s*(\d+)"
)
_SHRINK_DENOMINATOR = re.compile(r"EXPRESSED IN UNITS\s+OF DENOMINATOR\s+(\d+)")


@dataclass
class LoadedFile:
    """Result of :func:`load`: a structure and, if present, its phonon modes.

    ``modes`` is the Gamma-point set — what every consumer of a FREQCALC wants —
    and ``qmodes`` holds one set per sampled q, ``qmodes[0]`` being that same
    Gamma set. A plain frequency run has exactly one; a SCELPHONO run has as many
    as its supercell allows.
    """

    structure: Structure
    modes: Optional[PhononModes] = None
    qmodes: Optional[list] = None

    @property
    def has_phonons(self) -> bool:
        return self.modes is not None and len(self.modes) > 0

    @property
    def qpoints(self) -> list:
        """The sampled q-points' modes, or an empty list without phonons."""
        return list(self.qmodes) if self.qmodes else []


def has_phonons(path: str) -> bool:
    """Whether a CRYSTAL ``.out`` file contains a vibrational calculation.

    Returns False for ``.gui`` files (geometry only) and for outputs without a
    frequency table. Scans the text for CRYSTAL's frequency-run markers.
    """
    if _is_gui(path):
        return False
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if any(m.search(line) for m in _FREQ_MARKERS):
                return True
    return False


def load(path: str, initial: bool = False) -> LoadedFile:
    """Load a CRYSTAL file, auto-detecting whether it holds phonon modes.

    This is the single entry point the GUI uses: it returns the structure and,
    if the output is a frequency calculation, the phonon modes as well.
    """
    if not _is_gui(path) and not _is_cif(path) and has_phonons(path):
        structure, qmodes = load_dispersion(path)
        return LoadedFile(structure=structure, modes=qmodes[0], qmodes=qmodes)
    return LoadedFile(structure=load_structure(path, initial=initial), modes=None)


def load_structure(path: str, initial: bool = False) -> Structure:
    """Load a structure from a CRYSTAL ``.out``/``.gui`` file or a ``.cif``.

    Parameters
    ----------
    path:
        A CRYSTAL ``.out`` output, a ``.gui``/``fort.34`` geometry file, or a
        crystallographic ``.cif``.
    initial:
        For ``.out`` files, read the initial geometry instead of the last one
        (relevant after a geometry optimisation). Ignored for ``.gui``/``.cif``.
    """
    if _is_gui(path):
        return Structure.from_ase(_gui_to_ase(path))
    if _is_cif(path):
        return Structure.from_ase(_cif_to_ase(path))
    return Structure.from_ase(_out_to_ase(path, initial=initial))


def _structure_for_modes(out, natom_modes: int) -> Structure:
    """The geometry an open output's phonon eigenvectors are defined on.

    Normally the geometry of the run, but a SCELPHONO calculation builds a
    supercell to get its force constants and then reports the modes of the cell
    it expanded — 60 of them for a 20-atom cell whose geometry sections describe
    540 atoms. Reshaping the eigenvectors against the wrong one is what used to
    make such a file fail to open at all.

    Raises:
        ValueError: When neither cell has as many atoms as the modes span, so
            that the message names the counts instead of a numpy reshape.
    """
    atoms = _out_geometry_to_ase(out, initial=False)
    if len(atoms) == natom_modes:
        return Structure.from_ase(atoms)

    try:
        primitive = _out_primitive_to_ase(out, initial=False)
    except Exception:  # noqa: BLE001 - no primitive section to fall back on
        primitive = None
    if primitive is not None and len(primitive) == natom_modes:
        return Structure.from_ase(primitive)

    raise ValueError(
        f"This run's phonon modes span {natom_modes} atoms, but its geometry "
        f"has {len(atoms)}"
        + (f" and its primitive cell {len(primitive)}" if primitive is not None else "")
        + ". The modes cannot be placed on either cell."
    )


def load_phonons(
    path: str,
    qpoint: int = 0,
    keep_imaginary: bool = True,
) -> tuple[Structure, PhononModes]:
    """Load equilibrium geometry + phonon modes from a CRYSTAL ``.out`` file.

    Returns the structure the modes are defined on, alongside the modes for the
    requested q-point (Gamma by default). Imaginary/soft modes are kept by
    default so they remain animatable. Use :func:`load_dispersion` to get every
    sampled q in one pass — which is what reading the file twice would cost.

    Parameters
    ----------
    path:
        CRYSTAL ``.out`` file from a frequency calculation.
    qpoint:
        Index of the q-point to extract (0 = Gamma).
    keep_imaginary:
        If True, negative-frequency modes are retained (CRYSTALClear would
        otherwise NaN them out).
    """
    structure, qmodes = load_dispersion(path, keep_imaginary=keep_imaginary)
    if not 0 <= qpoint < len(qmodes):
        raise IndexError(
            f"q-point {qpoint} out of range: this run sampled {len(qmodes)}"
        )
    return structure, qmodes[qpoint]


def load_dispersion(
    path: str,
    keep_imaginary: bool = True,
) -> tuple[Structure, list]:
    """Load the geometry and **every** sampled q-point's modes from a ``.out``.

    Returns ``(structure, [PhononModes, ...])`` with the Gamma-point set first.
    A plain FREQCALC yields a single set; a SCELPHONO run yields one per q the
    supercell makes accessible, each mode carrying its complex eigenvector and
    its q in the structure's own reciprocal basis, ready to be tiled into a
    visible wave (``core.cells``).

    One pass over the output serves all of them: parsing a large frequency
    output is the expensive part of opening such a file, and re-reading it per
    q-point would make selecting one from the panel feel like loading a file.
    """
    from CRYSTALClear.crystal_io import Crystal_output

    out = Crystal_output(path)
    out.get_phonon(read_eigvt=True, rm_imaginary=not keep_imaginary)

    # frequency: (nqpoint, nmode) in THz -> converted to cm^-1 below.
    # eigenvector at a q-point comes back per mode either flattened as
    # (3*natom,) or already (natom, 3), and is complex: at Gamma the imaginary
    # part is zero, away from it it is CRYSTAL's "anti-phase" block.
    freqs = np.atleast_2d(np.asarray(out.frequency))
    eigvecs = np.asarray(out.eigenvector)
    nq = min(len(freqs), len(eigvecs))
    if nq == 0:
        raise ValueError("this output reports no phonon modes")

    structure = _structure_for_modes(out, int(np.asarray(eigvecs[0][0]).size) // 3)
    natom = len(structure)
    qpoints = _sampled_qpoints(out, nq)

    return structure, [
        _modes_at(freqs[q], eigvecs[q], natom, qpoints[q], activity=(q == 0), out=out)
        for q in range(nq)
    ]


def _modes_at(freqs, eigvecs, natom: int, qpoint, activity: bool, out) -> PhononModes:
    """The modes of one q-point, as :class:`PhononMode` objects.

    IR/Raman selection rules and intensities are printed for the Gamma block
    only — ``activity`` says whether this set is that block — so away from it the
    modes carry ``None`` and the panel's activity filter switches itself off
    rather than showing an empty list.
    """
    nmode = len(freqs)
    gamma = qpoint is None or not np.any(np.abs(np.asarray(qpoint, dtype=float)) > 1e-6)
    if activity and gamma:
        ir = _per_mode(getattr(out, "IR", None), nmode)
        raman = _per_mode(getattr(out, "Raman", None), nmode)
        intens = _per_mode(getattr(out, "intens", None), nmode, cast=float)
    else:
        ir = raman = intens = [None] * nmode

    modes = []
    for i in range(nmode):
        eigenvector = np.asarray(eigvecs[i]).reshape(natom, 3)
        # At Gamma the eigenvector is real up to a global phase and CRYSTAL
        # prints it real, so the (zero) imaginary part is dropped: everything
        # downstream that predates dispersion goes on seeing real arrays. An
        # unidentified q counts as Gamma here — with no q there is no phase to
        # apply, and the in-phase part is the honest half of the answer.
        if gamma:
            eigenvector = np.real(eigenvector)
        modes.append(
            PhononMode(
                frequency=float(np.real(freqs[i])) * THZ_TO_CM,
                eigenvector=eigenvector,
                ir_active=ir[i],
                raman_active=raman[i],
                ir_intensity=intens[i],
                qpoint=qpoint,
            )
        )
    return PhononModes(modes)


def _sampled_qpoints(out, nq: int) -> list:
    """The q of each mode set, in the order CRYSTALClear returns them.

    Read from the output text rather than from ``Crystal_output.qpoint``, which
    can't be lined up with the frequencies: it omits the Gamma block that heads
    a dispersion run (leaving ``nqpoint - 1`` entries for ``nqpoint`` mode sets)
    and divides all three components by the *first* shrinking factor, which is
    wrong the moment a SCELPHONO supercell isn't cubic.

    Anything that doesn't line up — no q header at all (a plain FREQCALC), or a
    QHA run whose "q-point" axis counts volumes rather than points in reciprocal
    space — falls back to ``None`` (Gamma) for every set. Unlabelled beats
    mislabelled: a wrong q would put a wrong phase on every image atom.
    """
    lines = getattr(out, "data", None)
    if lines is None:
        return [None] * nq
    end = getattr(out, "eoo", None)
    parsed = _qpoints_from_lines(lines[:end] if end else lines)
    if len(parsed) != nq:
        return [None] * nq
    return parsed


def _qpoints_from_lines(lines) -> list:
    """Fractional q of every ``DISPERSION K POINT`` header, in file order.

    Empty when the run reports no q headers, or when their denominators (the
    shrinking factors) never appear — without those the integer triples cannot
    be turned into coordinates.
    """
    shrink = None
    qpoints = []
    for line in lines:
        if shrink is None:
            factors = _SHRINK_FACTORS.search(line)
            if factors:
                shrink = np.array([int(f) for f in factors.groups()], dtype=float)
            else:
                single = _SHRINK_DENOMINATOR.search(line)
                if single:
                    shrink = np.full(3, float(single.group(1)))
        header = _QPOINT_HEADER.search(line)
        if header:
            qpoints.append(np.array([int(v) for v in header.groups()[1:]], dtype=float))
    if not qpoints or shrink is None or np.any(shrink <= 0):
        return []
    return [q / shrink for q in qpoints]


def load_adp(path: str) -> Optional[ADPSet]:
    """Load the atomic displacement parameters of a CRYSTAL ``.out``.

    Returns ``None`` — never raises — when the file has no ``ADP`` section,
    which is the common case: a frequency run only computes them if asked. The
    tensors come back cartesian and in Angstrom squared, on the same atom
    ordering as :func:`load_structure`.
    """
    if _is_gui(path) or _is_cif(path):
        return None
    try:
        from CRYSTALClear.crystal_io import Crystal_output

        out = Crystal_output(path).get_ADP()
    except Exception:  # noqa: BLE001 - absent section, or an output it can't read
        return None
    if out.adp.size == 0 or len(out.adp_temperature) == 0:
        return None
    return ADPSet(temperatures=out.adp_temperature, tensors=out.adp)


def _per_mode(values, nmode: int, cast=bool) -> list:
    """Normalise a CRYSTALClear per-mode array to a length-``nmode`` list.

    ``out.IR`` / ``out.Raman`` / ``out.intens`` are empty when the output has no
    IR/Raman analysis (dispersion runs, or FREQCALC without intensities), and
    the modes then simply carry ``None`` rather than a wrong ``False``. A
    length mismatch is treated the same way — better unlabelled than misaligned.
    """
    if values is None:
        return [None] * nmode
    arr = np.asarray(values).ravel()
    if arr.size != nmode:
        return [None] * nmode
    out = []
    for v in arr:
        value = cast(v)
        out.append(None if isinstance(value, float) and not np.isfinite(value) else value)
    return out


def save_structure_gui(structure: Structure, gui_file: str, symmetry: bool = True) -> None:
    """Write a :class:`Structure` to a CRYSTAL ``.gui`` file.

    A periodic cell is written at its own dimensionality (3D bulk / 2D slab /
    1D polymer); a non-periodic one becomes a 0D (molecule) gui. If symmetry
    can't be determined, a ``P1`` gui is written rather than failing the save.

    ``pbc`` is always passed explicitly: left to itself, ``cry_pmg2gui``
    round-trips a molecule through ``Molecule(species, coords, charge,
    site_properties)`` — positional arguments that no longer line up with
    current pymatgen, so it raises.
    """
    from CRYSTALClear.convert import cry_pmg2gui  # qualified path: safe

    if _has_lattice(structure):
        obj = _to_pymatgen(structure)
        pbc = tuple(bool(p) for p in structure.pbc)
    else:
        obj = _to_pymatgen_molecule(structure)
        pbc = (False, False, False)

    try:
        cry_pmg2gui(obj, gui_file=gui_file, pbc=pbc, symmetry=symmetry)
    except Exception:  # noqa: BLE001 - symmetry undeterminable: write an unreduced P1 gui
        cry_pmg2gui(obj, gui_file=gui_file, pbc=pbc, symmetry=False)


def save_structure_cif(structure: Structure, cif_file: str, symmetry: bool = True) -> None:
    """Write a :class:`Structure` to a CIF file.

    For a periodic cell this uses pymatgen's ``CifWriter``; with ``symmetry`` it
    detects the space group (``symprec`` search) and writes the symmetry-reduced
    CIF, otherwise a plain ``P1`` listing. A non-periodic system (a molecule)
    has no lattice for a crystallographic CIF, so it falls back to ASE's writer.
    If symmetry can't be determined, a ``P1`` CIF is written rather than failing
    the save.
    """
    if not _has_lattice(structure):
        from ase.io import write  # non-periodic: no lattice for a crystallographic CIF

        write(cif_file, structure.to_ase(), format="cif")
        return

    from pymatgen.io.cif import CifWriter

    pmg = _to_pymatgen(structure)
    try:
        CifWriter(pmg, symprec=0.01 if symmetry else None).write_file(cif_file)
    except Exception:  # noqa: BLE001 - symmetry undeterminable: write an unreduced P1 CIF
        CifWriter(pmg, symprec=None).write_file(cif_file)


def read_atoms(path: str) -> Atoms:
    """Read atoms from an ``.xyz`` / ``.pdb`` / ``.cif`` file for importing.

    Returns a bare ``ase.Atoms`` (symbols + cartesian positions); the caller
    appends these into the current structure. The source file's own cell (if any)
    is intentionally ignored — the atoms are added to the structure already loaded.
    A multi-frame file (e.g. an XYZ trajectory) yields its first frame.
    """
    from ase.io import read

    result = read(path, index=0)  # first frame if the file holds several
    if isinstance(result, list):  # some formats/return an explicit list
        result = result[0]
    return result


def output_properties(path: str) -> dict:
    """Extract what a CRYSTAL ``.out`` file says about itself, as (label, value).

    Returns an ordered ``{label: value_string}``: first how the calculation was
    set up (code, task, Hamiltonian and functional, k-point mesh, basis-set
    size, SCF thresholds), then what it computed (total energy, band gap, Fermi
    energy). Empty for ``.gui`` files. Every part is guarded independently, so
    anything this particular run doesn't report — or that CRYSTALClear can't
    parse — just omits its row rather than failing the whole panel.
    """
    if _is_gui(path):
        return {}
    from CRYSTALClear.crystal_io import Crystal_output

    out = Crystal_output(path)
    props: dict = {}

    def _try(label: str, fn, fmt) -> None:
        try:
            value = fn()
        except Exception:  # noqa: BLE001 - a getter failing must not break the rest
            return
        if value is None:
            return
        try:
            props[label] = fmt(value)
        except Exception:  # noqa: BLE001 - unexpected shape; skip this row
            return

    props.update(_setup_rows(out))
    _try("Total energy (eV)", out.get_final_energy, lambda v: f"{float(v):.6f}")
    _try("Band gap (eV)", out.get_band_gap, lambda v: f"{float(np.ravel(v)[0]):.4f}")
    _try("Fermi energy (eV)", out.get_fermi_energy, lambda v: f"{float(np.ravel(v)[0]):.4f}")
    return props


_SUPERSCRIPTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

# CRYSTAL spells functionals out in full ("PERDEW-BURKE-ERNZERHOF"), which
# wraps over three lines in the dock. Show the acronym everyone uses instead —
# matched exactly, so an unlisted functional keeps CRYSTAL's own wording rather
# than being mislabelled.
_FUNCTIONAL_ACRONYMS = {
    "DIRAC-SLATER LDA": "LDA",
    "VOSKO-WILK-NUSAIR": "VWN",
    "PERDEW-BURKE-ERNZERHOF": "PBE",
    "PERDEW-BURKE-ERNZERHOF (SOL)": "PBEsol",
    "BECKE 88": "B88",
    "LEE-YANG-PARR": "LYP",
    "PERDEW-WANG 91": "PW91",
    "PERDEW-WANG GGA": "PWGGA",
    "PERDEW-ZUNGER": "PZ",
    "PERDEW 86": "P86",
    "WU-COHEN": "WC",
    "VON BARTH-HEDIN": "VBH",
}


def _functional(name: Optional[str]) -> Optional[str]:
    """The short name for a CRYSTAL functional, or its own wording."""
    if not name:
        return None
    return _FUNCTIONAL_ACRONYMS.get(name.strip().upper(), name)


def _dispersion(name: Optional[str]) -> Optional[str]:
    """``DFT-D3(BJ)`` -> ``D3(BJ)``.

    The "DFT-" prefix says nothing next to a row already labelled Dispersion.
    """
    if not name:
        return None
    label = name.strip().upper()
    if label.startswith("DFT-"):
        label = label[4:]
    if "GRIMME" in label:  # older outputs name only the author, not the version
        return "yes"
    return label


def _superscript(exponent: int) -> str:
    """``-9`` -> ``⁻⁹``, so a power of ten reads as one in the panel."""
    return str(exponent).translate(_SUPERSCRIPTS)


def _setup_rows(out) -> dict:
    """Display rows describing how the calculation was set up.

    Reads ``Crystal_output.get_calculation_info`` (CRYSTALClear >= 0.2.16).
    Older CRYSTALClear releases don't have it, so the panel simply falls back
    to the computed-properties rows instead of erroring.
    """
    try:
        info = out.get_calculation_info()
    except Exception:  # noqa: BLE001 - absent method, or an output it can't read
        return {}

    rows: dict = {}

    def _put(label: str, value) -> None:
        """Add a row, dropping anything the output didn't report."""
        if value is None or value == "" or value == []:
            return
        rows[label] = str(value)

    _put("Code", info.get("code"))
    _put("Task", " + ".join(info.get("run_type") or []))
    if info.get("terminated") is False:
        # Worth flagging: energies and geometries below may be from a partial run.
        _put("Termination", "did not terminate normally")

    method = info.get("method")
    if method == "DFT" and info.get("hybrid_exchange"):
        method = f"DFT (hybrid, {float(info['hybrid_exchange']):g}% exact exchange)"
    _put("Method", method)
    _put("Exchange", _functional(info.get("exchange")))
    _put("Correlation", _functional(info.get("correlation")))
    _put("Dispersion", _dispersion(info.get("dispersion")))
    shell_type = info.get("shell_type")
    _put("Shell type", shell_type.capitalize() if shell_type else None)

    shrink = info.get("shrink")
    if shrink:
        # Mesh and count go on separate rows: together they were too wide for
        # the dock and the count got cut off. Non-breaking spaces keep the mesh
        # itself on one line in a narrow dock instead of one factor per line.
        _put("k-point mesh", "\u00a0\u00d7\u00a0".join(str(int(s)) for s in shrink))
    ibz = info.get("n_kpoints_ibz")
    _put("k points (IBZ)", int(ibz) if ibz else None)

    _put("Basis functions", info.get("n_ao"))
    _put("Shells", info.get("n_shells"))
    electrons, core = info.get("n_electrons"), info.get("n_core_electrons")
    if electrons is not None:
        _put("Electrons/cell", f"{int(electrons)} ({int(core)} core)" if core else electrons)
    _put("Symmetry ops", info.get("n_symmops"))

    toldee = info.get("toldee")
    _put("SCF tolerance", f"10{_superscript(-int(toldee))} Ha" if toldee else None)
    tolinteg = info.get("tolinteg")
    _put("TOLINTEG", " ".join(str(int(t)) for t in tolinteg) if tolinteg else None)
    grid = info.get("grid_points")
    _put("DFT grid", f"{int(grid):,}" if grid else None)
    return rows


# ── shared by the savers ──────────────────────────────────────────────────
def _has_lattice(structure: Structure) -> bool:
    """Whether ``structure`` should be written as a periodic cell at all.

    A structure with no periodic direction — or with a degenerate (all-zero)
    cell — is a molecule: there is no lattice for pymatgen to build on.
    """
    return bool(structure.is_periodic) and not np.allclose(structure.cell, 0.0)


def _to_pymatgen(structure: Structure):
    """A pymatgen ``Structure``, with boundary-completion duplicates merged out.

    The structure handed to a saver is the *displayed* cell, which may be
    "packed" (boundary atoms duplicated as periodic images). pymatgen's symmetry
    finder raises on those coincident sites, so they are collapsed first
    (``merge_sites`` returns a new Structure in current pymatgen; older versions
    mutate in place and return ``None``).
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    pmg = AseAtomsAdaptor().get_structure(structure.to_ase())
    merged = pmg.merge_sites(tol=0.01, mode="delete")
    return pmg if merged is None else merged


def _to_pymatgen_molecule(structure: Structure):
    """A pymatgen ``Molecule`` — for systems with no lattice to speak of."""
    from pymatgen.core.structure import Molecule

    atoms = structure.to_ase()
    return Molecule(atoms.get_chemical_symbols(), atoms.get_positions())


# ── internal conversions (avoid CRYSTALClear.convert's broken wrappers) ────
def _is_gui(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext == ".gui" or os.path.basename(path).startswith("fort.34")


def _is_cif(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".cif"


def _out_to_ase(path: str, initial: bool) -> Atoms:
    from CRYSTALClear.crystal_io import Crystal_output

    return _out_geometry_to_ase(Crystal_output(path), initial=initial)


def _out_geometry_to_ase(out, initial: bool) -> Atoms:
    """The geometry of an open ``Crystal_output``, with its true dimensionality.

    ``get_geometry`` returns a pymatgen ``Structure``, which is 3D-periodic by
    construction: a 2D slab comes back flagged periodic along c as well, where
    CRYSTAL has put a formal 500 Å vacuum vector. Consumers that trust ``pbc``
    then treat that vacuum as a real lattice direction — the cell is drawn 500 Å
    tall, and CrystalNN's Voronoi tessellation hangs on it. CRYSTAL states the
    dimensionality in the output, so use it (as the ``.gui`` reader already does).
    """
    atoms = _pmg_to_ase(out.get_geometry(initial=initial))
    atoms.set_pbc(_out_pbc(out))
    return atoms


def _out_primitive_to_ase(out, initial: bool) -> Atoms:
    """The primitive cell of an open ``Crystal_output``, with its dimensionality.

    The counterpart of :func:`_out_geometry_to_ase` for a run that expanded its
    cell: after SCELPHONO the geometry sections describe the supercell the force
    constants were computed in, while the modes CRYSTAL prints belong to the
    cell that was expanded.
    """
    atoms = _pmg_to_ase(out.get_primitive_geometry(initial=initial))
    atoms.set_pbc(_out_pbc(out))
    return atoms


def _out_pbc(out) -> tuple:
    """CRYSTAL dimensionality as ase ``pbc`` — 3=bulk, 2=slab, 1=polymer, 0=molecule.

    Falls back to fully periodic if the output doesn't state a dimensionality,
    which is what every consumer assumed before.
    """
    try:
        dim = int(out.get_dimensionality())
    except Exception:  # noqa: BLE001 - absent/unparsable: keep the old assumption
        return (True, True, True)
    return (dim >= 1, dim >= 2, dim >= 3)


def _cif_to_ase(path: str) -> Atoms:
    """Read a CIF into ase.Atoms via pymatgen (applies the file's symmetry).

    pymatgen's ``CifParser`` expands the asymmetric unit by the CIF's symmetry
    operations, so the full conventional cell is returned — matching what a
    crystallographer expects, and what the CRYSTAL readers already produce.
    """
    from pymatgen.io.cif import CifParser

    structures = CifParser(path).parse_structures(primitive=False)
    if not structures:
        raise ValueError("no structure found in the CIF file")
    return _pmg_to_ase(structures[0])


def _pmg_to_ase(pmg_obj) -> Atoms:
    """Convert a pymatgen Structure *or* Molecule to ase.Atoms."""
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor().get_atoms(pmg_obj)


def _gui_to_ase(path: str) -> Atoms:
    """Build ase.Atoms directly from a CRYSTAL .gui via Crystal_gui.read_gui."""
    from CRYSTALClear.crystal_io import Crystal_gui

    g = Crystal_gui().read_gui(path)
    numbers = [int(z) for z in g.atom_number]
    positions = np.asarray(g.atom_positions, dtype=float)  # cartesian Angstrom
    cell = np.asarray(g.lattice, dtype=float)
    dim = int(g.dimensionality)
    pbc = (dim >= 1, dim >= 2, dim >= 3)  # CRYSTAL: 3=bulk,2=slab,1=polymer,0=molecule
    return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)


__all__ = [
    "load_adp",
    "load_structure",
    "load_dispersion",
    "load_phonons",
    "save_structure_gui",
    "save_structure_cif",
    "read_atoms",
    "output_properties",
]
