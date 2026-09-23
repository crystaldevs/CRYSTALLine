"""Generate a CRYSTAL23 ``.d12`` input deck from a :class:`Structure`.

This is the write-side companion to :mod:`crystalline.crystalio.loader`: the app
already *reads* CRYSTAL geometries and *plots* CRYSTAL results, and this module
lets it *author* the input that produces them — closing the build → run →
visualise loop in one tool.

A CRYSTAL deck is three (or four) ``END``-terminated blocks:

1. **Geometry** — a title, a dimensionality keyword (``CRYSTAL``/``MOLECULE``/…),
   the space group, the *minimal* lattice parameters, and the **asymmetric unit**
   (one representative atom per symmetry orbit). Geometry keywords (``SCELPHONO``)
   and task blocks (``OPTGEOM``, ``FREQCALC``, ``EOS``, ``ELASTCON``, ``QHA``,
   ``CPHF``) live here too.
2. **Basis set** — one ``BASISSET`` card naming an internal library. ``BASISSET``
   also serves as block 1's terminator (it "replaces the final END").
3. **Hamiltonian / SCF** — an optional ``DFT…END`` sub-block (functional, grid),
   then ``SHRINK`` (k-mesh) and convergence keywords, closed by ``END``.

Design choices for this first cut (see the module's tests for the exact decks):

* **Geometry is derived, not asked for.** The asymmetric unit is found with the
  same pymatgen ``SpacegroupAnalyzer`` that ``loader.save_structure_gui`` uses,
  so the space group and cell agree with what the ``.gui`` writer would produce.
  We symmetrise the *conventional standard* cell (not the primitive input), so a
  cubic crystal prints one ``a`` and fractional coordinates in the standard
  setting with flags ``0 0 0``. Any failure falls back to a ``P1`` listing of
  every atom — a valid deck, never a hard error (mirroring the ``.gui`` writer).
* **Scope.** Every dimensionality is written, each with its own coordinate
  convention (3D fractional; 2D ``x,y`` fractional and ``z`` in Å; 1D ``x``
  fractional and ``y,z`` in Å; 0D all Å) and its own minimal set of lattice
  parameters. Symmetry reduction to the asymmetric unit is applied to 3D
  crystals; slabs and polymers are written in their trivial group with every
  atom listed, because the layer/rod-group orbits needed to reduce them are not
  available here. Cells whose periodic directions are not already the ones
  CRYSTAL expects (slab in ``xy``, polymer along ``x``) are rotated into place.
* **Basis sets** are CRYSTAL's internal libraries via ``BASISSET`` (all-electron,
  so plain atomic numbers, no ECP offset).

pymatgen is imported lazily so ``core`` stays importable (and unit-testable)
without a display, matching the rest of the package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from crystalline.core.measure import angle_between
from crystalline.core.structure import Structure

# CRYSTAL's internal basis-set libraries selectable by the ``BASISSET`` card
# (manual §2.2.2). ``CUSTOM`` is excluded — it needs explicit shell cards, which
# this builder doesn't author yet.
INTERNAL_BASIS_SETS: Tuple[str, ...] = (
    "POB-TZVP",
    "POB-TZVP-REV2",
    "POB-DZVP",
    "POB-DZVPP",
    "POB-DZVP-REV2",
    "STO-3G",
    "STO-6G",
)

# Stand-alone exchange-correlation keywords, grouped as the manual groups them
# (§4.1) and paired with the name people actually search for. The keyword is
# what CRYSTAL reads; the description is what makes it findable — PBE's
# stand-alone keyword is ``PBEXC``, so a flat list of keywords hides the most
# common functional in the set behind a spelling nobody looks for.
FUNCTIONAL_GROUPS: Tuple[Tuple[str, Tuple[Tuple[str, str], ...]], ...] = (
    ("LDA and GGA", (
        ("SVWN", "LDA — Slater exchange + VWN correlation"),
        ("BLYP", "GGA — Becke 88 + LYP"),
        ("PBE", "GGA — Perdew–Burke–Ernzerhof"),
        ("PBESOLXC", "GGA — PBEsol, PBE revised for solids"),
        ("SOGGAXC", "GGA — SOGGA"),
        ("SOGGA11", "GGA — SOGGA11"),
    )),
    ("Global hybrids", (
        ("B3LYP", "20% exact exchange — the common molecular default"),
        ("PBE0", "25% exact exchange on PBE — the common solid-state default"),
        ("PBESOL0", "25% exact exchange on PBEsol"),
        ("B3PW", "Becke 3-parameter with PW91 correlation"),
        ("B1WC", "one-parameter hybrid, WC exchange"),
        ("WC1LYP", "WC exchange + LYP correlation"),
        ("B97H", "B97 hybrid"),
        ("PBE0-13", "PBE0 with 1/3 exact exchange"),
        ("SOGGA11X", "SOGGA11 hybrid"),
        ("mPW1PW91", "modified PW91 hybrid"),
        ("mPW1K", "mPW1PW91 reparametrised for kinetics"),
    )),
    ("Range-separated hybrids", (
        ("HSE06", "screened hybrid — solids with a small gap"),
        ("HSEsol", "HSE06 on PBEsol"),
        ("SC-BLYP", "short-range corrected BLYP"),
        ("HISS", "middle-range corrected"),
        ("RSHXLDA", "range-separated LDA exchange"),
        ("LC-wPBE", "long-range corrected PBE"),
        ("LC-wPBEsol", "long-range corrected PBEsol"),
        ("LC-wBLYP", "long-range corrected BLYP"),
        ("wB97", "Head-Gordon wB97"),
        ("wB97X", "wB97 with short-range exact exchange"),
        ("LC-BLYP", "long-range corrected BLYP"),
        ("CAM-B3LYP", "Coulomb-attenuated B3LYP"),
        ("LC-PBE", "long-range corrected PBE"),
    )),
    ("meta-GGA", (
        ("M06L", "Minnesota local meta-GGA"),
        ("revM06L", "revised M06-L"),
        ("MN15L", "Minnesota MN15-L"),
        ("SCAN", "strongly constrained and appropriately normed"),
        ("r2SCAN", "regularised, restored SCAN"),
    )),
    ("meta-GGA hybrids", (
        ("B1B95", "one-parameter B95 hybrid"),
        ("mPW1B95", "modified PW + B95"),
        ("mPW1B1K", "mPW1B95 for kinetics"),
        ("PW6B95", "PW6B95"),
        ("PWB6K", "PW6B95 for kinetics"),
        ("M05", "Minnesota M05"),
        ("M052X", "M05 with doubled exact exchange"),
        ("M06", "Minnesota M06"),
        ("M062X", "M06 with doubled exact exchange"),
        ("M06HF", "full exact exchange"),
        ("MN15", "Minnesota MN15"),
        ("revM06", "revised M06"),
        ("SCAN0", "SCAN hybrid"),
        ("r2SCANh", "r2SCAN hybrid, 10%"),
        ("r2SCAN0", "r2SCAN hybrid, 25%"),
        ("r2SCAN50", "r2SCAN hybrid, 50%"),
    )),
)

# The flat list of keywords, for anything that just needs to know what is valid.
COMMON_FUNCTIONALS: Tuple[str, ...] = tuple(
    keyword for _group, entries in FUNCTIONAL_GROUPS for keyword, _why in entries
)

# What someone types when they mean a keyword spelled otherwise. The manual
# lists PBEXC as the stand-alone exchange+correlation keyword, but PBE is what
# the code takes and what everybody writes, so PBE is what is offered; PBEXC
# still resolves, for anyone copying from an older deck.
FUNCTIONAL_ALIASES = {
    "PBEXC": "PBE",
    "PBESOL": "PBESOLXC",
    "SOGGA": "SOGGAXC",
    "LDA": "SVWN",
}

# Separate potentials for the EXCHANGE / CORRELAT keywords (manual §4.1). Leaving
# exchange unset falls back to Hartree-Fock exchange; leaving correlation unset
# gives an exchange-only functional.
EXCHANGE_FUNCTIONALS: Tuple[str, ...] = (
    "LDA", "VBH", "BECKE", "mPW91", "PBE", "PBESOL", "PWGGA", "SOGGA", "WCGGA",
)
CORRELATION_FUNCTIONALS: Tuple[str, ...] = (
    "PWLSD", "PZ", "VBH", "VWN", "LYP", "P86", "PBE", "PBESOL", "PWGGA", "WL", "B95",
)

# A two-component SCF supports only pure and global-hybrid LDA/GGA functionals
# (manual ch. 6 preamble), so these families are rejected when SOC is requested.
_MGGA_AND_RSH = frozenset(
    name.upper()
    for name in (
        "HSE06", "HSEsol", "SC-BLYP", "HISS", "RSHXLDA", "LC-wPBE", "LC-wPBEsol",
        "LC-wBLYP", "wB97", "wB97X", "LC-BLYP", "CAM-B3LYP", "LC-PBE",
        "M06L", "revM06L", "MN15L", "SCAN", "r2SCAN", "B1B95", "mPW1B95", "mPW1B1K",
        "PW6B95", "PWB6K", "M05", "M052X", "M06", "M062X", "M06HF", "MN15", "revM06",
        "SCAN0", "r2SCANh", "r2SCAN0", "r2SCAN50",
    )
)

# Spin(-current) DFT formulation for a two-component run (manual §6.3).
NONCOLLINEAR_MODES: Tuple[str, ...] = ("COLLINEAR", "NONCOLC", "NONCOLSF")

# Starting guesses available inside a TWOCOMPON block (manual §6.2).
TWO_COMPONENT_GUESSES: Tuple[str, ...] = (
    "GUESSPAT", "GCOREROT", "GUESSPATNC", "GUESSPNOSO", "GUESSPSO",
)

# Functionals CRYSTAL has D3-parametrised (manual §5.1): the "-D3" suffix is a
# valid stand-alone keyword only for these. Compared case-insensitively.
_D3_FUNCTIONALS = frozenset(
    {"BLYP", "PBE", "B97", "B3LYP", "PBE0", "PW1PW", "M06", "HSE06", "HSESOL", "LC-WPBE"}
)

# Integration grids (manual §4.3). ``None`` keeps CRYSTAL's default (XLGRID).
GRIDS: Tuple[str, ...] = ("XLGRID", "XXLGRID", "XXXLGRID", "HUGEGRID")

# TOLDEE (SCF energy convergence, 10^-n) defaults by task (manual, TOLDEE table).
# CPHF is not in that table; it needs a well-converged wave function, so a tight
# value is used rather than the single-point default.
_TOLDEE_DEFAULTS = {
    "SP": 6,
    "OPTGEOM": 7,
    "FREQCALC": 10,
    "EOS": 8,
    "ELASTCON": 8,
    "CPHF": 9,
    "DISPERSION": 10,  # a FREQCALC run underneath
    "QHA": 10,  # phonons at several volumes, so the frequency tolerance applies
    "ANHAPES": 10,  # anharmonic PES on top of a harmonic FREQCALC
    "ANHARM": 10,  # numerical X-H stretching curve needs tight energies
}

# Task keywords that assume an optimised geometry and so can pre-optimise with a
# PREOPTGEOM sub-block; each opens a block-1 block closed by END.
_EQUILIBRIUM_TASKS = ("FREQCALC", "EOS", "ELASTCON", "DISPERSION", "QHA", "ANHAPES")

# The block-opening keyword each task actually writes. Phonon dispersion is not a
# keyword of its own: it is a FREQCALC run with the DISPERSI sub-keyword.
_TASK_BLOCK_KEYWORD = {
    "FREQCALC": "FREQCALC",
    "EOS": "EOS",
    "ELASTCON": "ELASTCON",
    "DISPERSION": "FREQCALC",
    "QHA": "QHA",
    "ANHAPES": "FREQCALC",
}

# What the five TOLINTEG truncation thresholds control (manual §18.3-18.4 and the
# TOLINTEG entry). Each is a power of ten: larger = tighter (more accurate, slower).
TOLINTEG_LABELS: Tuple[Tuple[str, str, str], ...] = (
    (
        "ITOL1",
        "Coulomb overlap",
        "A bielectronic Coulomb integral is neglected when the overlap between the "
        "two charge distributions is smaller than 10⁻ᴵᵀᴼᴸ¹.",
    ),
    (
        "ITOL2",
        "Coulomb penetration",
        "Decides whether a Coulomb integral is evaluated exactly or through the "
        "approximate multipolar expansion.",
    ),
    (
        "ITOL3",
        "HF exchange overlap",
        "Truncates the exchange series: terms whose charge overlap falls below "
        "10⁻ᴵᵀᴼᴸ³ are discarded.",
    ),
    (
        "ITOL4",
        "Exchange pseudo-overlap (g)",
        "Truncation of the g summation in the exchange series.",
    ),
    (
        "ITOL5",
        "Exchange pseudo-overlap (n)",
        "Truncation of the n summation. Must be more severe than ITOL4 — the manual "
        "recommends ITOL5 exceed it by 3 to 8 orders of magnitude to keep the SCF stable.",
    ),
)

# The manual's stability rule of thumb for the exchange series (§18.4).
_TOLINTEG_MIN_GAP = 3


class CrystalInputError(ValueError):
    """Raised when a deck can't be built (e.g. an unsupported dimensionality)."""


@dataclass
class GeometryOptions:
    """Block-1 geometry controls.

    ``supercell`` is the general ``SUPERCEL`` expansion matrix — new cell vectors
    as linear combinations of the primitive ones. It is distinct from the
    ``SCELPHONO`` matrix on :class:`TaskOptions`, which orders atoms the way a
    phonon-dispersion run needs; the two cannot be combined.
    """

    title: str = "Generated by CRYSTALLine"
    use_symmetry: bool = True  # False -> P1, every atom listed
    symprec: float = 1e-2  # pymatgen symmetry tolerance (matches crystallography.analyze)
    # A deliberately *lowered* symmetry: the point operators to keep, as integer
    # matrices in the conventional cell's basis. Empty means "whatever the
    # structure has", which is the ordinary case. Lowering it does not move an
    # atom — it makes atoms that the full group held equivalent independent, so
    # a distortion the symmetry forbids can develop. See
    # :mod:`crystalline.core.symmetry_reduction`.
    kept_rotations: Tuple[tuple, ...] = ()
    supercell: Optional[Sequence[Sequence[int]]] = None  # SUPERCEL expansion matrix
    supercell_noshift: bool = False  # NOSHIFT: keep the origin where it is


@dataclass
class BasisOptions:
    """Block-2 basis set (internal library only, for now)."""

    name: str = "POB-TZVP"


@dataclass
class MethodOptions:
    """Block-3 Hamiltonian: HF, or DFT with a functional (+ optional D3, grid).

    A DFT functional is given either as one stand-alone keyword
    (``functional_mode="COMBINED"``) or as separate ``EXCHANGE`` and ``CORRELAT``
    potentials (``"SPLIT"``). In split mode an unset exchange leaves CRYSTAL's
    Hartree-Fock exchange in place, and an unset correlation gives an
    exchange-only functional — both are documented defaults, not omissions.
    """

    kind: str = "DFT"  # "HF" or "DFT"
    functional_mode: str = "COMBINED"  # "COMBINED" (one keyword) or "SPLIT"
    functional: str = "PBE0"  # stand-alone keyword, used when COMBINED
    exchange: Optional[str] = None  # EXCHANGE keyword, used when SPLIT
    correlation: Optional[str] = None  # CORRELAT keyword, used when SPLIT
    hybrid_percent: Optional[int] = None  # HYBRID: exact-exchange percentage
    # NONLOCAL: (B, C) weights of the non-local exchange and correlation parts.
    # With HYBRID this reproduces Becke 3-parameter functionals exactly — the
    # manual gives B3LYP as BECKE/LYP + HYBRID 20 + NONLOCAL 0.9 0.81.
    nonlocal_weights: Optional[Tuple[float, float]] = None
    dispersion_d3: bool = False  # append "-D3" to the functional (DFT only)
    grid: Optional[str] = None  # None -> CRYSTAL default grid


@dataclass
class TwoComponentOptions:
    """TWOCOMPON: two-component spinor SCF and spin-orbit coupling (manual ch. 6).

    ``SOC`` needs a SOREP operator supplied through an effective core potential —
    one of CRYSTAL's internal libraries (STUTSC/STUTLC/STUTSH, COLUSC/COLULC/
    COLUSH) or manual input via ``INPSOC`` — so the basis set has to provide it.

    Opening the block deactivates DIIS and symmetry, and the implementation
    covers single-point energies only: geometry optimisation, frequencies and
    response properties are unavailable in the two-component spinor basis, as are
    meta-GGA and range-separated functionals. :func:`build_input` enforces both.
    """

    enabled: bool = False  # write the TWOCOMPON block at all
    soc: bool = False  # SOC: include the spin-orbit operator
    guess: str = "GUESSPAT"  # starting guess for the density matrix
    guess_angles: Tuple[float, float] = (0.0, 0.0)  # GCOREROT: (RTHETA, RPHI)
    second_variational: bool = False  # 2NDVARIAT
    second_variational_rot: int = 0  # IROT
    second_variational_angles: Tuple[float, float] = (0.0, 0.0)  # if IROT != 0
    print_energies: bool = False  # PRTENESOC
    spinorlock: Optional[Tuple[int, int]] = None  # SPINORLOCK: (NSPIN, NCYC)
    noncollinear: str = "COLLINEAR"  # COLLINEAR | NONCOLC | NONCOLSF (in the DFT block)
    extra_keywords: str = ""


@dataclass
class ScfOptions:
    """Block-3 SCF controls."""

    shrink: Optional[int] = None  # k-mesh; None -> suggested from the cell
    tolinteg: Tuple[int, int, int, int, int] = (7, 7, 7, 7, 14)
    toldee: Optional[int] = None  # None -> task default (SP 6, OPTGEOM 7)
    maxcycle: int = 100
    # SPIN inside the DFT block, or UHF for Hartree-Fock. Everything below is
    # only written when this is on: SPINLOCK and ATOMSPIN are meaningless in a
    # closed-shell run and CRYSTAL rejects them there.
    spin_polarized: bool = False
    # SPINLOCK: (NSPIN, NCYC) — hold n(alpha) - n(beta) at NSPIN for NCYC
    # cycles, then let the SCF relax. The usual way to reach a particular
    # magnetic state instead of whichever one the guess falls into.
    spinlock: Optional[Tuple[int, int]] = None
    # ATOMSPIN: (atom label, +1 or -1) per atom, setting the starting spin of
    # each. The labels are CRYSTAL's own numbering, as printed in the output —
    # which is the numbering the app shows in the structure panel.
    atomspin: Tuple[Tuple[int, int], ...] = ()


@dataclass
class CphfOptions:
    """Coupled-Perturbed HF/KS controls (manual ch. 10).

    Used both for a stand-alone ``CPHF`` task and for the ``INTCPHF`` sub-block
    that FREQCALC opens for analytical IR/Raman tensors — the two share the same
    keyword vocabulary. ``None`` means "leave CRYSTAL's default alone".

    The ``*2`` fields tune the second Self-Consistent Coupled-Perturbed cycle
    (CPHF2), which only runs for fourth-order calculations and for Raman
    intensities; they are meaningless for a plain second-order CPHF.
    """

    order: str = "SECOND"  # SECOND | THIRD (hyperpolarisability) | FOURTH
    fmixing: Optional[int] = None
    maxcycle: Optional[int] = None
    tolalpha: Optional[int] = None  # |Δα| < 10^-n
    fmixing2: Optional[int] = None
    maxcycle2: Optional[int] = None
    tolgamma: Optional[int] = None  # |ΔU| < 10^-n
    extra_keywords: str = ""


@dataclass
class FreqOptions:
    """FREQCALC sub-keywords (manual ch. 8).

    ``raman_intensities`` implies IR intensities computed analytically: the
    manual requires INTRAMAN to always accompany INTENS, and the Raman tensor
    comes from the CPHF block opened by INTCPHF. :func:`build_input` enforces
    that rather than emitting a deck CRYSTAL would reject.
    """

    numderiv: Optional[int] = None  # 1 = one-sided quotient, 2 = central difference
    stepsize: Optional[float] = None  # displacement for numerical derivatives (Å)
    ir_intensities: bool = False  # INTENS
    ir_technique: str = "INTPOL"  # INTPOL (Berry phase) | INTLOC (Wannier) | INTCPHF
    raman_intensities: bool = False  # INTRAMAN (forces INTENS + INTCPHF)
    ir_spectrum: bool = False  # IRSPEC block
    raman_spectrum: bool = False  # RAMSPEC block
    raman_experiment: Optional[Tuple[float, float]] = None  # RAMANEXP: (T in K, laser nm)
    temperature: Optional[Tuple[int, float, float]] = None  # TEMPERAT: (steps, T1, T2)
    pressure: Optional[Tuple[int, float, float]] = None  # PRESSURE: (steps, P1, P2)
    analysis: bool = False  # ANALYSIS of the vibrational modes
    print_hessian: bool = False  # PRINT (Hessian and eigenvectors)
    restart: bool = False  # RESTART from FREQINFO.DAT
    cphf: CphfOptions = field(default_factory=CphfOptions)
    extra_keywords: str = ""


@dataclass
class OptGeomOptions:
    """OPTGEOM convergence sub-keywords (manual §7.3)."""

    toldeg: Optional[float] = None  # RMS on gradient
    toldex: Optional[float] = None  # RMS on displacement
    maxcycle: Optional[int] = None
    extra_keywords: str = ""


@dataclass
class EosOptions:
    """EOS sub-keywords (manual ch. 12)."""

    volume_range: Optional[Tuple[float, float, int]] = None  # RANGE: (min, max, points)
    pressure_range: Optional[Tuple[float, float, int]] = None  # PRANGE: (min, max, points)
    extra_keywords: str = ""


@dataclass
class DispersionOptions:
    """Phonon-dispersion sub-keywords of FREQCALC (manual §8.8).

    ``DISPERSI`` computes frequencies across the Brillouin zone using the
    direct-space supercell built by ``SCELPHONO`` (see ``TaskOptions.supercell``),
    so a dispersion run without a supercell is only the Gamma point repeated.

    ``bands_path`` holds one ``I1 I2 I3 J1 J2 J3`` segment per line; ``NLINE`` is
    derived from how many segments are given.
    """

    noksymdisp: bool = False  # do not label phonons by irrep
    bands: bool = False  # BANDS: phonon band structure for plotting
    bands_shrink: int = 16  # ISS: shrinking factor the segment ends refer to
    bands_points: int = 30  # NSUB: k points along each line
    bands_path: str = ""  # one "I1 I2 I3 J1 J2 J3" segment per line
    interphess: Optional[Tuple[int, int, int, int]] = None  # (L1, L2, L3, IPRINT)
    wang: Optional[Sequence[float]] = None  # 9 dielectric-tensor elements (3D, polar)
    pdos: Optional[Tuple[float, int, int]] = None  # (NUMA, NBIN, LPRO)
    ins: Optional[Tuple[float, int, int]] = None  # (NUMA, NBIN, NWTYPE)
    extra_keywords: str = ""


@dataclass
class QhaOptions:
    """Quasi-harmonic approximation sub-keywords (manual ch. 9).

    QHA relaxes and computes phonons at several volumes, then fits an
    equation of state, so it is normally paired with a ``SCELPHONO`` supercell
    (see ``TaskOptions.supercell``) and started from a fully optimised geometry.
    """

    step: Optional[float] = None  # STEP: volume step, per cent (default 3)
    points: Optional[int] = None  # POINTS: number of volumes (4, 7 or 13)
    temperature: Optional[Tuple[int, float, float]] = None  # TEMPERAT: (steps, T1, T2)
    volume_range: Optional[Tuple[float, float, int]] = None  # VRANGE: (min, max, points)
    restart: bool = False  # RESTART from an incomplete run
    restart2: bool = False  # RESTART2 from a complete run (e.g. new temperatures)
    extra_keywords: str = ""


@dataclass
class AnharmOptions:
    """ANHARM: anharmonic X–H (X–D) stretching (manual §8.11).

    A stand-alone block-1 keyword — it does *not* live inside FREQCALC, and no
    harmonic frequency run is needed first. ``atom_label`` is the sequence number
    of the hydrogen (or deuterium) to displace, as CRYSTAL numbers atoms after
    reading the geometry; it moves along the direction to its first neighbour.
    """

    atom_label: int = 1  # LB
    keepsymm: bool = False  # displace all symmetry-equivalent X–H bonds together
    points26: bool = False  # 26 points instead of the default 7
    noguess: bool = False  # atomic-density SCF guess at each point
    print_extended: bool = False
    test_only: bool = False  # TESTANHA: check the neighbour, compute nothing
    isotopes: Sequence[Tuple[int, float]] = ()  # (atom label, mass in amu)
    extra_keywords: str = ""


@dataclass
class AnhapesOptions:
    """ANHAPES and the VSCF/VCI anharmonic states (manual §8.12).

    ``ANHAPES`` sits inside FREQCALC and evaluates the cubic/quartic terms of the
    potential energy surface for a chosen subset of normal modes; ``VSCF`` and
    ``VCI`` then solve the vibrational problem on that surface. Requesting VCI
    runs a VSCF first, whether or not ``vscf`` is set.

    ``modes`` lists the mode numbers from the harmonic calculation. Translational
    (and, for molecules, rotational) modes must be excluded.
    """

    modes: str = ""  # whitespace- or newline-separated mode numbers
    scheme: int = 3  # numerical scheme 1–4; 3 balances cost and accuracy
    step: float = 0.9  # step h; 0.9 is the manual's recommended value
    restart: bool = False  # RESTART: reuse a stored harmonic Hessian
    restart_pes: bool = False  # RESTPES: reuse anharmonic PES terms (SCANPES.DAT)
    vscf: bool = False
    vscf_tol: Optional[int] = None  # VSCFTOL: convergence at 10^-n cm^-1
    vscf_mix: Optional[int] = None  # VSCFMIX: Fock-mixing percentage
    vci: bool = False
    vci_quanta: int = 6  # N_quanta: max excitation quanta in a configuration
    vci_modes: int = 3  # N_modes: max simultaneously excited modes
    vci_guess: int = 1  # 0 = harmonic initial guess, 1 = VSCF (VCI@VSCF)
    extra_keywords: str = ""


@dataclass
class ElasticOptions:
    """ELASTCON sub-keywords (manual ch. 13)."""

    numderiv: Optional[int] = None  # points per strain
    stepsize: Optional[float] = None  # strain step
    clamped_ion: bool = False  # CLAMPION: skip the internal-strain relaxation
    extra_keywords: str = ""


@dataclass
class TaskOptions:
    """What to compute.

    ``kind`` is one of ``SP`` (single point), ``OPTGEOM`` (geometry optimisation),
    ``FREQCALC`` (harmonic frequencies), ``EOS`` (equation of state), ``ELASTCON``
    (elastic constants) or ``CPHF`` (dielectric/polarisability response). The
    equilibrium-requiring tasks (FREQCALC/EOS/ELASTCON) can pre-optimise the
    geometry with ``preoptimize``; per-task sub-keywords live in the nested
    option objects, mirroring CRYSTAL's own block nesting.
    """

    kind: str = "SP"
    optimize_cell: bool = True  # OPTGEOM: relax cell + atoms (CRYSTAL default); False -> ATOMONLY
    preoptimize: bool = False  # PREOPTGEOM (full relaxation first), where supported
    # SCELPHONO expansion matrix. A *geometry*-block keyword, written between the
    # atoms and the task block; used by phonon dispersion and QHA.
    supercell: Optional[Sequence[Sequence[int]]] = None
    optgeom: OptGeomOptions = field(default_factory=OptGeomOptions)
    anharm: AnharmOptions = field(default_factory=AnharmOptions)
    anhapes: AnhapesOptions = field(default_factory=AnhapesOptions)
    freq: FreqOptions = field(default_factory=FreqOptions)
    cphf: CphfOptions = field(default_factory=CphfOptions)
    eos: EosOptions = field(default_factory=EosOptions)
    elastic: ElasticOptions = field(default_factory=ElasticOptions)
    dispersion: DispersionOptions = field(default_factory=DispersionOptions)
    qha: QhaOptions = field(default_factory=QhaOptions)


@dataclass
class CrystalInputSpec:
    """The full specification the builder turns into a ``.d12`` deck.

    Structured as one dataclass per block so new options (ECPs, FREQCALC, elastic
    constants…) can be added to a section without disturbing the rest.
    """

    geometry: GeometryOptions = field(default_factory=GeometryOptions)
    basis: BasisOptions = field(default_factory=BasisOptions)
    method: MethodOptions = field(default_factory=MethodOptions)
    scf: ScfOptions = field(default_factory=ScfOptions)
    two_component: TwoComponentOptions = field(default_factory=TwoComponentOptions)
    task: TaskOptions = field(default_factory=TaskOptions)
    extra_keywords: str = ""  # free text appended to block 3 before its END


# ── public API ─────────────────────────────────────────────────────────────
def build_input(structure: Structure, spec: Optional[CrystalInputSpec] = None) -> str:
    """Return a complete CRYSTAL ``.d12`` deck (a string ending in a newline).

    Raises :class:`CrystalInputError` for an empty structure, a missing basis
    name, or a dimensionality not yet supported (1D/2D).
    """
    spec = spec or CrystalInputSpec()
    if len(structure) == 0:
        raise CrystalInputError("Cannot build an input for an empty structure.")
    if not spec.basis.name.strip():
        raise CrystalInputError("A basis set must be chosen.")
    _check_two_component(spec)
    _check_spin(structure, spec.scf)
    if spec.geometry.supercell is not None and spec.task.supercell is not None:
        raise CrystalInputError(
            "SUPERCEL and SCELPHONO both build a supercell and cannot be combined — "
            "SCELPHONO already orders atoms the way a phonon calculation needs."
        )

    lines: List[str] = []
    lines += _geometry_lines(structure, spec.geometry)
    lines += _supercel_geometry_lines(spec.geometry)
    lines += _task_lines(spec.task)  # OPTGEOM sits in block 1, before BASISSET
    lines += ["BASISSET", spec.basis.name.strip()]  # replaces block-1 END
    lines += _hamiltonian_lines(structure, spec)
    return "\n".join(lines) + "\n"


def write_input(
    structure: Structure, path: str, spec: Optional[CrystalInputSpec] = None
) -> None:
    """Write :func:`build_input` to ``path`` (conventionally a ``.d12`` file)."""
    with open(path, "w") as fh:
        fh.write(build_input(structure, spec))


def tolinteg_warnings(tolinteg: Sequence[int]) -> List[str]:
    """Advisory problems with a TOLINTEG setting (empty when it looks sound).

    These are warnings, not errors: CRYSTAL will still run. The exchange-series
    rule comes from manual §18.4, which requires the ``n``-summation threshold
    (ITOL5) to be more severe than the ``g``-summation one (ITOL4), by three to
    eight orders of magnitude, or the SCF may destabilise.
    """
    values = [int(v) for v in tolinteg]
    if len(values) != 5:
        return ["TOLINTEG needs exactly five values."]
    messages: List[str] = []
    if any(v <= 0 for v in values):
        messages.append("TOLINTEG values must be positive.")
    if values[4] - values[3] < _TOLINTEG_MIN_GAP:
        messages.append(
            f"ITOL5 ({values[4]}) should exceed ITOL4 ({values[3]}) by at least "
            f"{_TOLINTEG_MIN_GAP} — the manual recommends 3–8 — to keep the SCF stable."
        )
    return messages


def suggest_shrink(structure: Structure) -> int:
    """A reasonable ``SHRINK`` (Monkhorst–Pack) factor from the cell size.

    Denser sampling for small cells, sparser for large ones — the usual rule of
    thumb of keeping ``shrink × a`` roughly constant. Non-periodic systems need
    no k-mesh, so return 1.
    """
    if not structure.is_periodic:
        return 1
    cell = np.asarray(structure.cell, dtype=float)
    pbc = structure.pbc
    lengths = [float(np.linalg.norm(cell[i])) for i in range(3) if pbc[i]]
    longest = max(lengths) if lengths else 10.0
    if longest <= 0:
        return 4
    return int(np.clip(round(60.0 / longest), 2, 16))


# ── geometry block ─────────────────────────────────────────────────────────
def _geometry_lines(structure: Structure, opts: GeometryOptions) -> List[str]:
    ndim = int(sum(bool(p) for p in structure.pbc))
    if ndim == 3:
        return [opts.title, "CRYSTAL"] + _crystal_body(structure, opts)
    if ndim == 2:
        return [opts.title, "SLAB"] + _slab_body(structure, opts)
    if ndim == 1:
        return [opts.title, "POLYMER"] + _polymer_body(structure)
    return [opts.title, "MOLECULE"] + _molecule_body(structure)


# CRYSTAL fixes which Cartesian axes carry the periodicity: a slab lies in the
# xy plane, a polymer runs along x. These permutations bring an arbitrary cell
# into that convention (the same ones CRYSTALClear's gui writer applies, so the
# two writers agree). A permutation matrix is orthogonal, so applying it to both
# the cell and the positions is a rigid rotation of the whole system.
_AXIS_TO_LAST = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
_AXIS_TO_FIRST = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)


def _oriented(structure: Structure, periodic_axes: List[int], target: str):
    """Return ``(cell, positions, axes)`` rotated into CRYSTAL's convention.

    ``target`` is ``"slab"`` (aperiodic direction → z) or ``"polymer"`` (periodic
    direction → x). The returned ``axes`` are the indices the periodic vectors
    occupy afterwards.
    """
    atoms = structure.to_ase()
    cell = np.asarray(atoms.get_cell(), dtype=float)
    positions = atoms.get_positions()

    if target == "slab":
        aperiodic = next(i for i in range(3) if i not in periodic_axes)
        rot = {0: _AXIS_TO_LAST, 1: _AXIS_TO_FIRST, 2: None}[aperiodic]
        axes = [0, 1]
    else:
        rot = {0: None, 1: _AXIS_TO_LAST, 2: _AXIS_TO_FIRST}[periodic_axes[0]]
        axes = [0]

    if rot is None:
        return cell, positions, axes
    # Reorder the cell rows to match, so row `axes[k]` is a periodic vector.
    order = [(i - (3 - _shift(rot))) % 3 for i in range(3)]
    return (cell @ rot)[order], positions @ rot, axes


def _shift(rot: np.ndarray) -> int:
    """How far the permutation ``rot`` cycles the axes (1 or 2)."""
    return 1 if np.allclose(rot, _AXIS_TO_LAST) else 2


def _slab_body(structure: Structure, opts: GeometryOptions) -> List[str]:
    """Block-1 body for a 2D slab: layer group, in-plane cell, asymmetric unit.

    Coordinates follow the manual's 2D convention — ``x`` and ``y`` fractional
    along the two periodic vectors, ``z`` in Ångström perpendicular to the slab.

    A slab's symmetry is a *layer group*, not a space group, and the app finds
    it already; this writes it. Where it cannot be found, or cannot be checked,
    layer group 1 with every atom listed is the fallback — always true, just
    more work for CRYSTAL.
    """
    if opts.use_symmetry:
        body = _layer_group_body(structure, opts.symprec)
        if body is not None:
            return body
    return _p1_slab_body(structure)


def _layer_group_body(structure: Structure, symprec: float):
    """The slab in its own layer group, or ``None`` if that cannot be trusted.

    ``None`` rather than a guess: a layer group number names a particular
    arrangement of axes, and a deck written in the wrong one is a plausible
    file describing a different surface. The group is only used once its own
    operations have been shown to rebuild the slab from the asymmetric unit.
    """
    from crystalline.core import slab_symmetry

    symmetry = slab_symmetry.analyse(structure, symprec)
    if symmetry is None or not slab_symmetry.regenerates(symmetry):
        return None
    numbers, positions = slab_symmetry.asymmetric_unit(symmetry)
    coordinates = slab_symmetry.deck_coordinates(symmetry, positions)
    lines = [str(symmetry.number),
             " ".join(f"{v:.6f}" for v in slab_symmetry.cell_record(symmetry)),
             str(len(numbers))]
    for z_number, row in zip(numbers, coordinates):
        lines.append(_atom_line(int(z_number), row))
    return lines


def _p1_slab_body(structure: Structure) -> List[str]:
    """Layer group 1, every atom — the always-valid fallback."""
    pbc = [bool(p) for p in structure.pbc]
    cell, positions, axes = _oriented(structure, [i for i, p in enumerate(pbc) if p], "slab")
    va, vb = cell[axes[0]], cell[axes[1]]
    a, b = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    gamma = angle_between(va, vb)

    normal = np.cross(va, vb)
    normal = normal / np.linalg.norm(normal)
    basis = np.column_stack([va, vb])  # (3, 2): the in-plane vectors

    lines = ["1", " ".join(f"{v:.6f}" for v in (a, b, gamma)), str(len(positions))]
    for z_number, r in zip(structure.numbers, positions):
        height = float(r @ normal)
        frac, *_ = np.linalg.lstsq(basis, r - height * normal, rcond=None)
        lines.append(_atom_line(int(z_number), (frac[0], frac[1], height)))
    return lines


def _polymer_body(structure: Structure) -> List[str]:
    """Block-1 body for a 1D polymer: rod group, repeat length, atoms.

    Coordinates follow the manual's 1D convention — ``x`` fractional along the
    periodic vector, ``y`` and ``z`` in Ångström across it. Rod group 1 is
    written with every atom listed, for the same reason as a slab: spglib has no
    rod groups, so there is no orbit information to reduce with.
    """
    pbc = [bool(p) for p in structure.pbc]
    cell, positions, axes = _oriented(structure, [i for i, p in enumerate(pbc) if p], "polymer")
    va = cell[axes[0]]
    a = float(np.linalg.norm(va))
    e1, e2, e3 = _polymer_frame(va)

    lines = ["1", f"{a:.6f}", str(len(positions))]
    for z_number, r in zip(structure.numbers, positions):
        lines.append(
            _atom_line(int(z_number), (float(r @ e1) / a, float(r @ e2), float(r @ e3)))
        )
    return lines


def _polymer_frame(a_vec: np.ndarray):
    """An orthonormal frame whose first axis runs along ``a_vec``.

    Prefers the world y as the second axis when it is already perpendicular, so a
    polymer already aligned with x keeps its familiar y and z coordinates instead
    of being spun about its own axis.
    """
    e1 = a_vec / np.linalg.norm(a_vec)
    y_hat = np.array([0.0, 1.0, 0.0])
    if abs(float(e1 @ y_hat)) < 1e-8:
        e2 = y_hat
    else:
        helper = np.eye(3)[int(np.argmin(np.abs(e1)))]
        e2 = np.cross(e1, helper)
    e2 = e2 / np.linalg.norm(e2)
    return e1, e2, np.cross(e1, e2)


def _crystal_body(structure: Structure, opts: GeometryOptions) -> List[str]:
    """Block-1 body for a 3D crystal: flags, space group, cell, asymmetric unit.

    Uses the conventional standard cell so the printed parameters and fractional
    coordinates are in CRYSTAL's default setting (flags ``0 0 0``). On any
    symmetry-analysis failure — or when the user turns symmetry off — falls back
    to ``P1`` with every atom listed.
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    pmg = AseAtomsAdaptor().get_structure(structure.to_ase())

    if opts.use_symmetry and opts.kept_rotations:
        body = _reduced_crystal_body(structure, opts)
        if body is not None:
            return body
        # A reduction that cannot be written is not quietly replaced by the full
        # symmetry — that would silently hand back the constraint the user asked
        # to be rid of. Every atom, in group 1, is the honest fallback.
        return _p1_crystal_body(pmg)
    if opts.use_symmetry:
        try:
            return _symmetric_crystal_body(pmg, opts.symprec)
        except Exception:  # noqa: BLE001 - symmetry undeterminable: fall back to P1
            pass
    return _p1_crystal_body(pmg)


def _reduced_crystal_body(structure: Structure, opts: GeometryOptions):
    """Block-1 body under a lowered symmetry, or ``None`` if it cannot be written.

    ``None`` rather than an exception, and never a silent promotion back to the
    full group: the check is whether declaring this group and this asymmetric
    unit rebuilds the crystal we started from, and if it does not the deck would
    be a plausible file describing a different one.
    """
    from crystalline.core import symmetry_reduction as reduction

    symmetry = reduction.analyse(structure, opts.symprec)
    if symmetry is None:
        return None
    rotations = [np.asarray(r, dtype=int) for r in opts.kept_rotations]
    subgroup = reduction.descend(symmetry, rotations)
    if not subgroup.writable:
        return None

    numbers, coords = reduction.asymmetric_unit(symmetry, subgroup.rotations)
    system = _system_of(subgroup.number)
    lines = ["0 0 0", str(subgroup.number),
             _lattice_line(system, _lattice_of(symmetry.lattice)), str(len(numbers))]
    for z, position in zip(numbers, coords):
        lines.append(_atom_line(int(z), position))
    return lines


def _lattice_of(matrix):
    from pymatgen.core import Lattice

    return Lattice(matrix)


# The crystal system each space-group number belongs to, which decides how many
# lattice parameters CRYSTAL expects on the cell record.
def _system_of(number: int) -> str:
    for limit, system in ((2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"),
                          (142, "tetragonal"), (167, "trigonal"), (194, "hexagonal"),
                          (230, "cubic")):
        if number <= limit:
            return system
    return "triclinic"


def _symmetric_crystal_body(pmg, symprec: float) -> List[str]:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    # Symmetrise the *conventional* cell so orbits and parameters are in the
    # standard setting — the primitive input cell would give a rhombohedral
    # NaCl, not the cubic one CRYSTAL expects with a single "a".
    conventional = SpacegroupAnalyzer(pmg, symprec=symprec).get_conventional_standard_structure()
    sga = SpacegroupAnalyzer(conventional, symprec=symprec)
    number = sga.get_space_group_number()
    system = sga.get_crystal_system()
    sym = sga.get_symmetrized_structure()

    lines = ["0 0 0", str(number), _lattice_line(system, sym.lattice)]
    orbits = sym.equivalent_sites
    lines.append(str(len(orbits)))
    for orbit in orbits:
        lines.append(_atom_line(orbit[0].specie.Z, orbit[0].frac_coords))
    return lines


def _p1_crystal_body(pmg) -> List[str]:
    """Every atom, space group 1 — the always-valid fallback (and symmetry-off)."""
    latt = pmg.lattice
    lines = ["0 0 0", "1", _lattice_line("triclinic", latt), str(len(pmg))]
    for site in pmg:
        lines.append(_atom_line(site.specie.Z, site.frac_coords))
    return lines


def _molecule_body(structure: Structure) -> List[str]:
    """Block-1 body for a 0D molecule: point group 1 (C1), Cartesian Å."""
    atoms = structure.to_ase()
    numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    lines = ["1", str(len(atoms))]
    for z, pos in zip(numbers, positions):
        lines.append(_atom_line(int(z), pos))
    return lines


# The minimal lattice parameters CRYSTAL reads per crystal system (manual §2.1,
# "Comments on geometry input"). trigonal/hexagonal share (a, c) because the
# conventional standard cell is always the hexagonal setting (IFHR = 0).
def _lattice_line(system: str, lattice) -> str:
    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
    system = system.lower()
    if system == "cubic":
        vals: Sequence[float] = (a,)
    elif system in ("hexagonal", "trigonal", "tetragonal"):
        vals = (a, c)
    elif system == "orthorhombic":
        vals = (a, b, c)
    elif system == "monoclinic":
        vals = (a, b, c, beta)  # b-unique standard setting
    else:  # triclinic (and the P1 fallback)
        vals = (a, b, c, alpha, beta, gamma)
    return " ".join(f"{v:.6f}" for v in vals)


def _atom_line(z: int, coords) -> str:
    """A ``Z x y z`` atom record; coordinates fractional (3D) or Å (molecule)."""
    x, y, zc = (float(v) for v in coords)
    return f"{int(z):<3d} {x: .10f} {y: .10f} {zc: .10f}"


# ── task block (block 1) ───────────────────────────────────────────────────
def _task_lines(task: TaskOptions) -> List[str]:
    """Emit the block-1 task keyword block (empty for a single point).

    Every task block sits between the geometry and ``BASISSET``; each opens a
    sub-block closed by its own terminator, leaving ``BASISSET`` to close block 1.
    """
    # SCELPHONO belongs to the geometry block, so it precedes the task keyword.
    lines = _scelphono_lines(task.supercell)
    return lines + _task_block_lines(task)


def _check_spin(structure: Structure, scf: "ScfOptions") -> None:
    """Reject spin settings CRYSTAL would reject, while the user can still fix them.

    SPINLOCK and ATOMSPIN are only read in a spin-polarised run, and an
    ATOMSPIN label outside the structure is a silent mis-assignment rather than
    an error — CRYSTAL numbers atoms from 1, so an off-by-one puts the moment on
    the wrong atom and the run converges to the wrong magnetic state.
    """
    if not scf.spin_polarized:
        if scf.spinlock is not None or scf.atomspin:
            raise CrystalInputError(
                "SPINLOCK and ATOMSPIN only apply to a spin-polarised run. "
                "Turn on spin polarisation, or clear them."
            )
        return
    if scf.spinlock is not None and scf.spinlock[1] < 1:
        raise CrystalInputError("SPINLOCK needs a positive number of cycles.")
    for label, spin in scf.atomspin:
        if not 1 <= int(label) <= len(structure):
            raise CrystalInputError(
                f"ATOMSPIN refers to atom {label}, but this structure has "
                f"{len(structure)} atoms (numbered from 1)."
            )
        if int(spin) not in (1, -1):
            raise CrystalInputError(
                f"ATOMSPIN takes +1 or -1 for each atom; atom {label} was given {spin}."
            )
    seen = [label for label, _ in scf.atomspin]
    if len(set(seen)) != len(seen):
        raise CrystalInputError("ATOMSPIN lists the same atom more than once.")


def _check_two_component(spec: "CrystalInputSpec") -> None:
    """Reject the combinations chapter 6 says the 2c-SCF cannot do."""
    two = spec.two_component
    if not two.enabled:
        return
    if spec.task.kind != "SP":
        raise CrystalInputError(
            "A two-component (spin-orbit) SCF supports single-point energies only: "
            "geometry optimisation, frequencies and response properties are not "
            "available in the two-component spinor basis."
        )
    if spec.method.kind == "DFT" and spec.method.functional_mode == "COMBINED":
        if spec.method.functional.strip().upper() in _MGGA_AND_RSH:
            raise CrystalInputError(
                f"{spec.method.functional.strip()} is a meta-GGA or range-separated "
                "functional; a two-component SCF supports only pure and global-hybrid "
                "LDA/GGA functionals."
            )


def _supercel_geometry_lines(geometry: GeometryOptions) -> List[str]:
    """A general ``SUPERCEL`` expansion, optionally preceded by ``NOSHIFT``.

    NOSHIFT must come *before* SUPERCEL: without it CRYSTAL first shifts the
    origin to minimise symmetry operators with translational components.

    Not to be confused with :func:`_scelphono_lines`, which writes the *other*
    CRYSTAL supercell keyword — the two used to be named one letter apart, so
    swapping the call sites emitted a different calculation that still parsed.
    """
    if geometry.supercell is None:
        return []
    lines = ["NOSHIFT"] if geometry.supercell_noshift else []
    return lines + ["SUPERCEL"] + _matrix_rows(geometry.supercell, "SUPERCEL")


def _matrix_rows(matrix, keyword: str) -> List[str]:
    rows = [list(row) for row in matrix]
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise CrystalInputError(f"The {keyword} expansion matrix must be 3×3.")
    return [" ".join(str(int(v)) for v in row) for row in rows]


def _scelphono_lines(matrix) -> List[str]:
    """A ``SCELPHONO`` expansion matrix, written one row per line.

    The phonon supercell — the cell force constants are computed in — not the
    general geometry expansion :func:`_supercel_geometry_lines` writes.
    """
    if matrix is None:
        return []
    return ["SCELPHONO"] + _matrix_rows(matrix, "SCELPHONO")


def _task_block_lines(task: TaskOptions) -> List[str]:
    if task.kind == "SP":
        return []
    if task.kind == "OPTGEOM":
        lines = ["OPTGEOM"]
        if not task.optimize_cell:
            lines.append("ATOMONLY")  # default (omitted) relaxes cell + atoms
        lines += _optgeom_body(task.optgeom)
        lines.append("ENDOPT")
        return lines
    if task.kind == "CPHF":
        return ["CPHF"] + _cphf_body(task.cphf) + ["END"]
    if task.kind == "ANHARM":
        return ["ANHARM"] + _anharm_body(task.anharm) + ["END"]
    if task.kind in _EQUILIBRIUM_TASKS:
        lines = [_TASK_BLOCK_KEYWORD[task.kind]]
        # Inside PREOPTGEOM the default is atoms-only, so FULLOPTG *is* needed
        # here for a full cell + atoms pre-relaxation (manual §8.1.2).
        if task.preoptimize:
            lines += ["PREOPTGEOM", "FULLOPTG", "END"]
        if task.kind == "FREQCALC":
            lines += _freq_body(task.freq)
        elif task.kind == "DISPERSION":
            lines += _dispersion_body(task.dispersion)
        elif task.kind == "QHA":
            lines += _qha_body(task.qha)
        elif task.kind == "ANHAPES":
            lines += _anhapes_body(task.anhapes)
        elif task.kind == "EOS":
            lines += _eos_body(task.eos)
        else:
            lines += _elastic_body(task.elastic)
        lines.append("END")
        return lines
    raise CrystalInputError(f"Unknown task: {task.kind!r}")


def _dispersion_body(opts: DispersionOptions) -> List[str]:
    """DISPERSI and its sub-keywords, inside a FREQCALC block (manual §8.8).

    INTERPHESS is emitted directly after DISPERSI as the manual requires, with
    WANG next since the two are used together for polar (ionic) crystals.
    """
    lines = ["DISPERSI"]
    if opts.noksymdisp and not opts.bands:  # BANDS already implies NOKSYMDISP
        lines.append("NOKSYMDISP")
    if opts.interphess is not None:
        l1, l2, l3, iprint = opts.interphess
        lines += ["INTERPHESS", f"{int(l1)} {int(l2)} {int(l3)}", str(int(iprint))]
    if opts.wang is not None:
        tensor = [float(v) for v in opts.wang]
        if len(tensor) != 9:
            raise CrystalInputError("WANG needs the 9 elements of the dielectric tensor.")
        # Three records of three, one per row of the tensor — not nine on a
        # line. Every working deck writes it this way (e.g. the ZnO WANG runs
        # on the Desktop), and CRYSTAL reads three values per record: nine on
        # one line leaves it reading the *next* keyword as numbers.
        lines.append("WANG")
        lines += [" ".join(_fmt_float(v) for v in tensor[row:row + 3])
                  for row in (0, 3, 6)]
    if opts.bands:
        segments = _extra_lines(opts.bands_path)
        if not segments:
            raise CrystalInputError(
                "Phonon BANDS needs at least one path segment (I1 I2 I3 J1 J2 J3)."
            )
        # ISS and NSUB on one record, NLINE on the next. The manual's layout is
        # ambiguous about this and they were written on one line; a real working
        # deck (a SCELPHONO ZnO dispersion) settles it — CRYSTAL reads NLINE
        # from its own record, so the three-on-a-line form loses the path.
        lines += [
            "BANDS",
            f"{int(opts.bands_shrink)} {int(opts.bands_points)}",
            str(len(segments)),
            *segments,
        ]
    if opts.pdos is not None:
        numa, nbin, lpro = opts.pdos
        lines += ["PDOS", f"{_fmt_float(numa)} {int(nbin)}", str(int(lpro))]
    if opts.ins is not None:
        numa, nbin, nwtype = opts.ins
        lines += ["INS", f"{_fmt_float(numa)} {int(nbin)}", str(int(nwtype))]
    return lines + _extra_lines(opts.extra_keywords)


def _qha_body(opts: QhaOptions) -> List[str]:
    """QHA sub-keywords (manual §9.2); each value sits on its own line."""
    lines: List[str] = []
    if opts.restart:
        lines.append("RESTART")
    if opts.restart2:
        lines.append("RESTART2")
    if opts.step is not None:
        lines += ["STEP", _fmt_float(opts.step)]
    if opts.points is not None:
        lines += ["POINTS", str(int(opts.points))]
    if opts.temperature is not None:
        n, t1, t2 = opts.temperature
        lines += ["TEMPERAT", f"{int(n)} {_fmt_float(t1)} {_fmt_float(t2)}"]
    if opts.volume_range is not None:
        vmin, vmax, n = opts.volume_range
        lines += ["VRANGE", f"{_fmt_float(vmin)} {_fmt_float(vmax)} {int(n)}"]
    return lines + _extra_lines(opts.extra_keywords)


def _optgeom_body(opts: OptGeomOptions) -> List[str]:
    lines: List[str] = []
    if opts.toldeg is not None:
        lines += ["TOLDEG", _fmt_float(opts.toldeg)]
    if opts.toldex is not None:
        lines += ["TOLDEX", _fmt_float(opts.toldex)]
    if opts.maxcycle is not None:
        lines += ["MAXCYCLE", str(int(opts.maxcycle))]
    return lines + _extra_lines(opts.extra_keywords)


def _freq_body(opts: FreqOptions) -> List[str]:
    """FREQCALC sub-keywords, in the order the manual's worked examples use.

    IR/Raman come last because INTRAMAN must follow INTENS, and the INTCPHF
    block that supplies the Raman tensor nests between them.
    """
    lines: List[str] = []
    if opts.restart:
        lines.append("RESTART")
    if opts.numderiv is not None:
        lines += ["NUMDERIV", str(int(opts.numderiv))]
    if opts.stepsize is not None:
        lines += ["STEPSIZE", _fmt_float(opts.stepsize)]
    if opts.analysis:
        lines.append("ANALYSIS")
    if opts.print_hessian:
        lines.append("PRINT")
    if opts.temperature is not None:
        n, t1, t2 = opts.temperature
        lines += ["TEMPERAT", f"{int(n)} {_fmt_float(t1)} {_fmt_float(t2)}"]
    if opts.pressure is not None:
        n, p1, p2 = opts.pressure
        lines += ["PRESSURE", f"{int(n)} {_fmt_float(p1)} {_fmt_float(p2)}"]

    # Raman requires IR intensities computed through CPHF (manual §8.4): INTRAMAN
    # "should be always used together with the INTENS keyword", and the Raman
    # tensor is produced by the CPHF block that INTCPHF opens. The keyword order
    # below follows the manual's worked examples (§8.4, §8.7): INTENS, then
    # INTRAMAN, then the technique block.
    technique = "INTCPHF" if opts.raman_intensities else opts.ir_technique
    if opts.ir_intensities or opts.raman_intensities:
        lines.append("INTENS")
        if opts.raman_intensities:
            lines.append("INTRAMAN")
        lines.append(technique)
        if technique == "INTCPHF":
            lines += _cphf_body(opts.cphf) + ["END"]
    if opts.raman_intensities and opts.raman_experiment is not None:
        temp, laser = opts.raman_experiment
        lines += ["RAMANEXP", f"{_fmt_float(temp)} {_fmt_float(laser)}"]
    # Both spectrum keywords open their own block, closed by END (§8.6, §8.7).
    if opts.ir_spectrum:
        lines += ["IRSPEC", "END"]
    if opts.raman_spectrum:
        lines += ["RAMSPEC", "END"]
    return lines + _extra_lines(opts.extra_keywords)


def _cphf_body(opts: CphfOptions) -> List[str]:
    """The interior of a CPHF / INTCPHF block (no opening or closing keyword)."""
    lines: List[str] = []
    if opts.order in ("THIRD", "FOURTH"):
        lines.append(opts.order)
    for keyword, value in (
        ("FMIXING", opts.fmixing),
        ("MAXCYCLE", opts.maxcycle),
        ("TOLALPHA", opts.tolalpha),
        ("FMIXING2", opts.fmixing2),
        ("MAXCYCLE2", opts.maxcycle2),
        ("TOLGAMMA", opts.tolgamma),
    ):
        if value is not None:
            lines += [keyword, str(int(value))]
    return lines + _extra_lines(opts.extra_keywords)


def _anharm_body(opts: AnharmOptions) -> List[str]:
    """The interior of an ANHARM block: the atom to displace, then its options."""
    if int(opts.atom_label) < 1:
        raise CrystalInputError("ANHARM needs the sequence number of a hydrogen atom (1 or more).")
    lines = [str(int(opts.atom_label))]
    if opts.keepsymm:
        lines.append("KEEPSYMM")
    if opts.points26:
        lines.append("POINTS26")
    if opts.noguess:
        lines.append("NOGUESS")
    if opts.print_extended:
        lines.append("PRINT")
    if opts.test_only:
        lines.append("TESTANHA")
    isotopes = list(opts.isotopes)
    if isotopes:
        lines.append("ISOTOPES")
        lines.append(str(len(isotopes)))
        lines += [f"{int(label)} {_fmt_float(mass)}" for label, mass in isotopes]
    return lines + _extra_lines(opts.extra_keywords)


def _anhapes_body(opts: AnhapesOptions) -> List[str]:
    """ANHAPES plus any VSCF/VCI request, inside a FREQCALC block (manual §8.12)."""
    modes = _mode_numbers(opts.modes)
    lines: List[str] = []
    if opts.restart:
        lines.append("RESTART")
    if opts.restart_pes:
        lines.append("RESTPES")
    lines += [
        "ANHAPES",
        str(len(modes)),
        " ".join(str(m) for m in modes),
        f"{int(opts.scheme)} {_fmt_float(opts.step)}",
    ]
    # VCI runs a VSCF of its own, so VSCF is only written when asked for directly.
    if opts.vscf:
        lines.append("VSCF")
    if opts.vscf_tol is not None:
        lines += ["VSCFTOL", str(int(opts.vscf_tol))]
    if opts.vscf_mix is not None:
        lines += ["VSCFMIX", str(int(opts.vscf_mix))]
    if opts.vci:
        lines += [
            "VCI",
            f"{int(opts.vci_quanta)} {int(opts.vci_modes)}",
            str(int(opts.vci_guess)),
        ]
    return lines + _extra_lines(opts.extra_keywords)


def _mode_numbers(text: str) -> List[int]:
    """Parse and sanity-check the ANHAPES mode list.

    Modes 1-3 are the translations of any system and are never valid here (the
    manual excludes them explicitly, along with molecular rotations), so a list
    containing them is rejected rather than silently sent to CRYSTAL.
    """
    tokens = text.replace(",", " ").split()
    if not tokens:
        raise CrystalInputError("ANHAPES needs at least one mode number.")
    try:
        modes = [int(token) for token in tokens]
    except ValueError:
        raise CrystalInputError("ANHAPES mode numbers must be integers.") from None
    if any(m < 1 for m in modes):
        raise CrystalInputError("ANHAPES mode numbers start at 1.")
    acoustic = sorted({m for m in modes if m <= 3})
    if acoustic:
        raise CrystalInputError(
            "Modes 1-3 are translations and cannot be treated anharmonically "
            f"(got {', '.join(str(m) for m in acoustic)}). Molecular rotations "
            "must be excluded too — for a non-linear molecule start from mode 7."
        )
    return modes


def _eos_body(opts: EosOptions) -> List[str]:
    lines: List[str] = []
    if opts.volume_range is not None:
        vmin, vmax, n = opts.volume_range
        lines += ["RANGE", f"{_fmt_float(vmin)} {_fmt_float(vmax)} {int(n)}"]
    if opts.pressure_range is not None:
        pmin, pmax, n = opts.pressure_range
        lines += ["PRANGE", f"{_fmt_float(pmin)} {_fmt_float(pmax)} {int(n)}"]
    return lines + _extra_lines(opts.extra_keywords)


def _elastic_body(opts: ElasticOptions) -> List[str]:
    lines: List[str] = []
    if opts.numderiv is not None:
        lines += ["NUMDERIV", str(int(opts.numderiv))]
    if opts.stepsize is not None:
        lines += ["STEPSIZE", _fmt_float(opts.stepsize)]
    if opts.clamped_ion:
        lines.append("CLAMPION")
    return lines + _extra_lines(opts.extra_keywords)


def _fmt_float(value: float) -> str:
    """Compact fixed-point rendering (CRYSTAL reads free-format reals)."""
    return f"{float(value):g}"


def _extra_lines(text: str) -> List[str]:
    """Free-text keywords as individual lines, blanks stripped."""
    return [ln.rstrip() for ln in text.strip("\n").splitlines() if ln.strip()]


# ── Hamiltonian / SCF block (block 3) ──────────────────────────────────────
def _hamiltonian_lines(structure: Structure, spec: CrystalInputSpec) -> List[str]:
    method = spec.method
    scf = spec.scf
    lines: List[str] = []

    # TWOCOMPON precedes the DFT block, as the manual's worked examples show.
    lines += _two_component_lines(spec.two_component)

    if method.kind == "DFT":
        lines.append("DFT")
        # The non-collinear formulation is a DFT-block keyword and comes first.
        if spec.two_component.enabled and spec.two_component.noncollinear != "COLLINEAR":
            lines.append(spec.two_component.noncollinear)
        lines += _functional_lines(method)
        if method.hybrid_percent is not None:
            lines += ["HYBRID", str(int(method.hybrid_percent))]
        if method.nonlocal_weights is not None:
            b, c = method.nonlocal_weights
            lines += ["NONLOCAL", f"{_fmt_float(b)} {_fmt_float(c)}"]
        if method.grid:
            lines.append(method.grid)
        if scf.spin_polarized:
            lines.append("SPIN")
        lines.append("END")
    elif scf.spin_polarized:  # HF, spin-polarised
        lines.append("UHF")

    if structure.is_periodic:
        shrink = scf.shrink if scf.shrink is not None else suggest_shrink(structure)
        lines += ["SHRINK", f"{shrink} {shrink}"]

    # After SHRINK and outside the DFT block, which is where CRYSTAL wants them
    # (and where a working input puts them). Both are spin-only.
    if scf.spin_polarized:
        if scf.spinlock is not None:
            nspin, ncyc = scf.spinlock
            lines += ["SPINLOCK", f"{int(nspin)} {int(ncyc)}"]
        if scf.atomspin:
            lines.append("ATOMSPIN")
            lines.append(str(len(scf.atomspin)))
            lines += [f"{int(label)} {int(spin)}" for label, spin in scf.atomspin]

    lines += ["TOLINTEG", " ".join(str(int(t)) for t in scf.tolinteg)]
    toldee = scf.toldee if scf.toldee is not None else _TOLDEE_DEFAULTS[spec.task.kind]
    lines += ["TOLDEE", str(int(toldee))]
    lines += ["MAXCYCLE", str(int(scf.maxcycle))]

    lines += _extra_lines(spec.extra_keywords)
    lines.append("END")
    return lines


def _two_component_lines(opts: TwoComponentOptions) -> List[str]:
    """The TWOCOMPON block requesting an SCF in a two-component spinor basis."""
    if not opts.enabled:
        return []
    lines = ["TWOCOMPON"]
    if opts.soc:
        lines.append("SOC")
    if opts.guess == "GCOREROT":
        theta, phi = opts.guess_angles
        lines += ["GCOREROT", f"{_fmt_float(theta)} {_fmt_float(phi)}"]
    elif opts.guess and opts.guess != "GUESSPAT":  # GUESSPAT is the default
        lines.append(opts.guess)
    if opts.second_variational:
        lines += ["2NDVARIAT", str(int(opts.second_variational_rot))]
        if opts.second_variational_rot:  # a rotation needs its orientation
            theta, phi = opts.second_variational_angles
            lines.append(f"{_fmt_float(theta)} {_fmt_float(phi)}")
    if opts.print_energies:
        lines.append("PRTENESOC")
    if opts.spinorlock is not None:
        nspin, ncyc = opts.spinorlock
        lines += ["SPINORLOCK", f"{int(nspin)} {int(ncyc)}"]
    return lines + _extra_lines(opts.extra_keywords) + ["END"]


def _functional_lines(method: MethodOptions) -> List[str]:
    """The functional selection: one stand-alone keyword, or EXCHANGE + CORRELAT."""
    if method.functional_mode != "SPLIT":
        return [_functional_keyword(method)]
    lines: List[str] = []
    # An unset potential is meaningful: no EXCHANGE keeps Hartree-Fock exchange,
    # no CORRELAT gives an exchange-only functional (manual §4.1).
    if method.exchange:
        lines += ["EXCHANGE", method.exchange.strip()]
    if method.correlation:
        lines += ["CORRELAT", method.correlation.strip()]
    if not lines:
        raise CrystalInputError(
            "Choose an exchange or a correlation potential — CRYSTAL requires at "
            "least one of them in a DFT block."
        )
    return lines


def _functional_keyword(method: MethodOptions) -> str:
    func = method.functional.strip()
    if method.dispersion_d3 and func.upper() in _D3_FUNCTIONALS:
        return f"{func}-D3"
    return func


__all__ = [
    "CrystalInputError",
    "CrystalInputSpec",
    "GeometryOptions",
    "BasisOptions",
    "MethodOptions",
    "ScfOptions",
    "TwoComponentOptions",
    "TaskOptions",
    "OptGeomOptions",
    "FreqOptions",
    "CphfOptions",
    "EosOptions",
    "ElasticOptions",
    "DispersionOptions",
    "QhaOptions",
    "AnharmOptions",
    "AnhapesOptions",
    "build_input",
    "write_input",
    "suggest_shrink",
    "tolinteg_warnings",
    "INTERNAL_BASIS_SETS",
    "COMMON_FUNCTIONALS",
    "EXCHANGE_FUNCTIONALS",
    "CORRELATION_FUNCTIONALS",
    "NONCOLLINEAR_MODES",
    "TWO_COMPONENT_GUESSES",
    "GRIDS",
    "TOLINTEG_LABELS",
]
