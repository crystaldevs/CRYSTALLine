"""Build a CRYSTAL ``PROPERTIES`` deck (``.d3``).

A ``.d3`` runs *after* an SCF and reads its wave function, so it carries no
geometry and no basis — just a list of properties to compute. What it can ask
for here is deliberately what CRYSTALLine can then open again: band structures,
densities of states, and the crystalline orbitals the orbital viewer draws.

Two constraints from the manual (§14.3, §14.6, §14.12) shape the writer, and
both are silent failures rather than loud ones if got wrong:

* **NEWK must run before DOSS and before ORBITALS**, because both need the
  eigenvectors it computes. BAND does not, and must come *before* NEWK — the
  manual is explicit that ``NEWK BAND DOSS`` fails with "NEWK MUST BE CALLED
  BEFORE DOSS". So the order is fixed here rather than left to the caller.
* **A band path is written as integers over a shrinking factor** (``ISS``), so
  the fractional coordinates of the high-symmetry points have to share one
  denominator. Letter labels are also allowed (``ISS=0``), but CRYSTAL's letters
  differ per Bravais lattice and do not always match the ones a k-path library
  hands out, so integers are used: they cannot be misread.

Qt-free, like the rest of ``core``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

from crystalline.core.structure import Structure

# The largest shrinking factor a band path may be expressed over. Denominators
# come from a k-path library and are small (2, 3, 4, 8); anything past this
# means the path is not the set of special points it is supposed to be.
_MAX_BAND_ISS = 48

# Legendre polynomials used to expand the DOS. 12 is what the manual's own
# examples use; the maximum it allows is 25.
DEFAULT_NPOL = 12


class PropertiesInputError(ValueError):
    """Raised when the requested properties cannot be written as a valid deck."""


@dataclass
class NewkOptions:
    """NEWK: recompute eigenvectors on a (usually finer) k-mesh.

    Almost everything else needs it. ``shrink`` defaults to the SCF suggestion
    for the cell and ``shrink2`` to twice that, which is the ratio CRYSTAL's own
    examples use (``12 24``).
    """

    shrink: Optional[int] = None
    shrink2: Optional[int] = None
    fermi: bool = True      # IFE=1: recompute the Fermi energy
    printing: int = 0       # IPRINT


@dataclass
class BandOptions:
    """BAND: eigenvalues along a path through the Brillouin zone."""

    enabled: bool = False
    title: str = "Band structure"
    # (start, end) pairs in fractional reciprocal coordinates.
    segments: Tuple[Tuple[Tuple[float, float, float], Tuple[float, float, float]], ...] = ()
    labels: Tuple[Tuple[str, str], ...] = ()   # for the comment line only
    points: int = 200        # NSUB, total k points along the whole path
    first_band: int = 1      # INZB
    last_band: Optional[int] = None  # IFNB; None -> filled in from the basis
    store: bool = True       # IPLO=1: write fort.25/BAND.DAT for plotting
    print_eigenvalues: bool = False  # LPR66


@dataclass
class DossOptions:
    """DOSS: total and projected densities of states."""

    enabled: bool = False
    points: int = 300        # NPT
    first_band: int = 1      # INZB
    last_band: Optional[int] = None  # IFNB
    store: bool = True       # IPLO=2 -> DOSS.DAT
    npol: int = DEFAULT_NPOL
    # Energy window in hartree. Only written when the band range is given as
    # -1/-1, which is how the manual says to ask for an energy-defined range.
    window: Optional[Tuple[float, float]] = None
    # Each projection is a list of atom numbers (projected onto all their AOs).
    # An empty tuple gives the total DOS alone.
    projections: Tuple[Tuple[int, ...], ...] = ()


@dataclass
class OrbitalsOptions:
    """ORBITALS: crystalline orbitals written in Molden format (manual §14.12).

    This is what feeds CRYSTALLine's own orbital viewer, which is why the
    filename matters: the app finds the files by the stem CRYSTAL writes.
    """

    enabled: bool = False
    name: str = "orbitals"
    fractional: bool = True   # ICAR: 1 for periodic systems, 0 for molecules
    wannier: bool = False     # ILOC=1 needs LOCALI to have run first


@dataclass
class PropertiesSpec:
    """Everything a ``.d3`` deck needs."""

    newk: NewkOptions = field(default_factory=NewkOptions)
    band: BandOptions = field(default_factory=BandOptions)
    doss: DossOptions = field(default_factory=DossOptions)
    orbitals: OrbitalsOptions = field(default_factory=OrbitalsOptions)
    ppan: bool = False        # Mulliken population analysis
    extra_keywords: str = ""


# ── the band path ─────────────────────────────────────────────────────────
def band_path(structure: Structure) -> Tuple[List[Tuple[str, str]], List[Tuple[tuple, tuple]]]:
    """The conventional high-symmetry path for this lattice.

    Returns ``(labels, segments)`` — the label pairs for the title comment and
    the fractional endpoints. Raises :class:`PropertiesInputError` rather than
    guessing when the lattice cannot be classified, because a wrong path is a
    band structure of the wrong crystal and looks perfectly plausible.
    """
    if not structure.is_periodic:
        raise PropertiesInputError("A band structure needs a periodic structure.")
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.bandstructure import HighSymmKpath

        pmg = AseAtomsAdaptor.get_structure(structure.to_ase())
        kpath = HighSymmKpath(pmg)
        points = kpath.kpath["kpoints"]
        walks = kpath.kpath["path"]
    except Exception as exc:  # noqa: BLE001 - surfaced with what to do instead
        raise PropertiesInputError(
            f"Could not work out the high-symmetry path for this lattice ({exc}). "
            f"Enter the path by hand, or run the band structure on the "
            f"conventional cell."
        ) from exc

    labels: List[Tuple[str, str]] = []
    segments: List[Tuple[tuple, tuple]] = []
    for walk in walks:
        for start, end in zip(walk, walk[1:]):
            labels.append((_tidy_label(start), _tidy_label(end)))
            segments.append((tuple(points[start]), tuple(points[end])))
    if not segments:
        raise PropertiesInputError("The high-symmetry path came back empty.")
    return labels, segments


def _tidy_label(label: str) -> str:
    """``\\Gamma`` -> ``G``, which is what CRYSTAL calls it."""
    name = label.replace("\\", "").strip()
    return "G" if name.lower() == "gamma" else name


def band_shrink(segments: Sequence[Tuple[tuple, tuple]]) -> int:
    """The smallest ISS that makes every coordinate of the path an integer.

    CRYSTAL writes a path as integers over a common shrinking factor, so this is
    the least common multiple of the denominators. Special points have small
    ones — halves, thirds, eighths — so a large answer means the path is not
    made of special points and the deck would be wrong in a way nothing checks.
    """
    denominators = set()
    for start, end in segments:
        for point in (start, end):
            for value in point:
                # limit_denominator *rounds* to the nearest simple fraction, so
                # an arbitrary coordinate comes back looking like a special
                # point. Check it actually is one, or the deck is written with a
                # k-point somewhere else in the zone and nothing says so.
                ratio = Fraction(float(value)).limit_denominator(24)
                if abs(float(ratio) - float(value)) > 1e-6:
                    raise PropertiesInputError(
                        f"The path point {value:g} is not a simple fraction of a "
                        f"reciprocal lattice vector, so it is not one of the "
                        f"lattice's special points. Check the path."
                    )
                denominators.add(ratio.denominator)
    shrink = 1
    for d in denominators:
        shrink = shrink * d // _gcd(shrink, d)
    if shrink > _MAX_BAND_ISS:
        raise PropertiesInputError(
            f"This path needs a shrinking factor of {shrink}, which means its "
            f"points are not the special points of the lattice. Check the path."
        )
    return shrink


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


# ── the deck ──────────────────────────────────────────────────────────────
def build_properties_input(
    structure: Structure, spec: Optional[PropertiesSpec] = None
) -> str:
    """Render a ``.d3`` deck. Raises :class:`PropertiesInputError` if it can't."""
    spec = spec or PropertiesSpec()
    asked = [spec.band.enabled, spec.doss.enabled, spec.orbitals.enabled, spec.ppan]
    if not any(asked) and not spec.extra_keywords.strip():
        raise PropertiesInputError("Choose at least one property to compute.")

    lines: List[str] = []
    # BAND first: the manual is explicit that NEWK before DOSS is required and
    # that BAND has to precede NEWK, or the run stops with
    # "NEWK MUST BE CALLED BEFORE DOSS".
    if spec.band.enabled:
        lines += _band_lines(structure, spec.band)
    if _needs_newk(spec):
        lines += _newk_lines(structure, spec.newk)
    if spec.doss.enabled:
        lines += _doss_lines(spec.doss)
    if spec.orbitals.enabled:
        lines += _orbitals_lines(structure, spec.orbitals)
    if spec.ppan:
        lines.append("PPAN")
    lines += [line.strip() for line in spec.extra_keywords.splitlines() if line.strip()]
    lines.append("END")
    return "\n".join(lines) + "\n"


def _needs_newk(spec: PropertiesSpec) -> bool:
    """DOSS and ORBITALS both read NEWK's eigenvectors; BAND and PPAN do not."""
    return spec.doss.enabled or spec.orbitals.enabled


def _newk_lines(structure: Structure, opts: NewkOptions) -> List[str]:
    from crystalline.core.crystal_input import suggest_shrink

    shrink = opts.shrink if opts.shrink is not None else suggest_shrink(structure)
    shrink2 = opts.shrink2 if opts.shrink2 is not None else shrink * 2
    return ["NEWK", f"{int(shrink)} {int(shrink2)}",
            f"{1 if opts.fermi else 0} {int(opts.printing)}"]


def _band_lines(structure: Structure, opts: BandOptions) -> List[str]:
    segments = list(opts.segments)
    labels = list(opts.labels)
    if not segments:
        labels, segments = band_path(structure)
    shrink = band_shrink(segments)
    last = opts.last_band if opts.last_band is not None else _band_ceiling(structure)
    if last < opts.first_band:
        raise PropertiesInputError(
            f"The last band ({last}) is below the first ({opts.first_band})."
        )

    title = (opts.title or "Band structure")[:72]
    if labels:
        title = f"{title} ({_route(labels)})"[:72]

    lines = ["BAND", title,
             f"{len(segments)} {shrink} {int(opts.points)} {int(opts.first_band)} "
             f"{int(last)} {1 if opts.store else 0} "
             f"{1 if opts.print_eigenvalues else 0}"]
    for start, end in segments:
        lines.append(" ".join(str(_to_integer(v, shrink)) for v in (*start, *end)))
    return lines


def _route(labels: Sequence[Tuple[str, str]]) -> str:
    """The path as a readable chain, e.g. ``G X W K G | U X``.

    A conventional path is not always one connected walk — pymatgen returns
    ``[[G, X, W, K, G, L, U, W, L, K], [U, X]]`` for an fcc lattice — so a plain
    join of the segment ends silently loses where it jumped.
    """
    parts = [labels[0][0]]
    for index, (start, end) in enumerate(labels):
        if index and start != labels[index - 1][1]:
            parts += ["|", start]
        parts.append(end)
    return " ".join(parts)


def _to_integer(value: float, shrink: int) -> int:
    scaled = value * shrink
    nearest = int(round(scaled))
    if abs(scaled - nearest) > 1e-6:
        raise PropertiesInputError(
            f"The path point {value} is not a whole number of 1/{shrink}."
        )
    return nearest


def _band_ceiling(structure: Structure) -> int:
    """A safe default for the last band: enough to cover the occupied states.

    The number of basis functions is what really bounds it, and that is in the
    basis set rather than in the structure — so this counts electrons instead
    and adds room for the empty bands people actually want to see. Too small
    hides the conduction band; too large is only slower.
    """
    electrons = int(sum(int(z) for z in structure.numbers))
    return max(4, electrons // 2 + max(4, electrons // 4))


def _doss_lines(opts: DossOptions) -> List[str]:
    if opts.npol > 25:
        raise PropertiesInputError("DOSS uses at most 25 Legendre polynomials.")
    window = opts.window
    if window is not None:
        first, last = -1, -1  # the manual's way of asking for an energy range
    else:
        first = int(opts.first_band)
        last = int(opts.last_band if opts.last_band is not None else 0)
        if last and last < first:
            raise PropertiesInputError(
                f"The last DOS band ({last}) is below the first ({first})."
            )
    lines = ["DOSS",
             f"{len(opts.projections)} {int(opts.points)} {first} {last} "
             f"{2 if opts.store else 0} {int(opts.npol)} 0"]
    if window is not None:
        low, high = window
        if low >= high:
            raise PropertiesInputError("The DOS energy window is empty.")
        lines.append(f"{low:.6g} {high:.6g}")
    for atoms in opts.projections:
        if not atoms:
            raise PropertiesInputError("A DOS projection needs at least one atom.")
        # A negative count projects onto all the AOs of the listed atoms, which
        # is what "the DOS on this atom" means to anyone asking for it.
        lines.append(" ".join([str(-len(atoms))] + [str(int(a)) for a in atoms]))
    return lines


def _orbitals_lines(structure: Structure, opts: OrbitalsOptions) -> List[str]:
    name = opts.name.strip()
    if not name:
        raise PropertiesInputError("The orbital files need a name to be written under.")
    if any(c in name for c in " \t/\\"):
        raise PropertiesInputError(
            "The orbital file name becomes a file name on disk, so it cannot "
            "contain spaces or path separators."
        )
    fractional = opts.fractional and structure.is_periodic
    return ["ORBITALS", name, "1" if fractional else "0",
            "1" if opts.wannier else "0", "END"]


def write_properties_input(
    structure: Structure, path: str, spec: Optional[PropertiesSpec] = None
) -> str:
    """Write the deck to ``path`` and return what was written."""
    text = build_properties_input(structure, spec)
    with open(path, "w") as handle:
        handle.write(text)
    return text


__all__ = [
    "BandOptions",
    "DossOptions",
    "NewkOptions",
    "OrbitalsOptions",
    "PropertiesInputError",
    "PropertiesSpec",
    "band_path",
    "band_shrink",
    "build_properties_input",
    "write_properties_input",
]
