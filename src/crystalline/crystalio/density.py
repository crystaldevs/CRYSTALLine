"""Scalar fields on a 3D grid: the charge density, the spin density, the potential.

An ECH3 or POT3 run (manual §14.8 and §14.14) samples a property over the
primitive cell and writes it twice, in two formats that say the same thing:

* **Gaussian CUBE** — ``DENS_CUBE.DAT``, ``SPIN_CUBE.DAT``, ``POT_CUBE.DAT``,
  and for a 2c-SCF run the magnetisation and current components as well. This
  is the format other programs read, so it is the one to prefer.
* **``fort.31``**, the DLV format of Appendix D (p. 447): a title, the three
  point counts, the origin, the three step vectors, then the values five to a
  record. It carries no atoms, but it is written by every run that writes a
  cube, which makes it a useful check on one.

Two things about the grid are worth stating, because assuming otherwise
produces a picture that looks plausible and is wrong:

* **It is not axis-aligned.** The steps run along the *primitive lattice
  vectors*, so for a hexagonal, rhombohedral, monoclinic or triclinic cell the
  sampled box is sheared. A field is therefore given by an origin and three
  step *vectors*, never by a scalar spacing.
* **Whether the far face is sampled is not stated** by either format. A grid of
  ``n`` points along a spans the lattice vector either in ``n - 1`` steps (both
  faces sampled, the far one a duplicate of the near one) or in ``n`` (the far
  face left to the neighbouring cell). Which one CRYSTAL writes decides the
  lattice the field tiles on, so :func:`lattice_of` settles it against the
  structure's own cell rather than guessing.

Values are kept in the atomic units the files use — electron/bohr³ for a
density, hartree/electron for a potential — and lengths are converted to the
Angstrom the rest of the app works in.

Qt-free, like the rest of ``crystalio``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

BOHR = 0.5291772109038  # Angstrom, CODATA — the unit both formats are written in

# What a field *is*. The kind decides how it should be drawn: a charge density
# is non-negative and wants one surface, a spin density is signed and wants
# two, a potential is signed and is usually painted onto another field.
CHARGE = "charge"
SPIN = "spin"
POTENTIAL = "potential"
MAGNETISATION = "magnetisation"
CURRENT = "current"
# A file whose content does not say what it holds. Its name is not consulted.
FIELD = "field"
# One field minus another; what it is a difference *of* is kept alongside.
DIFFERENCE = "difference"

KIND_NAMES = {
    CHARGE: "Charge density",
    SPIN: "Spin density",
    POTENTIAL: "Electrostatic potential",
    MAGNETISATION: "Magnetisation",
    CURRENT: "Particle current",
    FIELD: "Scalar field",
    DIFFERENCE: "Difference",
}

# The unit each kind is written in, for a label on a colour bar.
KIND_UNITS = {
    CHARGE: "e/bohr³",
    SPIN: "e/bohr³",
    POTENTIAL: "hartree/e",
    MAGNETISATION: "bohr⁻³",
    CURRENT: "a.u.",
    FIELD: "a.u.",
}

SIGNED_KINDS = (SPIN, POTENTIAL, MAGNETISATION, CURRENT, DIFFERENCE)

# How a field is drawn. A surface of constant value is what a density wants; a
# coloured cut is what a potential or a difference usually wants, since both
# take signs and a surface then shows only where they cross one value.
ISOSURFACE = "isosurface"
SLICE = "slice"
VIEWS = ((ISOSURFACE, "Isosurface"), (SLICE, "Lattice plane (hkl)"))

# The isovalue to start from, in e/bohr³: 0.1 sits in the bonding region of a
# typical solid.
DEFAULT_ISOVALUE = 0.1

POSITIVE_COLOUR = "#1f77b4"
NEGATIVE_COLOUR = "#d62728"
DEFAULT_CMAP = "coolwarm"


@dataclass
class DensityOptions:
    """How a scalar field should be drawn.

    Kept here rather than in the renderer for the same reason the band and DOS
    options are: the dialog chooses, the renderer draws, and the choice can be
    tested without a window.
    """

    view: str = ISOSURFACE
    isovalue: float = DEFAULT_ISOVALUE     # in the field's own units
    opacity: float = 0.85
    positive: str = POSITIVE_COLOUR
    negative: str = NEGATIVE_COLOUR
    # Paint a second field — the electrostatic potential — onto the surface of
    # the first. Both must be on the same grid; see :func:`same_grid`.
    colour_by: Optional["ScalarField"] = None
    cmap: str = DEFAULT_CMAP
    # A slice is a lattice plane, named by its Miller indices in the
    # conventional cell, at a fraction of the interplanar spacing from the
    # origin: 0 and 1 are the same plane, so the range covers the family once.
    miller: Tuple[int, int, int] = (0, 0, 1)
    offset: float = 0.0
    # Remove what lies in front of the plane — atoms, bonds, cell edges — so
    # the plane is seen instead of the crystal around it.
    cutaway: bool = True
    # A density spans orders of magnitude between a bond and a core, so a
    # linear colour map shows a white cell with dots on the nuclei.
    logarithmic: bool = True
    # Off by default. The grid runs along the lattice vectors, so it already
    # is the cell — clipping only cuts the surfaces at the faces, and what
    # keeps the picture to the structure is that pieces belonging to no atom on
    # screen are dropped.
    clip_to_cell: bool = False
    # A colour bar beside the view, for whatever is drawn as a colour map — a
    # slice, or a second field painted onto a surface. A plain surface has one
    # colour and nothing to key.
    colour_bar: bool = False


# The symbol each kind is written with on a colour bar.
_SYMBOLS = {
    CHARGE: "ρ",
    SPIN: "ρα − ρβ",
    POTENTIAL: "V",
    MAGNETISATION: "m",
    CURRENT: "j",
    FIELD: "value",
}


def bar_title(field: "ScalarField", options: "DensityOptions") -> str:
    """What a colour bar is headed with: the quantity, its unit, and its scale.

    The bar keys whatever the colours are made of — the painted field when there
    is one — and says so when a slice is coloured by the logarithm, since the
    numbers on it are then exponents and would be misread as densities.
    """
    shown = options.colour_by if options.colour_by is not None else field
    if shown.kind == DIFFERENCE:
        symbol = "Δ" + _SYMBOLS.get(shown.of, "value")
    else:
        symbol = _SYMBOLS.get(shown.kind, "value")
    logarithmic = (options.view == SLICE and options.logarithmic
                   and options.colour_by is None and not field.signed)
    quantity = f"log₁₀ {symbol}" if logarithmic else symbol
    return f"{quantity} ({shown.unit})"


class DensityError(ValueError):
    """A file that should have held a scalar field does not."""


@dataclass(frozen=True)
class ScalarField:
    """A property sampled on a regular — but not necessarily orthogonal — grid.

    ``values`` has shape ``(na, nb, nc)``, one value per grid point. The point
    ``(i, j, k)`` sits at ``origin + i*steps[0] + j*steps[1] + k*steps[2]``,
    in cartesian Angstrom. ``kind`` is one of the constants above and decides
    how the field is drawn; ``unit`` labels the values, which are left in the
    atomic units the file was written in.
    """

    values: np.ndarray
    origin: np.ndarray
    steps: np.ndarray           # (3, 3): the step along a, b and c, in Angstrom
    kind: str = CHARGE
    name: str = ""
    source: str = ""
    # The atoms the cube carried, if it carried any: ``numbers`` (Z) and
    # ``positions`` in Angstrom. ``fort.31`` has none.
    numbers: Optional[np.ndarray] = None
    positions: Optional[np.ndarray] = None
    # For a difference: the kind of the two fields it was taken between.
    of: Optional[str] = None

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(int(n) for n in self.values.shape)

    @property
    def unit(self) -> str:
        return KIND_UNITS.get(self.of if self.kind == DIFFERENCE else self.kind, "a.u.")

    @property
    def signed(self) -> bool:
        """Whether the field takes both signs, and so wants two surfaces."""
        return self.kind in SIGNED_KINDS or float(self.values.min()) < 0.0

    @property
    def peak(self) -> float:
        """The largest magnitude in the field — the scale an isovalue is read against."""
        return float(np.abs(self.values).max()) if self.values.size else 0.0

    def span(self) -> np.ndarray:
        """The three edges of the sampled box, in Angstrom.

        Geometry, not convention: ``n`` points laid ``step`` apart reach
        ``(n - 1) * step``, whatever the far face is taken to mean.
        """
        counts = np.array(self.shape, dtype=float) - 1.0
        return self.steps * counts[:, None]

    def points(self) -> np.ndarray:
        """Every grid point, shape ``(na, nb, nc, 3)``, in cartesian Angstrom."""
        na, nb, nc = self.shape
        i, j, k = np.meshgrid(np.arange(na), np.arange(nb), np.arange(nc), indexing="ij")
        fractions = np.stack([i, j, k], axis=-1).astype(float)
        return self.origin + fractions @ self.steps

    def in_angstrom_cubed(self) -> "ScalarField":
        """The same field with its values per Angstrom³ rather than per bohr³."""
        if (self.of if self.kind == DIFFERENCE else self.kind) not in (CHARGE, SPIN,
                                                                        MAGNETISATION):
            return self
        return replace(self, values=self.values / BOHR ** 3)


# ── reading ──────────────────────────────────────────────────────────────
def read_cube(path, kind: Optional[str] = None, name: str = "") -> ScalarField:
    """Read a Gaussian CUBE file.

    The format: two comment lines, then the atom count with the origin, then
    one line per axis giving its point count and step vector, then one line per
    atom, then the values with the *last* index varying fastest.

    Two sign conventions live in those numbers. A negative atom count means a
    line of orbital indices follows the atoms (CRYSTAL does not write one, but
    a file handed to us might). A negative point count means that axis is in
    Angstrom rather than bohr — CRYSTAL writes atomic units throughout, but the
    convention is part of the format and costs one line to honour.
    """
    lines = _lines(path)
    if len(lines) < 6:
        raise DensityError(f"{Path(path).name} is too short to be a CUBE file.")
    try:
        head = lines[2].split()
        count = int(head[0])
        origin = np.array([float(v) for v in head[1:4]], dtype=float)
        axes, counts = [], []
        for row in lines[3:6]:
            fields = row.split()
            n = int(fields[0])
            step = np.array([float(v) for v in fields[1:4]], dtype=float)
            # A negative count is the format's way of saying "this axis is in
            # Angstrom"; the step is then already in the unit we want.
            axes.append(step if n < 0 else step * BOHR)
            counts.append(abs(n))
    except (IndexError, ValueError) as error:
        raise DensityError(f"{Path(path).name} is not a CUBE file: {error}") from error

    natoms = abs(count)
    start = 6 + natoms + (1 if count < 0 else 0)
    numbers, positions = _cube_atoms(lines[6:6 + natoms])
    values = _floats(lines[start:])
    wanted = counts[0] * counts[1] * counts[2]
    if values.size < wanted:
        raise DensityError(
            f"{Path(path).name} holds {values.size} values, not the "
            f"{counts[0]}×{counts[1]}×{counts[2]} = {wanted} its header promises."
        )
    if kind is None:
        kind, guessed = kind_of_title(lines[0])
        name = name or guessed
    return ScalarField(
        values=values[:wanted].reshape(counts),   # last index fastest: C order
        origin=origin * BOHR,
        steps=np.array(axes, dtype=float),
        kind=kind,
        name=name or KIND_NAMES.get(kind, "Scalar field"),
        source=str(path),
        numbers=numbers,
        positions=positions,
    )


def read_fort31(path, kind: Optional[str] = None, name: str = "") -> ScalarField:
    """Read the DLV format of Appendix D (p. 447).

    Records: a title, ``npa npb npc``, the cartesian origin, the step along a,
    the step along b, the step along c, then the values five to a record. All
    atomic units.

    The manual does not say in which order the values are written. They are
    read here as Fortran writes a 3D array — the first index varying fastest —
    which is the assumption to check first if a field ever comes out
    transposed. A run writes its ``fort.31`` and its cube together, so the two
    can be compared point by point; :func:`agree` does that.
    """
    lines = _lines(path)
    if len(lines) < 6:
        raise DensityError(f"{Path(path).name} is too short to be a fort.31.")
    try:
        counts = [int(v) for v in lines[1].split()[:3]]
        origin = np.array([float(v) for v in lines[2].split()[:3]], dtype=float)
        steps = np.array(
            [[float(v) for v in lines[row].split()[:3]] for row in (3, 4, 5)],
            dtype=float,
        )
    except (IndexError, ValueError) as error:
        raise DensityError(f"{Path(path).name} is not a fort.31: {error}") from error
    if len(counts) != 3 or min(counts) < 2:
        raise DensityError(f"{Path(path).name} does not name three grid dimensions.")

    values = _floats(lines[6:])
    wanted = counts[0] * counts[1] * counts[2]
    if values.size < wanted:
        raise DensityError(
            f"{Path(path).name} holds {values.size} values, not the "
            f"{counts[0]}×{counts[1]}×{counts[2]} = {wanted} its header promises."
        )
    title = lines[0].strip()
    if kind is None:
        kind, _label = kind_of_title(title)
    return ScalarField(
        values=values[:wanted].reshape(counts, order="F"),  # first index fastest
        origin=origin * BOHR,
        steps=steps * BOHR,
        kind=kind,
        name=name or title or KIND_NAMES.get(kind, "Scalar field"),
        source=str(path),
    )


def read_field(path) -> ScalarField:
    """Read whichever of the two formats ``path`` is in."""
    form = file_format(path)
    if form == "cube":
        return read_cube(path)
    if form == "fort31":
        return read_fort31(path)
    raise DensityError(f"{Path(path).name} is not a CUBE file or a fort.31.")


def _cube_atoms(rows: Sequence[str]):
    """The atoms of a cube: ``Z, charge, x, y, z`` per line, in bohr."""
    numbers, positions = [], []
    for row in rows:
        fields = row.split()
        if len(fields) < 5:
            continue
        try:
            numbers.append(int(float(fields[0])))
            positions.append([float(v) * BOHR for v in fields[2:5]])
        except ValueError:
            continue
    if not numbers:
        return None, None
    return np.array(numbers, dtype=int), np.array(positions, dtype=float)


def _lines(path) -> List[str]:
    with open(path, "r", errors="ignore") as handle:
        return handle.read().splitlines()


def _floats(rows: Sequence[str]) -> np.ndarray:
    """Every number in ``rows``, in order.

    Fortran writes ``0.1234-102`` for a number too small for its exponent
    field, dropping the ``E``. Left as it is, that reads as two numbers and
    shifts every value after it.
    """
    out: List[float] = []
    for row in rows:
        for token in row.split():
            try:
                out.append(float(token))
            except ValueError:
                fixed = _repair_exponent(token)
                if fixed is None:
                    return np.array(out, dtype=float)
                out.append(fixed)
    return np.array(out, dtype=float)


def _repair_exponent(token: str) -> Optional[float]:
    """``0.1234-102`` -> ``0.1234E-102``, or ``None`` if it is not a number."""
    for index in range(1, len(token)):
        if token[index] in "+-" and token[index - 1] not in "eEdD+-":
            try:
                return float(token[:index] + "E" + token[index:])
            except ValueError:
                return None
    return None


# ── finding them ─────────────────────────────────────────────────────────
def file_format(path) -> Optional[str]:
    """``"cube"``, ``"fort31"`` or ``None``, from what the file holds.

    Names are a weak guide — ``DENS_CUBE.DAT`` shares its extension with
    ``BAND.DAT`` — so the shape of the header decides: a cube's third line is
    an integer followed by three reals, a fort.31's second line is three
    integers and nothing else.
    """
    p = Path(path)
    try:
        with open(p, "r", errors="ignore") as handle:
            head = [handle.readline(400) for _ in range(6)]
    except OSError:
        return None
    if len(head) < 6 or not head[5]:
        return None
    if _is_count_and_vector(head[2]) and all(_is_count_and_vector(row) for row in head[3:6]):
        return "cube"
    # A fort.31 opens with a title — free text, but never one of the markers
    # the other PROPERTIES formats open with — and then three point counts,
    # each of which must be at least 2 for the grid to have any extent. Without
    # both tests a BAND.DAT ("# NKPT 10" over a line of three integers) reads
    # as a fort.31.
    title = head[0].strip()
    if title.startswith("#") or title.startswith("-%-"):
        return None
    second = head[1].split()
    if len(second) == 3 and all(_is_integer(v) for v in second):
        if min(int(v) for v in second) >= 2 and all(_is_vector(row) for row in head[2:6]):
            return "fort31"
    return None


def _first_line(path) -> str:
    try:
        with open(path, "r", errors="ignore") as handle:
            return handle.readline(200)
    except OSError:
        return ""


def _is_integer(token: str) -> bool:
    try:
        int(token)
    except ValueError:
        return False
    return True


def _is_vector(row: str) -> bool:
    fields = row.split()
    if len(fields) < 3:
        return False
    try:
        [float(v) for v in fields[:3]]
    except ValueError:
        return False
    return True


def _is_count_and_vector(row: str) -> bool:
    fields = row.split()
    if len(fields) < 4 or not _is_integer(fields[0]):
        return False
    try:
        [float(v) for v in fields[1:4]]
    except ValueError:
        return False
    return True


# What CRYSTAL writes at the start of a grid file's first line, per property.
_TITLES = (
    ("charge density", CHARGE, "Charge density"),
    ("spin density", SPIN, "Spin density"),
    ("potential", POTENTIAL, "Electrostatic potential"),
    ("electrostatic potential", POTENTIAL, "Electrostatic potential"),
    ("magnetization", MAGNETISATION, "Magnetisation"),
    ("magnetisation", MAGNETISATION, "Magnetisation"),
    ("current", CURRENT, "Particle current"),
)


def kind_of_title(title: str) -> Tuple[str, str]:
    """The property a grid file holds, from its first line — and from nothing else.

    CRYSTAL states it there: "Charge density - 3D GRID", "Potential - 3D GRID".
    A file's name is not evidence. Outputs are renamed as soon as they leave the
    scratch directory — the tutorial's are ``mgo_ech3.cube`` and
    ``mgo_pot3.cube`` — and guessing from names took every one of those
    potentials for a charge density. A title that names nothing known gives
    :data:`FIELD`: drawn by what its values do, labelled as what it is, an
    unidentified field.
    """
    head = (title or "").strip().lower()
    for words, kind, label in _TITLES:
        if head.startswith(words):
            return kind, label
    return FIELD, "Scalar field"


# The extensions a grid file comes with: CRYSTAL writes DENS_CUBE.DAT and
# fort.31, and cubes are commonly renamed to .cube or .cub.
_GRID_EXTENSIONS = (".dat", ".cube", ".cub", ".31")


def find_fields(folder, cell=None) -> List[str]:
    """The scalar-field files in ``folder``, best first.

    Files are picked out by extension — CRYSTAL's ``.DAT`` and ``fort.31``, and
    the ``.cube`` they are usually renamed to — and their first lines decide
    whether they are grids and of what. Never the rest of the name, which is
    rewritten freely.

    Best first: a grid of the structure that is open — its lattice is ``cell``,
    which the grid's own header is compared with, since a folder shared by
    several calculations would otherwise offer whichever of them ran last —
    then a cube over a ``fort.31`` saying the same thing (it carries the atoms,
    and its value order is not a guess), then a charge density over a spin
    density over a potential, then the newest.

    """
    try:
        entries = [p for p in Path(folder).iterdir()
                   if p.is_file() and p.suffix.lower() in _GRID_EXTENSIONS]
    except OSError:
        return []
    found = []
    for path in entries:
        form = file_format(path)
        if form is not None:
            found.append((path, form, kind_of_title(_first_line(path))[0]))
    order = {CHARGE: 0, SPIN: 1, POTENTIAL: 2}
    reference = None if cell is None else np.asarray(cell, dtype=float)

    def rank(item):
        path, form, kind = item
        try:
            age = -path.stat().st_mtime
        except OSError:
            age = 0.0
        ours = reference is None or _same_lattice(grid_lattice(path, form), reference)
        return (not ours, form != "cube", order.get(kind, 3), age, str(path))

    return [str(path) for path, _form, _kind in sorted(found, key=rank)]


def grid_lattice(path, form: Optional[str] = None) -> Optional[np.ndarray]:
    """The lattice a grid file spans, in Angstrom, read from its header alone.

    Enough to tell which calculation a file belongs to without reading a million
    values: the step vectors and the point counts are in the first six lines.
    """
    form = form or file_format(path)
    try:
        with open(path, "r", errors="ignore") as handle:
            head = [handle.readline(400) for _ in range(6)]
        if form == "cube":
            rows = [row.split() for row in head[3:6]]
            counts = np.array([abs(int(row[0])) for row in rows], dtype=float)
            steps = np.array([[float(v) for v in row[1:4]] for row in rows])
            steps = np.where(np.array([int(row[0]) < 0 for row in rows])[:, None],
                             steps, steps * BOHR)
        elif form == "fort31":
            counts = np.array([int(v) for v in head[1].split()[:3]], dtype=float)
            steps = np.array([[float(v) for v in head[r].split()[:3]] for r in (3, 4, 5)]) * BOHR
        else:
            return None
    except (OSError, ValueError, IndexError):
        return None
    return steps * (counts - 1)[:, None]


def _same_lattice(first, second, tolerance: float = 2e-3) -> bool:
    """Whether two lattices are the same cell, whichever of its vectors comes first.

    Compared as sets of vector lengths and the angles between them, so that the
    same cell written in another order still matches, and a different cell
    that merely shares a volume does not.
    """
    if first is None or second is None:
        return False

    def shape(lattice):
        lattice = np.asarray(lattice, dtype=float)
        lengths = np.linalg.norm(lattice, axis=1)
        cosines = [abs(float(lattice[i] @ lattice[j] / (lengths[i] * lengths[j])))
                   for i, j in ((0, 1), (0, 2), (1, 2))]
        return np.sort(lengths), np.sort(cosines)

    (la, ca), (lb, cb) = shape(first), shape(second)
    return bool(np.allclose(la, lb, atol=tolerance * max(la.max(), 1.0))
                and np.allclose(ca, cb, atol=tolerance))


# ── comparing and combining ──────────────────────────────────────────────
def same_grid(first: ScalarField, second: ScalarField, tolerance: float = 1e-6) -> bool:
    """Whether two fields were sampled at the same points.

    Anything that combines two fields — a difference, a potential painted onto
    a density — needs this to hold, and two runs that differ in their ``NP``,
    their cell or their ``SCALE`` will quietly produce grids that do not line
    up point for point.
    """
    return (
        first.shape == second.shape
        and bool(np.allclose(first.origin, second.origin, atol=tolerance))
        and bool(np.allclose(first.steps, second.steps, atol=tolerance))
    )


def agree(first: ScalarField, second: ScalarField, tolerance: float = 1e-6) -> bool:
    """Whether two fields hold the same values on the same grid.

    A run writes its ``fort.31`` and its cube together, so reading both and
    asking this is how the value order assumed for ``fort.31`` gets checked
    against a format that states its own.
    """
    if not same_grid(first, second, tolerance):
        return False
    scale = max(first.peak, second.peak, 1.0)
    return bool(np.allclose(first.values, second.values, atol=tolerance * scale))


def sample(field: ScalarField, points: np.ndarray) -> np.ndarray:
    """``field`` at arbitrary cartesian points, taken from the nearest grid point.

    Used to paint one field onto another's surface. Nearest rather than
    interpolated: the surface's vertices are dense compared with the grid, and
    trilinear interpolation across a sheared cell costs more than it shows.
    Points outside the sampled box take the value of the nearest face, which is
    what happens at a cell boundary the surface crosses.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    fractional = (points - field.origin) @ np.linalg.inv(field.steps)
    index = np.rint(fractional).astype(int)
    limit = np.array(field.shape, dtype=int) - 1
    index = np.clip(index, 0, limit)
    return field.values[index[:, 0], index[:, 1], index[:, 2]]


def miller_plane(cell: np.ndarray, miller: Sequence[int], offset: float = 0.0):
    """The plane ``(hkl)`` of ``cell``, as ``(point, unit normal, spacing)``.

    The normal of ``(hkl)`` is the reciprocal lattice vector
    ``h a* + k b* + l c*``, and the planes of the family are ``1/|G|`` apart.
    ``offset`` places the plane that far along the normal, in units of that
    spacing, from the cell's origin — so ``0`` is the plane through the origin
    and ``0.5`` the one halfway to the next.

    ``cell`` should be the *conventional* cell: that is what Miller indices are
    quoted in, and MgO's ``(001)`` in its primitive fcc cell is a ``{111}``
    plane holding one kind of atom.
    """
    h, k, l = (int(v) for v in miller)
    if (h, k, l) == (0, 0, 0):
        raise DensityError("(000) is not a plane: give at least one non-zero index.")
    cell = np.asarray(cell, dtype=float)
    reciprocal = np.linalg.inv(cell).T          # rows a*, b*, c* (no 2π)
    g = h * reciprocal[0] + k * reciprocal[1] + l * reciprocal[2]
    length = float(np.linalg.norm(g))
    normal = g / length
    spacing = 1.0 / length
    return float(offset) * spacing * normal, normal, spacing


def sample_periodic(field: ScalarField, points: np.ndarray, cell=None) -> np.ndarray:
    """``field`` at arbitrary points anywhere in space, interpolated.

    The grid covers one cell, but a plane through a crystal runs through many,
    so the field is continued periodically and read by trilinear interpolation.
    When the grid samples the far face of the cell — CRYSTAL's does — that
    layer is a copy of the first and is dropped before wrapping, or every
    cell boundary would carry a doubled row.
    """
    from scipy.ndimage import map_coordinates

    points = np.atleast_2d(np.asarray(points, dtype=float))
    lattice = lattice_of(field, cell)
    counts = np.array(field.shape)
    inclusive = np.allclose(lattice, field.steps * (counts - 1)[:, None], atol=1e-6)
    values = field.values[:-1, :-1, :-1] if inclusive else field.values
    period = np.array(values.shape, dtype=float)
    fractional = (points - field.origin) @ np.linalg.inv(lattice)
    indices = (fractional % 1.0) * period[None, :]
    return map_coordinates(values, indices.T, order=1, mode="grid-wrap")


def difference(first: ScalarField, second: ScalarField, name: str = "") -> ScalarField:
    """``first - second``, for a deformation or a difference density.

    The result is signed whatever the two were, so it is drawn with the two
    surfaces of a signed field rather than the one of a density.
    """
    if not same_grid(first, second):
        raise DensityError(
            "The two fields are on different grids, so they cannot be "
            "subtracted: they must come from runs with the same cell, the same "
            "number of points and the same extents."
        )
    return replace(
        first,
        values=first.values - second.values,
        kind=DIFFERENCE,
        of=first.of if first.kind == DIFFERENCE else first.kind,
        name=name or f"{first.name} − {second.name}",
        source="",
    )


def lattice_of(field: ScalarField, cell: Optional[np.ndarray] = None,
               tolerance: float = 1e-3) -> np.ndarray:
    """The lattice the field repeats on, in Angstrom.

    Neither format says whether the far face of the cell was sampled, and the
    two readings differ by one step — enough to make a tiled picture drift. If
    the structure's ``cell`` is at hand the question is settled by trying both
    against it; otherwise the inclusive reading is taken, which is what
    CRYSTAL's grid over a primitive cell is.
    """
    counts = np.array(field.shape, dtype=float)
    inclusive = field.steps * (counts - 1.0)[:, None]    # far face sampled
    exclusive = field.steps * counts[:, None]            # far face left out
    if cell is not None:
        cell = np.asarray(cell, dtype=float)
        for candidate in (inclusive, exclusive):
            if np.allclose(np.abs(candidate), np.abs(cell), atol=tolerance):
                return candidate
    return inclusive
