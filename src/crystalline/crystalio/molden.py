"""Read the Molden files CRYSTAL's ``ORBITALS`` keyword writes.

A PROPERTIES run with ``ORBITALS`` writes one Molden file per sampled k-point,
holding the cell, the atoms, the Gaussian basis and the crystalline-orbital
coefficients — everything needed to evaluate an orbital in real space.

Two conventions in these files are worth stating, because getting either wrong
produces a plausible-looking but wrong orbital:

* **A crystalline orbital is complex away from Γ.** CRYSTAL does not write
  complex coefficients; it writes the real and imaginary parts as *two separate
  files*, ``…_K100_real.molden`` and ``…_K100_complex.molden``, each with real
  coefficients. Γ needs only the real one. :func:`k_label` and
  :func:`companion_part` read that off the file name.
* **The coefficients are for one cell.** ψ_k is a Bloch sum over the lattice, so
  evaluating it means summing the reference cell's basis functions over
  neighbouring cells with the e^{ik·T} phase — see
  :mod:`crystalline.core.orbitals`. The file alone is the reference cell.

Kept Qt- and PyVista-free, like the rest of ``crystalio``: parsing is pure text
and numpy, and is unit-tested directly.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

BOHR_PER_ANGSTROM = 1.8897261254535

# Atomic orbitals per shell, in the spherical convention these files declare with
# [5D]/[7F]. "sp" is a Pople combined shell: one s and one p sharing exponents.
SHELL_SIZE = {"s": 1, "p": 3, "sp": 4, "d": 5, "f": 7, "g": 9}

# CRYSTAL names each file "<stem>_<n>_K<hkl>_<part>.molden"; the k label and the
# part are the only things distinguishing the files of one run.
_FILENAME = re.compile(r"_(\d+)_K(\w+?)_(real|complex)\.molden$", re.IGNORECASE)

# CRYSTAL prints the sampled k-points as integer ("oblique") coordinates in
# units of the shrinking factor, e.g.
#
#     *** K POINTS COORDINATES (OBLIQUE COORDINATES IN UNITS OF IS =  3)
#       1-R(  0  0  0)   2-C(  1  0  0)   3-C(  1  1  0)   ...
#
# The Molden file names carry only the digits ("K110"), so the shrinking factor
# — without which they are not a k-vector — has to come from the output.
_KPOINT_HEADER = re.compile(r"IN UNITS OF IS\s*=\s*(\d+)", re.IGNORECASE)
_KPOINT_ENTRY = re.compile(r"(\d+)-[A-Z]\(\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\)")
# Where to look for it: a properties run writes its own output beside the SCF
# one, and it is that run's NEWK net the Molden files belong to.
_OUTPUT_SUFFIXES = (".out", ".outp", ".log")


@dataclass(frozen=True)
class Shell:
    """One contracted Gaussian shell, sitting on ``atom``.

    ``coefficients`` is ``(nprim,)`` for a plain shell and ``(nprim, 2)`` for an
    ``sp``, whose two columns are the s and p contractions over shared
    ``exponents``. The values are as the file gives them — Molden folds the
    primitive normalisation into the contraction coefficients, so they are used
    as-is rather than renormalised.
    """

    atom: int
    kind: str
    exponents: np.ndarray
    coefficients: np.ndarray

    @property
    def size(self) -> int:
        """How many atomic orbitals this shell contributes."""
        return SHELL_SIZE[self.kind]


@dataclass(frozen=True)
class Orbital:
    """One crystalline orbital: its energy, occupation and AO coefficients."""

    energy: float          # as printed (Hartree)
    occupation: float
    coefficients: np.ndarray   # (nao,)
    spin: str = "ALPHA+BETA"
    symmetry: str = ""

    @property
    def is_occupied(self) -> bool:
        return self.occupation > 0.0


@dataclass(frozen=True)
class MoldenOrbitals:
    """A parsed ORBITALS file: geometry, basis and the orbitals on it."""

    numbers: np.ndarray            # (natom,) atomic numbers
    positions: np.ndarray          # (natom, 3) cartesian Angstrom
    cell: Optional[np.ndarray]     # (3, 3) lattice vectors as rows, Angstrom
    shells: List[Shell]
    orbitals: List[Orbital]
    spherical_d: bool = True
    spherical_f: bool = True
    kpoint_label: str = ""         # "000", "100", … from the file name
    part: str = "real"             # "real" or "complex" (the imaginary part)

    @property
    def n_ao(self) -> int:
        return sum(shell.size for shell in self.shells)

    @property
    def is_periodic(self) -> bool:
        return self.cell is not None

    @property
    def homo_index(self) -> Optional[int]:
        """Index of the highest occupied orbital, or ``None`` if none is."""
        occupied = [i for i, mo in enumerate(self.orbitals) if mo.is_occupied]
        return occupied[-1] if occupied else None


def load(path: str) -> MoldenOrbitals:
    """Parse a Molden file written by CRYSTAL's ``ORBITALS``."""
    with open(path, "r", errors="ignore") as handle:
        text = handle.read()
    if "[Molden Format]" not in text:
        raise ValueError(f"{os.path.basename(path)} is not a Molden file")

    sections = _sections(text)
    cell = _cell(sections)
    numbers, positions = _atoms(sections, cell)
    shells = _basis(sections.get("gto", ""))
    n_ao = sum(shell.size for shell in shells)
    orbitals = _orbitals(sections.get("mo", ""), n_ao)
    label, part = _name_parts(path)
    return MoldenOrbitals(
        numbers=numbers,
        positions=positions,
        cell=cell,
        shells=shells,
        orbitals=orbitals,
        spherical_d="5d" in sections,
        spherical_f="7f" in sections,
        kpoint_label=label,
        part=part,
    )


def _sections(text: str) -> dict:
    """Split the file on its ``[Section]`` headers, keyed by lowercased name.

    The header can carry a qualifier — ``[Atoms] (Fractional)`` — which decides
    how the body is read, so it is kept as the first line of the body rather
    than thrown away with the name.
    """
    out: dict = {}
    parts = re.split(r"^\[([^\]]+)\]([^\n]*)\n?", text, flags=re.M)
    for i in range(1, len(parts) - 2, 3):
        name = parts[i].strip().lower()
        out[name] = parts[i + 1].strip() + "\n" + parts[i + 2]
    return out


def _cell(sections: dict) -> Optional[np.ndarray]:
    """Lattice vectors as rows in Angstrom, or ``None`` for a molecule."""
    body = sections.get("cellaxes")
    if not body:
        return None
    rows = [
        [float(v) for v in line.split()]
        for line in body.splitlines()
        if len(line.split()) == 3
    ]
    if len(rows) != 3:
        return None
    return np.asarray(rows, dtype=float)


def _atoms(sections: dict, cell) -> Tuple[np.ndarray, np.ndarray]:
    """``(atomic numbers, cartesian Angstrom positions)``.

    The coordinates may be fractional, Angstrom or atomic units — the qualifier
    on the ``[Atoms]`` header says which, and CRYSTAL writes fractional.
    """
    body = sections.get("atoms", "")
    qualifier = body.splitlines()[0].lower() if body else ""
    numbers, coords = [], []
    for line in body.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 6:
            continue
        numbers.append(int(fields[2]))
        coords.append([float(v) for v in fields[3:6]])
    if not numbers:
        return np.empty(0, dtype=int), np.empty((0, 3), dtype=float)

    positions = np.asarray(coords, dtype=float)
    if "fractional" in qualifier:
        if cell is None:
            raise ValueError("fractional atom coordinates but no [CellAxes] to resolve them")
        positions = positions @ cell
    elif "au" in qualifier or "bohr" in qualifier:
        positions = positions / BOHR_PER_ANGSTROM
    return np.asarray(numbers, dtype=int), positions


def _basis(body: str) -> List[Shell]:
    """The contracted shells of every atom, in file order.

    ``[GTO]`` is a per-atom block: an atom index on its own line, then that
    atom's shells, each a ``<kind> <nprim>`` header followed by its primitives.
    """
    shells: List[Shell] = []
    atom = -1
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        fields = lines[i].split()
        i += 1
        if not fields:
            continue
        if len(fields) <= 2 and fields[0].isdigit():
            atom = int(fields[0]) - 1  # the file counts atoms from 1
            continue
        kind = fields[0].lower()
        if kind not in SHELL_SIZE or len(fields) < 2:
            continue
        nprim = int(fields[1])
        rows = []
        for _ in range(nprim):
            if i >= len(lines):
                break
            rows.append([float(v.replace("D", "E").replace("d", "e"))
                         for v in lines[i].split()])
            i += 1
        block = np.asarray(rows, dtype=float)
        if block.size == 0:
            continue
        coefficients = block[:, 1:3] if kind == "sp" else block[:, 1]
        shells.append(
            Shell(atom=atom, kind=kind, exponents=block[:, 0], coefficients=coefficients)
        )
    return shells


def _orbitals(body: str, n_ao: int) -> List[Orbital]:
    """Every orbital in the ``[MO]`` section, in file order."""
    orbitals: List[Orbital] = []
    energy = occupation = None
    spin = symmetry = ""
    coefficients: List[float] = []

    def flush() -> None:
        if energy is None or not coefficients:
            return
        vector = np.zeros(n_ao, dtype=float)
        take = min(n_ao, len(coefficients))
        vector[:take] = coefficients[:take]
        orbitals.append(
            Orbital(
                energy=float(energy),
                occupation=float(occupation or 0.0),
                coefficients=vector,
                spin=spin or "ALPHA+BETA",
                symmetry=symmetry,
            )
        )

    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip().lower()
        if key in ("sym", "ene", "spin", "occup"):
            if key == "sym":
                flush()
                coefficients = []
                energy = occupation = None
                spin = symmetry = ""
                symmetry = value.strip()
            elif key == "ene":
                energy = float(value)
            elif key == "spin":
                spin = value.strip()
            else:
                occupation = float(value)
            continue
        fields = stripped.split()
        if len(fields) == 2:
            try:
                coefficients.append(float(fields[1]))
            except ValueError:
                continue
    flush()
    return orbitals


# ── the files of one run ──────────────────────────────────────────────────
def _name_parts(path: str) -> Tuple[str, str]:
    match = _FILENAME.search(os.path.basename(path))
    return (match.group(2), match.group(3).lower()) if match else ("", "real")


def _name_index(path: str) -> Optional[int]:
    """The k-point's ordinal in the run — ``2`` for ``…_00002_K100_real.molden``.

    It is how a file is matched to a row of the output's k-point table, which is
    the only place the shrinking factor is written.
    """
    match = _FILENAME.search(os.path.basename(path))
    return int(match.group(1)) if match else None


def _kpoint_tables(text: str) -> List[Tuple[int, dict]]:
    """Every ``K POINTS COORDINATES`` table in ``text``, as ``(shrink, rows)``.

    A run prints one per net it builds — the SCF's, then NEWK's — and they have
    different shrinking factors, so which one a Molden file belongs to is
    decided by matching its ordinal *and* its coordinates, not by taking the
    first or the last.
    """
    tables: List[Tuple[int, dict]] = []
    headers = list(_KPOINT_HEADER.finditer(text))
    for n, header in enumerate(headers):
        end = headers[n + 1].start() if n + 1 < len(headers) else len(text)
        rows = {
            int(entry.group(1)): tuple(int(entry.group(k)) for k in (2, 3, 4))
            for entry in _KPOINT_ENTRY.finditer(text[header.end():end])
        }
        if rows:
            tables.append((int(header.group(1)), rows))
    return tables


def k_vector(path: str) -> Optional[np.ndarray]:
    """The k-point of ``path`` in fractional (reciprocal-lattice) coordinates.

    A crystalline orbital is a Bloch sum, ``psi_k(r) = sum_T e^{2*pi*i*k.T} ...``,
    so k *is* the phase relation between one cell and the next — the thing that
    makes an orbital away from Γ different from a molecular one. It is not in
    the Molden file: the file name gives the integer coordinates and the CRYSTAL
    output gives the shrinking factor they are in units of, so both are needed.

    ``None`` when the output can't be found or doesn't line up with the files —
    the caller then has an orbital it cannot phase, and should say so rather
    than quietly draw it at Γ.
    """
    label, _part = _name_parts(path)
    if not label:
        return None
    if set(label) <= {"0"}:
        return np.zeros(3)  # Γ is the origin whatever the shrinking factor is
    index = _name_index(path)
    net = _kpoint_net(os.path.dirname(os.path.abspath(path)) or ".")
    if net is None:
        return None
    shrink, rows = net
    coordinates = rows.get(index)
    if coordinates is None or "".join(str(c) for c in coordinates) != label:
        return None
    return np.asarray(coordinates, dtype=float) / float(shrink)


def _kpoint_net(directory: str) -> Optional[Tuple[int, dict]]:
    """The k-point table the Molden files in ``directory`` were written from.

    A run prints one table per net it builds — the SCF's, then NEWK's — with
    different shrinking factors, and a single file cannot always tell them
    apart: the first point off Γ is ``(1 0 0)`` and is labelled ``K100`` in
    both, so reading it against a 6×6×6 net instead of the 3×3×3 one NEWK asked
    for halves k and draws the wrong orbital. The table is therefore chosen
    against *all* the run's files at once — the right one accounts for every
    label — and later tables win a tie, NEWK's being the last printed.
    """
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return None
    wanted = {}
    for name in names:
        if not name.lower().endswith(".molden"):
            continue
        label, _part = _name_parts(name)
        index = _name_index(name)
        if label and index is not None:
            wanted[index] = label
    if not wanted:
        return None

    best = None
    for name in names:
        if not name.lower().endswith(_OUTPUT_SUFFIXES):
            continue
        try:
            with open(os.path.join(directory, name), "r", errors="ignore") as handle:
                text = handle.read()
        except OSError:
            continue
        for order, (shrink, rows) in enumerate(_kpoint_tables(text)):
            if shrink <= 0:
                continue
            matched = sum(
                1 for index, label in wanted.items()
                if index in rows and "".join(str(c) for c in rows[index]) == label
            )
            if matched and (best is None or (matched, order) >= (best[0], best[1])):
                best = (matched, order, shrink, rows)
    return (best[2], best[3]) if best is not None else None


def k_label(path: str) -> str:
    """``"100"`` for ``…_00002_K100_real.molden`` (``""`` if unnamed that way)."""
    return _name_parts(path)[0]


def companion_part(path: str) -> Optional[str]:
    """The matching imaginary-part file for a ``_real`` one, when it exists.

    Away from Γ a crystalline orbital is complex and CRYSTAL writes its two parts
    as separate files; plotting the modulus needs both.
    """
    label, part = _name_parts(path)
    if part != "real" or not label:
        return None
    candidate = re.sub(r"_real\.molden$", "_complex.molden", path, flags=re.IGNORECASE)
    return candidate if os.path.isfile(candidate) else None


def find_orbital_files(path: str) -> List[str]:
    """Every ORBITALS Molden file sitting beside ``path``, sorted by k then part.

    ``path`` may be the CRYSTAL output, or any one of the Molden files. A run
    writes one pair per sampled k-point, and CRYSTAL can write the same set twice
    under two stems, so byte-identical duplicates are dropped.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return []
    found = [
        os.path.join(directory, name)
        for name in names
        if name.lower().endswith(".molden") and _FILENAME.search(name)
    ]
    return _drop_duplicates(found)


def _drop_duplicates(paths: List[str]) -> List[str]:
    """Keep one file per (k-point, part); a run can write the same set twice."""
    seen: dict = {}
    for path in paths:
        key = _name_parts(path)
        seen.setdefault(key, path)
    return [seen[key] for key in sorted(seen)]


__all__ = [
    "BOHR_PER_ANGSTROM",
    "MoldenOrbitals",
    "Orbital",
    "Shell",
    "companion_part",
    "find_orbital_files",
    "k_label",
    "k_vector",
    "load",
]
