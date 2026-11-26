"""Check CRYSTALLine's rod groups against CRYSTAL's own, all 99 in one batch.

Two steps, with a CRYSTAL run in between:

    python tools/rod_group_check.py write rodcheck/
    (run every deck in rodcheck/ with CRYSTAL: rodcheck/run_all.sh does it)
    python tools/rod_group_check.py check rodcheck/*.out

``write`` makes one deck per rod group: two atoms in general positions, in
POLYMER input, with ``TESTGEOM`` so that CRYSTAL stops once the geometry is
read — a second a deck, no basis set, no SCF.

``check`` reads any CRYSTAL polymer output, these or any other, and takes from
it the group number, the repeat length and the atoms it was given, all as
CRYSTAL itself prints them. It then builds the chain from those with
CRYSTALLine's operators and compares it, atom for atom, with the chain CRYSTAL
built: the operators, the axis, the orientation across the chain and the origin
along it are all in that comparison. Each output gets one line; any that does
not agree says by how much.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from crystalline.core import rod_groups as rg  # noqa: E402

_A = 8.0                     # Å, the repeat length of every deck
_TOLERANCE = 1e-3            # Å
_APART = 1.2                 # Å: no two atoms closer, in any of the 99 chains


# ── write ─────────────────────────────────────────────────────────────────
def chain(number: int, unit, a: float):
    """``(numbers, positions)``: every atom of one repeat, x fractional in [0, 1)."""
    group = rg.rod_group(number)
    numbers = np.array([z for z, _ in unit])
    start = np.array([position for _, position in unit], dtype=float)
    images = np.concatenate([rg._image(group, k, start) for k in range(group.order)])
    kinds = np.tile(numbers, group.order)
    images[:, 0] %= 1.0
    same = rg._matches(images, kinds, images, kinds, a, _TOLERANCE)
    keep = [i for i in range(len(images)) if not same[i, :i].any()]
    return kinds[keep], images[keep]


def _closest(positions, a: float) -> float:
    cartesian = positions.copy()
    cartesian[:, 0] *= a
    gaps = cartesian[:, None, :] - cartesian[None, :, :]
    gaps[..., 0] = (gaps[..., 0] / a + 0.5) % 1.0 * a - 0.5 * a
    distance = np.linalg.norm(gaps, axis=2)
    np.fill_diagonal(distance, np.inf)
    return float(distance.min())


def starting_atoms():
    """Two atoms, x fractional and y, z in Å, that every group spreads apart.

    In general position — on no mirror, no axis, no centre — and far enough
    from all of them that no chain puts two atoms within :data:`_APART` of
    each other, which CRYSTAL would refuse: 2.5–4 Å off the axis, since the
    mirrors of a hexagonal group are 30° apart and an atom between two of them
    is only half its distance from the axis away from its own image. Found by a
    search from a fixed seed, so every run writes the same decks.
    """
    rng = np.random.default_rng(0)
    for _attempt in range(20000):
        atoms = []
        for z in (6, 8):
            radius, angle = rng.uniform(2.5, 4.0), rng.uniform(0.0, 2 * np.pi)
            atoms.append((z, (round(float(rng.random()), 4),
                              round(float(radius * np.cos(angle)), 4),
                              round(float(radius * np.sin(angle)), 4))))
        atoms = tuple(atoms)
        if all(_closest(chain(number, atoms, _A)[1], _A) >= _APART
               for number, *_rest in rg.TABLE):
            return atoms
    raise SystemExit("no two starting atoms fit every group; widen the search")


def write(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    atoms = starting_atoms()
    for number, symbol, setting, _space_group in rg.TABLE:
        lines = [f"Rod group {number} {symbol} ({setting}) - CRYSTALLine check",
                 "POLYMER", str(number), f"{_A:.6f}", str(len(atoms))]
        lines += [f"{z} {x:.6f} {y:.6f} {zc:.6f}" for z, (x, y, zc) in atoms]
        lines += ["TESTGEOM", "END"]
        (folder / f"rod_{number:02d}.d12").write_text("\n".join(lines) + "\n")
    script = folder / "run_all.sh"
    script.write_text(
        "#!/bin/sh\n"
        "# Usage: sh run_all.sh [path to the serial crystal executable]\n"
        'CRYSTAL="${1:-crystal}"\n'
        'cd "$(dirname "$0")"\n'
        "for deck in rod_*.d12; do\n"
        '  "$CRYSTAL" < "$deck" > "${deck%.d12}.out" 2>&1\n'
        "done\n")
    print(f"Wrote 99 decks and run_all.sh to {folder}")


# ── check ─────────────────────────────────────────────────────────────────
_GROUP = re.compile(r"POLYMER GROUP N\.\s*(\d+)")
_NUMBER = r"(-?\d+\.\d+(?:[EeDd][-+]?\d+)?)"
_ROW = re.compile(r"^\s*\d+\s+(\d+)\s+" + r"\s+".join([_NUMBER] * 3) + r"\s*$")
_TABLE_ROW = re.compile(r"^\s*\d+\s+[TF]\s+(\d+)\s+[A-Z]{1,2}\s+"
                        + r"\s+".join([_NUMBER] * 3) + r"\s*$")
_EQUIVALENT_ROW = re.compile(r"^\s*\d+\s+\d+\s+\d+\s+(\d+)\s+[A-Z]{1,2}\s+"
                             + r"\s+".join([_NUMBER] * 3) + r"\s*$")
_PRIMITIVE_A = re.compile(r"^\s*(\d+\.\d+)\s+500\.0+\s+500\.0+\s")


def _float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def read(path: Path):
    """``(group, a, given, built)`` as CRYSTAL prints them, or a reason it cannot."""
    lines = path.read_text(errors="ignore").splitlines()
    group = a = None
    given, built, equivalent = [], [], []
    for index, line in enumerate(lines):
        if group is None and (match := _GROUP.search(line)):
            group = int(match.group(1))
        elif a is None and (match := _PRIMITIVE_A.match(line)):
            a = float(match.group(1))
        elif "INPUT COORDINATES" in line and not given:
            for row in lines[index + 1:]:
                if (match := _ROW.match(row)):
                    given.append((int(match.group(1)) % 100,
                                  tuple(_float(match.group(k)) for k in (2, 3, 4))))
                elif given:
                    break
        elif "COORDINATES OF THE EQUIVALENT ATOMS" in line and not equivalent:
            # Printed straight after the input, x in Å: there even if the run
            # stops before the table below.
            for row in lines[index + 1:]:
                if (match := _EQUIVALENT_ROW.match(row)):
                    equivalent.append((int(match.group(1)) % 100,
                                       tuple(_float(match.group(k)) for k in (2, 3, 4))))
                elif equivalent and row.strip().startswith(("NUMBER", "*")):
                    break
        elif "ATOMS IN THE ASYMMETRIC UNIT" in line and not built:
            for row in lines[index + 1:]:
                if (match := _TABLE_ROW.match(row)):
                    built.append((int(match.group(1)) % 100,
                                  tuple(_float(match.group(k)) for k in (2, 3, 4))))
                elif built and row.strip() and not row.lstrip().startswith("*"):
                    break
    if group is None:
        return "no POLYMER GROUP line: not a polymer, or CRYSTAL stopped before it"
    if not built and equivalent and a:
        built = [(z, (x / a, y, zc)) for z, (x, y, zc) in equivalent]
    if a is None or not given or not built:
        return "the cell, the input atoms or the generated atoms are missing"
    return group, a, given, built


def check_one(path: Path) -> bool:
    found = read(path)
    if isinstance(found, str):
        print(f"{path.name}: cannot check — {found}")
        return False
    group, a, given, built = found
    numbers, ours = chain(group, given, a)
    theirs_numbers = np.array([z for z, _ in built])
    theirs = np.array([position for _, position in built], dtype=float)
    theirs[:, 0] %= 1.0
    symbol = rg.rod_group(group).symbol
    if len(ours) != len(theirs):
        print(f"{path.name}: rod group {group} {symbol} — MISMATCH: CRYSTAL builds "
              f"{len(theirs)} atoms, CRYSTALLine {len(ours)}")
        return False
    hits = rg._matches(ours, numbers, theirs, theirs_numbers, a, _TOLERANCE)
    if hits.any(axis=1).all() and hits.any(axis=0).all():
        print(f"{path.name}: rod group {group} {symbol} — agrees ({len(ours)} atoms)")
        return True
    along = (ours[:, None, 0] - theirs[None, :, 0] + 0.5) % 1.0 - 0.5
    gap = np.sqrt((along * a) ** 2 + ((ours[:, None, 1:] - theirs[None, :, 1:]) ** 2).sum(2))
    worst = float(gap.min(axis=1).max())
    print(f"{path.name}: rod group {group} {symbol} — MISMATCH: an atom of "
          f"CRYSTALLine's chain is {worst:.4f} Å from any of CRYSTAL's")
    return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("write", help="write the 99 decks").add_argument("folder", type=Path)
    commands.add_parser("check", help="compare CRYSTAL outputs").add_argument(
        "outputs", type=Path, nargs="+")
    arguments = parser.parse_args(argv)
    if arguments.command == "write":
        write(arguments.folder)
        return 0
    results = [check_one(path) for path in sorted(arguments.outputs)]
    print(f"\n{sum(results)} of {len(results)} agree")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
