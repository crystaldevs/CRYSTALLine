"""Crystallographic analysis of a :class:`Structure` — space group, lattice,
density, etc. — derived with pymatgen (the same engine CRYSTALClear uses).

Kept Qt-free and with a lazy pymatgen import so it can be unit-tested without a
display, matching the rest of ``core``.

**Reduced-dimensionality systems.** CRYSTAL writes a slab (2D), polymer (1D) or
molecule (0D) with the non-periodic lattice parameters *formally set to 500 Å*
(it prints "NON PERIODIC DIRECTION: LATTICE PARAMETER FORMALLY SET TO 500"). That
500 Å is a placeholder, not a real cell edge, so reporting a 3D space group, a
``c`` of 500, a cell volume or a density for a slab is meaningless (the density
comes out ~70× too low, diluted by vacuum). Instead a slab is analysed with
spglib's *layer group* (the 2D analogue of a space group), and only its in-plane
metrics — ``a``, ``b``, ``γ`` and the cell area — are reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from crystalline.core.structure import Structure


@dataclass(frozen=True)
class CrystalInfo:
    """Crystallographic summary of a structure (fields are ``None`` when N/A)."""

    formula: str
    n_atoms: int
    periodic: bool
    dimensionality: str
    ndim: Optional[int] = None  # number of periodic directions (0–3)
    space_group_symbol: Optional[str] = None
    space_group_number: Optional[int] = None
    layer_group_symbol: Optional[str] = None  # slab (2D) analogue of a space group
    layer_group_number: Optional[int] = None
    crystal_system: Optional[str] = None
    point_group: Optional[str] = None
    z: Optional[int] = None
    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None
    area: Optional[float] = None  # in-plane cell area for a slab (2D)
    volume: Optional[float] = None
    density: Optional[float] = None

    def rows(self) -> List[Tuple[str, str]]:
        """(label, value) pairs for display, skipping fields that don't apply.

        The lattice block is dimensionality-aware: a slab (2D) never shows the
        formal 500 Å ``c``, the vacuum-inflated volume or the meaningless density —
        it reports its in-plane metrics instead.
        """
        rows: List[Tuple[str, str]] = [
            ("Formula", self.formula),
            ("Atoms", str(self.n_atoms)),
            ("Dimensionality", self.dimensionality),
        ]
        if self.layer_group_symbol is not None:
            rows.append(("Layer group", f"{self.layer_group_symbol} (No. {self.layer_group_number})"))
        elif self.space_group_symbol is not None:
            rows.append(("Space group", f"{self.space_group_symbol} (No. {self.space_group_number})"))
        if self.crystal_system is not None:
            rows.append(("Crystal system", self.crystal_system.capitalize()))
        if self.point_group is not None:
            rows.append(("Point group", self.point_group))
        if self.z is not None:
            rows.append(("Formula units Z", str(self.z)))
        if self.ndim == 2:
            if self.a is not None:
                rows.append(("a, b (Å)", f"{self.a:.4f}, {self.b:.4f}"))
                rows.append(("γ (°)", f"{self.gamma:.3f}"))
            if self.area is not None:
                rows.append(("Cell area (Å²)", f"{self.area:.3f}"))
        elif self.ndim == 1:
            if self.a is not None:
                rows.append(("a (Å)", f"{self.a:.4f}"))
        else:
            if self.a is not None:
                rows.append(("a, b, c (Å)", f"{self.a:.4f}, {self.b:.4f}, {self.c:.4f}"))
                rows.append(("α, β, γ (°)", f"{self.alpha:.3f}, {self.beta:.3f}, {self.gamma:.3f}"))
            if self.volume is not None:
                rows.append(("Cell volume (Å³)", f"{self.volume:.3f}"))
            if self.density is not None:
                rows.append(("Density (g/cm³)", f"{self.density:.3f}"))
        return rows


_DIMENSIONALITY = {0: "Molecule (0D)", 1: "Polymer (1D)", 2: "Slab (2D)", 3: "Bulk crystal (3D)"}


def analyze(structure: Structure, symprec: float = 1e-2) -> CrystalInfo:
    """Return a :class:`CrystalInfo` for ``structure``.

    For a 3D crystal the reported lattice/volume/density/Z are those of the
    conventional (crystallographic) cell — what a crystallographer expects in a
    summary table — regardless of which cell was loaded. A slab (2D) is reported
    by its layer group and in-plane metrics, never the formal 500 Å ``c`` CRYSTAL
    fills the vacuum direction with. Non-periodic systems get just the composition.
    """
    ase_atoms = structure.to_ase()
    n_atoms = len(ase_atoms)
    ndim = int(sum(bool(p) for p in ase_atoms.get_pbc()))

    if n_atoms == 0:
        return CrystalInfo(formula="—", n_atoms=0, periodic=False, dimensionality="—", ndim=0)

    from pymatgen.core import Composition

    composition = Composition(ase_atoms.get_chemical_formula())
    formula = composition.reduced_formula
    dimensionality = _DIMENSIONALITY.get(ndim, "Bulk crystal (3D)")

    if not structure.is_periodic:
        return CrystalInfo(
            formula=formula, n_atoms=n_atoms, periodic=False,
            dimensionality=_DIMENSIONALITY.get(ndim, "Molecule (0D)"), ndim=ndim,
        )

    if ndim == 2:
        return _analyze_slab(ase_atoms, formula, n_atoms, dimensionality, symprec)
    if ndim == 1:
        return _analyze_polymer(ase_atoms, formula, n_atoms, dimensionality)

    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    pmg = AseAtomsAdaptor().get_structure(ase_atoms)

    try:
        sga = SpacegroupAnalyzer(pmg, symprec=symprec)
        conventional = sga.get_conventional_standard_structure()
        lattice = conventional.lattice
        _reduced, factor = conventional.composition.get_reduced_composition_and_factor()
        return CrystalInfo(
            formula=formula,
            n_atoms=n_atoms,
            periodic=True,
            dimensionality=dimensionality,
            ndim=3,
            space_group_symbol=sga.get_space_group_symbol(),
            space_group_number=sga.get_space_group_number(),
            crystal_system=sga.get_crystal_system(),
            point_group=sga.get_point_group_symbol(),
            z=int(round(factor)),
            a=lattice.a, b=lattice.b, c=lattice.c,
            alpha=lattice.alpha, beta=lattice.beta, gamma=lattice.gamma,
            volume=conventional.volume,
            density=float(conventional.density),
        )
    except Exception:  # noqa: BLE001 - symmetry analysis can fail; still show the lattice
        lattice = pmg.lattice
        return CrystalInfo(
            formula=formula, n_atoms=n_atoms, periodic=True, dimensionality=dimensionality, ndim=3,
            a=lattice.a, b=lattice.b, c=lattice.c,
            alpha=lattice.alpha, beta=lattice.beta, gamma=lattice.gamma,
            volume=pmg.volume, density=float(pmg.density),
        )


def _analyze_slab(ase_atoms, formula, n_atoms, dimensionality, symprec) -> CrystalInfo:
    """Layer-group analysis of a 2D slab, using its two periodic axes only.

    spglib's ``get_layergroup`` is the 2D counterpart of a space group; it takes
    the index of the aperiodic axis so the formal 500 Å vacuum vector is ignored
    rather than treated as a real (huge) lattice edge. If it is unavailable or
    fails, the in-plane metrics are still reported, just without the group.
    """
    from pymatgen.core import Composition

    pbc = [bool(p) for p in ase_atoms.get_pbc()]
    periodic_axes = [i for i, p in enumerate(pbc) if p]
    aperiodic_dir = next(i for i, p in enumerate(pbc) if not p)
    cell = np.asarray(ase_atoms.get_cell(), dtype=float)
    a, b, gamma, area = _inplane_params(cell, periodic_axes)
    _, factor = Composition(ase_atoms.get_chemical_formula()).get_reduced_composition_and_factor()

    symbol = number = point_group = None
    try:
        import spglib

        spg_cell = (cell, ase_atoms.get_scaled_positions(), ase_atoms.get_atomic_numbers())
        dataset = spglib.get_layergroup(spg_cell, aperiodic_dir=aperiodic_dir, symprec=symprec)
        if dataset is not None:
            symbol = dataset.international
            number = dataset.number
            point_group = dataset.pointgroup
    except Exception:  # noqa: BLE001 - layer-group detection is best-effort
        pass

    return CrystalInfo(
        formula=formula, n_atoms=n_atoms, periodic=True, dimensionality=dimensionality, ndim=2,
        layer_group_symbol=symbol, layer_group_number=number, point_group=point_group,
        z=int(round(factor)), a=a, b=b, gamma=gamma, area=area,
    )


def _analyze_polymer(ase_atoms, formula, n_atoms, dimensionality) -> CrystalInfo:
    """1D polymer: report the one periodic repeat length; the other two axes are
    CRYSTAL's formal 500 Å placeholders. (spglib has no rod groups, so no group.)"""
    from pymatgen.core import Composition

    pbc = [bool(p) for p in ase_atoms.get_pbc()]
    axis = next(i for i, p in enumerate(pbc) if p)
    length = float(np.linalg.norm(np.asarray(ase_atoms.get_cell(), dtype=float)[axis]))
    _, factor = Composition(ase_atoms.get_chemical_formula()).get_reduced_composition_and_factor()
    return CrystalInfo(
        formula=formula, n_atoms=n_atoms, periodic=True, dimensionality=dimensionality, ndim=1,
        z=int(round(factor)), a=length,
    )


def _inplane_params(cell: np.ndarray, axes: List[int]) -> Tuple[float, float, float, float]:
    """In-plane ``(a, b, γ, area)`` for the two periodic lattice vectors of a slab."""
    v1, v2 = cell[axes[0]], cell[axes[1]]
    a = float(np.linalg.norm(v1))
    b = float(np.linalg.norm(v2))
    cos_gamma = float(np.dot(v1, v2) / (a * b)) if a and b else 0.0
    gamma = float(np.degrees(np.arccos(np.clip(cos_gamma, -1.0, 1.0))))
    area = float(np.linalg.norm(np.cross(v1, v2)))
    return a, b, gamma, area


__all__ = ["CrystalInfo", "analyze"]
