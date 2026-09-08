"""Evaluate a crystalline orbital in real space, on a grid.

A crystalline orbital is a Bloch sum over the lattice,

    psi_k(r) = sum_T exp(i k.T) * sum_mu c_mu phi_mu(r - T)

where the ``c_mu`` are the reference cell's coefficients (what a Molden file
holds) and ``phi_mu`` are contracted Gaussians on the cell's atoms. Dropping the
lattice sum would draw a molecule, not a crystal: the orbital would fall to zero
at the cell boundary instead of continuing into the neighbouring cell.

Conventions, since each is a way to get a plausible-looking but wrong picture:

* **Normalisation lives in the coefficients.** Molden's contraction coefficients
  already carry each primitive's normalisation, so a primitive is evaluated as a
  bare ``exp(-alpha r^2)`` times its solid harmonic. Verified end to end by
  integrating ``|psi|^2`` over the cell, which comes out at 1.
* **Spherical harmonics in Molden's order.** ``[5D]`` is
  ``d0, d+1, d-1, d+2, d-2`` and ``[7F]`` is ``f0, f+1, f-1, f+2, f-2, f+3,
  f-3``; a different order silently permutes the lobes.
* **An ``sp`` shell is one s and one p** sharing exponents, contributing four
  atomic orbitals in the order ``s, px, py, pz``.

Qt- and PyVista-free, like the rest of ``core``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from crystalline.crystalio.molden import BOHR_PER_ANGSTROM, MoldenOrbitals, Shell

# A primitive is ignored past the radius where it has decayed to this fraction
# of its peak. Gaussians die quickly, so this buys a large speed-up for a
# negligible change: at 1e-8 the neglected tail is far below any isovalue worth
# drawing.
_PRIMITIVE_TOLERANCE = 1e-8

# How a complex orbital is turned into something drawable. See :func:`evaluate`.
AMPLITUDE = "amplitude"   # Re(psi): signed, ± lobes
MODULUS = "modulus"       # |psi|: one surface, no sign, identical in every cell
PHASE = "phase"           # |psi| as the surface, arg(psi) as its colour
DRAW_MODES = (AMPLITUDE, MODULUS, PHASE)

@dataclass(frozen=True)
class OrbitalField:
    """A crystalline orbital sampled on a regular grid.

    ``values`` is the (real) amplitude with shape ``shape``; ``origin`` and
    ``spacing`` place it in cartesian Angstrom, so it can be handed straight to a
    uniform-grid isosurface. ``modulus`` says whether the values are an amplitude
    (signed, with lobes) or a modulus (non-negative).
    """

    values: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray
    shape: Tuple[int, int, int]
    modulus: bool = False
    # The Bloch phase of the *cell* each grid point falls in — one value per
    # cell, not per point — when the phase view was asked for. Drawn as colour
    # on a modulus surface: |psi| is lattice-periodic, so the surface is
    # identical in every cell and the colour is the only thing that changes,
    # changing by exactly k.T. This is the phonon panel's per-atom phase colour,
    # applied to the same quantity.
    #
    # Pointwise arg(psi) was tried here first and is worse to look at: a Bloch
    # state is e^(i*k.r) times something cell-periodic, so its phase also ramps
    # *within* each cell and every lobe comes out a smear of three or four
    # colours, with no flat colour anywhere to read the relation off.
    phases: Optional[np.ndarray] = None
    # The parallelepiped the orbital belongs to — one cell, or the supercell
    # being shown. The sampled box is its *bounding* box, which for a
    # non-orthogonal cell is appreciably larger, so the renderer clips the
    # isosurface back to this.
    cell: Optional[np.ndarray] = None

    @property
    def peak(self) -> float:
        """Largest absolute amplitude — the scale an isovalue is a fraction of."""
        return float(np.abs(self.values).max()) if self.values.size else 0.0


# ── real solid harmonics, in Molden's order ───────────────────────────────
def _solid_harmonics(kind: str, d: np.ndarray) -> List[np.ndarray]:
    """The angular factors of a shell at displacements ``d`` (n, 3), in bohr.

    Real solid harmonics ``S_lm``, ordered as Molden lists them. The radial
    ``r^l`` is included (these are solid, not spherical, harmonics), which is
    what multiplies the Gaussian directly.
    """
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    if kind == "s":
        return [np.ones_like(x)]
    if kind == "p":
        return [x, y, z]
    if kind == "d":
        r2 = x * x + y * y + z * z
        # 1/sqrt(3) is the (2l-1)!! that Molden's normalisation leaves to the
        # angular part for l >= 2. Without it every d orbital comes out sqrt(3)
        # too large *relative to s and p*, which does not merely rescale the
        # picture — it changes the shape of any orbital mixing d with s or p.
        # Pinned by integrating a single d atomic orbital, which must give 1.
        scale = 1.0 / np.sqrt(3.0)
        return [
            scale * (3.0 * z * z - r2) / 2.0,          # d0
            scale * np.sqrt(3.0) * x * z,              # d+1
            scale * np.sqrt(3.0) * y * z,              # d-1
            scale * np.sqrt(3.0) / 2.0 * (x * x - y * y),  # d+2
            scale * np.sqrt(3.0) * x * y,              # d-2
        ]
    if kind == "f":
        r2 = x * x + y * y + z * z
        scale = 1.0 / np.sqrt(15.0)  # (2l-1)!! = 15 for l = 3
        return [
            scale * z * (5.0 * z * z - 3.0 * r2) / 2.0,                 # f0
            scale * np.sqrt(6.0) / 4.0 * x * (5.0 * z * z - r2),        # f+1
            scale * np.sqrt(6.0) / 4.0 * y * (5.0 * z * z - r2),        # f-1
            scale * np.sqrt(15.0) / 2.0 * z * (x * x - y * y),          # f+2
            scale * np.sqrt(15.0) * x * y * z,                          # f-2
            scale * np.sqrt(10.0) / 4.0 * x * (x * x - 3.0 * y * y),    # f+3
            scale * np.sqrt(10.0) / 4.0 * y * (3.0 * x * x - y * y),    # f-3
        ]
    raise ValueError(f"unsupported shell type {kind!r}")


def _shell_cutoff(shell: Shell) -> float:
    """Radius (bohr) past which every primitive of ``shell`` is negligible."""
    alpha = float(np.min(shell.exponents))
    return float(np.sqrt(-np.log(_PRIMITIVE_TOLERANCE) / alpha))


def _shell_amplitudes(shell: Shell, d: np.ndarray) -> List[np.ndarray]:
    """This shell's atomic orbitals evaluated at displacements ``d`` (bohr).

    Returns one array per atomic orbital, in the order the coefficients expect:
    ``s`` first for an ``sp``, then its three p components.
    """
    r2 = np.einsum("ij,ij->i", d, d)
    exponentials = np.exp(-shell.exponents[None, :] * r2[:, None])  # (n, nprim)

    if shell.kind == "sp":
        radial_s = exponentials @ shell.coefficients[:, 0]
        radial_p = exponentials @ shell.coefficients[:, 1]
        return [radial_s] + [radial_p * angular
                             for angular in _solid_harmonics("p", d)]

    radial = exponentials @ shell.coefficients
    return [radial * angular for angular in _solid_harmonics(shell.kind, d)]


# ── the grid ──────────────────────────────────────────────────────────────
# A ceiling on the sampled grid, so asking for a large tiling at a fine density
# cannot quietly ask for tens of gigabytes: three arrays of this many doubles is
# a few hundred megabytes, and the evaluation is linear in the point count.
# Past it the density is dropped uniformly rather than the box being cut.
_MAX_GRID_POINTS = 4_000_000


def _bounds(cell: np.ndarray, padding: float):
    """The axis-aligned box enclosing a cell, as ``(low, high)``."""
    cell = np.asarray(cell, dtype=float)
    corners = np.asarray(
        [sum(c) for c in itertools.product(*[(np.zeros(3), v) for v in cell])]
    )
    return corners.min(axis=0) - padding, corners.max(axis=0) + padding


def cell_grid(cell: np.ndarray, samples: int, padding: float = 0.0, matched_to=None):
    """A regular cartesian grid spanning the cell's bounding box.

    Returns ``(origin, spacing, shape)`` rather than the points themselves: the
    grid is regular, so every consumer here wants the description, not half a
    million coordinates.

    The box is axis-aligned, so a non-orthogonal cell is sampled over a slightly
    larger region than itself — which is what an isosurface wants anyway, since
    an orbital does not stop at the cell face.

    ``matched_to`` is a smaller cell whose sampling *density* this grid should
    match: ``samples`` then counts the points across that box, and this one gets
    proportionally more. Without it a 3x3x1 supercell would be drawn with the
    same 60x60x60 points as a single cell — the same grid stretched over two and
    a half times the distance in each direction, so the very tiling asked for to
    see the orbital better would be what made it coarse.
    """
    low, high = _bounds(cell, padding)
    if matched_to is None:
        shape = np.full(3, int(samples), dtype=int)
    else:
        reference_low, reference_high = _bounds(matched_to, padding)
        ratio = (high - low) / np.maximum(reference_high - reference_low, 1e-12)
        shape = np.maximum(np.rint((int(samples) - 1) * ratio).astype(int) + 1, 2)
    excess = float(np.prod(shape.astype(float))) / _MAX_GRID_POINTS
    if excess > 1.0:
        shape = np.maximum((shape / excess ** (1.0 / 3.0)).astype(int), 2)

    shape = tuple(int(n) for n in shape)
    axes = [np.linspace(low[k], high[k], shape[k]) for k in range(3)]
    spacing = np.array([
        (axes[k][1] - axes[k][0]) if shape[k] > 1 else 1.0 for k in range(3)
    ])
    return low, spacing, shape


# ── evaluation ────────────────────────────────────────────────────────────
def evaluate(
    data: MoldenOrbitals,
    index: int,
    samples: int = 60,
    padding: float = 0.0,
    repeat: Tuple[int, int, int] = (1, 1, 1),
    box_cell: Optional[np.ndarray] = None,
    imaginary: Optional[MoldenOrbitals] = None,
    kpoint: Optional[np.ndarray] = None,
    draw: str = AMPLITUDE,
    phase: float = 0.0,
) -> OrbitalField:
    """Sample crystalline orbital ``index`` of ``data`` on a grid.

    ``kpoint`` is k in fractional (reciprocal-lattice) coordinates, and is what
    makes this a *crystalline* orbital rather than a molecule repeated: each
    lattice image enters the sum with its phase ``exp(2*pi*i*k.T)``, so the
    orbital's sign and shape change from one cell to the next exactly as a
    phonon's displacement pattern does at the same q. At Γ (or when k is
    unknown) every image enters in phase and one cell tells the whole story.

    ``box_cell`` is the region to sample and clip to, when that is not the
    Molden file's own cell — the crystallographic (conventional) cell, above all.
    The two are different regions of the *same* crystal, and the lattice sum runs
    over the file's own lattice either way, so an orbital defined on a primitive
    cell can be drawn over the conventional one it belongs to. This is what lets
    the view stay crystallographic instead of being forced to the primitive cell.

    ``repeat`` spans a supercell of that region — the way to see the phase
    relation, which needs at least two cells to be a relation at all.
    :func:`crystalline.core.phonons.commensurate_repeats` gives the tiling in
    which one whole period of ``kpoint`` fits. It is *not* a tiling of the
    single-cell result: the orbital is evaluated over the larger box directly,
    because only its modulus repeats cell to cell. ``padding`` (Angstrom) grows
    the sampled box beyond that.

    How far the Bloch sum reaches is worked out from the box and the basis rather
    than being a setting: every lattice translation whose shells can touch the
    box contributes, and none of them is optional.

    ``imaginary`` is the companion ``_complex`` file for a k-point away from
    Gamma, where the orbital is complex and CRYSTAL writes its two parts
    separately. Without it only the real coefficients are known, and the drawn
    orbital is missing half of itself away from Γ.

    What is drawn from the complex ``psi`` is then a choice of ``draw``:

    * :data:`AMPLITUDE` draws ``Re(psi * e^{i*phase})`` — a signed field with the
      ± lobes. ``phase`` rotates through the family of such sections the way a
      phonon animation steps through its cycle; which member is "the" real part
      is a convention, the relation between cells is not.
    * :data:`MODULUS` draws ``|psi|``, which is lattice-periodic and has no sign:
      every cell looks identical, and the phase is gone.
    * :data:`PHASE` draws that same ``|psi|`` surface and carries each cell's
      Bloch phase ``k.T`` alongside it, for the renderer to colour the surface
      with — one flat colour per cell. This is the one that shows the phase
      *relation*: the surface is identical in every cell, so the colour is the
      only thing that changes, and it changes by exactly ``k.T``. It is what the
      phonon panel does with its cyclic colour per atom, for the same reason —
      a relation is only legible when the thing being related looks the same.
      ``phase`` turns the whole colour wheel, and ``box_cell`` is the cell the
      colours are counted in, since that is the one drawn on screen.
    """
    if not 0 <= index < len(data.orbitals):
        raise IndexError(f"orbital {index} out of range (this file has {len(data.orbitals)})")
    if data.cell is None:
        raise ValueError("this file carries no cell; only crystalline orbitals are supported")

    repeat = tuple(max(1, int(r)) for r in repeat)
    region = np.asarray(data.cell if box_cell is None else box_cell, dtype=float)
    if region.shape != (3, 3) or abs(np.linalg.det(region)) < 1e-8:
        region = np.asarray(data.cell, dtype=float)
    unit_region = region
    region = region * np.asarray(repeat, dtype=float)[:, None]
    # ``samples`` is a density per cell, so a tiling is drawn as finely as one
    # cell is rather than being the same grid stretched over more of the crystal.
    low, spacing, shape = cell_grid(region, samples, padding, matched_to=unit_region)

    coefficients = data.orbitals[index].coefficients
    imaginary_coefficients = None
    if imaginary is not None:
        if len(imaginary.orbitals) != len(data.orbitals):
            raise ValueError("the real and imaginary files describe different orbital sets")
        imaginary_coefficients = imaginary.orbitals[index].coefficients
    real, imag = _amplitude(
        data, coefficients, low, spacing, shape,
        kpoint=kpoint, imaginary_coefficients=imaginary_coefficients,
    )

    # Scaled by the *complex* norm, so an isovalue means the same thing whichever
    # of the three views is on screen, and whether one cell or a supercell is
    # drawn.
    scale = _norm_scale(real, imag, low, spacing, shape, data.cell)
    phases = None
    if imag is None:
        # A real orbital: there is nothing for the other two views to show that
        # the signed one doesn't, so they collapse onto it.
        values, drawn_modulus = real * scale, False
    elif draw == MODULUS:
        values, drawn_modulus = np.sqrt(real * real + imag * imag) * scale, True
    elif draw == PHASE:
        values, drawn_modulus = np.sqrt(real * real + imag * imag) * scale, True
        phases = _cell_phases(low, spacing, shape, unit_region, kpoint, data.cell, phase)
    else:
        values = (real * np.cos(phase) - imag * np.sin(phase)) * scale
        drawn_modulus = False

    return OrbitalField(
        values=values, origin=low, spacing=spacing, shape=shape,
        modulus=drawn_modulus, cell=region, phases=phases,
    )


def commensurate_repeats_in(kpoint, cell, box_cell=None, limit: int = 12) -> tuple:
    """The tiling of ``box_cell`` in which one whole period of ``kpoint`` fits.

    ``kpoint`` is fractional in the reciprocal basis of ``cell`` — the Molden
    file's own (primitive) lattice, which is the only basis k is ever quoted in.
    The cell on screen is usually a different one: the crystallographic cell of
    the same crystal, whose vectors are integer combinations ``M`` of the
    primitive ones. The phase across one step of the *displayed* cell is then
    ``exp(2*pi*i*(M.k))``, so it is ``M.k``, not ``k``, whose denominators say
    how many cells a period takes.

    Skipping that transformation is not a small error for a rhombohedral crystal
    drawn on its hexagonal cell: k = (2/3, 1/3, 0) asks for 3x3x1 primitive
    cells, which is one hexagonal cell and no tiling at all.
    """
    from crystalline.core.phonons import commensurate_repeats

    if kpoint is None:
        return (1, 1, 1)
    k = np.asarray(kpoint, dtype=float).ravel()
    if box_cell is None:
        return tuple(commensurate_repeats(k, limit=limit))
    cell = np.asarray(cell, dtype=float)
    box_cell = np.asarray(box_cell, dtype=float)
    if cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-8:
        return tuple(commensurate_repeats(k, limit=limit))
    transform = box_cell @ np.linalg.inv(cell)
    return tuple(commensurate_repeats(transform @ k, limit=limit))


def _cell_phases(origin, spacing, shape, box_cell, kpoint, cell, offset: float):
    """The Bloch phase of the cell each grid point falls in, wrapped to ±pi.

    Constant across a cell and stepping by ``2*pi*(M.k)_j`` from one to the next,
    ``M`` taking the lattice ``kpoint`` is quoted against to the cell being drawn
    (see :func:`commensurate_repeats_in`). A colour map on this is the picture of
    ``psi(r + T) = e^(2*pi*i*k.T) psi(r)``.
    """
    if kpoint is None:
        return np.zeros(shape, dtype=float)
    box_cell = np.asarray(box_cell, dtype=float)
    transform = box_cell @ np.linalg.inv(np.asarray(cell, dtype=float))
    step = 2.0 * np.pi * (transform @ np.asarray(kpoint, dtype=float))

    inverse = np.linalg.inv(box_cell)  # r @ inverse -> fractional in the drawn cell
    axes = [
        (origin[j] + spacing[j] * np.arange(shape[j])).reshape(
            [-1 if a == j else 1 for a in range(3)]
        )
        for j in range(3)
    ]
    total = np.full(shape, float(offset))
    for j in range(3):
        if not step[j]:
            continue
        fractional = sum(axes[a] * inverse[a, j] for a in range(3))
        total = total + np.floor(fractional) * step[j]
    return np.angle(np.exp(1j * total))  # wrapped to (-pi, pi]


def _norm_scale(real, imag, origin, spacing, shape, cell) -> float:
    """``1/sqrt(int |psi|^2 over one cell)`` — 1.0 if that can't be measured.

    CRYSTAL's Molden coefficients carry their own overall normalisation, which is
    not "one electron per cell" — evaluated as written they integrate to about
    0.15 here. That constant is the same for every orbital of a file, so it says
    nothing about the orbital and only makes isovalues incomparable between runs.
    """
    density = real * real if imag is None else real * real + imag * imag
    total = _integrate(density, origin, spacing, shape, cell)
    if not np.isfinite(total) or total <= 0.0:
        return 1.0
    return float(1.0 / np.sqrt(total))


def _amplitude(
    data: MoldenOrbitals,
    coefficients: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    shape: Tuple[int, int, int],
    kpoint: Optional[np.ndarray] = None,
    imaginary_coefficients: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """The Bloch sum on the grid, as ``(real, imaginary)`` fields.

    ``imaginary`` is ``None`` when the orbital is real everywhere — Γ with no
    companion file — which is the common case and halves the work.

    Each lattice image contributes with its phase ``exp(2*pi*i*k.n)``, ``n``
    being the image's integer translation. Leaving that factor out is not a
    small error away from Γ: it sums every cell in phase, which is a *different*
    orbital (the one at Γ built from these coefficients) and is lattice-periodic
    when the real one is not.

    Each shell is evaluated only on the block of grid points its Gaussians can
    reach. Because the grid is regular, that block is found by arithmetic on the
    indices rather than by testing every point — which is the difference between
    this being interactive and not: masking the whole grid per shell meant
    ~1350 passes over half a million points, and dominated the cost entirely.
    The two channels share that evaluation; only the accumulation is doubled.
    """
    cell_bohr = np.asarray(data.cell, dtype=float) * BOHR_PER_ANGSTROM
    centres = np.asarray(data.positions, dtype=float) * BOHR_PER_ANGSTROM
    origin_bohr = np.asarray(origin, dtype=float) * BOHR_PER_ANGSTROM
    spacing_bohr = np.asarray(spacing, dtype=float) * BOHR_PER_ANGSTROM
    axes = [origin_bohr[k] + spacing_bohr[k] * np.arange(shape[k]) for k in range(3)]

    k = None if kpoint is None else np.asarray(kpoint, dtype=float)
    if k is not None and not np.any(k):
        k = None  # Γ: every image in phase, so the real path applies
    complex_orbital = k is not None or imaginary_coefficients is not None

    cutoffs = [_shell_cutoff(shell) for shell in data.shells]
    translations = _translations(cell_bohr, centres, axes, max(cutoffs, default=0.0))

    real = np.zeros(shape, dtype=float)
    imag = np.zeros(shape, dtype=float) if complex_orbital else None
    for image, translation in translations:
        if k is None:
            weight_re, weight_im = 1.0, 0.0
        else:
            angle = 2.0 * np.pi * float(np.dot(k, image))
            weight_re, weight_im = float(np.cos(angle)), float(np.sin(angle))
        offset = 0
        for shell, cutoff in zip(data.shells, cutoffs):
            size = shell.size
            block = coefficients[offset:offset + size]
            block_imaginary = (
                None if imaginary_coefficients is None
                else imaginary_coefficients[offset:offset + size]
            )
            offset += size
            if not np.any(block) and (block_imaginary is None or not np.any(block_imaginary)):
                continue  # this shell contributes nothing to this orbital
            centre = centres[shell.atom] + translation

            # The index window this shell can reach, per axis.
            window = []
            for k_axis in range(3):
                lo = int(np.ceil(
                    (centre[k_axis] - cutoff - origin_bohr[k_axis]) / spacing_bohr[k_axis]
                ))
                hi = int(np.floor(
                    (centre[k_axis] + cutoff - origin_bohr[k_axis]) / spacing_bohr[k_axis]
                )) + 1
                window.append((max(lo, 0), min(hi, shape[k_axis])))
            if any(lo >= hi for lo, hi in window):
                continue  # this image is out of the sampled box entirely

            slices = tuple(slice(lo, hi) for lo, hi in window)
            sub = np.meshgrid(*[axes[a][slices[a]] - centre[a] for a in range(3)],
                              indexing="ij")
            displacement = np.stack([s.ravel() for s in sub], axis=1)
            sub_shape = sub[0].shape
            amplitudes = _shell_amplitudes(shell, displacement)
            for position, amplitude in enumerate(amplitudes):
                value = float(block[position])
                value_imaginary = (
                    0.0 if block_imaginary is None else float(block_imaginary[position])
                )
                if not value and not value_imaginary:
                    continue
                # (weight_re + i weight_im) * (value + i value_imaginary)
                contribution = weight_re * value - weight_im * value_imaginary
                field = amplitude.reshape(sub_shape)
                if contribution:
                    real[slices] += contribution * field
                if imag is not None:
                    contribution = weight_im * value + weight_re * value_imaginary
                    if contribution:
                        imag[slices] += contribution * field
    return real, imag


def _translations(cell, centres, axes, reach) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Every lattice translation whose basis functions can touch the sampled box.

    Each is ``(image, shift)``: the integer triple, which the Bloch phase is
    computed from, and the cartesian displacement that places the cell.

    Worked out rather than configured: a supercell box needs more of them than a
    single cell, and leaving one out silently thins the orbital near a face.
    The range comes from taking the box (grown by the basis's reach) into
    fractional coordinates and covering it.
    """
    low = np.array([axis[0] for axis in axes]) - reach
    high = np.array([axis[-1] for axis in axes]) + reach
    span = np.stack([centres.min(axis=0), centres.max(axis=0)])
    corners = np.asarray(
        list(itertools.product(*zip(low - span[1], high - span[0]))), dtype=float
    )
    fractional = corners @ np.linalg.inv(cell)
    lower = np.floor(fractional.min(axis=0)).astype(int)
    upper = np.ceil(fractional.max(axis=0)).astype(int)
    return [
        (np.asarray(t, dtype=float), np.asarray(t, dtype=float) @ cell)
        for t in itertools.product(*[range(lower[k], upper[k] + 1) for k in range(3)])
    ]


def integrate_density(field: OrbitalField, cell: np.ndarray) -> float:
    """``int |psi|^2`` over one cell's worth of the sampled box.

    The check that the whole chain is right: normalisation, harmonic ordering and
    the lattice sum all have to be correct for this to come out at 1. It is a
    Riemann sum over the *box*, so only the points inside the cell are counted.

    Note that this squares whatever the field holds, which for a real *section*
    of a complex orbital is not ``|psi|^2`` — a section carries about half the
    density, the rest sitting in the part not drawn.
    """
    return _integrate(field.values ** 2, field.origin, field.spacing, field.shape, cell)


def _integrate(density: np.ndarray, origin, spacing, shape, cell) -> float:
    """Riemann sum of ``density`` over the points of the box inside ``cell``."""
    cell = np.asarray(cell, dtype=float)
    axes = [origin[k] + spacing[k] * np.arange(shape[k]) for k in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([m.ravel() for m in mesh], axis=1)
    fractional = points @ np.linalg.inv(cell)
    inside = np.all((fractional >= 0.0) & (fractional < 1.0), axis=1)
    voxel = float(np.prod(spacing))
    return float(density.ravel()[inside].sum() * voxel)


__all__ = [
    "AMPLITUDE",
    "DRAW_MODES",
    "MODULUS",
    "OrbitalField",
    "PHASE",
    "cell_grid",
    "commensurate_repeats_in",
    "evaluate",
    "integrate_density",
]
