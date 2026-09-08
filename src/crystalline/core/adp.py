"""Atomic displacement parameters: the thermal ellipsoid of each atom.

An ADP is the mean-square displacement tensor ``U = <u u^T>`` of an atom about
its site, in Angstrom squared. Drawn as the surface enclosing a chosen
probability of finding the atom, it is the ellipsoid every published crystal
structure carries — and the one quantity a calculation and a diffraction
refinement can be compared on directly.

Tensors here are always **cartesian** — what CRYSTAL prints, and what a renderer
needs since the ellipsoid is drawn in real space. A CIF's ``U_ij`` is a different
convention (defined through the structure factor, and equal to the cartesian form
only for an orthogonal cell aligned with the cartesian axes); nothing here writes
CIFs, so no conversion between the two is provided.

Qt- and PyVista-free, like the rest of ``core``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import numpy as np

# Crystallographic default: the surface enclosing half the probability density.
DEFAULT_PROBABILITY = 0.5


@dataclass(frozen=True)
class ADPSet:
    """ADP tensors for every atom, at each temperature the run reported.

    Attributes
    ----------
    temperatures:
        ``(ntemp,)`` in K, in the order the output printed them.
    tensors:
        ``(ntemp, natom, 3, 3)`` cartesian ADP tensors in Angstrom squared.
    """

    temperatures: np.ndarray
    tensors: np.ndarray

    def __post_init__(self) -> None:
        temperatures = np.asarray(self.temperatures, dtype=float).ravel()
        tensors = np.asarray(self.tensors, dtype=float)
        if tensors.ndim != 4 or tensors.shape[0] != len(temperatures) or tensors.shape[2:] != (3, 3):
            raise ValueError(
                f"tensors must be (ntemp, natom, 3, 3) with ntemp={len(temperatures)}, "
                f"got {tensors.shape}"
            )
        object.__setattr__(self, "temperatures", temperatures)
        object.__setattr__(self, "tensors", tensors)

    def __len__(self) -> int:
        return len(self.temperatures)

    @property
    def n_atoms(self) -> int:
        return self.tensors.shape[1]

    def at(self, index: int) -> np.ndarray:
        """The ``(natom, 3, 3)`` tensors at temperature ``index`` (clamped)."""
        if len(self) == 0:
            return np.empty((0, 3, 3))
        return self.tensors[int(np.clip(index, 0, len(self) - 1))]

    def label(self, index: int) -> str:
        """``"300 K"`` for the temperature at ``index`` — for a picker."""
        if len(self) == 0:
            return ""
        value = self.temperatures[int(np.clip(index, 0, len(self) - 1))]
        return f"{value:g} K"


@lru_cache(maxsize=None)
def probability_scale(probability: float = DEFAULT_PROBABILITY) -> float:
    """How far out to draw the ellipsoid to enclose ``probability``.

    The displacement is a trivariate Gaussian with covariance ``U``, so
    ``u^T U^-1 u`` follows chi-squared with three degrees of freedom and the
    surface enclosing probability ``p`` sits at ``sqrt(chi2.ppf(p, 3))``. The
    crystallographic 50% gives 1.5382 — the number ORTEP and VESTA use.

    Cached because the answer depends on nothing but ``probability``, while the
    callers ask for it per *atom*: :func:`ellipsoid_axes` is called in a loop
    over every atom that gets an ellipsoid, and each call was importing
    ``scipy.stats`` and re-solving the quantile. That was 61 ms of the 87 ms a
    1728-atom ellipsoid field cost to build — 70% of it, for one number. The
    probability only ever comes from a settings field with a handful of
    values, so the cache stays tiny.

    An invalid probability raises rather than returning, and ``lru_cache`` does
    not memoise exceptions, so a bad value is rejected every time it is passed.
    """
    from scipy.stats import chi2

    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between 0 and 1")
    return float(np.sqrt(chi2.ppf(probability, 3)))


def ellipsoid_axes(
    u_cart: np.ndarray, probability: float = DEFAULT_PROBABILITY
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """``(radii, axes, positive_definite)`` for one atom's ellipsoid.

    ``radii`` are the three semi-axis lengths in Angstrom at the requested
    probability; ``axes`` has the corresponding principal directions as **rows**
    (a proper rotation, so it can be used as a drawing transform directly).

    A tensor with a non-positive eigenvalue is not a physical displacement
    distribution — crystallographers call such an atom "NPD", and it happens
    with under-converged sampling as readily as with bad data. Rather than
    raising or drawing an imaginary axis, the offending eigenvalue is clamped
    to zero (a flat ellipsoid, which is what the number is saying) and the
    returned flag is False so a caller can mark it.
    """
    u_cart = np.asarray(u_cart, dtype=float)
    if u_cart.shape != (3, 3):
        raise ValueError(f"expected a (3, 3) tensor, got {u_cart.shape}")

    # eigh needs symmetry; CRYSTAL prints it symmetric, but a tensor that has
    # been converted between bases can pick up rounding asymmetry.
    eigenvalues, eigenvectors = np.linalg.eigh((u_cart + u_cart.T) / 2.0)
    positive_definite = bool(np.all(eigenvalues > 0.0))
    radii = probability_scale(probability) * np.sqrt(np.clip(eigenvalues, 0.0, None))

    axes = eigenvectors.T  # eigh returns eigenvectors as columns
    if np.linalg.det(axes) < 0:  # keep it a rotation, not a reflection
        axes[0] = -axes[0]
    return radii, axes, positive_definite


def ellipsoid_radii(
    u_cart: np.ndarray, probability: float = DEFAULT_PROBABILITY
) -> np.ndarray:
    """``(natom, 3)`` ascending semi-axis lengths for a *stack* of ADP tensors.

    The batched counterpart of :func:`ellipsoid_axes`'s first return value, for
    callers that need every atom's ellipsoid *size* but not its orientation —
    deciding which atoms are big enough to draw, above all. Equivalent to
    ``[ellipsoid_axes(u, probability)[0] for u in u_cart]`` and used in its place
    because that loop is not cheap at scale: it re-solves the chi-squared
    quantile and a 3x3 eigenproblem per atom, and the renderer runs it on every
    animation frame. ``numpy`` diagonalises the whole stack at once, which
    measured ~80x faster at 2000 atoms (89 ms -> 1.1 ms).

    Non-positive eigenvalues are clamped to zero exactly as
    :func:`ellipsoid_axes` does, so an NPD atom reports a flat ellipsoid rather
    than an imaginary axis.
    """
    tensors = np.asarray(u_cart, dtype=float)
    if tensors.ndim != 3 or tensors.shape[1:] != (3, 3):
        raise ValueError(f"expected (natom, 3, 3) tensors, got {tensors.shape}")
    if len(tensors) == 0:
        return np.empty((0, 3), dtype=float)

    # Symmetrise for the same reason ellipsoid_axes does: a tensor converted
    # between bases can pick up rounding asymmetry, and eigvalsh assumes none.
    symmetric = (tensors + np.swapaxes(tensors, -1, -2)) / 2.0
    eigenvalues = np.linalg.eigvalsh(symmetric)  # ascending, per tensor
    return probability_scale(probability) * np.sqrt(np.clip(eigenvalues, 0.0, None))


__all__ = [
    "ADPSet",
    "DEFAULT_PROBABILITY",
    "ellipsoid_axes",
    "ellipsoid_radii",
    "probability_scale",
]
