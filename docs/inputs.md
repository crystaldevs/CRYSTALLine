# CRYSTAL input builder

Two decks can be written from the structure on screen:

- **File → Build CRYSTAL input (.d12)…** — the calculation.
- **File → Build properties input (.d3)…** — the analysis of the wave function
  that follows it.

Both show the text of the file as it will be written.

:::{div} crystal-shot
![The CRYSTAL input builder](dialog-input.png)
:::

## The geometry block

The geometry is derived from the structure:

| Structure | Keyword |
| --- | --- |
| crystal | `CRYSTAL` — space group and asymmetric unit |
| slab | `SLAB` — [layer group](symmetry.md#slabs) and asymmetric unit |
| polymer | `POLYMER` |
| molecule | `MOLECULE` |

Only the lattice parameters left free by the group are written — a hexagonal
layer group requires *a* alone — and only the symmetry-inequivalent atoms. When
the symmetry cannot be determined, the deck is written in group 1 with all
atoms listed.

A [reduced symmetry](symmetry.md#reducing-the-symmetry), if one has been
chosen, is used here.

## The calculation

The method (Hartree–Fock, or DFT with a single keyword or with separate
exchange and correlation functionals), the basis set and the SCF parameters are
selected in the dialog, together with the type of calculation: single point,
geometry optimisation, frequencies with infrared and Raman intensities, phonon
dispersion, quasi-harmonic approximation, equation of state, elastic constants,
coupled-perturbed Hartree–Fock, anharmonic calculations and spin–orbit
coupling.

Phonon band structures belong to this deck (`BANDS` within `FREQCALC`) and use
the path editor described in [Band paths](band-paths.md).

## The properties deck

:::{div} crystal-shot
![The properties input builder](dialog-properties.png)
:::

The dialog writes `NEWK`, `BAND`, `DOSS`, `COOP` and `COHP`, `ECHG`, `PPAN`,
`ORBITALS`, `EMD` and `XRD`.

Two points are handled by the program. The keywords are written in the order
required by the manual, `BAND` – `NEWK` – `DOSS`, since `NEWK` placed before
`BAND` stops the run. And the band path is written as integers over a shrinking
factor: a factor which would leave a coordinate fractional is refused rather
than rounded, so that the path contains the points intended.

## Running the calculation

The deck is saved and run as usual. CRYSTALLine does not run CRYSTAL; it
prepares the input and reads the output.
