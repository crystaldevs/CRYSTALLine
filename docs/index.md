# Welcome to the CRYSTALLine home page!

::::{div} crystal-hero

```{image} logo.png
:alt: CRYSTALLine
:width: 420px
:align: center
:class: crystal-logo-light
```

```{image} _static/logo-dark.png
:alt: CRYSTALLine
:width: 420px
:align: center
:class: crystal-logo-dark
```

::::

CRYSTALLine is a program to build, display and manipulate the structures used by
the [CRYSTAL](https://www.crystal.unito.it/) quantum-chemistry code, to animate
the vibrational modes computed from them, and to plot the properties that a
calculation returns. It runs on Linux, macOS and Windows, and is built on the
[CRYSTALClear](https://github.com/crystaldevs/CRYSTALClear) library.

It follows in the tradition of
[MOLDRAW](https://www.moldraw.unito.it/), written by Prof. Piero Ugliengo, which
served the CRYSTAL community for over twenty years: a program on your own
machine, for looking at a structure and changing it, rather than a script to be
written each time.

CRYSTALLine is free software under the
[GNU General Public License v3](https://github.com/crystaldevs/CRYSTALLine/blob/main/LICENSE).

:::{div} crystal-shot
![The main window](screen1.png)
:::

:::{div} crystal-caption
The main window: the crystallography of the loaded structure on the left, the
interactive view in the centre, the vibrational modes on the right.
:::

## Main features

1. **Structure display.** Atoms, bonds, hydrogen bonds, coordination polyhedra
   and the unit cell, drawn in an interactive 3D view. Every element of the
   drawing can be configured from the Display panel.
2. **Crystallography.** The space group of a crystal, the layer group of a slab
   or the point group of a molecule, with the lattice parameters, cell volume,
   density and formula; a polymer is reported by its repeat length, since no
   library names rod groups. These are recomputed whenever the structure is
   edited.
3. **Structure manipulation.** Atoms can be selected, moved, added, deleted,
   duplicated and changed in element; cells can be converted between primitive
   and crystallographic settings, expanded into supercells and edited through
   their lattice parameters. All operations can be undone.
4. **Vibrational modes.** Modes are read from the CRYSTAL output and animated in
   place. Modes computed away from Γ by a dispersion calculation are animated
   as travelling waves, and the cell can be tiled over one period. Animations
   are exported as GIF, as a sequence of frames, or as MP4, MOV and WebM.
5. **Input preparation.** Input decks for both CRYSTAL (`.d12`) and PROPERTIES 
   (`.d3`) are written from the structure on screen, with a
   preview of the file before it is saved. The band path can be defined by
   clicking the points of the Brillouin zone.
6. **Property plots.** Electronic band structures and densities of states, IR
   and Raman spectra (harmonic and anharmonic), elastic properties, equations of
   state, phonon bands and densities of states, and simulated XRD patterns.
7. **Densities in space.** Crystalline orbitals, the charge and spin densities
   and the electrostatic potential are drawn over the structure itself, as
   isosurfaces or as slices through the cell, on the grid CRYSTAL computed them
   on.

## Where to start

[Installation](install.md) describes how to install the program and what to do
if it does not start. [First steps](first-steps.md) describes the main window
and how to open a file. The remaining pages of the User guide describe each part
of the program, and the Reference pages list the menus, the file formats and the
known problems.

## Acknowledgments

CRYSTALLine is built on [CRYSTALClear](https://github.com/crystaldevs/CRYSTALClear),
which reads the CRYSTAL output files and draws the property plots, and it uses
[ASE](https://wiki.fysik.dtu.dk/ase/), [pymatgen](https://pymatgen.org/),
[spglib](https://spglib.readthedocs.io/) and [PyVista](https://pyvista.org/).

It was developed with the assistance of
[Claude](https://www.anthropic.com/claude) (Anthropic), using
[Claude Code](https://www.claude.com/product/claude-code).

```{toctree}
:hidden:
:caption: Getting started

install
first-steps
```

```{toctree}
:hidden:
:caption: User guide

viewer
editing
symmetry
phonons
inputs
band-paths
plots
```

```{toctree}
:hidden:
:caption: Reference

shortcuts
formats
troubleshooting
```
