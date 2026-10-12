# Overview

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

CRYSTALLine builds, displays and edits structures for the
[CRYSTAL](https://www.crystal.unito.it/) quantum-chemistry code, animates
vibrational modes computed from them, and plots properties a calculation
returns. It runs on Linux, macOS and Windows, and is built on
[CRYSTALClear](https://github.com/crystaldevs/CRYSTALClear).

It follows [MOLDRAW](https://www.moldraw.unito.it/), written by Prof. Piero
Ugliengo, which served the CRYSTAL community for over twenty years: a program
on your own machine for looking at a structure and changing it, rather than a
script written each time.

CRYSTALLine is free software under the
[GNU General Public License v3](https://github.com/crystaldevs/CRYSTALLine/blob/main/LICENSE).

:::{div} crystal-shot
![The main window](screen1.png)
:::

:::{div} crystal-caption
Main window: crystallography on the left, an interactive view in the centre,
vibrational modes on the right.
:::

## Main features

1. **Crystals, slabs, polymers and molecules.** A space group for a crystal, a
   layer group for a slab, a point group for a molecule, with lattice
   parameters, volume, density and formula beside it, all recomputed when you
   change anything.
2. **Structures you edit by hand.** Drag an atom across the view and bonds,
   polyhedra and hydrogen bonds follow it. Add, delete, duplicate or
   change atoms; expand a supercell; swap between primitive and
   crystallographic settings; retype lattice parameters. Undo reaches back
   through all of it.
3. **Animated vibrational modes.** Modes are read from a CRYSTAL output and
   animated on your structure. Away from Γ they travel as waves: tile a cell
   over one period and watch one pass through it. Save the animation as a GIF,
   as numbered frames, or as MP4, MOV or WebM.
4. **Inputs written from the structure on screen.** CRYSTAL (`.d12`) and
   PROPERTIES (`.d3`) decks, shown in full before you save them. Set a band path by
   clicking corners of a Brillouin zone instead of looking coordinates up.
5. **Plots of the results.** Electronic band structures and densities of
   states, IR and Raman spectra (harmonic and anharmonic), elastic surfaces and
   sections through them, equations of state, phonon bands and densities of
   states, simulated XRD patterns.
6. **Densities drawn over the atoms.** Crystalline orbitals, charge and spin
   densities, electrostatic potential, as isosurfaces or sliced open along a
   lattice plane, on whatever grid CRYSTAL computed them on.
7. **A view you control.** Atoms, bonds, hydrogen bonds, coordination
   polyhedra and cell edges are each switched and styled from the Display
   panel.

## Not supported yet

**Symmetry of a polymer (1D).** A polymer is read, drawn, edited and given a
band path like any other structure, but its symmetry is a rod group, and spglib
names space groups and layer groups only. A polymer is therefore reported by its
repeat length, its symmetry cannot be lowered, and its decks are written in rod
group 1 with every atom listed.

**TOPOND.** CRYSTAL's [TOPOND](https://www.crystal.unito.it/topond.html) module
writes files of its own for the topological analysis of the electron density,
and none of them is read here. [TopIso3D Viewer](http://www.topiso3d.ufpb.br/)
is written for that output and draws it as three-dimensional maps. Densities written by `ECH3` and `POT3` are
another matter and are drawn here; see
[Property plots](plots.md#electron-density-and-electrostatic-potential).

## Where to start

[Installation](install.md) covers installing CRYSTALLine and what to do if it
will not start. [First steps](first-steps.md) walks through the window and a
file opened in it. The other **User guide** pages take each part of the
program in turn; the **Reference** pages list menus, file formats and known
problems.

## Acknowledgments

CRYSTALLine is built on [CRYSTALClear](https://github.com/crystaldevs/CRYSTALClear),
which reads the CRYSTAL output files and draws the property plots, and it uses
[ASE](https://ase-lib.org/), [pymatgen](https://pymatgen.org/),
[spglib](https://spglib.readthedocs.io/en/stable/) and
[PyVista](https://pyvista.org/).

It was developed with the assistance of
[Claude](https://claude.com/product/overview) (Anthropic), using
[Claude Code](https://claude.com/product/claude-code).

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
release-notes
```
