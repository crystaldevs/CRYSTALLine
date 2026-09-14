# Property plots

The entries of the **Plot** menu draw the properties computed by CRYSTAL. The
figures are produced by [CRYSTALClear](https://github.com/crystaldevs/CRYSTALClear)
and appear in the **Plots** panel at the bottom of the window, one tab per
figure.

Some entries use the output file already open; the others read the data files
written by a properties run, and their names end with `…`. Those files are
looked for, and asked for, in the folder of the open output — see [working in
the calculation's folder](first-steps.md#working-in-the-calculations-folder).
**Plot → Plot font…** sets the font of the figures drawn afterwards.

## Electronic band structures and densities of states

**Plot → Electronic bands & DOS…** draws a band structure, a density of states,
or the two side by side.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item}
```{image} dialog-electronic.png
:alt: The Electronic bands and DOS dialog
:class: crystal-framed
```
:::

:::{grid-item}
```{image} plot-bands-dos.png
:alt: Band structure and density of states of silicon
:class: crystal-framed
```
+++
Silicon, along Γ-X-W-L-Γ.
:::

::::

**The files.** `BAND.DAT`, `DOSS.DAT`, `*.BAND`, `*.DOSS`, `fort.25` and
`*.f25` are read. Those in the folder of the open output are listed, recognised
by their contents rather than by their names, so that a charge-density map or a
COOP file is not offered as a band structure. When a folder holds several runs,
the band file and the density of states are paired by their Fermi energy, which
identifies the SCF they come from; a disagreement is reported.

**The energies.** CRYSTAL writes them relative to the Fermi level. They can be
plotted that way or as absolute energies, in which case the Fermi energy is
added back and the line marking it moves with the data. Units are eV or
Hartree. The window is taken from the contents of the file — from the first
wide gap below the Fermi level, which separates the valence bands from the core
levels, to as far above — and **Suggested window** restores it.

**The rest.** The projections of the density of states, with their names and
colours; the treatment of the spin-down component; the labels of the path; and
the colours, styles and widths of the lines.

## Vibrational spectra

**Plot → Vibrational spectra…** lists the curves contained in the output:
infrared, Raman with its polarisations, and their anharmonic (VSCF, VCI)
counterparts where present.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item}
```{image} dialog-spectra.png
:alt: The vibrational spectra dialog
:class: crystal-framed
```
:::

:::{grid-item}
```{image} plot-spectrum.png
:alt: Harmonic and anharmonic infrared spectra
:class: crystal-framed
```
+++
Brucite: the harmonic spectrum and its VCI counterpart, which shows the
anharmonic shift.
:::

::::

Several curves may be selected at once, which is the usual case: the
single-crystal Raman components on the same axes show the anisotropy, and a
harmonic curve under its anharmonic counterpart shows the shift. The broadening
is chosen in the same dialog — pseudo-Voigt, Lorentzian, Gaussian or a stick
spectrum — and only the widths used by the chosen lineshape are enabled.

## Anharmonic calculations

- **VCI states…** — the states of a VCI calculation, as a heatmap or a Sankey
  diagram.
- **Anharmonic scan…** — the scanned potential.
- **Anharmonic PES…** — one- and two-dimensional cuts of the potential energy
  surface, with the wavefunctions and probability densities of a double well.

## Properties read from the output

These entries use the open output file and are enabled when it contains the
data they need.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item}

- **Elastic properties** — Young's modulus, linear compressibility, shear
  modulus and Poisson's ratio, as three-dimensional surfaces and as sections
  through them.
- **Equation of state.**
- **Phonon band structures and densities of states**, from the data files of a
  dispersion calculation.
- **Simulated XRD patterns.**

:::

:::{grid-item}
```{image} plot-elastic.png
:alt: Young's modulus surface of coesite
:class: crystal-framed
```
+++
The directional Young's modulus of coesite.
:::

::::

## Crystalline orbitals

**Plot → Crystalline orbitals…** reads the files written by the `ORBITALS`
keyword and draws the orbital in the 3D view rather than in the plot panel;
**Clear orbital** removes it.

```{image} screen2.png
:alt: A crystalline orbital across a graphite supercell
:class: crystal-framed
:width: 70%
:align: center
```

## Electron density and electrostatic potential

**Plot → Electron density & potential…** draws the 3D grids written by a
properties run with `ECH3` or `POT3`. Like an orbital, the field is drawn in
the 3D view rather than in the plot panel; **Clear field** removes it.

**The files.** `DENS_CUBE.DAT` (charge density), `SPIN_CUBE.DAT` (spin
density), `POT_CUBE.DAT` (electrostatic potential) and `fort.31`, which holds
the same numbers in the DLV format. They are picked out by extension and
recognised by what they contain, never by the rest of their name — CRYSTAL
writes the property on a cube's first line, and that is what is read, so renamed
files such as `mgo_pot3.cube` are recognised all the same. They are looked for in the folder of the open output —
see [working in the calculation's
folder](first-steps.md#working-in-the-calculations-folder). A run that wrote
both a density and a potential opens with the density chosen and the potential
ready as the second field.

**What is drawn.** CRYSTAL samples the property over the primitive cell, with
the grid running along the lattice vectors — so in a hexagonal or a triclinic
cell the grid is not a box, and it is drawn as the sheared grid it is, with no
resampling in between. Values are left in the atomic units the files are
written in: electrons per bohr³ for a density, hartree per electron for a
potential.

**Isosurface** draws a surface of constant value. The slider opens on the level
that wraps the densest fifth of the cell, which comes out of the drawn atoms
whatever the system is: the hexagonal shells of beryllium, the oxide ion
dwarfing its cation in MgO, silicon's bonds, a molecular envelope around each
urea. A field that takes both signs — a spin density, a potential, a difference
— is drawn as two surfaces, one per sign.

**Clip to the cell** cuts the surfaces at the faces of the outlined cell;
left off, they are drawn whole around every atom on screen.

The field is lattice-periodic, so it is drawn in every cell the atoms on screen
occupy, and only the parts of it that an atom lies against are kept. A crystal's
density fills space, but a view holds only the atoms that were asked for, and
density around the others reads as the field being in the wrong place.

**Lattice plane (hkl)** draws the field on a plane named by its Miller indices,
which refer to the conventional cell — MgO's (001) is the familiar layer of
alternating Mg and O. **Position** moves the plane along its normal as a fraction
of the interplanar spacing d(hkl): 0 passes through the cell origin, and the
dialog shows d and the plane's distance from the origin. The plane is laid out
as a rectangle over everything on screen and coloured by the field, on a
logarithmic scale by default: a density runs over four orders of magnitude
between a void and a nucleus, and on a linear scale everything but the cores is
one colour. The colour range is set by the bulk of the plane rather than by the
few points on the nuclei.

With **Cut away the crystal in front of the plane**, the atoms, bonds and cell
edges on the near side are not drawn, and the camera turns to face the plane, so
that it is not buried inside the crystal. Atoms lying in the plane are marked by
small dots in their element colours — every atom of the crystal on the plane, not
only the ones drawn. The cutaway only changes what is drawn: the structure
itself, and what can be selected or measured, are untouched. To cut away the
other side, change the sign of the indices.

**Show a colour bar** keys the colour map with a bar at the right of the view,
headed with the quantity and its unit — and with log₁₀ when a plane is coloured
on the logarithmic scale, since its numbers are then exponents. It is offered for
a plane and for a surface coloured by a second field; a plain surface has a
single colour and nothing to key.

**A second field** gives the two pictures that need two grids:

- *Colour the surface by a second field* paints the electrostatic potential
  onto the density surface — the surface says where the electrons end, the
  colour what a charge would feel there.
- *Subtract a second field* draws the difference of two grids: a deformation
  density, or one calculation against another. For a deformation density, run
  the properties deck twice — once as it is, once with **PATO** ticked — and
  subtract the second density from the first. Both runs write
  `DENS_CUBE.DAT`, so rename one of them in between; the property each cube
  holds is read from its content, not its name. The
  difference takes both signs, so it is drawn with two surfaces.

  Both need the two runs to share a grid — the same cell, the same number of
  points and the same extents — and the dialog says so when they do not.

The field follows the atoms on screen: it is drawn around every atom, in
whichever periodic image the atom lies, and pieces of it around atoms that are
not drawn are left out. To see more of it, build a supercell.

The grids come from a `.d3` deck with `ECH3` or `POT3`, which the [properties
input builder](inputs.md#charge-density-and-electrostatic-potential-on-a-grid)
writes.
