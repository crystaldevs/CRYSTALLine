# First steps

## Opening a file

**File → Open…** ({kbd}`Ctrl+O`) reads a CRYSTAL output (`.out`), a geometry
file (`.gui`, `.f34`) or a crystallographic information file (`.cif`). A file
dropped on the window is opened in the same way.

No output to hand? [Installation](install.md#an-example-to-open) offers a small
MgO calculation to open.

If the output contains a vibrational calculation, phonon modes are read with it.
If it does not, the geometry alone is shown and the remaining functions are
unaffected.

## Working in the calculation's folder

CRYSTAL writes more than the output file. Several external units might be 
generated depending on the specific task. CRYSTALLine looks for these beside 
the output that is open, and works better when they are all in one folder:

- The **Electronic bands & DOS** dialog lists the band and density-of-states
  files it finds there, recognises them by their contents, and pairs a band 
  structure with a density of states computed from the same SCF.
- The file dialogs of the other plots open in that folder.
- Anharmonic wavefunctions are found without being asked for.

So keeping a calculation and everything it produced in one directory, rather
than moving the output somewhere on its own, is what allows the program to
offer the right file instead of an empty dialog.

## The main window

![The main window](screen1.png)

The window is divided into panels:

| Position | Panel | Contents |
| --- | --- | --- |
| Left | **Info** | space group or layer group, lattice parameters, density, and a summary of the calculation |
| Left | **Display** | the settings of the 3D view |
| Left | **Geometry** | distances, angles and dihedrals, lattice planes, and the atom tools |
| Right | **Phonons** | the list of vibrational modes and the animation controls |
| Bottom | **Plots** | the property plots, in tabs |

**View → Panels** lists every panel with a tick beside it: clearing the tick
puts that panel away and setting it again brings it back, so the window can be
narrowed to the panels a particular job needs. **View → Restore all panels**
brings back everything at once.

## Controlling the view

The structure is rotated by dragging, zoomed with the scroll wheel and
translated by dragging with the middle button (or with {kbd}`Shift` held down).

The toolbar contains three groups: **VIEW**, which orients the structure along
the **a**, **b** or **c** axis or, as in VESTA, along **a\***, **b\*** or
**c\***, the normal to the bc, ca or ab plane, which shows that face of the
cell square-on, and returns it to a view of the whole; **ROTATE**, which turns
it by a fixed step, 15° unless changed in the adjacent box; and **CONV. CELL**,
which switches between the crystallographic and the primitive cell. In an
orthogonal cell a and a\* coincide; in a monoclinic or triclinic one, looking
down a shows the bc face foreshortened and looking down a\* shows it at its true
shape. For a slab, **c\*** looks straight down onto the layer.

**View → Appearance** selects the light or dark interface, or follows the
system setting.

## Next

[The 3D view](viewer.md) describes the Display panel, [Editing
structures](editing.md) the modification of a structure, [Symmetry](symmetry.md)
the analysis and reduction of its symmetry, [Phonons](phonons.md) the
vibrational modes, [CRYSTAL input builder](inputs.md) the preparation of new
calculations and [Property plots](plots.md) the plotting of the results.
