# First steps

## Opening a file

**File → Open…** ({kbd}`Ctrl+O`) reads a CRYSTAL output (`.out`), a geometry
file (`.gui`, `.34`) or a crystallographic information file (`.cif`). A file
dropped on the window is opened in the same way.

If the output contains a vibrational calculation, the modes are read with it.
If it does not, the geometry alone is shown and the remaining functions are
unaffected.

## The main window

![The main window](screen1.png)

The window is divided into dockable panels:

| Position | Panel | Contents |
| --- | --- | --- |
| Left | **Info** | space group or layer group, lattice parameters, density, and a summary of the calculation |
| Left | **Display** | the settings of the 3D view |
| Left | **Geometry** | distances, angles, dihedrals and planes |
| Right | **Phonons** | the list of vibrational modes and the animation controls |
| Bottom | **Plots** | the property plots, in tabs |

Panels can be moved, stacked, resized or closed. **View → Panels** restores a
single panel and **View → Restore all panels** restores the original
arrangement.

## Controlling the view

The structure is rotated by dragging, zoomed with the scroll wheel and
translated by dragging with the middle button (or with {kbd}`Shift` held down).

The toolbar contains three groups: **VIEW**, which orients the structure along
the **a**, **b** or **c** axis and returns it to a view of the whole;
**ROTATE**, which turns it by a fixed step, 15° unless changed in the adjacent
box; and **CONV. CELL**, which switches between the crystallographic and the
primitive cell.

**View → Appearance** selects the light or dark interface, or follows the
system setting.

## Next

[The 3D view](viewer.md) describes the Display panel, [Editing
structures](editing.md) the modification of a structure, [Symmetry](symmetry.md)
the analysis and reduction of its symmetry, [Phonons](phonons.md) the
vibrational modes, [CRYSTAL input builder](inputs.md) the preparation of new
calculations and [Property plots](plots.md) the plotting of the results.
