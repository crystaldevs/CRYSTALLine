# Editing structures

Editing is enabled by **Edit → Editing mode** ({kbd}`Ctrl+E`). Every operation
can be undone ({kbd}`Ctrl+Z`; redo with {kbd}`Ctrl+Shift+Z` or {kbd}`Ctrl+Y`) —
a supercell and a change of lattice parameters included, so edits made before
one are still there underneath it. Switching between the crystallographic and
primitive settings is not an undo step: it is a way of looking at a crystal
rather than a change to it. It does discard edits made to the cell on screen,
and *that* is undoable. **Edit → Restore geometry** ({kbd}`Ctrl+R`) returns a
structure to the state in which it was read.

## Selection

An atom is selected by clicking it and added to the selection with
{kbd}`Ctrl` held down; clicking the background clears the selection.
**Edit → Select all** ({kbd}`Ctrl+A`), **Clear selection** and **Invert
selection** act on the whole structure.

## Moving atoms

A selected atom is moved by dragging it; its periodic images move with it, and
a selection of several atoms moves as a unit. The arrow keys displace the
selection by a small step, and **Edit → Translate selected…** applies a given
displacement.

## Adding and removing atoms

**Add atom** in the Structure panel adds one atom of the element chosen beside
it. It appears in the middle of the structure — the middle of the cell along
each direction that repeats, and among the atoms along each direction that does
not, so that a new atom on a slab or a polymer lands on it rather than out in
the vacuum — and is selected, ready to be dragged or given exact coordinates.

- **Delete selected** ({kbd}`Del`)
- **Duplicate selected** ({kbd}`Ctrl+D`)
- **Set element of selected…**, which opens a periodic table

## The cell

The **Cell** menu acts on the lattice rather than on the atoms:

- **Crystallographic cell** and **Primitive cell** select which of the two is
  displayed.
- **Lattice parameters…** edits *a*, *b*, *c*, α, β and γ.
- **Supercell…** expands the cell.
- **Complete molecules at cell boundary** completes the molecules cut by the
  edges of the cell. This affects the drawing; the deck written for CRYSTAL
  contains the cell itself.
- **Point symmetry analysis** and **Brillouin zone…** are described in
  [Symmetry](symmetry.md) and [Band paths](band-paths.md).

The Info panel is recomputed after each operation, so the space group shown is
that of the current structure.

## Measurements

The **Geometry** panel is in three sections — **Measure**, **Lattice planes**
and **Atoms** — each folded or unfolded with a click on its title. The ones left
open stay open in every tab and the next time the program is started.

**Measure** measures the current selection:

| Atoms selected | Quantity                 |
| -------------- | ------------------------ |
| 1              | the position of the atom |
| 2              | a distance               |
| 3              | an angle                 |
| 4              | a dihedral angle         |

Measurements remain drawn in the view and can be coloured individually or by
type. The **Thickness** slider sets how thick the lines of distances, angles and
dihedrals are drawn, in Å: those selected in the list, all of them when none is
selected, and the ones measured next. A plane through the selected atoms is
fitted under **Lattice planes**.

## Lattice planes

A crystallographic plane is drawn from its Miller indices, with no atoms
selected, in the **Lattice planes** section of the Geometry panel. The
indices are quoted in the conventional cell; hexagonal and trigonal crystals 
are written with four indices (*h k i l*), *i* = −(*h* + *k*) following from 
the other two.

The plane is placed either at a **position** along its normal, in units of the
interplanar spacing *d*(hkl) from the plane through the origin — from −1 to 1:
0 is the plane through the origin, 1 and −1 its neighbours on either side, 0.5
lies halfway to the next one — or **through the selected atom**. Negative
positions matter for a plane with a negative index, such as (1 −1 0) or
(−1 0 0): the cell lies partly or wholly on the negative side of the plane
through the origin, and a plane is drawn only where it cuts the cell. **Whole family** draws every plane of the family across the cell on
screen, *d*(hkl) apart. Each plane is drawn where it cuts the displayed cell
(for a slab, the layer and 1 Å either side of it) and is listed with *d*(hkl)
and the number of atoms lying on it, within 0.15 Å — the same test that marks
atoms on a density slice; **Select atoms** selects them, ready to be measured.
The **Opacity** slider sets how see-through the
planes selected in the list are — every plane when none is selected — and the
opacity new planes are drawn with; at 0 only the outline of a plane is drawn.
Planes are kept through a change of view or supercell and cleared when another
file is opened.

**Fit to selected atoms** goes the other way, from atoms to a plane: a fitted plane
is coloured, made see-through, used to select atoms and removed like any other, but 
has no family. 
