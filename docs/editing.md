# Editing structures

Editing is enabled by **Edit → Editing mode** ({kbd}`Ctrl+E`). Every operation
can be undone ({kbd}`Ctrl+Z`; redo with {kbd}`Ctrl+Shift+Z` or {kbd}`Ctrl+Y`),
a supercell and a change of lattice parameters included, and both keep the
edits made before them. **Edit → Restore geometry** ({kbd}`Ctrl+R`) returns a
structure to the state in which it was read.

Switching between the crystallographic and primitive settings is a way of
looking at a crystal rather than a change to it, so on its own it is not an undo
step. Edits are carried into the other cell, and then it is one. An edit to one
atom of the primitive cell reaches every copy of it in the crystallographic
cell. The other way, an edit that treats differently atoms the centring makes
copies of (one of two centred chains turned) leaves no smaller primitive cell to
show it in, and you are asked before it is dropped, which is undoable.

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

**Edit → Rotate selected…** (also **Rotate…** in the Geometry panel) turns the
selection about an axis. The axis can lie along a lattice vector of the cell on
screen, along *x*, *y* or *z*, through two selected atoms (a bond, or the length
of a chain), or along a typed direction; it passes through the centroid of the
selection, one of its atoms, the origin or a typed point. The view follows the
angle as it is changed, and nothing is applied until **OK**; the rotation is
then a single undo step. Positive angles turn anticlockwise looking down the
axis. Periodic images turn with their atoms, so a chain selected across a face
of the cell turns as one piece and the crystal stays periodic.

To turn one chain of a polymer crystal about its own length, select the chain
and rotate it **along** the chain's lattice vector, **through** the centroid of
the selection.

## Adding and removing atoms

**Add atom** in the Structure panel adds one atom of the element chosen beside
it. It appears in the middle of the structure (the middle of the cell along
each direction that repeats, and among the atoms along each direction that does
not, so that a new atom on a slab or a polymer lands on it rather than out in
the vacuum) and is selected, ready to be dragged or given exact coordinates.

- **Delete selected** ({kbd}`Del`)
- **Duplicate selected** ({kbd}`Ctrl+D`)
- **Set element of selected…**, which opens a periodic table

## The cell

The **Cell** menu acts on the lattice rather than on the atoms:

- **Crystallographic cell** and **Primitive cell** select which of the two is
  displayed.
- **Lattice parameters…** edits *a*, *b*, *c*, α, β and γ of the cell on
  screen: the crystallographic cell when it is shown, the primitive one
  otherwise. The atoms keep their fractional coordinates, edits included, and
  the crystal keeps facing the way it did. **Restore geometry** returns to the
  cell that was read.
- **Supercell…** expands the cell. Edits are carried into it. An edit made in
  one cell of a supercell only — an atom removed from one of them — is carried
  into a supercell that is a whole multiple of it; for any other size you are
  asked whether to start again from the structure as read.
- **Complete coordination at cell boundary** draws what the cell holds with its
  coordination complete. A molecule is drawn whole wherever it touches the
  cell. A framework (silicon, rock salt) is drawn with the atoms on the faces,
  edges and corners of the cell, and with all the ligands of each cation in it
  (up to eight), so every Na of rock salt has its six Cl. A chain or a layer in
  a crystal (a polymer, brucite) is drawn as a framework is, and each of its
  coordination centres is given all its neighbours, so chains run one bond past
  each face and every cation keeps its full coordination. The atoms added are
  atoms of the structure like any other: counted, selectable, and moved by the
  phonon modes. This affects the drawing; the deck written
  for CRYSTAL contains the cell itself. Edits are kept when it is switched on or
  off.
- **Point symmetry analysis** and **Brillouin zone…** are described in
  [Symmetry](symmetry.md) and [Band paths](band-paths.md).

The Info panel is recomputed after each operation, so the space group shown is
that of the current structure.

## Measurements

The **Geometry** panel is in three sections, **Measure**, **Lattice planes**
and **Atoms**, each folded or unfolded with a click on its title. The ones left
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
selected, in the **Lattice planes** section of the Geometry panel. The indices
are quoted in the conventional cell. Hexagonal and trigonal crystals are
written with four indices (*h k i l*), where *i* = −(*h* + *k*).

The plane is placed either at a **position** along its normal, in units of the
interplanar spacing *d*(hkl), or **through the selected atom**. Position 0 is
the plane through the origin, 1 and −1 its neighbours on either side. Negative
positions are needed for planes with a negative index, such as (1 −1 0), whose
family lies on that side of the origin.

**Whole family** draws every plane of the family across the cell, *d*(hkl)
apart. Each is listed with its *d*(hkl) and the number of atoms lying on it,
within 0.15 Å; **Select the atoms on it** selects those atoms, ready to be
measured. The **Opacity** slider sets how see-through the selected planes are,
or all of them when none is selected.

**Fit to selected atoms** goes the other way, from atoms to a plane. A fitted
plane behaves like any other, but has no family. 
