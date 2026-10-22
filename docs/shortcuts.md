# Menus and shortcuts

:::{note}
The shortcuts are written with {kbd}`Ctrl`, as they are on Linux and Windows. On macOS press {kbd}`⌘` wherever {kbd}`Ctrl` is written: {kbd}`Ctrl+O`
here means {kbd}`⌘O`.
:::

## Mouse

| | |
| --- | --- |
| Drag | rotate |
| Scroll | zoom |
| Middle-drag, or {kbd}`Shift` and drag | pan |
| Click an atom | select it (in editing mode) |
| {kbd}`Ctrl` and click | add to the selection |
| Drag a selected atom | move it, with its periodic images |
| Click empty space | clear the selection |

## File

| | |
| --- | --- |
| Open… | {kbd}`Ctrl+O`, each file in a tab of its own |
| Open recent | |
| Close tab | {kbd}`Ctrl+W` |
| Import atoms into structure… | |
| Save structure as .gui… | |
| Save structure as .cif… | |
| Build CRYSTAL input (.d12)… | |
| Build properties input (.d3)… | |
| Export image… | |
| Export phonon animation… | |

## Cell

| | |
| --- | --- |
| Crystallographic cell / Primitive cell | which cell is shown |
| Lattice parameters… | |
| Supercell… | |
| Complete coordination at cell boundary | toggle |
| Point symmetry analysis | elements, and **Reduce symmetry…** |
| Brillouin zone… | the zone, read-only |

## Edit

| | |
| --- | --- |
| Undo | {kbd}`Ctrl+Z` |
| Redo | {kbd}`Ctrl+Shift+Z` or {kbd}`Ctrl+Y` |
| Editing mode | {kbd}`Ctrl+E` |
| Select all | {kbd}`Ctrl+A` |
| Clear selection / Invert selection | |
| Delete selected | {kbd}`Del` |
| Duplicate selected | {kbd}`Ctrl+D` |
| Translate selected… | |
| Rotate selected… | about a lattice vector, a cartesian axis, two atoms or a typed direction |
| Set element of selected… | |
| Restore geometry | {kbd}`Ctrl+R` |
| Nudge the selection | {kbd}`Arrow keys` |

The commands which modify atoms act on the current selection and are enabled
only in editing mode.

## View

| | |
| --- | --- |
| Appearance | Match system, Light, Dark |
| Display settings | the Display panel |
| Panels | show or hide each panel |
| Restore all panels | put every panel back |
| Next tab / Previous tab | {kbd}`Ctrl+Tab` / {kbd}`Ctrl+Shift+Tab` |
| Along a/b/c axis, Along a\*/b\*/c\* | the same as the **VIEW** chips |

On macOS the tabs are stepped through with {kbd}`⌃Tab` and {kbd}`⌃⇧Tab`, as in Safari, or with {kbd}`⌘⇧]` and {kbd}`⌘⇧[`.

## Plot

Elastic properties ▸ · Elastic properties (2D) ▸ · Equation of state ·
Simulated XRD pattern… · Electronic bands & DOS… · Phonon bands & DOS… ·
Vibrational spectra… · VCI states… · Anharmonic scan… · Anharmonic PES… ·
Crystalline orbitals… · Clear orbital · Electron density & potential… · Clear
field · Plot font…

The entries are enabled when the output contains the corresponding data; see
[Property plots](plots.md).

## The toolbar

| Group | What |
| --- | --- |
| **VIEW** | look down **a**, **b** or **c**, or down **a\***, **b\*** or **c\*** (the bc, ca or ab face square-on); fit everything back in view |
| **ROTATE** | turn the scene by a fixed step (15° by default, set beside it) |
| **CONV. CELL** | crystallographic cell on, primitive cell off |

The Brillouin-zone window has the same groups, for **k**<sub>x</sub>,
**k**<sub>y</sub> and **k**<sub>z</sub>.
