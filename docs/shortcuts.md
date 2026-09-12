# Menus and shortcuts

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
| Open… | {kbd}`Ctrl+O` |
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
| Complete molecules at cell boundary | toggle |
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
| Panels | show one panel |
| Restore all panels | put every panel back |

## Plot

Electronic bands & DOS… · Vibrational spectra… · VCI states… · Anharmonic
scan… · Anharmonic PES… · elastic properties · equation of state · phonon band
structure… · phonon density of states… · simulated XRD… · Crystalline
orbitals… · Clear orbital · Plot font…

The entries are enabled when the output contains the corresponding data; see
[Property plots](plots.md).

## The toolbar

| Group | What |
| --- | --- |
| **VIEW** | look down **a**, **b** or **c**; fit everything back in view |
| **ROTATE** | turn the scene by a fixed step (15° by default, set beside it) |
| **CONV. CELL** | crystallographic cell on, primitive cell off |

The Brillouin-zone window has the same groups, for **k**<sub>x</sub>,
**k**<sub>y</sub> and **k**<sub>z</sub>.
