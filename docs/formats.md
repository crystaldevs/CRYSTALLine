# File formats

## Files read

| Format | Contents read |
| --- | --- |
| CRYSTAL `.out` | structure and symmetry, the parameters and results of the calculation, vibrational modes, elastic tensor, equation of state and anharmonic data, according to what the run contains |
| CRYSTAL `.gui` / `.34` | structure and symmetry |
| `.cif` | structure |

These files may also be opened by dropping them on the window.

## Import into the current structure

**File → Import atoms into structure…** reads `.xyz`, `.pdb` and `.cif` files
and adds their atoms to the structure already loaded, without replacing it. A
file dropped on a window which already contains a structure is imported in the
same way.

## Data files read for the plots

The files a PROPERTIES (`.d3`) run leaves beside the output:

| | |
| --- | --- |
| `BAND.DAT`, `*.BAND` | electronic band structures |
| `DOSS.DAT`, `*.DOSS` | densities of states |
| `fort.25`, `*.f25` | either of those, and charge-density maps |
| `.d3` decks | the band path's own point names |

These files are recognised by their first line rather than by their name, so
that the appropriate file is proposed when a folder contains several runs. A
`fort.25` file does not record the shrinking factor of the tick labels of a band
structure; the corners of the path then keep their coordinates unless the deck
which produced them is present.

## Files written

| | |
| --- | --- |
| `.gui`, `.cif` | the structure, symmetry-reduced |
| `.d12` | a CRYSTAL calculation |
| `.d3` | a properties run |
| PNG, JPEG, TIFF, SVG, PDF, EPS | the 3D view or the Brillouin zone |
| GIF, numbered frames | phonon animations |
| MP4, MOV, WebM | phonon animations, with `CRYSTALLine[video]` |
