# File formats

## Files read

| Format | Contents read |
| --- | --- |
| CRYSTAL `.out` | structure and symmetry, the parameters and results of the calculation, vibrational modes, elastic tensor, equation of state and anharmonic data, according to what the run contains |
| CRYSTAL `.gui` / `.f34` | structure and symmetry |
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
| `DENS_CUBE.DAT`, `SPIN_CUBE.DAT`, `POT_CUBE.DAT`, `*.cube` | charge density, spin density and electrostatic potential on a 3D grid |
| `fort.31`, `*.f31` | the same grids in the DLV format |
| Molden files | crystalline orbitals |
| `.d3` decks | the band path's own point names |

These files are searched for in the folder of the output that is open, which is
where CRYSTAL leaves them, and are recognised by their extension and their
content — never by the rest of their name, which is rewritten freely — so that the appropriate file is proposed when a folder contains
several runs. A
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
