# Installation

CRYSTALLine is on [PyPI](https://pypi.org/project/CRYSTALLine/). It needs
**Python 3.11 or newer** — the floor comes from pymatgen, not from CRYSTALLine
itself.

It should be installed in an environment of its own rather than in the system
Python. Either a conda environment or a virtual environment may be used; conda
also provides the Python version, whereas a virtual environment uses the
`python` already installed.

::::{tab-set}

:::{tab-item} conda
```sh
conda create --name crystal python=3.12
conda activate crystal
pip install CRYSTALLine
```
:::

:::{tab-item} venv
```sh
python -m venv ~/.venvs/crystal
source ~/.venvs/crystal/bin/activate    # Windows: ~\.venvs\crystal\Scripts\activate
pip install CRYSTALLine
```
:::

::::

Then start it:

```sh
crystalline
```

Any version from 3.11 onwards may be used, including 3.14; releases are made
with 3.12.

## Optional components

The export of animations as MP4, MOV or WebM requires an encoder. GIF and
numbered frames do not.

```sh
pip install CRYSTALLine[video]
```

## Dependencies

These are installed automatically if not already present:

| Package | Why |
| --- | --- |
| PySide6-Essentials ≥ 6.5 | the Qt GUI (Essentials, not the 847 MB metapackage) |
| pyvista ≥ 0.43, pyvistaqt ≥ 0.11.4, vtk ≥ 9.1 | the 3D viewport |
| numpy ≥ 1.23, scipy ≥ 1.9 | geometry, bonds, statistics |
| ase ≥ 3.23, pymatgen ≥ 2023.11.10, spglib ≥ 2.5 | structures, symmetry, Brillouin zones |
| CRYSTALClear ≥ 0.2.16 | reading CRYSTAL output and drawing the property plots |
| matplotlib ≥ 3.6, Pillow ≥ 9.0 | plots and images |

## Checking the installation

```sh
crystalline --check
```

reports the Python version, the dependencies and the versions installed,
whether Qt can open a window and whether a VTK render window can be created,
and states what is missing. It returns a non-zero exit status if an essential
component failed, so that it can be used in a script.

## Platform notes

**Linux.** The system libraries required by Qt and VTK (`libGL`,
`libxkbcommon`) are not always present. Under Wayland, VTK draws through
XWayland and the program requests Qt's `xcb` plugin; if `QT_QPA_PLATFORM` has
been set to `wayland`, it must be unset. Ubuntu 24.04 also requires:

```sh
sudo apt install libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
  libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 \
  libxcb-xkb1 libxkbcommon-x11-0
```

**macOS and Windows.** No additional components are required. On Windows the
`crystalline` command is a graphical entry point without a console; if it fails
silently, `python -m crystalline` shows the error.

[Troubleshooting](troubleshooting.md) describes what to do if the program does not start.

## Installation from source

```sh
git clone https://github.com/crystaldevs/CRYSTALLine.git
cd CRYSTALLine
pip install -e .
pytest            # the suite runs headless
```
