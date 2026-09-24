# Troubleshooting

## It does not start

```sh
crystalline --check
```

reports the Python version, the dependencies and the versions installed,
whether Qt can open a window and whether a VTK render window can be created,
and states what is missing.

The messages produced by Qt itself are not always helpful: on a machine without
a display it advises reinstalling the application. Two further sources of
information are available.

`python -m crystalline` runs the program from a terminal, where the errors are
printed. This matters on Windows, where the `crystalline` command is a
graphical entry point without a console. Errors are also written to a log file:

| System | Log |
| --- | --- |
| macOS | `~/Library/Logs/CRYSTALLine/errors.log` |
| Windows | `%LOCALAPPDATA%\CRYSTALLine\Logs\errors.log` |
| Linux | `~/.local/state/CRYSTALLine/errors.log` |

`crystalline --check` prints the path.

## Linux

The system libraries required by Qt and VTK (`libGL`, `libxkbcommon`) are not
always present after installation; `--check` reports which stage failed.

**Wayland.** VTK draws into an X11 window, and the program therefore requests
Qt's `xcb` plugin and runs through XWayland. If `QT_QPA_PLATFORM` has been set
to `wayland`, the program fails to start with `BadWindow (invalid Window
parameter)`; unset the variable, or run `QT_QPA_PLATFORM=xcb crystalline`.

**Ubuntu 24.04** needs the `xcb` libraries pip does not pull in:

```sh
sudo apt install libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
  libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 \
  libxcb-xkb1 libxkbcommon-x11-0
```

## A menu entry is greyed out

The entries of the Plot menu are enabled only when the output contains the
corresponding data: the elastic surfaces require an elastic tensor. The editing
commands require **Editing mode** ({kbd}`Ctrl+E`), and the phonon controls
require an output containing vibrational modes.

## No modes are listed

The output must contain a frequency calculation. A single point or a geometry
optimisation which ends before `FREQCALC` contains none.

## The band path is refused

The path is written as integers divided by a shrinking factor. A factor which
would leave a coordinate fractional is refused rather than rounded; use `auto`,
or a multiple of the factor proposed. 

## A plot fails

The plots are produced by CRYSTALClear. A failure usually means that the file
does not contain the data required, and the message reports which file and what
was missing. If a band or DOS file is not listed in the dialog, its first line
should be checked: the files are recognised by their contents, and a truncated
file is skipped.

## Reporting a problem

Problems may be reported at
[github.com/crystaldevs/CRYSTALLine/issues](https://github.com/crystaldevs/CRYSTALLine/issues)
Please include the output of `crystalline --check` and, if the program
started, the log file.
