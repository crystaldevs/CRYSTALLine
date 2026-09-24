"""Entry point: ``python -m crystalline`` (or the ``crystalline`` command).

``--check`` exists because the installed command is a ``gui_scripts`` entry: on
Windows that runs under ``pythonw`` with no console at all, so a startup failure
prints to nothing and the command looks like it did nothing. A user with a
problem needs one thing they can run that talks back.
"""

from __future__ import annotations

import sys

_USAGE = """\
usage: crystalline [file] [--check] [--version]

  file       a CRYSTAL .out/.gui/.f34 or a .cif to open on startup
  --check    check this machine has what CRYSTALLine needs, and say what is
             missing if not
  --version  print the version and exit
"""


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--help" in argv or "-h" in argv:
        print(_USAGE)
        return 0
    if "--version" in argv:
        from crystalline import __version__

        print(__version__)
        return 0
    if "--check" in argv or "--doctor" in argv:
        from crystalline.diagnose import report

        return report()

    from crystalline.app import run

    return run()


if __name__ == "__main__":
    sys.exit(main())
