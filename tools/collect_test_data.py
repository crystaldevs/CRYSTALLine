"""Collect the sample CRYSTAL files the test suite reads into ``tests/data``.

The suite reads these from the repository, not from anyone's own machine, so
that its tests run anywhere. They are small — the parsers want the data files a
PROPERTIES run writes, not the wavefunctions or the grids beside them — and this
script is how they get there.

Run it to refresh them from a local copy of the source material:

    python tools/collect_test_data.py [--dry-run]

Sources are given as globs under ``$HOME``; anything missing is reported and
skipped, so a partial local copy still refreshes what it can.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

HOME = Path.home()
DEST = Path(__file__).resolve().parent.parent / "tests" / "data"

# destination directory -> source directory, files
MANIFEST = {
    # github.com/crystaldevs/MSSC2025, Basic, Day 4 — one-electron properties
    # of two simple solids.
    "silicon": (
        "MSSC2025/Basic/Day4/OneElectronProperties/Silicon",
        ["BAND.DAT", "DOSS.DAT", "PPAN.DAT", "si_band.d3", "si_band.f25",
         "si_doss.d3", "si_doss.f25", "si_echg.d3", "si_echg.f25"],
    ),
    "beryllium": (
        "MSSC2025/Basic/Day4/OneElectronProperties/Berillium",
        ["BAND.DAT", "DOSS.DAT", "PPAN.DAT", "be_band.d3", "be_band.f25",
         "be_doss.d3", "be_doss.f25", "be_echg.d3", "be_echg.f25"],
    ),
    # github.com/crystaldevs/MSSC2025, Advanced, Day 1 — the same for MgO,
    # with projected densities.
    "mgo": (
        "MSSC2025/Advanced/Day1/one-electron-properties",
        ["mgo_band.BAND", "mgo_band.d3", "mgo_band.f25", "mgo_band_b3lyp.BAND",
         "mgo_doss_totalao.DOSS", "mgo_doss_totalao.d3", "mgo_doss_totalao.f25",
         "mgo_doss_partialao.DOSS", "mgo_doss_partialao.d3",
         "mgo_doss_partialao.f25", "mgo_anbd.d3", "mgo_anbd_2.d3",
         "mgo_echg.d3", "mgo_echg.f25", "mgo_pban_core.d3", "mgo_pban_core.f25",
         "mgo_pban_valence.d3", "mgo_pban_valence.f25",
         "mgo_pdide_valence.d3", "mgo_pdide_valence.f25",
         "mof6_ech3_pot3.d3"],
    ),
    # github.com/crystaldevs/MSSC2025, Basic, Day 5 — an F centre, and so a
    # spin-polarised band file.
    "lif": (
        "MSSC2025/Basic/Day5/Defects/4_LiF_2_2_2_F_center_band",
        ["BAND.DAT", "band.d3", "fort.25"],
    ),
    # github.com/crystaldevs/QMMC2026 — a BAND written under its other name.
    "qmmc": ("QMMC2026/OneElectronProperties/output", ["mgo_band_dat.BAND"]),
    # A spin-polarised density of states, and a phonon band f25.
    "ito": ("Desktop/PyCrystal/CRYSTALpytools/examples/data", ["doss_ito-cu.DOSS"]),
    "zno": ("Desktop/PyCrystal/ZnO", ["ZnO_scelphono_333_shrink_22_tight_bands.f25"]),
    "corundum": ("Desktop/PyCrystal/corundum_orbitals", ["corundum.outp"]),
    # Small outputs read whole: an elastic tensor, an anharmonic scan, a
    # harmonic+VCI spectrum.
    "coesite": ("Desktop/PyCrystal/coesite", ["coesite_ela.out"]),
    "methane": ("Desktop/PyCrystal/anharmonic_freq", ["CH4_anarmonic.out"]),
    "thiourea": ("Desktop/PyCrystal/anharmonic_freq",
                 ["thiourea_pbed3_Ahlrichs-pVTZ_freqcalc_80K.out"]),
}


# The runs that are too big to ship: megabytes of orbitals, density grids and
# anharmonic output. ``--link`` builds a directory of symlinks to wherever they
# sit on this machine, for CRYSTALLINE_TEST_DATA to point at, so the tests that
# read them go on running here without any of it entering the repository.
BIG = {
    "corundum": "Desktop/PyCrystal/corundum_orbitals",
    "co2": "Desktop/PyCrystal/anharmonic_freq/CO2_molecule",
    "qmmc": "QMMC2026/OneElectronProperties/output",
}
LINKS = Path.home() / ".crystalline-test-data"


def link_big(dry_run: bool) -> int:
    """Point a CRYSTALLINE_TEST_DATA directory at the big runs, where they are."""
    made = 0
    for name, source in sorted(BIG.items()):
        src = HOME / source
        if not src.is_dir():
            print(f"  missing  {src}")
            continue
        link = LINKS / name
        print(f"  {name:10s} -> {src}")
        if dry_run:
            continue
        LINKS.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src)
        made += 1
    if made:
        print(f"\nNow export CRYSTALLINE_TEST_DATA={LINKS}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be copied, and copy nothing")
    parser.add_argument("--link", action="store_true",
                        help="symlink the runs too big to ship, for "
                             "CRYSTALLINE_TEST_DATA to point at")
    args = parser.parse_args()

    if args.link:
        return link_big(args.dry_run)

    copied = missing = 0
    total = 0
    for folder, (source, names) in sorted(MANIFEST.items()):
        target = DEST / folder
        for name in names:
            src = HOME / source / name
            if not src.is_file():
                print(f"  missing  {src}")
                missing += 1
                continue
            size = src.stat().st_size
            total += size
            print(f"  {size / 1e3:8.1f} kB  {folder}/{name}")
            if not args.dry_run:
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target / name)
            copied += 1
    print(f"\n{copied} files, {total / 1e6:.2f} MB"
          + (f"   ({missing} missing)" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
