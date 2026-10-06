# Sample CRYSTAL files

What the test suite reads. They are here so that the suite is the same suite on
every machine: it used to read these from the author's own directories, and 87
of its tests ran nowhere else.

Nothing here is a wavefunction or a density grid — the parsers want the data
files a PROPERTIES run writes, which are small. `tools/collect_test_data.py`
copied them in and refreshes them from a local copy of their sources.

| Folder | What it is | Source |
| --- | --- | --- |
| `silicon`, `beryllium` | band structures, densities of states, Mulliken populations, charge-density maps | [MSSC2025](https://github.com/crystaldevs/MSSC2025), `Basic/Day4/OneElectronProperties` |
| `mgo` | the same for MgO, with projected and partial densities | [MSSC2025](https://github.com/crystaldevs/MSSC2025), `Advanced/Day1/one-electron-properties` |
| `lif` | an F centre, and so a spin-polarised band file | [MSSC2025](https://github.com/crystaldevs/MSSC2025), `Basic/Day5/Defects` |
| `qmmc` | a `.BAND` written under another name | [QMMC2026](https://github.com/crystaldevs/QMMC2026), `OneElectronProperties` |
| `ito` | a spin-polarised density of states | [CRYSTALpytools](https://github.com/crystal-code-tools/CRYSTALpytools) examples |
| `zno` | a phonon band `fort.25` | a SCELPHONO run |
| `coesite` | an elastic tensor | |
| `methane` | an anharmonic scan (`ANHAPES`) | |
| `thiourea` | a harmonic frequency run | |
| `corundum` | the output of an `ORBITALS` run — the orbitals themselves are too big to ship | |
| `locmodes` | `h2o2_synthetic.out`: a `LOCMODES` section for H₂O₂ in a 5 Å box, with made-up force constants, laid out as CRYSTAL23 lays out the blocks the reader needs — not a run | written for the tests |

The school material comes from the CRYSTAL schools' own public repositories —
[MSSC2024](https://github.com/crystaldevs/MSSC2024),
[MSSC2025](https://github.com/crystaldevs/MSSC2025) and
[QMMC2026](https://github.com/crystaldevs/QMMC2026) — and is reproduced here
unchanged. The rest are runs by the CRYSTALLine authors.

## Runs that are too big to ship

Some tests need whole runs — crystalline orbitals, density cubes, anharmonic
output — which are megabytes apiece. Those look for `CRYSTALLINE_TEST_DATA`, and
skip cleanly when it is not set:

```
CRYSTALLINE_TEST_DATA=~/my-crystal-runs pytest
```

expecting these folders inside it:

| Folder | What the tests read from it |
| --- | --- |
| `corundum` | `corundum_0000*_K*.molden` — the orbitals of an `ORBITALS` run |
| `co2` | `*/co2_anh_*.out` — anharmonic runs with VSCF/VCI levels |
| `qmmc` | `mgo_ech3.cube`, `mgo_pot3.cube` — a charge density and a potential |

If those runs are already somewhere on your machine, `python
tools/collect_test_data.py --link` builds that directory out of symlinks to
them, so nothing is copied and nothing large goes near the repository.
