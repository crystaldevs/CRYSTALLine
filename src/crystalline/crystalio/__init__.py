"""Adapters over CRYSTALClear for loading/saving CRYSTAL files.

Named ``crystalio`` (not ``io``) to avoid shadowing Python's stdlib ``io``.
This is the *only* place CRYSTALLine touches CRYSTALClear's API, so if that API
shifts, changes are localised here rather than spread through the UI.
"""

from crystalline.crystalio.loader import (
    LoadedFile,
    has_phonons,
    load,
    load_adp,
    load_dispersion,
    load_structure,
    load_phonons,
    output_properties,
    read_atoms,
    save_structure_gui,
    save_structure_cif,
)
from crystalline.crystalio.spectra import (
    LINESHAPES,
    SpectrumKind,
    available_spectra,
    load_spectra,
    plot_spectra,
)
from crystalline.crystalio.plotting import (
    PlotKind,
    PlotUnavailable,
    available_plots,
    crystalclear_available,
    output_availability,
)
from crystalline.crystalio.vci import (
    REPRESENTATIONS,
    VCIRun,
    has_vci,
    load_vci,
    plot_vci,
    vci_run,
)
from crystalline.crystalio.anscan import (
    WF_FILTER,
    AnscanRun,
    anscan_run,
    find_wavefunctions,
    has_anscan,
    load_anscan,
    plot_anscan,
)
from crystalline.crystalio.pes import (
    DIMENSIONS,
    QUANTITIES,
    PESMode,
    PESPair,
    PESRun,
    has_pes,
    load_pes,
    pes_run,
    plot_pes,
)

__all__ = [
    "load",
    "LoadedFile",
    "has_phonons",
    "load_adp",
    "load_dispersion",
    "load_structure",
    "load_phonons",
    "output_properties",
    "read_atoms",
    "save_structure_gui",
    "save_structure_cif",
    "PlotKind",
    "PlotUnavailable",
    "available_plots",
    "crystalclear_available",
    "output_availability",
    "LINESHAPES",
    "SpectrumKind",
    "available_spectra",
    "load_spectra",
    "plot_spectra",
    "REPRESENTATIONS",
    "VCIRun",
    "has_vci",
    "load_vci",
    "plot_vci",
    "vci_run",
    "WF_FILTER",
    "AnscanRun",
    "anscan_run",
    "find_wavefunctions",
    "has_anscan",
    "load_anscan",
    "plot_anscan",
    "DIMENSIONS",
    "QUANTITIES",
    "PESMode",
    "PESPair",
    "PESRun",
    "has_pes",
    "load_pes",
    "pes_run",
    "plot_pes",
]
