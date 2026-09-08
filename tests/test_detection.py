"""Auto-detection of vibrational (frequency) CRYSTAL outputs.

Uses tiny synthetic files so it runs without CRYSTALClear or the user's data —
``has_phonons`` only scans text for CRYSTAL's frequency-run markers.
"""

from crystalline.crystalio.loader import has_phonons

_FREQ_HEADER = "    MODES         EIGV          FREQUENCIES     IRREP  IR   INTENS    RAMAN\n"
_SYMM_MARKER = " +++ SYMMETRY ADAPTION OF VIBRATIONAL MODES +++\n"


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_detects_frequency_table(tmp_path):
    f = _write(tmp_path, "freq.out", "SCF ENDED\n" + _FREQ_HEADER + "1- 3  0.0  0.0\n")
    assert has_phonons(f) is True


def test_detects_symmetry_adaption_marker(tmp_path):
    f = _write(tmp_path, "freq2.out", "blah\n" + _SYMM_MARKER + "more\n")
    assert has_phonons(f) is True


def test_plain_scf_output_has_no_phonons(tmp_path):
    f = _write(tmp_path, "scf.out", "SCF ENDED\nTOTAL ENERGY -100.0\nOPT END\n")
    assert has_phonons(f) is False


def test_gui_file_is_never_phonons(tmp_path):
    # even if a .gui somehow contained the words, geometry files carry no modes
    f = _write(tmp_path, "geom.gui", _FREQ_HEADER)
    assert has_phonons(f) is False
