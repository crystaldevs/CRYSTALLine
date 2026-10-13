"""Reads remembered by the version of the file they were made from.

Opening the same plot dialog on the same output again — which is how a second
plot is made — read the whole output again. What is remembered has to be
handed back only while the file is the one it was read from: its real path,
its size and its modification time, never its name.
"""

import os

import pytest

from crystalline.crystalio import remembered


@pytest.fixture(autouse=True)
def fresh():
    remembered.forget()
    yield
    remembered.forget()


def _counting(value="read"):
    calls = []

    def reader():
        calls.append(1)
        return value

    return reader, calls


def test_the_same_file_is_read_once(tmp_path):
    path = tmp_path / "run.out"
    path.write_text("an output")
    reader, calls = _counting()

    assert remembered.read("spectra", (str(path),), reader) == "read"
    assert remembered.read("spectra", (str(path),), reader) == "read"
    assert len(calls) == 1


def test_a_file_written_again_is_read_again(tmp_path):
    path = tmp_path / "run.out"
    path.write_text("an output")
    reader, calls = _counting()
    remembered.read("spectra", (str(path),), reader)

    stamp = os.stat(path).st_mtime_ns + 1_000_000
    os.utime(path, ns=(stamp, stamp))                   # rewritten, same size

    remembered.read("spectra", (str(path),), reader)
    assert len(calls) == 2


def test_it_is_the_file_that_counts_not_the_name_it_is_reached_by(tmp_path):
    path = tmp_path / "run.out"
    path.write_text("an output")
    link = tmp_path / "another name.out"
    link.symlink_to(path)
    reader, calls = _counting()

    remembered.read("spectra", (str(path),), reader)
    remembered.read("spectra", (str(link),), reader)    # the same file, by another name
    assert len(calls) == 1

    copy = tmp_path / "copy.out"
    copy.write_text("an output")                        # the same text, another file
    remembered.read("spectra", (str(copy),), reader)
    assert len(calls) == 2


def test_what_is_read_for_one_purpose_is_not_handed_out_for_another(tmp_path):
    path = tmp_path / "run.out"
    path.write_text("an output")
    remembered.read("spectra", (str(path),), lambda: "curves")

    assert remembered.read("pes", (str(path),), lambda: "constants") == "constants"


def test_a_read_that_fails_is_tried_again(tmp_path):
    path = tmp_path / "run.out"
    path.write_text("an output")
    calls = []

    def broken():
        calls.append(1)
        raise ValueError("no PES constants")

    for _ in range(2):
        with pytest.raises(ValueError):
            remembered.read("pes", (str(path),), broken)
    assert len(calls) == 2


def test_a_file_that_cannot_be_looked_at_is_never_remembered(tmp_path):
    reader, calls = _counting()
    missing = str(tmp_path / "gone.out")

    remembered.read("spectra", (missing,), reader)
    remembered.read("spectra", (missing,), reader)
    assert len(calls) == 2


def test_only_a_few_whole_reads_are_kept_but_many_small_facts(tmp_path):
    """A whole read holds the parsed output, lines and all."""
    paths = []
    for index in range(70):
        path = tmp_path / f"f{index}.out"
        path.write_text(str(index))
        paths.append(str(path))
    for path in paths:
        remembered.read("pes", (path,), lambda: "whole")
        remembered.read("bands-info", (path,), lambda: "fact", light=True)

    def kept(kind, light):
        return sum(remembered.get(remembered.key(kind, p), light) is not remembered.MISSING
                   for p in paths)

    assert kept("pes", False) == 4
    assert kept("bands-info", True) == 64


# ── the window's side ─────────────────────────────────────────────────────
def _window():
    from crystalline.ui.main_window import MainWindow

    class _Stub:
        _read_for_dialog = MainWindow._read_for_dialog

        def __init__(self):
            self.busy = []

        def _run_busy(self, work, message, done, _failed):
            self.busy.append(message)
            done(work())               # the thread, without the thread

    return _Stub()


def test_a_first_read_is_behind_the_spinner_and_a_second_is_at_once(tmp_path):
    """Not with the window frozen under a wait cursor, as it used to be; and
    the spinner does not flash up for a read that is already in hand."""
    path = tmp_path / "run.out"
    path.write_text("an output")
    window = _window()
    shown = []
    reader, calls = _counting(("run", "out"))

    for _ in range(2):
        window._read_for_dialog("pes", (str(path),), reader, "Reading the PES constants…",
                                shown.append, "PES failed")

    assert shown == [("run", "out"), ("run", "out")]
    assert window.busy == ["Reading the PES constants…"]      # once: the second was at hand
    assert len(calls) == 1


def test_the_electronic_dialog_reads_each_file_once_however_often_it_is_opened(monkeypatch):
    """It reads every band file in the folder to rank them, then the chosen
    one to pair it and to set itself up — and all of it again on reopening."""
    pytest.importorskip("PySide6")
    pytest.importorskip("CRYSTALClear")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from crystalline.crystalio import electronic as el
    from crystalline.ui.panels.electronic_dialog import ElectronicDialog

    QApplication.instance() or QApplication([])
    folder = os.path.join(os.path.dirname(__file__), "data", "mgo")
    reads = []
    real = el._read_bands
    monkeypatch.setattr(el, "_read_bands", lambda path: reads.append(path) or real(path))

    ElectronicDialog(folder=folder)
    first = list(reads)
    ElectronicDialog(folder=folder)

    assert len(first) == len(set(first))     # each band file read once, the first time
    assert reads == first                    # and not at all the second
