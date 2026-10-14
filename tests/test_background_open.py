"""Opening a file reads it off the main thread, then builds its tab from what was read.

With no loading screen: the first of several files comes to the front to be
worked in, and the rest are set behind it as they are read.

Most of these exercise the parts apart: :func:`_read_file`, which is Qt-free
and does all the reading but the Info panel's rows about the run
(:func:`_read_output_props`, read once the tab is up); the queue in
:meth:`MainWindow._load_path`; and where a read file lands
(:meth:`MainWindow._show_read_file`) — on stand-ins carrying the window's own
methods, with the tab-building stubbed out. The last few open files on the
real window (``_window.py``), with only its 3D views stood in for.
"""

import threading

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import sample_data as sample  # noqa: E402
from crystalline.ui import main_window as mw  # noqa: E402
from crystalline.ui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ── reading ───────────────────────────────────────────────────────────────
def test_an_output_is_read_whole_for_its_tab():
    """The structure, and everything the window used to read again afterwards:
    what each menu needs to know about the output. The Info panel's rows about
    the run are read apart, once the tab is up: half the reading on a large
    output, and nothing else waits for them."""
    path = sample.path("coesite/coesite_ela.out")
    if path is None:
        pytest.skip("sample output not available")

    read = mw._read_file(path)

    assert len(read.loaded.structure) > 0
    assert read.output_path == path
    assert read.output_props is None                         # still to read...
    assert mw._read_output_props(path)                       # ...and there to be read
    assert set(read.capabilities) == set(mw._PROBES)
    for key, found in read.capabilities.items():             # the window's own answer
        assert found == mw._probe(key, path), key


def test_a_geometry_file_has_no_output_behind_it(tmp_path):
    from ase.build import bulk

    from crystalline.core.structure import Structure
    from crystalline.crystalio import save_structure_gui

    path = str(tmp_path / "mgo.gui")
    save_structure_gui(Structure.from_ase(bulk("MgO", "rocksalt", a=4.21)), path)

    read = mw._read_file(path)

    assert set(read.loaded.structure.symbols) == {"Mg", "O"}
    assert read.output_path is None                          # nothing to plot from
    assert read.adps is None
    assert read.output_props == {}                           # and no rows to wait for


def test_a_file_that_will_not_read_raises_rather_than_opening_a_tab(tmp_path):
    path = tmp_path / "nothing.out"
    path.write_text("not a CRYSTAL output\n")
    with pytest.raises(Exception):
        mw._read_file(str(path))


def test_what_was_found_off_the_thread_answers_the_window(monkeypatch):
    """Found once, while reading: the window must not walk the output again."""

    class _Tab:
        capabilities = {("vci", "/runs/a.out"): True}

    class _Stub:
        _capability = MainWindow._capability
        _tab = _Tab()
        _output_path = "/runs/a.out"

    asked = []
    monkeypatch.setattr(mw, "_probe", lambda key, path: asked.append(key) or False)
    assert _Stub()._capability("vci") is True
    assert asked == []
    assert _Stub()._capability("pes") is False               # not found yet: asked now
    assert asked == ["pes"]


# ── the queue ─────────────────────────────────────────────────────────────
class _Label:
    """The status bar's reading label: what it said, and whether it is shown."""

    def __init__(self):
        self.texts = []
        self.visible = False

    def setText(self, text):
        self.texts.append(text)

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False


def _window(events=None):
    """The window's queue, with the tab-building recorded instead.

    No busy overlay, no worker list, no tab bar: reading must use none of them —
    the user goes on working in the window while files are read. With
    ``events``, a file shown is handed a tab (its path), and what happens is
    recorded there in order — the rows about the run among it.
    """

    class _Stub:
        _load_path = MainWindow._load_path
        _read_next = MainWindow._read_next
        _read_output_rows = MainWindow._read_output_rows
        _update_reading_status = MainWindow._update_reading_status
        _start_reader = MainWindow._start_reader
        _lay_out_now = MainWindow._lay_out_now

        def __init__(self):
            self._reading_status = _Label()
            self._pending_reads = []
            self._reading = False
            self._reading_path = None
            self._reader = None
            self.shown = []

        def _show_read_file(self, path, read, front):
            self.shown.append((path, read, front, threading.current_thread()))
            if events is not None:
                events.append(("shown", path))
                return path

        def _show_output_rows(self, tab, read, props):
            events.append(("rows shown", tab, props))

    return _Stub()


def test_files_are_read_off_the_main_thread_one_at_a_time_in_order(qapp, qtbot, monkeypatch):
    read_on = {}

    def fake_read(path):
        read_on[path] = threading.current_thread()
        if path.endswith("broken.out"):
            raise ValueError("Geometry information not found.")
        return f"read {path}"

    reported = []
    monkeypatch.setattr(mw, "_read_file", fake_read)
    monkeypatch.setattr(QMessageBox, "critical",
                        staticmethod(lambda _parent, title, text: reported.append((title, text))))
    window = _window()

    window._load_path("/runs/a.out", front=True)
    window._load_path("/runs/broken.out", front=False)
    window._load_path("/runs/c.gui", front=False)
    qtbot.waitUntil(lambda: not window._reading, timeout=5000)

    main = threading.main_thread()
    assert all(thread is not main for thread in read_on.values())     # read off the main thread
    assert [(p, r, f) for p, r, f, _t in window.shown] == [           # tabs built in order...
        ("/runs/a.out", "read /runs/a.out", True), ("/runs/c.gui", "read /runs/c.gui", False)]
    assert all(thread is main for *_rest, thread in window.shown)     # ...on the main thread
    assert reported == [("Load failed", "broken.out:\nGeometry information not found.")]


def test_the_status_bar_says_what_is_being_read_and_how_much_is_left(qapp, qtbot, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(mw, "_read_file", lambda path: release.wait(5) and path)
    window = _window()

    for path in ("/runs/a.out", "/runs/b.out", "/runs/c.out"):
        window._load_path(path, front=path.endswith("a.out"))
    label = window._reading_status
    assert label.visible and label.texts[-1] == "Reading a.out…  2 more to come"
    release.set()
    qtbot.waitUntil(lambda: not window._reading, timeout=5000)
    assert "Reading b.out…  1 more to come" in label.texts
    assert "Reading c.out…" in label.texts
    assert not label.visible                                          # gone when all are read


def test_a_file_opened_while_another_is_read_waits_its_turn(qapp, qtbot, monkeypatch):
    started = []
    release = threading.Event()

    def slow_read(path):
        started.append(path)
        release.wait(5)
        return path

    monkeypatch.setattr(mw, "_read_file", slow_read)
    window = _window()
    window._load_path("/runs/first.out")
    qtbot.waitUntil(lambda: started == ["/runs/first.out"], timeout=5000)
    window._load_path("/runs/second.out")                    # arrives mid-read
    assert started == ["/runs/first.out"]                    # not read alongside it
    release.set()
    qtbot.waitUntil(lambda: not window._reading, timeout=5000)
    assert [p for p, *_rest in window.shown] == ["/runs/first.out", "/runs/second.out"]


def test_the_rows_about_a_run_are_read_after_its_tab_is_up_and_before_the_next_file(
        qapp, qtbot, monkeypatch):
    """A file's tab is shown as soon as its structure is read; its Info rows
    are read next, still off the main thread, and only then the next file —
    so the tab in front is not left waiting behind the rest of a batch."""
    events = []

    class _Read:
        def __init__(self, path):
            self.output_props = None                         # still to read

    def fake_read(path):
        events.append(("read", path))
        return _Read(path)

    def fake_rows(path):
        on_worker = threading.current_thread() is not threading.main_thread()
        events.append(("rows read", path, on_worker))
        return {"Code": path}

    monkeypatch.setattr(mw, "_read_file", fake_read)
    monkeypatch.setattr(mw, "_read_output_props", fake_rows)
    window = _window(events)
    window._load_path("/runs/a.out", front=True)
    window._load_path("/runs/b.out", front=False)
    qtbot.waitUntil(lambda: not window._reading, timeout=5000)

    assert events == [
        ("read", "/runs/a.out"), ("shown", "/runs/a.out"),
        ("rows read", "/runs/a.out", True), ("rows shown", "/runs/a.out", {"Code": "/runs/a.out"}),
        ("read", "/runs/b.out"), ("shown", "/runs/b.out"),
        ("rows read", "/runs/b.out", True), ("rows shown", "/runs/b.out", {"Code": "/runs/b.out"}),
    ]


def test_rows_that_will_not_read_leave_the_tab_without_them(qapp, qtbot, monkeypatch):
    events = []

    class _Read:
        output_props = None

    def broken(_path):
        raise ValueError("no SCF block")

    monkeypatch.setattr(mw, "_read_file", lambda path: _Read())
    monkeypatch.setattr(mw, "_read_output_props", broken)
    window = _window(events)
    window._load_path("/runs/a.out")
    window._load_path("/runs/b.out", front=False)
    qtbot.waitUntil(lambda: not window._reading, timeout=5000)
    assert ("rows shown", "/runs/a.out", {}) in events          # none, and no message
    assert ("shown", "/runs/b.out") in events                   # nor does it stop the queue


# ── where a file that has been read goes ──────────────────────────────────
class _Tab:
    def __init__(self, name, blank=False):
        self.name, self.blank = name, blank
        self.path = None
        self.notice = None
        self.pending = None

    def is_blank(self):
        return self.blank


def _router(blank_on_screen=False):
    """The window's placing of a read file, with the tab-building recorded."""

    class _Stub:
        _show_read_file = MainWindow._show_read_file
        _show_notice = MainWindow._show_notice
        _acting_on = MainWindow._acting_on

        def __init__(self):
            self.on_screen = _Tab("on screen", blank=blank_on_screen)
            self._tab = self.on_screen
            self.calls = []

        def _take_tab_for_file(self):
            if not self._tab.is_blank():
                self._tab = _Tab("new, in front")
            self.calls.append(("take", self._tab.name))
            return self._tab

        def _add_tab(self, show=True):
            self.calls.append(("add", show))
            self.placed = _Tab("new, behind")
            return self.placed

        def _update_tab_labels(self):
            self.calls.append(("labels",))

        def _fill_tab(self, tab, path, read):
            # filled while the window acts on it — whichever tab is on screen
            self.calls.append(("fill", tab.name, self._tab.name))
            tab.path = path
            tab.notice = read

        def _refresh_chrome(self):
            self.calls.append(("menus for", self._tab.name))

        def _settle_tab(self, tab):
            self.calls.append(("settle", tab.name))

    return _Stub()


@pytest.fixture
def warnings_said(monkeypatch):
    said = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda _parent, title, text: said.append((title, text))))
    return said


def test_the_first_file_comes_to_the_front_and_says_what_it_left_out(qapp, warnings_said):
    window = _router()
    window._show_read_file("/runs/a.out", "modes could not be read", front=True)

    assert window.calls == [("take", "new, in front"), ("fill", "new, in front", "new, in front"),
                            ("settle", "new, in front")]   # laid out and drawn before it is seen
    assert window._tab.name == "new, in front"
    assert warnings_said == [("Modes not read", "a.out:\nmodes could not be read")]


def test_the_others_are_set_behind_without_taking_the_screen(qapp, warnings_said):
    """A tab set behind the one being worked in holds its file and waits: it is
    not filled, and nothing is built or drawn for it, until it is looked at —
    see test_file_tabs.py for the realising. Nothing pops up over the work."""
    window = _router()
    window._show_read_file("/runs/b.out", "modes could not be read", front=False)

    assert window.calls == [("add", False), ("labels",)]     # added and named, not filled
    assert window.placed.path == "/runs/b.out"
    assert window.placed.pending == ("/runs/b.out", "modes could not be read")
    assert window._tab is window.on_screen                   # still the one worked in
    assert warnings_said == []                               # said when it is looked at


def test_a_file_read_into_an_empty_window_fills_its_empty_tab(qapp, warnings_said):
    window = _router(blank_on_screen=True)
    window._show_read_file("/runs/b.out", None, front=False)
    assert window.calls == [("take", "on screen"), ("fill", "on screen", "on screen"),
                            ("settle", "on screen")]


def test_a_tab_read_in_behind_says_what_it_left_out_when_first_shown(qapp, warnings_said):
    import inspect

    window = _router()
    tab = _Tab("behind")
    tab.path, tab.notice = "/runs/b.out", "modes could not be read"
    window._show_notice(tab)
    window._show_notice(tab)                                 # once, not every visit
    assert warnings_said == [("Modes not read", "b.out:\nmodes could not be read")]
    assert "self._show_notice(tab)" in inspect.getsource(MainWindow._activate_tab)


# ── on the real window ────────────────────────────────────────────────────
_THIOUREA = "thiourea/thiourea_pbed3_Ahlrichs-pVTZ_freqcalc_80K.out"
_COESITE = "coesite/coesite_ela.out"
_COMING = [("", "Reading the output…")]


@pytest.fixture
def window(monkeypatch, tmp_path):
    from _window import close_window, make_window

    made = make_window(monkeypatch, tmp_path)
    yield made
    close_window(made)


def _rows(info_panel):
    """The Info panel's CRYSTAL-output rows as (label, value); [] while hidden."""
    from PySide6.QtWidgets import QFormLayout

    if info_panel._output_group.isHidden():
        return []
    form = info_panel._output_form
    rows = []
    for row in range(form.rowCount()):
        label = form.itemAt(row, QFormLayout.LabelRole)
        value = form.itemAt(row, QFormLayout.FieldRole)
        rows.append((label.widget().text() if label else "", value.widget().text()))
    return rows


def _held_rows(monkeypatch):
    """Hold the rows about a run back until the event returned is set."""
    release = threading.Event()
    real = mw._read_output_props
    monkeypatch.setattr(mw, "_read_output_props", lambda path: release.wait(10) and real(path))
    return release, real


def test_a_tab_comes_up_before_the_rows_about_its_run(window, qtbot, monkeypatch):
    """Opened as File ▸ Open does: the tab is up with its structure while the
    rows are read — the panel says they are coming, the status bar that the
    file is still being read — and they come in when they are."""
    release, real = _held_rows(monkeypatch)
    path = sample.path(_THIOUREA)
    window._load_path(path)
    qtbot.waitUntil(lambda: window._tab.path == path and window._tab.built(), timeout=10000)

    assert window._reading and window._reading_status.text().startswith("Reading thiourea")
    assert _rows(window.info_panel) == _COMING
    assert len(window.structure) > 0
    release.set()
    qtbot.waitUntil(lambda: not window._reading, timeout=10000)
    assert _rows(window.info_panel) == [(f"{k}:", str(v)) for k, v in real(path).items()]


def test_rows_read_for_a_tab_behind_are_there_when_it_is_first_shown(window, qtbot):
    """Read into the tab's waiting file, and filled in with it on the first look."""
    from _window import show_tab

    front, behind = sample.path(_THIOUREA), sample.path(_COESITE)
    window._load_path(front)
    window._load_path(behind, front=False)
    qtbot.waitUntil(lambda: not window._reading, timeout=20000)
    tab = next(t for t in window._tabs if t.path == behind)
    assert not tab.built() and tab.pending[1].output_props    # waiting with the file

    show_tab(window, tab)
    rows = _rows(window.info_panel)
    assert rows != _COMING and len(rows) == len(mw._read_output_props(behind))


def test_a_tab_closed_before_its_rows_are_read_takes_nothing(window, qtbot, monkeypatch):
    """The rows arrive for a tab that has gone; the next file is read all the same."""
    release, _ = _held_rows(monkeypatch)
    first, second = sample.path(_THIOUREA), sample.path(_COESITE)
    window._load_path(first)
    qtbot.waitUntil(lambda: window._tab.path == first, timeout=10000)
    window._load_path(second)                                # queued behind the rows
    closed = window._tab
    window._close_current_tab()
    release.set()
    qtbot.waitUntil(lambda: not window._reading, timeout=20000)

    assert closed not in window._tabs
    assert window._tab.path == second
    assert _rows(window.info_panel) and _rows(window.info_panel) != _COMING
