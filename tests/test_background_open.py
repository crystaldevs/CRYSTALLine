"""Opening a file reads it off the main thread, then builds its tab from what was read.

With no loading screen: the first of several files comes to the front to be
worked in, and the rest are set behind it as they are read.

The window itself needs a real display (VTK), so these exercise the parts
apart: :func:`_read_file`, which is Qt-free and does all the reading; the
queue in :meth:`MainWindow._load_path`; and where a read file lands
(:meth:`MainWindow._show_read_file`) — on stand-ins carrying the window's own
methods, with the tab-building stubbed out.
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
def test_an_output_is_read_as_far_as_its_tab_needs():
    """The structure, and what each menu needs to know about the output. The
    Info panel's account of the run is not among it: it costs more than the
    structure, and is read afterwards (see the details tests below)."""
    path = sample.path("coesite/coesite_ela.out")
    if path is None:
        pytest.skip("sample output not available")

    read = mw._read_file(path)

    assert len(read.loaded.structure) > 0
    assert read.output_path == path
    assert read.output_props is None                         # still to be read
    assert mw._read_details(path)                            # ...and there to be read
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


def _window():
    """The window's queue, with the tab-building recorded instead.

    No busy overlay, no worker list, no tab bar: reading must use none of them —
    the user goes on working in the window while files are read.
    """

    class _Stub:
        _load_path = MainWindow._load_path
        _read_next = MainWindow._read_next
        _update_reading_status = MainWindow._update_reading_status
        _details_queue = MainWindow._details_queue
        _read_details_next = MainWindow._read_details_next

        def __init__(self):
            self._reading_status = _Label()
            self._pending_reads = []
            self._reading = False
            self._reading_path = None
            self._reader = None
            self.shown = []

        def _show_read_file(self, path, read, front):
            self.shown.append((path, read, front, threading.current_thread()))

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


# ── an output's details, read after its structure ─────────────────────────
class _Read:
    """What _read_file hands over, as far as the queue looks at it."""

    def __init__(self, path):
        self.output_path = None if path.endswith(".gui") else path
        self.output_props = None


class _DetailsTab:
    def __init__(self, path, behind):
        self.path = path
        self.pending = None

    def __repr__(self):
        return f"<tab {self.path}>"


def _details_window():
    """The queue, with tabs that record what reaches them."""

    class _Stub:
        _load_path = MainWindow._load_path
        _read_next = MainWindow._read_next
        _update_reading_status = MainWindow._update_reading_status
        _details_queue = MainWindow._details_queue
        _read_details_next = MainWindow._read_details_next
        _deliver_details = MainWindow._deliver_details
        _routed = MainWindow._routed
        _acting_on = MainWindow._acting_on

        def __init__(self):
            self._reading_status = _Label()
            self._pending_reads = []
            self._reading = False
            self._reading_path = None
            self._reader = None
            self._tabs = []
            self._tab = None
            self.delivered = []

        def _show_read_file(self, path, read, front):
            tab = _DetailsTab(path, behind=not front)
            self._tabs.append(tab)
            if front or self._tab is None:
                self._tab = tab
            else:
                tab.pending = (path, read)      # read in behind, not built yet
            return tab

        def _set_output_props(self, props):
            self.delivered.append((self._tab, props))

        def _refresh_chrome(self):
            pass

    return _Stub()


@pytest.fixture
def reads(monkeypatch):
    """Record the order things are read in; details name the output they are of."""
    order = []

    def read_file(path):
        order.append(("file", path))
        return _Read(path)

    def read_details(path):
        order.append(("details", path))
        return {"Total energy (eV)": path}

    monkeypatch.setattr(mw, "_read_file", read_file)
    monkeypatch.setattr(mw, "_read_details", read_details)
    return order


def test_every_file_is_shown_before_any_output_s_details_are_read(qapp, qtbot, reads):
    """The structure is what was asked for; the run's own account of itself
    costs more than it does, so it waits until nothing else is."""
    window = _details_window()
    for path, front in (("/runs/a.out", True), ("/runs/b.out", False), ("/runs/c.gui", False)):
        window._load_path(path, front=front)
    qtbot.waitUntil(lambda: not window._reading, timeout=5000)

    assert reads == [("file", "/runs/a.out"), ("file", "/runs/b.out"), ("file", "/runs/c.gui"),
                     ("details", "/runs/a.out"), ("details", "/runs/b.out")]   # no output behind a .gui


def test_details_reach_the_tab_of_their_output(qapp, qtbot, reads):
    """Through the tab's own routing, whichever tab is in front by then; and
    for a tab read in behind and not looked at yet, into what it will be
    built from."""
    window = _details_window()
    window._load_path("/runs/a.out", front=True)
    window._load_path("/runs/b.out", front=False)
    qtbot.waitUntil(lambda: not window._reading, timeout=5000)
    front, behind = window._tabs

    assert window.delivered == [(front, {"Total energy (eV)": "/runs/a.out"})]
    assert behind.pending[1].output_props == {"Total energy (eV)": "/runs/b.out"}


def test_details_for_a_tab_closed_meanwhile_go_nowhere(qapp, qtbot, monkeypatch, reads):
    window = _details_window()
    release = threading.Event()
    monkeypatch.setattr(mw, "_read_details",
                        lambda path: release.wait(5) and {"Total energy (eV)": path})
    window._load_path("/runs/a.out", front=True)
    qtbot.waitUntil(lambda: bool(window._tabs), timeout=5000)
    window._tabs.clear()                                   # closed while its details are read
    release.set()
    qtbot.waitUntil(lambda: not window._reading, timeout=5000)

    assert window.delivered == []


def test_details_are_read_without_a_word(qapp, qtbot, monkeypatch, reads):
    """The structure is on screen: "Reading a.out…" would say it was not."""
    window = _details_window()
    release = threading.Event()
    monkeypatch.setattr(mw, "_read_details", lambda path: release.wait(5) and {})
    window._load_path("/runs/a.out", front=True)
    qtbot.waitUntil(lambda: bool(window._tabs), timeout=5000)

    assert window._reading and not window._reading_status.visible
    window._load_path("/runs/b.out", front=False)          # asked for meanwhile: named
    assert window._reading_status.visible
    assert window._reading_status.texts[-1] == "Reading b.out…"
    release.set()
    qtbot.waitUntil(lambda: not window._reading, timeout=5000)


def test_details_wait_a_moment_after_the_file_is_shown(qapp, qtbot, reads):
    """For its 3D view to settle: read straight away, the details held up the
    confirmation of the view's new size for 400 ms, and the structure was
    shown stretched meanwhile."""
    import time

    window = _details_window()
    window._load_path("/runs/a.out", front=True)
    qtbot.waitUntil(lambda: bool(window._tabs), timeout=5000)
    shown = time.perf_counter()
    assert ("details", "/runs/a.out") not in reads

    qtbot.waitUntil(lambda: ("details", "/runs/a.out") in reads, timeout=5000)
    assert time.perf_counter() - shown >= mw._DETAILS_PAUSE_MS / 1000 * 0.8
