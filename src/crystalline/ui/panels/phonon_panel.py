"""Phonon panel: list vibrational modes and drive the animation.

Owns the ``QTimer`` that advances the animation phase (the animator itself is
Qt-free). Selecting a mode sets it on the :class:`PhononAnimator`; the Play/Stop
icons at the top of the panel toggle the timer; amplitude scales the
displacement and speed scales how fast the phase advances. A filter narrows the
list to the IR- and/or Raman-active modes when the output reports selection
rules.

A SCELPHONO run reports modes at several q-points, so the panel also carries a
q-point selector; it hides itself for the ordinary Gamma-only frequency run,
where a chooser with one entry is just clutter. Away from Gamma the modes are
travelling waves, invisible on a single cell — the button beside the selector
asks the window for the smallest supercell that shows one whole period.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QSizePolicy,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from crystalline.core.mode_analysis import ModeCharacter, mode_character
from crystalline.core.phonons import (
    PhononMode,
    PhononModes,
    commensurate_repeats,
    qpoint_label,
)
from crystalline.viz.phonon_animator import PhononAnimator

# Activity filters: (label, predicate over a PhononMode). ``None`` keeps every
# mode; the others are only selectable when the output carries the analysis.
_FILTERS = (
    ("All modes", None),
    ("IR active", lambda m: bool(m.ir_active)),
    ("Raman active", lambda m: bool(m.raman_active)),
    ("IR or Raman active", lambda m: bool(m.ir_active) or bool(m.raman_active)),
)

# The mode list must be able to shrink to this, or its size hint pushes the
# controls out of a short dock (reported on Windows, where the dock is laid out
# tighter). It keeps its scrollbar and stretches when there's room.
_LIST_MIN_HEIGHT = 70

# Two lines' worth, so the composition summary can wrap without the amplitude
# and speed controls below it shifting as the selection changes.
_CHARACTER_MIN_HEIGHT = 32

# Animation timing. The timer stays at ~30 fps whatever the speed — speed
# changes how far the phase moves per tick, so the motion stays smooth instead
# of turning into a slideshow at low speed.
_FRAME_INTERVAL_MS = 33
_FRAMES_PER_CYCLE = 60  # at speed 1.0: one full vibration in ~2 s
_PHASE_STEP = 2.0 * math.pi / _FRAMES_PER_CYCLE

# The largest share of the event loop the animation may take. A frame is a VTK
# rebuild plus a synchronous render, and on a large cell that costs more than the
# frame interval — at which point the timer is always overdue, fires back to
# back, and leaves nothing for input: the whole window goes sluggish, camera
# rotation stops tracking the mouse, and buttons take a visible moment to
# respond. Holding the animation to this share guarantees the rest of the time
# is there for Qt to deliver events.
#
# The animation degrades instead of the UI: a cell whose frames cost 60 ms
# animates at ~8 fps rather than freezing everything else. A cheap frame costs
# far less than the interval, so small structures still run at the full ~30 fps
# and never notice this.
_FRAME_DUTY = 0.5


def _mode_label(index: int, mode: PhononMode) -> str:
    """One list row: index, frequency and the tags that apply to the mode.

    Composition deliberately stays out of the row — it belongs in the tooltip
    and the summary line, where it has room to be read rather than squeezed
    against the frequency.
    """
    tags = []
    if mode.is_imaginary:
        tags.append("imag")
    if mode.ir_active:
        tags.append("IR")
    if mode.raman_active:
        tags.append("R")
    suffix = f"  [{', '.join(tags)}]" if tags else ""
    return f"{index}: {mode.frequency:9.2f} cm⁻¹{suffix}"


class PhononPanel(QWidget):
    """Choose a phonon mode and animate it in the viewport.

    ``qpoint_selected`` asks the window to load another q-point's modes (the
    view has to be rebuilt around them, which only the window can do), and
    ``tile_requested`` asks it for a supercell — carrying the ``(na, nb, nc)``
    that makes the selected q's wave a whole period.
    """

    mode_selected = Signal(int)
    qpoint_selected = Signal(int)
    tile_requested = Signal(tuple)

    def __init__(self, animator: PhononAnimator, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._animator = animator
        self._modes: Optional[PhononModes] = None
        self._qpoints: list = []  # one entry per sampled q, as fractional coords
        self._supercell = (1, 1, 1)  # the tiling on screen, so Tile can say "done"
        self._restore = None  # what Untile goes back to; None when Tile didn't tile it
        self._equilibrium: Optional[np.ndarray] = None
        self._rows: list[int] = []  # list row -> index into self._modes
        self._numbers: Optional[np.ndarray] = None      # atomic numbers of the geometry
        self._characters: list[ModeCharacter] = []      # per mode, parallel to self._modes

        self._phase = 0.0
        self._speed = 1.0
        # Earliest time the next frame may be drawn, as a perf_counter reading.
        # Set from how long the last frame actually took — see _on_timer.
        self._next_frame_at = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self._on_timer)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Transport controls sit at the top, where they're always in view even
        # if the dock is short enough to clip the bottom of the panel.
        head_row = QHBoxLayout()
        self.play_btn = self._transport_button(QStyle.SP_MediaPlay, "Play", "▶")
        self.play_btn.clicked.connect(self._play)
        self.stop_btn = self._transport_button(QStyle.SP_MediaStop, "Stop", "■")
        self.stop_btn.clicked.connect(self._stop)
        head_row.addWidget(self.play_btn)
        head_row.addWidget(self.stop_btn)
        head_row.addSpacing(6)
        head_row.addWidget(QLabel("Vibrational modes"), 1)
        layout.addLayout(head_row)

        # q-point row: only a dispersion (SCELPHONO) run has anything to choose,
        # so the whole row is hidden for the ordinary Gamma-only calculation.
        self.qpoint_row = QWidget(self)
        qpoint_layout = QHBoxLayout(self.qpoint_row)
        qpoint_layout.setContentsMargins(0, 0, 0, 0)
        qpoint_layout.setSpacing(4)
        qpoint_layout.addWidget(QLabel("q"))
        self.qpoint_box = QComboBox()
        self.qpoint_box.setToolTip(
            "Which q-point's modes to show. A SCELPHONO supercell gives access\n"
            "to the q commensurate with it; Γ is the zone centre, the only\n"
            "q an ordinary frequency calculation reaches."
        )
        self.qpoint_box.currentIndexChanged.connect(self._on_qpoint_changed)
        qpoint_layout.addWidget(self.qpoint_box, 1)
        self.tile_btn = QToolButton(self)
        self.tile_btn.setText("Tile")
        self.tile_btn.setAutoRaise(True)
        self.tile_btn.clicked.connect(self._on_tile_clicked)
        qpoint_layout.addWidget(self.tile_btn)
        layout.addWidget(self.qpoint_row)
        self.qpoint_row.setVisible(False)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Show"))
        self.filter_box = QComboBox()
        self.filter_box.addItems([label for label, _pred in _FILTERS])
        self.filter_box.currentIndexChanged.connect(lambda _i: self._populate())
        filter_row.addWidget(self.filter_box, 1)
        layout.addLayout(filter_row)

        self.mode_list = QListWidget()
        self.mode_list.setMinimumHeight(_LIST_MIN_HEIGHT)
        self.mode_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mode_list.currentRowChanged.connect(self._on_mode_changed)
        layout.addWidget(self.mode_list, 1)  # the only widget that takes the slack

        # What the selected mode is made of. Wraps rather than widening the dock,
        # and keeps its height when empty so the controls below don't jump about
        # as the selection moves between modes with longer and shorter summaries.
        self.character_label = QLabel("")
        self.character_label.setWordWrap(True)
        self.character_label.setStyleSheet("color: palette(mid);")
        self.character_label.setMinimumHeight(_CHARACTER_MIN_HEIGHT)
        self.character_label.setAlignment(Qt.AlignTop)
        self.character_label.setToolTip(
            "Share of the mode's kinetic energy carried by each element,\n"
            "and how many atoms are effectively in motion."
        )
        layout.addWidget(self.character_label)

        # Amplitude and speed share a form so their labels and fields line up.
        controls = QFormLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(4)

        self.amp_box = QDoubleSpinBox()
        self.amp_box.setRange(0.01, 5.0)
        self.amp_box.setSingleStep(0.05)
        self.amp_box.setDecimals(2)
        self.amp_box.setValue(self._animator.amplitude)
        self.amp_box.setToolTip(
            "Peak displacement of the most-displaced atom, in Angstrom.\n"
            "Independent of how many atoms the cell has."
        )
        self.amp_box.valueChanged.connect(self._on_amplitude)
        controls.addRow("Amplitude (Å)", self.amp_box)

        self.speed_box = QDoubleSpinBox()
        self.speed_box.setRange(0.1, 10.0)
        self.speed_box.setSingleStep(0.1)
        self.speed_box.setDecimals(1)
        self.speed_box.setSuffix(" ×")
        self.speed_box.setValue(self._speed)
        self.speed_box.setToolTip(
            "How fast the vibration is played back.\n"
            "1.0× is one full cycle every two seconds; it changes nothing physical."
        )
        self.speed_box.valueChanged.connect(self._on_speed)
        controls.addRow("Speed", self.speed_box)
        layout.addLayout(controls)

        self.setEnabled(False)

    def _transport_button(self, pixmap, tooltip: str, fallback: str) -> QToolButton:
        """A flat icon button for Play/Stop, captioned if the style has no icon."""
        button = QToolButton(self)
        icon = self.style().standardIcon(pixmap)
        if icon.isNull():  # some minimal styles ship no media icons
            button.setText(fallback)
        else:
            button.setIcon(icon)
            button.setIconSize(QSize(18, 18))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setAutoRaise(True)
        return button

    # ── data ────────────────────────────────────────────────────────────
    def set_modes(
        self, equilibrium: np.ndarray, modes: PhononModes, numbers=None
    ) -> None:
        """Load ``modes`` defined on the geometry at ``equilibrium``.

        ``numbers`` are that geometry's atomic numbers; with them each mode is
        labelled by the element carrying it and by how localised it is. Without
        them the list falls back to bare frequencies — the panel stays usable if
        a caller has only positions to hand.
        """
        self._stop()
        # Drop the old rows before repopulating: a stale row would otherwise be
        # read back as a "keep this mode" hint for a different set of modes.
        self._rows = []
        self.mode_list.blockSignals(True)
        self.mode_list.clear()
        self.mode_list.blockSignals(False)
        self._modes = modes
        self._equilibrium = np.asarray(equilibrium, dtype=float)
        self._numbers = None if numbers is None else np.asarray(numbers, dtype=int)
        self._characters = self._analyse(modes)
        self._set_filter_available(modes.has_activity)
        self._populate()
        self.setEnabled(len(modes) > 0)

    def set_qpoints(self, qpoints, current: int = 0) -> None:
        """List the q-points the run sampled, showing ``current`` as selected.

        ``qpoints`` is one fractional q per mode set in load order (``None`` for
        Gamma), i.e. what ``crystalio.load`` returns for a dispersion run. Fewer
        than two and the row hides itself: a Gamma-only frequency calculation has
        no choice to offer. Selecting an entry only *announces* the choice — the
        window reloads that q's modes and hands them back through
        :meth:`set_modes`.
        """
        self._qpoints = list(qpoints or [])
        self.qpoint_box.blockSignals(True)
        self.qpoint_box.clear()
        for i, q in enumerate(self._qpoints):
            self.qpoint_box.addItem(f"{i + 1}: {qpoint_label(q)}")
        if self._qpoints:
            self.qpoint_box.setCurrentIndex(min(max(current, 0), len(self._qpoints) - 1))
        self.qpoint_box.blockSignals(False)
        self.qpoint_row.setVisible(len(self._qpoints) > 1)
        self._update_tile_button()

    def current_qpoint_index(self) -> int:
        """Row of the selected q-point (0, i.e. Gamma, when there is no choice)."""
        return max(self.qpoint_box.currentIndex(), 0)

    def current_qpoint(self):
        """Fractional q of the selected q-point, or ``None`` for Gamma."""
        index = self.current_qpoint_index()
        if 0 <= index < len(self._qpoints):
            return self._qpoints[index]
        return None

    def commensurate_supercell(self) -> tuple:
        """The tiling that shows one whole period of the selected q's wave."""
        return commensurate_repeats(self.current_qpoint())

    def select_qpoint(self, index: int) -> bool:
        """Select q-point ``index``, announcing it as a user choice would.

        Returns whether there was such a q-point. Used to answer a question that
        belongs to a particular q — clicking an IR peak asks about Gamma, so the
        panel goes there rather than reading the click against another q's modes.
        """
        if not 0 <= index < len(self._qpoints):
            return False
        self.qpoint_box.setCurrentIndex(index)  # a no-op if it is already there
        return True

    def _on_qpoint_changed(self, index: int) -> None:
        self._update_tile_button()
        if 0 <= index < len(self._qpoints):
            self.qpoint_selected.emit(index)

    def _on_tile_clicked(self) -> None:
        self.tile_requested.emit(self._tile_target())

    def _tile_target(self) -> tuple:
        """The supercell the button asks for: a whole period, or the way back."""
        reps = self.commensurate_supercell()
        return self._restore if self._offers_untile(reps) else reps

    def _offers_untile(self, reps) -> bool:
        """Whether the button is currently an "undo my tiling" button.

        Tiling has to be a round trip: pick a q, tile it, then move to another q
        (or back to Γ) and put the cell back the way it was. Undo is offered once
        the panel's own tiling is on screen and there is nothing left to tile
        *for the q now selected* — otherwise "Tile 1×1×3" is the more useful
        thing to show, and the restore point is simply kept for afterwards.
        """
        return (
            self._restore is not None
            and self._restore != self._supercell
            and self._shows_whole_period(reps)
        )

    def set_supercell(self, reps, restore=None) -> None:
        """Tell the panel what tiling is on screen, and what Tile would undo to.

        ``restore`` is the supercell the view had before *this panel* tiled it,
        or ``None`` when the tiling on screen is the user's own (set from Cell ▸
        Supercell…, or never changed). Only a tiling the panel caused is one the
        panel offers to take back — the button must not quietly discard a
        supercell someone chose deliberately.
        """
        self._supercell = tuple(int(r) for r in reps)
        self._restore = None if restore is None else tuple(int(r) for r in restore)
        self._update_tile_button()

    def _shows_whole_period(self, reps) -> bool:
        """Whether the tiling on screen already spans whole periods of ``reps``.

        A multiple counts: someone looking at a 2x2x4 cell of a mode that repeats
        every 1x1x4 is already seeing whole waves, and telling them to re-tile
        would shrink their view.
        """
        return all(n % r == 0 for n, r in zip(self._supercell, reps))

    def _update_tile_button(self) -> None:
        """Label the tile button with the supercell it would build — or undo to.

        Nothing to do at Gamma, where every cell moves in phase and one cell
        already tells the whole story, nor once the view already spans a whole
        period the user built themselves. The button stays in place and says so
        rather than disappearing, so the row doesn't change width as q changes.
        """
        reps = self.commensurate_supercell()
        gamma = reps == (1, 1, 1)
        done = self._shows_whole_period(reps)
        if self._offers_untile(reps):
            self.tile_btn.setText("Untile")
            self.tile_btn.setEnabled(True)
            self.tile_btn.setToolTip(
                "Put the cell back to {}×{}×{}, as it was before tiling.".format(*self._restore)
            )
            return
        self.tile_btn.setEnabled(not gamma and not done)
        self.tile_btn.setText("Tile" if gamma else "Tile {}×{}×{}".format(*reps))
        if gamma:
            tip = "At Γ every cell moves in phase — one cell shows the whole mode."
        elif done:
            tip = "The cell on screen already spans a whole period of this q."
        else:
            tip = (
                "Show this mode as the wave it is: repeat the cell "
                "{}×{}×{}, one full period of q.".format(*reps)
            )
        self.tile_btn.setToolTip(tip)

    def _analyse(self, modes: PhononModes) -> list:
        """Composition of every mode, or an empty list if the geometry is unknown.

        Cheap (a few flops per atom per mode) and done once per file, so the list
        and the summary line can be built without re-deriving anything.
        """
        if self._numbers is None:
            return []
        return [mode_character(mode, self._numbers) for mode in modes]

    def character(self, index: int) -> Optional[ModeCharacter]:
        """Composition of mode ``index``, or ``None`` when it wasn't analysed."""
        if 0 <= index < len(self._characters):
            return self._characters[index]
        return None

    def has_mode(self) -> bool:
        """Whether a mode is currently selected (so it can be animated/exported)."""
        return self.current_mode_index() is not None

    def current_mode_index(self) -> Optional[int]:
        """Index into the loaded modes of the selected row, or ``None``.

        The list can be filtered, so the row is *not* the mode index.
        """
        if self._modes is None or self._equilibrium is None:
            return None
        row = self.mode_list.currentRow()
        if row < 0 or row >= len(self._rows):
            return None
        return self._rows[row]

    def current_selection(self):
        """``(equilibrium, PhononMode)`` for the selected mode, or ``None``."""
        index = self.current_mode_index()
        if index is None:
            return None
        return self._equilibrium, self._modes[index]

    def clear(self) -> None:
        """Reset the panel (e.g. after loading a file with no phonons).

        Only the timer is stopped — we deliberately do NOT call the animator's
        ``reset`` here: clear() runs after the geometry has already been redrawn
        (possibly with a different atom count), so pushing the old equilibrium
        back into the renderer would mismatch and is unnecessary.
        """
        self._timer.stop()
        self._animator.clear_mode()  # takes the displacement arrows off the view
        self._modes = None
        self._equilibrium = None
        self._numbers = None
        self._characters = []
        self._rows = []
        self.mode_list.clear()
        self.character_label.clear()
        self._set_filter_available(False)
        self.set_qpoints([])  # the new file's q-points are the window's to supply
        self.setEnabled(False)

    # ── list building ───────────────────────────────────────────────────
    def _set_filter_available(self, available: bool) -> None:
        """Offer the activity filters only when the output labels the modes."""
        self.filter_box.blockSignals(True)
        if not available:
            self.filter_box.setCurrentIndex(0)
        self.filter_box.blockSignals(False)
        self.filter_box.setEnabled(available)
        self.filter_box.setToolTip(
            "" if available else "This output has no IR/Raman activity analysis"
        )

    def _populate(self) -> None:
        """Rebuild the list under the current filter, keeping the selected mode.

        Reselecting is by *mode index*, not row, so switching filters doesn't
        jump to a different mode when the one in view survives the filter.
        """
        keep = self.current_mode_index()
        predicate = _FILTERS[self.filter_box.currentIndex()][1]

        self._rows = []
        self.mode_list.blockSignals(True)
        self.mode_list.clear()
        if self._modes is not None:
            for i, mode in enumerate(self._modes):
                if predicate is not None and not predicate(mode):
                    continue
                character = self.character(i)
                self._rows.append(i)
                self.mode_list.addItem(_mode_label(i, mode))
                if character is not None:
                    self.mode_list.item(self.mode_list.count() - 1).setToolTip(
                        character.summary(limit=6)
                    )
        self.mode_list.blockSignals(False)

        if not self._rows:
            self._stop()  # nothing left to animate under this filter
            self.character_label.clear()
            return
        row = self._rows.index(keep) if keep in self._rows else 0
        # Select quietly, then drive the change by hand: setCurrentRow emits
        # nothing when the row number happens to be unchanged, even though the
        # mode under it may not be.
        self.mode_list.blockSignals(True)
        self.mode_list.setCurrentRow(row)
        self.mode_list.blockSignals(False)
        self._on_mode_changed(row)

    # ── interaction ─────────────────────────────────────────────────────
    def _on_mode_changed(self, row: int) -> None:
        if self._modes is None or row < 0 or row >= len(self._rows):
            return
        index = self._rows[row]
        self._animator.set_mode(self._equilibrium, self._modes[index])
        self._phase = 0.0
        self._animator.set_frame(0.0)
        character = self.character(index)
        self.character_label.setText("" if character is None else character.summary(limit=4))
        self.mode_selected.emit(index)

    def select_mode(self, index: int) -> bool:
        """Select mode ``index`` in the list, returning whether it could be shown.

        The list may be filtered, so the requested mode isn't always on it —
        clicking an IR peak while the list shows only Raman-active modes, say.
        Rather than silently doing nothing, the filter is dropped back to
        "All modes" so the mode the user asked for is the one they get.
        """
        if self._modes is None or not 0 <= index < len(self._modes):
            return False
        if index not in self._rows:
            self.filter_box.setCurrentIndex(0)  # triggers _populate under "All modes"
        if index not in self._rows:
            return False
        self.mode_list.setCurrentRow(self._rows.index(index))
        return True

    def frequencies(self) -> Optional[np.ndarray]:
        """The loaded modes' frequencies in cm⁻¹, or ``None`` if none are loaded."""
        return None if self._modes is None else self._modes.frequencies

    def _on_amplitude(self, value: float) -> None:
        self._animator.amplitude = value

    def _on_speed(self, value: float) -> None:
        self._speed = value

    def stop(self) -> None:
        """Public stop — called when the user starts editing the structure."""
        self._stop()

    def invalidate_on_edit(self, positions: np.ndarray) -> None:
        """Reconcile the panel after a structural edit.

        Stops a running animation. If the atom count changed, the loaded modes
        are stale and dropped. Otherwise the animation is re-anchored to the
        *edited* geometry — crucially WITHOUT resetting atoms to the old
        equilibrium, which would snap them back and undo the edit just made
        (e.g. an atom dropped after a drag reappearing at its original spot).
        """
        if self._modes is None:
            return
        positions = np.asarray(positions, dtype=float)
        # Stop the timer only — do NOT animator.reset() to the stale equilibrium.
        self._timer.stop()
        if self._equilibrium is None or len(self._equilibrium) != len(positions):
            self.clear()  # atom count changed -> modes no longer apply
            return
        self._equilibrium = positions
        index = self.current_mode_index()
        if index is not None:
            self._animator.set_mode(self._equilibrium, self._modes[index])

    def _play(self) -> None:
        if self._modes is not None and self.has_mode():
            self._next_frame_at = 0.0  # draw the first frame straight away
            self._timer.start()

    def _stop(self) -> None:
        self._timer.stop()
        # Back to phase 0 as well as to the equilibrium geometry, so the next
        # Play starts from rest instead of jumping into mid-cycle.
        self._phase = 0.0
        self._animator.reset()

    def _on_timer(self) -> None:
        """Draw a frame, unless doing so would crowd the UI off the event loop.

        The timer keeps its steady ~30 fps beat; this decides whether each beat
        becomes a frame. After a frame costing ``t`` the next one is held off for
        ``t * (1/_FRAME_DUTY - 1)``, so the animation never occupies more than
        :data:`_FRAME_DUTY` of the loop and input always has room. Skipped beats
        return in microseconds, which is the point — that is the idle time.

        Kept separate from :meth:`_tick` so the pacing and the frame itself can be
        reasoned about (and tested) independently.
        """
        from time import perf_counter

        if perf_counter() < self._next_frame_at:
            return  # yield this beat to the event loop
        started = perf_counter()
        self._tick()
        cost = perf_counter() - started
        self._next_frame_at = perf_counter() + cost * (1.0 / _FRAME_DUTY - 1.0)

    def _tick(self) -> None:
        """Draw one frame and advance the phase (no pacing — see :meth:`_on_timer`)."""
        self._animator.set_frame(self._phase)
        self._phase = (self._phase + _PHASE_STEP * self._speed) % (2.0 * math.pi)


__all__ = ["PhononPanel"]
