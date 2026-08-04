"""Display panel: tune the 3D appearance live (VESTA-style).

A dockable panel (not a modal dialog) so it sits alongside the view and updates
it as controls change. Every edit rebuilds a fresh immutable
:class:`RenderSettings` and streams it back through the ``on_change`` callback.

Continuous values (sizes, opacities) pair a slider with a spin box so they can
be dragged for feel or typed for precision; colours use a swatch button backed
by the native colour picker.
"""

from __future__ import annotations

from typing import Callable, Optional

from ase.data import chemical_symbols
from ase.data.colors import jmol_colors
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from crystalline.viz.render_settings import RenderSettings


# Probability levels offered for the ellipsoid surface. 50% is what ORTEP and
# every structure report use; the higher ones are for seeing a small tensor.
_ADP_PROBABILITIES = (0.50, 0.75, 0.90, 0.99)

# How long a *continuous* control (a slider or its spin box) waits for the drag
# to settle before reporting the change. A slider spans 1000 steps, so dragging
# one across its range emits hundreds of times, and each of those can cost the
# renderer a full scene rebuild — dragging "Atom size" on a 1728-atom cell used
# to take 16 s for ten ticks. Coalescing to ~16 updates a second still tracks the
# thumb closely enough to feel live while collapsing the rest.
#
# Only the continuous rows debounce. Checkboxes, combos and colour pickers fire
# once per click, so waiting would add latency for nothing; they emit
# immediately, flushing any pending tick first so changes never arrive reordered.
_SLIDER_SETTLE_MS = 60


def _closest_index(values, target: float) -> int:
    """The entry of ``values`` nearest ``target`` — settings needn't be exact."""
    return min(range(len(values)), key=lambda i: abs(values[i] - target))


def _jmol_hex(z: int) -> str:
    """The default Jmol colour of element ``z`` as ``#rrggbb``."""
    r, g, b = (int(round(c * 255)) for c in jmol_colors[int(z)])
    return f"#{r:02x}{g:02x}{b:02x}"


class DisplayPanel(QWidget):
    """Edit :class:`RenderSettings` in a dock and stream changes back live."""

    def __init__(
        self,
        settings: RenderSettings,
        on_change: Callable[[RenderSettings], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._on_change = on_change
        self._loading = True
        # Coalesces the stream of changes a slider drag produces (see
        # _SLIDER_SETTLE_MS). Single-shot: each tick restarts the wait, so the
        # emit happens once the drag pauses.
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(_SLIDER_SETTLE_MS)
        self._settle.timeout.connect(self._emit)
        self._bg_color = settings.background_color
        self._measure_point_color = settings.measure_point_color
        self._measure_line_color = settings.measure_line_color
        self._measure_plane_color = settings.measure_plane_color
        self._symmetry_axis_color = settings.symmetry_axis_color
        self._symmetry_plane_color = settings.symmetry_plane_color
        self._symmetry_point_color = settings.symmetry_point_color
        self._mode_arrow_color = settings.mode_arrow_color
        # Which temperature to select the first time a file brings some. After
        # that the picker's own index carries across files.
        self._initial_adp_index = int(settings.adp_temperature_index)
        # Per-element colour overrides {Z: "#rrggbb"} and their swatch buttons.
        self._atom_colors: dict = {int(z): c for z, c in settings.atom_colors}
        self._elem_buttons: dict = {}

        # A scroll area keeps the (deliberately generous) set of controls usable
        # even in a short dock.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer.addWidget(scroll)

        body = QWidget()
        scroll.setWidget(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Atoms ──
        atoms = self._group(layout, "Atoms")
        self._atom_scale = self._float_row(atoms, "Size", settings.atom_scale, 0.1, 2.0, 0.05)
        self._atom_opacity = self._float_row(atoms, "Opacity", settings.atom_opacity, 0.05, 1.0, 0.05)
        self._show_labels = self._check(atoms, "Element labels", settings.show_atom_labels)
        self._label_size = self._int_row(atoms, "Label size", settings.atom_label_size, 6, 40)

        # ── Element colours (per-element swatches; populated per structure) ──
        elem_group = QGroupBox("Element colours")
        elem_outer = QVBoxLayout(elem_group)
        self._elem_form = QFormLayout()
        self._elem_form.setLabelAlignment(Qt.AlignRight)
        elem_outer.addLayout(self._elem_form)
        self._elem_hint = QLabel("Load a structure to recolour its elements.")
        self._elem_hint.setEnabled(False)
        elem_outer.addWidget(self._elem_hint)
        reset_btn = QPushButton("Reset to default colours")
        reset_btn.clicked.connect(self._reset_element_colors)
        elem_outer.addWidget(reset_btn)
        layout.addWidget(elem_group)

        # ── Bonds ──
        bonds = self._group(layout, "Bonds")
        self._show_bonds = self._check(bonds, "Show bonds", settings.show_bonds)
        self._bond_radius = self._float_row(bonds, "Radius (Å)", settings.bond_radius, 0.02, 1.2, 0.02)
        self._bond_tol = self._float_row(bonds, "Tolerance", settings.bond_tolerance, 1.0, 2.0, 0.05)
        self._show_hbonds = self._check(
            bonds, "Show hydrogen bonds", settings.show_hydrogen_bonds
        )

        # ── Cell & axes ──
        cell = self._group(layout, "Cell & axes")
        self._show_cell = self._check(cell, "Cell edges", settings.show_cell)
        self._show_axes = self._check(cell, "a/b/c gizmo", settings.show_lattice_vectors)
        self._show_orient = self._check(cell, "Orientation marker", settings.show_orientation_axes)

        # ── Polyhedra ──
        poly = self._group(layout, "Coordination polyhedra")
        self._show_poly = self._check(poly, "Show polyhedra", settings.show_polyhedra)
        self._poly_opacity = self._float_row(poly, "Opacity", settings.polyhedra_opacity, 0.05, 1.0, 0.05)
        self._poly_min = self._int_row(poly, "Min. coordination", settings.polyhedra_min_vertices, 3, 12)

        # ── Thermal ellipsoids ── (ADP; only meaningful for a run that has them)
        adp = self._group(layout, "Thermal ellipsoids (ADP)")
        self._show_adp = self._check(adp, "Show ellipsoids", settings.show_adp_ellipsoids)
        self._adp_temp = self._combo(adp, "Temperature", [], 0)
        self._adp_probability = self._combo(
            adp, "Probability",
            [f"{int(p * 100)} %" for p in _ADP_PROBABILITIES],
            _closest_index(_ADP_PROBABILITIES, settings.adp_probability),
        )
        self._adp_probability.setToolTip(
            "Fraction of the displacement distribution the drawn surface encloses.\n"
            "50% is the crystallographic convention (ORTEP)."
        )
        self._adp_opacity = self._float_row(adp, "Opacity", settings.adp_opacity, 0.05, 1.0, 0.05)
        self._adp_group = adp
        self.set_adp_temperatures([])  # nothing loaded yet

        # ── Phonon arrows ── (the selected mode's eigenvector, drawn on the atoms)
        arrows = self._group(layout, "Phonon displacement arrows")
        self._show_arrows = self._check(arrows, "Show arrows", settings.show_mode_arrows)
        self._arrow_scale = self._float_row(
            arrows, "Arrow length (Å)", settings.mode_arrow_scale, 0.2, 5.0, 0.1
        )
        self._arrow_proportional = self._check(
            arrows, "Scale by displacement", settings.mode_arrow_proportional
        )
        self._arrow_proportional.setToolTip(
            "Draw each arrow as long as that atom's displacement, the longest at\n"
            "the length above — instead of one length for every atom.\n"
            "This is what shows a mode away from Γ as a wave in a still image:\n"
            "its cells differ in phase, so only the arrow lengths tell them apart."
        )
        self._arrow_phase_colors = self._check(
            arrows, "Colour by phase (q ≠ Γ)", settings.mode_arrow_phase_colors
        )
        self._arrow_phase_colors.setToolTip(
            "Colour each arrow by the phase of the cell its atom is in, cycling\n"
            "once per wavelength — the one thing that distinguishes the cells of\n"
            "a travelling wave, whose amplitude is the same in every one.\n"
            "Replaces the colour below; nothing to show at Γ."
        )
        self._arrow_color_btn = self._color_row(arrows, "Colour", "_mode_arrow_color")

        # ── Measurements ── (Geometry panel overlays: dots, paths, plane patches)
        measure = self._group(layout, "Measurements")
        self._measure_point_btn = self._color_row(measure, "Dots", "_measure_point_color")
        self._measure_line_btn = self._color_row(measure, "Lines", "_measure_line_color")
        self._measure_plane_btn = self._color_row(measure, "Planes", "_measure_plane_color")

        # ── Point symmetry ── (Point symmetry panel overlays: axes, planes, centre)
        symmetry = self._group(layout, "Point symmetry")
        self._symmetry_axis_btn = self._color_row(
            symmetry, "Rotation axes", "_symmetry_axis_color"
        )
        self._symmetry_plane_btn = self._color_row(
            symmetry, "Mirror planes", "_symmetry_plane_color"
        )
        self._symmetry_point_btn = self._color_row(
            symmetry, "Inversion centre", "_symmetry_point_color"
        )

        # ── Scene ──
        scene = self._group(layout, "Scene")
        self._bg_color_btn = self._color_row(scene, "Background", "_bg_color")
        self._projection = self._combo(
            scene, "Projection", ["Perspective", "Parallel (orthographic)"],
            1 if settings.parallel_projection else 0,
        )

        layout.addStretch(1)
        self._loading = False

    # ── section + widget builders (each wired to emit on change) ────────
    def _group(self, layout: QVBoxLayout, title: str) -> QFormLayout:
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)
        layout.addWidget(box)
        return form

    def _float_row(self, form, label, value, lo, hi, step) -> QDoubleSpinBox:
        """A slider + spin box bound together over ``[lo, hi]``."""
        box = QDoubleSpinBox()
        box.setRange(lo, hi)
        box.setSingleStep(step)
        box.setDecimals(2)
        box.setValue(value)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 1000)

        def to_slider(v: float) -> int:
            return int(round((v - lo) / (hi - lo) * 1000))

        def from_slider(s: int) -> float:
            return lo + (s / 1000.0) * (hi - lo)

        slider.setValue(to_slider(value))
        guard = {"lock": False}

        def on_box(v: float) -> None:
            if guard["lock"]:
                return
            guard["lock"] = True
            slider.setValue(to_slider(v))
            guard["lock"] = False
            self._emit_soon()

        def on_slider(s: int) -> None:
            if guard["lock"]:
                return
            guard["lock"] = True
            box.setValue(from_slider(s))
            guard["lock"] = False
            self._emit_soon()

        box.valueChanged.connect(on_box)
        slider.valueChanged.connect(on_slider)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(slider, 1)
        h.addWidget(box)
        form.addRow(label, row)
        return box

    def _int_row(self, form, label, value, lo, hi) -> QSpinBox:
        # Debounced like the sliders: holding a spin box's arrow auto-repeats, and
        # "Min. coordination" re-runs the coordination analysis on every step.
        box = QSpinBox()
        box.setRange(lo, hi)
        box.setValue(value)
        box.valueChanged.connect(self._emit_soon)
        form.addRow(label, box)
        return box

    def _check(self, form, label, value) -> QCheckBox:
        box = QCheckBox()
        box.setChecked(value)
        box.toggled.connect(self._emit)
        form.addRow(label, box)
        return box

    def _combo(self, form, label, items, index) -> QComboBox:
        box = QComboBox()
        box.addItems(items)
        box.setCurrentIndex(index)
        box.currentIndexChanged.connect(self._emit)
        form.addRow(label, box)
        return box

    def _color_row(self, form, label, attr: str) -> QPushButton:
        """A swatch button that opens the colour picker; writes to ``self.<attr>``."""
        button = QPushButton()
        button.setFixedWidth(64)
        self._paint_swatch(button, getattr(self, attr))
        button.clicked.connect(lambda: self._pick_color(button, attr))
        form.addRow(label, button)
        return button

    def _pick_color(self, button: QPushButton, attr: str) -> None:
        current = QColor(getattr(self, attr))
        chosen = QColorDialog.getColor(current, self, "Choose colour")
        if chosen.isValid():
            setattr(self, attr, chosen.name())
            self._paint_swatch(button, chosen.name())
            self._emit()

    @staticmethod
    def _paint_swatch(button: QPushButton, color: str) -> None:
        button.setText(color)
        # A readable text colour on top of the swatch (dark text on light fills).
        c = QColor(color)
        text = "#000000" if c.lightnessF() > 0.5 else "#ffffff"
        button.setStyleSheet(f"background-color: {color}; color: {text}; padding: 3px;")

    # ── thermal ellipsoids (rebuilt for each file's temperatures) ───────
    def set_adp_temperatures(self, labels, autoshow: bool = False) -> None:
        """List the temperatures the loaded run reported ADPs for.

        An empty list means the file has none, and the whole group is disabled
        rather than hidden: the controls stay where the user learnt they are,
        greyed out with a reason.

        ``autoshow`` switches the ellipsoids on — a file that went to the trouble
        of computing ADPs is a file whose ADPs you want to see. Only the file
        loader passes it; a cell-view or supercell change re-lists the same
        temperatures and must not overrule a user who has switched them off.
        """
        labels = list(labels)
        keep = self._adp_temp.currentIndex()
        if keep < 0:  # never populated: fall back to the index the settings asked for
            keep = self._initial_adp_index
        # Save and restore rather than clear: this also runs from __init__, where
        # the panel is still being built and an emit would read widgets that
        # don't exist yet.
        was_loading = self._loading
        self._loading = True
        self._adp_temp.clear()
        self._adp_temp.addItems(labels)
        if labels:
            self._adp_temp.setCurrentIndex(min(max(keep, 0), len(labels) - 1))
        self._loading = was_loading

        available = bool(labels)
        for row in range(self._adp_group.rowCount()):
            for role in (QFormLayout.LabelRole, QFormLayout.FieldRole):
                item = self._adp_group.itemAt(row, role)
                if item is not None and item.widget() is not None:
                    item.widget().setEnabled(available)
        self._show_adp.setToolTip(
            "" if available else "This output has no ADP data (needs the ADP keyword)"
        )
        # Both of these emit, which is the point: one clears stale ellipsoids,
        # the other draws the new file's.
        if not available and self._show_adp.isChecked():
            self._show_adp.setChecked(False)
        elif available and autoshow and not self._show_adp.isChecked():
            self._show_adp.setChecked(True)

    # ── per-element colours (rebuilt for each structure's elements) ─────
    def set_elements(self, numbers) -> None:
        """Show a colour swatch per distinct element in the current structure."""
        zs = sorted({int(z) for z in numbers})
        while self._elem_form.rowCount():
            self._elem_form.removeRow(0)
        self._elem_buttons = {}
        self._elem_hint.setVisible(not zs)
        for z in zs:
            button = QPushButton()
            button.setFixedWidth(64)
            self._paint_swatch(button, self._atom_colors.get(z, _jmol_hex(z)))
            button.clicked.connect(lambda _c=False, zz=z, b=button: self._pick_element_color(zz, b))
            self._elem_buttons[z] = button
            self._elem_form.addRow(chemical_symbols[z], button)

    def _pick_element_color(self, z: int, button: QPushButton) -> None:
        current = QColor(self._atom_colors.get(z, _jmol_hex(z)))
        chosen = QColorDialog.getColor(current, self, f"Colour for {chemical_symbols[z]}")
        if chosen.isValid():
            self._atom_colors[z] = chosen.name()
            self._paint_swatch(button, chosen.name())
            self._emit()

    def _reset_element_colors(self) -> None:
        self._atom_colors = {}
        for z, button in self._elem_buttons.items():
            self._paint_swatch(button, _jmol_hex(z))
        self._emit()

    def _emit_soon(self) -> None:
        """Report a continuous control's change once the drag settles."""
        if self._loading:
            return
        self._settle.start()  # restarts the wait if one is already running

    def _emit(self) -> None:
        # A discrete control emitting now must not be overtaken by a slider tick
        # still sitting on the timer, which would push the older settings last.
        self._settle.stop()
        if self._loading:
            return
        self._on_change(
            RenderSettings(
                atom_scale=self._atom_scale.value(),
                atom_opacity=self._atom_opacity.value(),
                atom_colors=tuple(sorted((int(z), c) for z, c in self._atom_colors.items())),
                show_atom_labels=self._show_labels.isChecked(),
                atom_label_size=self._label_size.value(),
                show_bonds=self._show_bonds.isChecked(),
                bond_radius=self._bond_radius.value(),
                bond_tolerance=self._bond_tol.value(),
                show_hydrogen_bonds=self._show_hbonds.isChecked(),
                show_cell=self._show_cell.isChecked(),
                show_lattice_vectors=self._show_axes.isChecked(),
                show_orientation_axes=self._show_orient.isChecked(),
                show_polyhedra=self._show_poly.isChecked(),
                polyhedra_opacity=self._poly_opacity.value(),
                polyhedra_min_vertices=self._poly_min.value(),
                show_adp_ellipsoids=self._show_adp.isChecked(),
                adp_probability=_ADP_PROBABILITIES[self._adp_probability.currentIndex()],
                adp_opacity=self._adp_opacity.value(),
                adp_temperature_index=max(self._adp_temp.currentIndex(), 0),
                show_mode_arrows=self._show_arrows.isChecked(),
                mode_arrow_scale=self._arrow_scale.value(),
                mode_arrow_proportional=self._arrow_proportional.isChecked(),
                mode_arrow_phase_colors=self._arrow_phase_colors.isChecked(),
                mode_arrow_color=self._mode_arrow_color,
                measure_point_color=self._measure_point_color,
                measure_line_color=self._measure_line_color,
                measure_plane_color=self._measure_plane_color,
                symmetry_axis_color=self._symmetry_axis_color,
                symmetry_plane_color=self._symmetry_plane_color,
                symmetry_point_color=self._symmetry_point_color,
                background_color=self._bg_color,
                parallel_projection=self._projection.currentIndex() == 1,
            )
        )


__all__ = ["DisplayPanel"]
