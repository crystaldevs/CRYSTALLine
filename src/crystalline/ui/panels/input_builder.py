"""A dialog that authors a CRYSTAL ``.d12`` input deck from the current structure.

The tabbed form on the left drives
:func:`crystalline.core.crystal_input.build_input`; the pane on the right shows
the exact deck live, so the picky CRYSTAL format can be eyeballed before it is
written. Geometry is derived from the structure (space group + asymmetric unit),
so the form only covers the *choices* — method, basis, SCF and task — plus a
free-text escape hatch per block for anything not exposed yet.

Numeric fields that CRYSTAL gives a default for are "optional spin boxes": their
minimum doubles as an *unset* sentinel (shown as "Default"), and unset fields are
simply not written into the deck rather than being pinned to a guessed value.

All the deck logic lives in ``core.crystal_input`` (Qt-free and unit-tested);
this module is only the widget wiring.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase, QFontMetrics
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QAbstractSpinBox,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from crystalline.core.crystal_input import (
    FUNCTIONAL_ALIASES,
    FUNCTIONAL_GROUPS,
    GRIDS,
    INTERNAL_BASIS_SETS,
    CORRELATION_FUNCTIONALS,
    EXCHANGE_FUNCTIONALS,
    NONCOLLINEAR_MODES,
    TOLINTEG_LABELS,
    TWO_COMPONENT_GUESSES,
    BasisOptions,
    CphfOptions,
    CrystalInputError,
    CrystalInputSpec,
    AnhapesOptions,
    AnharmOptions,
    DispersionOptions,
    ElasticOptions,
    EosOptions,
    FreqOptions,
    GeometryOptions,
    MethodOptions,
    OptGeomOptions,
    QhaOptions,
    ScfOptions,
    TaskOptions,
    TwoComponentOptions,
    build_input,
    suggest_shrink,
    tolinteg_warnings,
    write_input,
)
from crystalline.core.structure import Structure
from crystalline.ui.panels.band_path_editor import BandPathEditor

_GRID_DEFAULT_LABEL = "Default"
# Sentinels for the split-functional combos: both "unset" states are meaningful
# CRYSTAL defaults, not omissions (§4.1).
_HF_EXCHANGE = "Hartree–Fock exchange"
_NO_CORRELATION = "None (exchange only)"
_UNSET = "Default"  # shown when an optional numeric field is left unset

# (label shown in the combo, CRYSTAL task kind)
_TASKS = (
    ("Single point", "SP"),
    ("Geometry optimisation", "OPTGEOM"),
    ("Frequencies (FREQCALC)", "FREQCALC"),
    ("Equation of state (EOS)", "EOS"),
    ("Elastic constants (ELASTCON)", "ELASTCON"),
    ("Dielectric response (CPHF)", "CPHF"),
    ("Phonon dispersion (DISPERSI)", "DISPERSION"),
    ("Quasi-harmonic (QHA)", "QHA"),
    ("Anharmonic lattice dynamics", "ANHAPES"),
    ("Anharmonic X–H stretch (ANHARM)", "ANHARM"),
)
_EQUILIBRIUM_TASKS = {"FREQCALC", "EOS", "ELASTCON", "DISPERSION", "QHA", "ANHAPES"}

# IR intensity techniques (manual §8.3). Raman forces INTCPHF regardless.
_IR_TECHNIQUES = (
    ("Berry phase (INTPOL)", "INTPOL"),
    ("Wannier functions (INTLOC)", "INTLOC"),
    ("CPHF/CPKS (INTCPHF)", "INTCPHF"),
)
_INTCPHF_INDEX = 2  # position of INTCPHF in _IR_TECHNIQUES


# ── small widget helpers ───────────────────────────────────────────────────
def _fit_special_text(box) -> None:
    """Widen a spin box so its "unset" text is not clipped.

    Qt derives a spin box's width from the numeric range and suffix and ignores
    ``specialValueText`` entirely, so the sentinel word gets cut off unless the
    minimum width is set from the text itself.
    """
    metrics = QFontMetrics(box.font())
    box.setMinimumWidth(metrics.horizontalAdvance(_UNSET) + 46)  # + arrows and padding


def _optional_spin(minimum: int, maximum: int, suffix: str = "") -> QSpinBox:
    """An integer spin box whose minimum means "leave CRYSTAL's default alone"."""
    box = QSpinBox()
    box.setRange(minimum - 1, maximum)
    box.setValue(minimum - 1)
    box.setSpecialValueText(_UNSET)
    if suffix:
        box.setSuffix(suffix)
    _fit_special_text(box)
    return box


def _optional_double(minimum: float, maximum: float, decimals: int, step: float) -> QDoubleSpinBox:
    """A float spin box whose minimum means "leave CRYSTAL's default alone"."""
    box = QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(minimum - step, maximum)
    box.setSingleStep(step)
    box.setValue(minimum - step)
    box.setSpecialValueText(_UNSET)
    _fit_special_text(box)
    return box


def _value_of(box) -> Optional[float]:
    """The spin box's value, or ``None`` when it is parked on its unset sentinel."""
    if box.value() <= box.minimum():
        return None
    return box.value()


def _int_of(box) -> Optional[int]:
    value = _value_of(box)
    return None if value is None else int(value)


class _CphfGroup(QGroupBox):
    """The CPHF/CPKS convergence controls, reused by the CPHF task and INTCPHF.

    The ``*2`` fields drive the second coupled-perturbed cycle, which only runs
    for fourth-order calculations and for Raman intensities; the owner enables
    them via :meth:`set_second_order_enabled`.
    """

    def __init__(self, title: str = "CPHF convergence", with_order: bool = False) -> None:
        super().__init__(title)
        form = _form()
        self.setLayout(form)

        self.order: Optional[QComboBox] = None
        if with_order:
            self.order = QComboBox()
            self.order.addItems(["Second order (polarisability)", "Third order", "Fourth order"])
            form.addRow("Perturbative order", self.order)

        self.fmixing = _optional_spin(0, 100, " %")
        self.maxcycle = _optional_spin(1, 999)
        self.tolalpha = _optional_spin(1, 12)
        form.addRow("FMIXING", self.fmixing)
        form.addRow("MAXCYCLE", self.maxcycle)
        form.addRow("TOLALPHA (10⁻ⁿ)", self.tolalpha)

        self.fmixing2 = _optional_spin(0, 100, " %")
        self.maxcycle2 = _optional_spin(1, 999)
        self.tolgamma = _optional_spin(1, 12)
        form.addRow("FMIXING2", self.fmixing2)
        form.addRow("MAXCYCLE2", self.maxcycle2)
        form.addRow("TOLGAMMA (10⁻ⁿ)", self.tolgamma)

        self.extra = QPlainTextEdit()
        self.extra.setPlaceholderText("Extra CPHF keywords, one per line")
        self.extra.setFixedHeight(48)
        form.addRow("Extra", self.extra)

    def set_second_order_enabled(self, enabled: bool) -> None:
        for widget in (self.fmixing2, self.maxcycle2, self.tolgamma):
            widget.setEnabled(enabled)

    def options(self) -> CphfOptions:
        order = "SECOND"
        if self.order is not None:
            order = ("SECOND", "THIRD", "FOURTH")[self.order.currentIndex()]
        return CphfOptions(
            order=order,
            fmixing=_int_of(self.fmixing),
            maxcycle=_int_of(self.maxcycle),
            tolalpha=_int_of(self.tolalpha),
            fmixing2=_int_of(self.fmixing2),
            maxcycle2=_int_of(self.maxcycle2),
            tolgamma=_int_of(self.tolgamma),
            extra_keywords=self.extra.toPlainText(),
        )

    def widgets(self) -> list:
        found = [self.fmixing, self.maxcycle, self.tolalpha,
                 self.fmixing2, self.maxcycle2, self.tolgamma]
        return found + ([self.order] if self.order is not None else [])


class _SupercellGroup(QGroupBox):
    """The SCELPHONO expansion matrix used by dispersion and QHA runs.

    Phonons away from Gamma are obtained from a direct-space supercell, so this
    is the control that decides which q points a dispersion run actually samples.
    """

    def __init__(
        self, checked: bool = False, diagonal: int = 2, title: str = "Supercell (SCELPHONO)"
    ) -> None:
        super().__init__(title)
        self.setCheckable(True)
        self.setChecked(checked)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 6, 10, 10)
        hint = QLabel("Expansion matrix; rows are the new cell vectors.")
        hint.setWordWrap(True)
        outer.addWidget(hint)

        self.cells = []
        for r in range(3):
            row = QHBoxLayout()
            for c in range(3):
                # A diagonal expansion, so switching the group on gives a real
                # supercell rather than the no-op identity.
                box = _plain_spin(diagonal if r == c else 0, -9, 9)
                box.setFixedWidth(56)
                row.addWidget(box)
                self.cells.append(box)
            row.addStretch(1)
            outer.addLayout(row)
        self._outer = outer

    def add_row(self, widget: QWidget) -> None:
        """Append an extra control below the matrix (e.g. SUPERCEL's NOSHIFT)."""
        self._outer.addWidget(widget)

    def matrix(self) -> Optional[list]:
        """The 3×3 matrix, or ``None`` when the group is switched off."""
        if not self.isChecked():
            return None
        values = [box.value() for box in self.cells]
        return [values[0:3], values[3:6], values[6:9]]


class InputBuilderDialog(QDialog):
    """Build and save a CRYSTAL ``.d12`` deck for ``structure``."""

    def __init__(self, structure: Structure, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Build CRYSTAL input")
        self._structure = structure
        self._technique_before_raman: Optional[int] = None
        self._syncing = False

        self._build_widgets()
        self._connect_signals()
        self._sync_enabled()
        self._refresh_preview()

    # ── construction ────────────────────────────────────────────────────
    def _build_widgets(self) -> None:
        # Each tab scrolls: the task pages (FREQCALC especially) are taller than
        # any sensible dialog, and without this their size hint would force the
        # window open at full height and refuse to shrink below it.
        tabs = QTabWidget()
        tabs.addTab(_scrollable(self._method_tab()), "Method")
        tabs.addTab(_scrollable(self._scf_tab()), "SCF")
        tabs.addTab(_scrollable(self._task_tab()), "Task")
        tabs.setMinimumWidth(320)

        self._preview = QPlainTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self._preview.setMinimumWidth(240)
        self._preview.setLineWrapMode(QPlainTextEdit.NoWrap)

        # A splitter (rather than a fixed layout) lets the form/preview divider
        # be dragged, so a long deck can be given the whole window if wanted.
        columns = QSplitter(Qt.Horizontal)
        columns.addWidget(tabs)
        columns.addWidget(self._preview)
        columns.setStretchFactor(0, 0)
        columns.setStretchFactor(1, 1)
        columns.setChildrenCollapsible(False)

        self._save_btn = QPushButton("Save .d12…")
        self._save_btn.setDefault(True)
        close_btn = QPushButton("Close")
        self._save_btn.clicked.connect(self._save)
        close_btn.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        buttons.addWidget(self._save_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(columns, 1)
        layout.addLayout(buttons)

        # Only now do the tabs belong to this dialog, so this is the earliest
        # point findChildren sees them. Spin boxes stretched to the full form
        # width read as unbalanced next to the text fields, so cap them all.
        for box in self.findChildren(QAbstractSpinBox):
            box.setMaximumWidth(150)
        self.resize(880, 560)  # comfortable start; the dialog stays freely resizable

    def _method_tab(self) -> QWidget:
        form = _form()
        self._title = QLineEdit("Generated by CRYSTALLine")
        form.addRow("Title", self._title)

        self._method = QComboBox()
        self._method.addItems(["DFT", "Hartree–Fock"])
        form.addRow("Method", self._method)

        self._functional_mode = QComboBox()
        self._functional_mode.addItems(
            ["One combined keyword", "Separate exchange + correlation"]
        )
        form.addRow("Functional given as", self._functional_mode)

        # Grouped, and each entry names the functional as well as the keyword:
        # PBE's stand-alone keyword is PBEXC, so a flat list of keywords hid the
        # most-wanted functional in the set behind a spelling nobody looks for.
        self._functional = QComboBox()
        self._functional.setEditable(True)  # any CRYSTAL keyword is still allowed
        _fill_functionals(self._functional)
        self._functional.setCurrentIndex(self._functional.findData("PBE0"))
        self._functional.setToolTip(
            "The keyword written into the DFT block. Typing is allowed for anything "
            "not listed — and common spellings are accepted: PBE, PBEsol, SOGGA and "
            "LDA are translated to the stand-alone keywords CRYSTAL wants."
        )
        form.addRow("Functional", self._functional)

        self._exchange = QComboBox()
        self._exchange.setEditable(True)
        self._exchange.addItem(_HF_EXCHANGE)  # unset -> CRYSTAL keeps HF exchange
        self._exchange.addItems(EXCHANGE_FUNCTIONALS)
        form.addRow("EXCHANGE", self._exchange)

        self._correlation = QComboBox()
        self._correlation.setEditable(True)
        self._correlation.addItem(_NO_CORRELATION)  # unset -> exchange-only functional
        self._correlation.addItems(CORRELATION_FUNCTIONALS)
        form.addRow("CORRELAT", self._correlation)

        self._hybrid = _optional_spin(0, 100, " %")
        self._hybrid.setToolTip(
            "HYBRID: percentage of exact Hartree–Fock exchange. Also overrides the "
            "fraction built into an existing global-hybrid functional."
        )
        form.addRow("HYBRID", self._hybrid)

        self._nonlocal = QCheckBox("Non-local weights (NONLOCAL)")
        self._nonlocal.setToolTip(
            "Weights of the non-local exchange and correlation parts. With HYBRID "
            "these reproduce Becke 3-parameter functionals — B3LYP is BECKE/LYP "
            "with HYBRID 20 and NONLOCAL 0.9 0.81."
        )
        form.addRow(self._nonlocal)
        self._nonlocal_b = _plain_double(0.9, 0.01, 3, maximum=2.0)
        self._nonlocal_c = _plain_double(0.81, 0.01, 3, maximum=2.0)
        form.addRow("Exchange weight B", self._nonlocal_b)
        form.addRow("Correlation weight C", self._nonlocal_c)

        self._d3 = QCheckBox("Grimme DFT-D3 dispersion")
        form.addRow(self._d3)

        self._grid = QComboBox()
        self._grid.addItem(_GRID_DEFAULT_LABEL)
        self._grid.addItems(GRIDS)
        form.addRow("Integration grid", self._grid)

        self._basis = QComboBox()
        self._basis.addItems(INTERNAL_BASIS_SETS)
        form.addRow("Basis set", self._basis)

        self._symmetry = QCheckBox("Reduce to the asymmetric unit")
        self._symmetry.setChecked(True)
        form.addRow(self._symmetry)

        form.addRow(_gap())
        self._supercel = _SupercellGroup(title="Supercell (SUPERCEL)")
        self._supercel.setToolTip(
            "New cell vectors as linear combinations of the primitive ones. For a "
            "phonon dispersion or QHA run use that task's SCELPHONO instead."
        )
        self._supercel_noshift = QCheckBox("Keep the origin (NOSHIFT)")
        self._supercel.add_row(self._supercel_noshift)
        form.addRow(self._supercel)

        form.addRow(_gap())
        form.addRow(self._two_component_group())
        return _page(form)

    def _two_component_group(self) -> QGroupBox:
        """TWOCOMPON: the two-component spinor SCF that spin-orbit coupling needs."""
        group = QGroupBox("Spin–orbit coupling (TWOCOMPON)")
        group.setCheckable(True)
        group.setChecked(False)
        self._twocompon = group
        form = _form()
        group.setLayout(form)

        note = QLabel(
            "A two-component SCF gives single-point energies only, with pure or "
            "global-hybrid LDA/GGA functionals. It also switches off DIIS and symmetry."
        )
        note.setWordWrap(True)
        form.addRow(note)

        self._soc = QCheckBox("Include the spin–orbit operator (SOC)")
        self._soc.setChecked(True)
        self._soc.setToolTip(
            "Needs a SOREP operator from an ECP — one of CRYSTAL's internal "
            "libraries (STUTSC/STUTLC/STUTSH, COLUSC/COLULC/COLUSH) or INPSOC."
        )
        form.addRow(self._soc)

        self._noncollinear = QComboBox()
        self._noncollinear.addItems(NONCOLLINEAR_MODES)
        self._noncollinear.setToolTip("Spin(-current) DFT formulation, written in the DFT block.")
        form.addRow("Formulation", self._noncollinear)

        self._soc_guess = QComboBox()
        self._soc_guess.addItems(TWO_COMPONENT_GUESSES)
        form.addRow("Starting guess", self._soc_guess)
        self._soc_theta = _plain_double(90.0, 5.0, 2, maximum=360.0)
        self._soc_phi = _plain_double(0.0, 5.0, 2, maximum=360.0)
        form.addRow("θ (deg)", self._soc_theta)
        form.addRow("φ (deg)", self._soc_phi)

        self._soc_2nd = QCheckBox("Second-variational (2NDVARIAT)")
        form.addRow(self._soc_2nd)
        self._soc_rot = _plain_spin(0, 0, 1)
        self._soc_rot.setToolTip("IROT: non-zero rotates the magnetization off the z axis.")
        form.addRow("IROT", self._soc_rot)
        self._soc_rot_theta = _plain_double(90.0, 5.0, 2, maximum=360.0)
        self._soc_rot_phi = _plain_double(90.0, 5.0, 2, maximum=360.0)
        form.addRow("Rotation θ (deg)", self._soc_rot_theta)
        form.addRow("Rotation φ (deg)", self._soc_rot_phi)

        self._soc_print = QCheckBox("Print energy contributions (PRTENESOC)")
        form.addRow(self._soc_print)

        self._soc_lock = QCheckBox("Lock occupied spinors (SPINORLOCK)")
        form.addRow(self._soc_lock)
        self._soc_nspin = _plain_spin(8, 1, 10000)
        self._soc_ncyc = _plain_spin(10, 1, 999)
        form.addRow("Spinors", self._soc_nspin)
        form.addRow("Cycles", self._soc_ncyc)

        self._soc_extra = _extra_box("Extra TWOCOMPON keywords, one per line")
        form.addRow("Extra", self._soc_extra)
        return group

    def _scf_tab(self) -> QWidget:
        form = _form()

        self._shrink = QSpinBox()
        self._shrink.setRange(1, 48)
        self._shrink.setValue(suggest_shrink(self._structure))
        form.addRow("SHRINK (k-mesh)", self._shrink)

        # TOLINTEG: five thresholds in a row, with an explanation on demand.
        self._tolinteg = []
        row = QHBoxLayout()
        for default in (7, 7, 7, 7, 14):
            box = QSpinBox()
            box.setRange(1, 30)
            box.setValue(default)
            box.setFixedWidth(52)
            row.addWidget(box)
            self._tolinteg.append(box)
        info = QPushButton("?")
        info.setFixedWidth(28)
        info.setToolTip("What do these five numbers mean?")
        info.clicked.connect(self._explain_tolinteg)
        row.addWidget(info)
        row.addStretch(1)
        form.addRow("TOLINTEG", _row(row))

        for box, (name, short, _) in zip(self._tolinteg, TOLINTEG_LABELS):
            box.setToolTip(f"{name} — {short}")

        self._tolinteg_warning = QLabel()
        self._tolinteg_warning.setWordWrap(True)
        self._tolinteg_warning.setStyleSheet("color: #b26b00;")
        self._tolinteg_warning.hide()  # shown only when there is something to say
        form.addRow(self._tolinteg_warning)

        self._toldee = _optional_spin(1, 20)
        self._toldee.setToolTip("SCF convergence on total energy, 10⁻ⁿ. Unset uses the task default.")
        form.addRow("TOLDEE (10⁻ⁿ)", self._toldee)

        self._maxcycle = QSpinBox()
        self._maxcycle.setRange(1, 9999)
        self._maxcycle.setValue(100)
        form.addRow("Max SCF cycles", self._maxcycle)

        self._spin = QCheckBox("Spin-polarised (open shell)")
        self._spin.setToolTip(
            "SPIN inside the DFT block, or UHF for Hartree–Fock. The two settings "
            "below are only read in a spin-polarised run."
        )
        form.addRow(self._spin)

        self._spinlock = QCheckBox("Lock the total spin (SPINLOCK)")
        self._spinlock.setToolTip(
            "Hold n(α) − n(β) at a chosen value for the first cycles, then let the "
            "SCF relax. The usual way to reach a particular magnetic state rather "
            "than whichever one the starting guess happens to fall into."
        )
        form.addRow(self._spinlock)

        spinlock_row = QHBoxLayout()
        self._spinlock_nspin = _plain_spin(0, -200, 200)
        self._spinlock_nspin.setToolTip("NSPIN: n(α) − n(β), i.e. 2S. 0 is antiferromagnetic.")
        self._spinlock_ncyc = _plain_spin(50, 1, 9999)
        self._spinlock_ncyc.setToolTip("NCYC: how many cycles to hold it for.")
        spinlock_row.addWidget(QLabel("n(α) − n(β)"))
        spinlock_row.addWidget(self._spinlock_nspin)
        spinlock_row.addSpacing(10)
        spinlock_row.addWidget(QLabel("for"))
        spinlock_row.addWidget(self._spinlock_ncyc)
        spinlock_row.addWidget(QLabel("cycles"))
        spinlock_row.addStretch(1)
        form.addRow("", _row_widget(spinlock_row))

        self._atomspin = QLineEdit()
        self._atomspin.setPlaceholderText("e.g.  5 +1, 6 -1     (atom number, then +1 or -1)")
        self._atomspin.setToolTip(
            "ATOMSPIN: the starting spin of individual atoms, which is what sets up "
            "an antiferromagnetic arrangement. Atom numbers are CRYSTAL's own, "
            "counting from 1 — the same numbers the Structure panel shows.\n\n"
            "Anything separable works: '5 1, 6 -1' or '5 +1 6 -1' or one pair a line."
        )
        form.addRow("ATOMSPIN", self._atomspin)

        self._extra = QPlainTextEdit()
        self._extra.setPlaceholderText("Extra block-3 keywords, one per line (optional)")
        self._extra.setFixedHeight(60)
        form.addRow("Extra keywords", self._extra)

        return _page(form)

    def _task_tab(self) -> QWidget:
        outer = QVBoxLayout()
        form = _form()

        self._task = QComboBox()
        self._task.addItems([label for label, _ in _TASKS])
        form.addRow("Task", self._task)

        self._preopt = QCheckBox("Pre-optimise geometry first (PREOPTGEOM)")
        form.addRow(self._preopt)
        outer.addLayout(form)

        # One page of options per task, swapped by the combo above.
        self._task_pages = QStackedWidget()
        self._task_pages.addWidget(QWidget())  # SP: nothing to configure
        self._task_pages.addWidget(self._optgeom_page())
        self._task_pages.addWidget(self._freq_page())
        self._task_pages.addWidget(self._eos_page())
        self._task_pages.addWidget(self._elastic_page())
        self._task_pages.addWidget(self._cphf_page())
        self._task_pages.addWidget(self._dispersion_page())
        self._task_pages.addWidget(self._qha_page())
        self._task_pages.addWidget(self._anhapes_page())
        self._task_pages.addWidget(self._anharm_page())
        outer.addWidget(self._task_pages)
        outer.addStretch(1)

        return _wrap(outer)

    def _optgeom_page(self) -> QWidget:
        form = _form()
        self._opt_scope = QComboBox()
        self._opt_scope.addItems(["Atoms only (ATOMONLY)", "Cell + atoms (default)"])
        self._opt_scope.setCurrentIndex(1)  # CRYSTAL relaxes cell + atoms by default
        form.addRow("Optimise", self._opt_scope)

        self._opt_toldeg = _optional_double(0.00001, 0.1, 5, 0.00001)
        self._opt_toldex = _optional_double(0.00001, 0.1, 5, 0.00001)
        self._opt_maxcycle = _optional_spin(1, 999)
        form.addRow("TOLDEG (gradient)", self._opt_toldeg)
        form.addRow("TOLDEX (displacement)", self._opt_toldex)
        form.addRow("Max opt. cycles", self._opt_maxcycle)

        self._opt_extra = _extra_box("Extra OPTGEOM keywords, one per line")
        form.addRow("Extra", self._opt_extra)
        return _page(form)

    def _freq_page(self) -> QWidget:
        form = _form()
        self._freq_numderiv = QComboBox()
        self._freq_numderiv.addItems([_UNSET, "1 — one-sided", "2 — central difference"])
        form.addRow("NUMDERIV", self._freq_numderiv)

        self._freq_stepsize = _optional_double(0.0001, 0.1, 4, 0.0001)
        form.addRow("STEPSIZE (Å)", self._freq_stepsize)

        self._freq_ir = QCheckBox("IR intensities (INTENS)")
        form.addRow(self._freq_ir)

        self._freq_technique = QComboBox()
        self._freq_technique.addItems([label for label, _ in _IR_TECHNIQUES])
        form.addRow("IR technique", self._freq_technique)

        self._freq_raman = QCheckBox("Raman intensities (INTRAMAN)")
        self._freq_raman.setToolTip(
            "Requires IR intensities computed analytically: CRYSTAL needs INTENS "
            "and the CPHF block opened by INTCPHF, which are selected automatically."
        )
        form.addRow(self._freq_raman)

        self._freq_ramanexp = QCheckBox("Match experimental conditions (RAMANEXP)")
        form.addRow(self._freq_ramanexp)
        self._freq_temp = QDoubleSpinBox()
        self._freq_temp.setRange(0.0, 5000.0)
        self._freq_temp.setValue(298.15)
        self._freq_temp.setDecimals(2)
        self._freq_temp.setSuffix(" K")
        form.addRow("Temperature", self._freq_temp)
        self._freq_laser = QDoubleSpinBox()
        self._freq_laser.setRange(1.0, 5000.0)
        self._freq_laser.setValue(532.0)
        self._freq_laser.setDecimals(1)
        self._freq_laser.setSuffix(" nm")
        form.addRow("Laser wavelength", self._freq_laser)

        self._freq_irspec = QCheckBox("IR spectrum (IRSPEC)")
        self._freq_ramspec = QCheckBox("Raman spectrum (RAMSPEC)")
        form.addRow(self._freq_irspec)
        form.addRow(self._freq_ramspec)

        self._freq_analysis = QCheckBox("Mode analysis (ANALYSIS)")
        self._freq_print = QCheckBox("Print Hessian (PRINT)")
        self._freq_restart = QCheckBox("Restart (RESTART)")
        for box in (self._freq_analysis, self._freq_print, self._freq_restart):
            form.addRow(box)

        self._freq_cphf = _CphfGroup("CPHF block (INTCPHF)")
        form.addRow(_gap())  # keep the group box off the check box above it
        form.addRow(self._freq_cphf)

        self._freq_extra = _extra_box("Extra FREQCALC keywords, one per line")
        form.addRow("Extra", self._freq_extra)
        return _page(form)

    def _eos_page(self) -> QWidget:
        form = _form()
        self._eos_range = QCheckBox("Volume range (RANGE)")
        form.addRow(self._eos_range)
        self._eos_vmin = _plain_double(0.90, 0.01, 3)
        self._eos_vmax = _plain_double(1.10, 0.01, 3)
        self._eos_vn = QSpinBox()
        self._eos_vn.setRange(2, 100)
        self._eos_vn.setValue(10)
        form.addRow("V min (×V₀)", self._eos_vmin)
        form.addRow("V max (×V₀)", self._eos_vmax)
        form.addRow("Points", self._eos_vn)

        self._eos_prange = QCheckBox("Pressure range (PRANGE)")
        form.addRow(self._eos_prange)
        self._eos_pmin = _plain_double(-5.0, 0.5, 2, minimum=-1000.0)
        self._eos_pmax = _plain_double(10.0, 0.5, 2, minimum=-1000.0)
        self._eos_pn = QSpinBox()
        self._eos_pn.setRange(2, 100)
        self._eos_pn.setValue(20)
        form.addRow("P min (GPa)", self._eos_pmin)
        form.addRow("P max (GPa)", self._eos_pmax)
        form.addRow("Points", self._eos_pn)

        self._eos_extra = _extra_box("Extra EOS keywords, one per line")
        form.addRow("Extra", self._eos_extra)
        return _page(form)

    def _elastic_page(self) -> QWidget:
        form = _form()
        self._el_numderiv = _optional_spin(2, 9)
        self._el_stepsize = _optional_double(0.001, 0.1, 4, 0.001)
        self._el_clampion = QCheckBox("Clamped-ion constants (CLAMPION)")
        form.addRow("NUMDERIV (points)", self._el_numderiv)
        form.addRow("STEPSIZE (strain)", self._el_stepsize)
        form.addRow(self._el_clampion)
        self._el_extra = _extra_box("Extra ELASTCON keywords, one per line")
        form.addRow("Extra", self._el_extra)
        return _page(form)

    def _dispersion_page(self) -> QWidget:
        form = _form()
        self._disp_cell = _SupercellGroup(checked=True)
        form.addRow(self._disp_cell)
        form.addRow(_gap())

        self._disp_noksym = QCheckBox("Do not label phonons by irrep (NOKSYMDISP)")
        form.addRow(self._disp_noksym)

        self._disp_interp = QCheckBox("Fourier interpolation (INTERPHESS)")
        form.addRow(self._disp_interp)
        self._disp_interp_l = [_plain_spin(2, 1, 24) for _ in range(3)]
        row = QHBoxLayout()
        for box in self._disp_interp_l:
            row.addWidget(box)
        row.addStretch(1)
        form.addRow("Expansion L₁ L₂ L₃", _row(row))
        self._disp_interp_print = QCheckBox("Print each k point")
        form.addRow(self._disp_interp_print)

        self._disp_wang = QCheckBox("Long-range correction (WANG)")
        self._disp_wang.setToolTip(
            "For polar 3D crystals: supply the dielectric tensor so the dynamical "
            "matrices account for long-range Coulomb interactions."
        )
        form.addRow(self._disp_wang)
        # One field per row, because that is one record per row in the deck.
        # A single nine-number field invited writing them on one line, which
        # CRYSTAL misreads.
        self._disp_wang_rows = []
        for index, default in enumerate(("1 0 0", "0 1 0", "0 0 1")):
            row = QLineEdit(default)
            row.setToolTip("Three elements of this row of the dielectric tensor.")
            form.addRow("Dielectric tensor" if index == 0 else "", row)
            self._disp_wang_rows.append(row)

        self._disp_bands = QCheckBox("Phonon bands (BANDS)")
        form.addRow(self._disp_bands)
        self._disp_bands_points = _plain_spin(30, 2, 500)
        form.addRow("Points per line (NSUB)", self._disp_bands_points)
        # The same editor as the .d3 builder's: a phonon path and an electron
        # path are the same object, and the integers CRYSTAL reads mean nothing
        # without the shrinking factor they are written over — which the editor
        # owns, so the two cannot be set to disagree.
        self._disp_bands_editor = BandPathEditor(self._structure)
        self._disp_bands_editor.changed.connect(self._on_form_changed)
        form.addRow("Path", self._disp_bands_editor)

        self._disp_pdos = QCheckBox("Phonon DOS (PDOS)")
        form.addRow(self._disp_pdos)
        self._disp_pdos_max = _plain_double(2500.0, 50.0, 1, maximum=100000.0)
        self._disp_pdos_bins = _plain_spin(250, 10, 10000)
        self._disp_pdos_proj = QCheckBox("Projected atomic DOS")
        form.addRow("Max frequency (cm⁻¹)", self._disp_pdos_max)
        form.addRow("Bins", self._disp_pdos_bins)
        form.addRow(self._disp_pdos_proj)

        self._disp_ins = QCheckBox("Neutron-weighted DOS (INS)")
        form.addRow(self._disp_ins)
        self._disp_ins_max = _plain_double(3000.0, 50.0, 1, maximum=100000.0)
        self._disp_ins_bins = _plain_spin(300, 10, 10000)
        self._disp_ins_type = QComboBox()
        self._disp_ins_type.addItems(["Coherent", "Incoherent", "Coherent + incoherent"])
        form.addRow("Max frequency (cm⁻¹)", self._disp_ins_max)
        form.addRow("Bins", self._disp_ins_bins)
        form.addRow("Cross-section", self._disp_ins_type)

        self._disp_extra = _extra_box("Extra FREQCALC keywords, one per line")
        form.addRow("Extra", self._disp_extra)
        return _page(form)

    def _qha_page(self) -> QWidget:
        form = _form()
        self._qha_cell = _SupercellGroup(checked=False)
        form.addRow(self._qha_cell)
        form.addRow(_gap())

        self._qha_step = _optional_double(0.5, 20.0, 2, 0.5)
        self._qha_step.setSuffix(" %")
        self._qha_step.setToolTip("Volume step s: the range runs from −s% to +2s%.")
        form.addRow("STEP", self._qha_step)

        self._qha_points = QComboBox()
        self._qha_points.addItems([_UNSET, "4", "7", "13"])
        self._qha_points.setToolTip("Number of volumes at which phonons are computed.")
        form.addRow("POINTS", self._qha_points)

        self._qha_temp = QCheckBox("Temperature range (TEMPERAT)")
        form.addRow(self._qha_temp)
        self._qha_nt = _plain_spin(100, 2, 1000)
        self._qha_t1 = _plain_double(10.0, 10.0, 1, maximum=10000.0)
        self._qha_t2 = _plain_double(1200.0, 10.0, 1, maximum=10000.0)
        form.addRow("Steps", self._qha_nt)
        form.addRow("T min (K)", self._qha_t1)
        form.addRow("T max (K)", self._qha_t2)

        self._qha_vrange = QCheckBox("Custom volume range (VRANGE)")
        form.addRow(self._qha_vrange)
        self._qha_vmin = _plain_double(0.94, 0.01, 3)
        self._qha_vmax = _plain_double(1.06, 0.01, 3)
        self._qha_vn = _plain_spin(7, 2, 100)
        form.addRow("V min (×V₀)", self._qha_vmin)
        form.addRow("V max (×V₀)", self._qha_vmax)
        form.addRow("Points", self._qha_vn)

        self._qha_restart = QCheckBox("Restart, incomplete run (RESTART)")
        self._qha_restart2 = QCheckBox("Restart, complete run (RESTART2)")
        form.addRow(self._qha_restart)
        form.addRow(self._qha_restart2)

        self._qha_extra = _extra_box("Extra QHA keywords, one per line")
        form.addRow("Extra", self._qha_extra)
        return _page(form)

    def _anhapes_page(self) -> QWidget:
        form = _form()
        self._anh_modes = _extra_box("e.g.  4 5 6 7 8 9 10 11 12")
        self._anh_modes.setToolTip(
            "Mode numbers from the harmonic calculation. Translations (modes 1–3) "
            "must be excluded, and molecular rotations too — for a non-linear "
            "molecule start from mode 7."
        )
        form.addRow("Modes", self._anh_modes)

        self._anh_scheme = QComboBox()
        self._anh_scheme.addItems(
            ["1 — energies only", "2 — energies only, denser",
             "3 — energies + gradients (recommended)", "4 — energies + gradients, denser"]
        )
        self._anh_scheme.setCurrentIndex(2)
        form.addRow("Numerical scheme", self._anh_scheme)

        self._anh_step = _plain_double(0.9, 0.1, 2, minimum=0.1, maximum=10.0)
        form.addRow("Step h", self._anh_step)

        self._anh_restart = QCheckBox("Reuse harmonic Hessian (RESTART)")
        self._anh_restpes = QCheckBox("Reuse anharmonic PES (RESTPES)")
        form.addRow(self._anh_restart)
        form.addRow(self._anh_restpes)

        self._anh_vscf = QCheckBox("Vibrational SCF (VSCF)")
        form.addRow(self._anh_vscf)
        self._anh_vscftol = _optional_spin(1, 12)
        self._anh_vscfmix = _optional_spin(0, 100, " %")
        form.addRow("VSCFTOL (10⁻ⁿ cm⁻¹)", self._anh_vscftol)
        form.addRow("VSCFMIX", self._anh_vscfmix)

        self._anh_vci = QCheckBox("Vibrational CI (VCI)")
        self._anh_vci.setToolTip("Runs a VSCF of its own first, so VSCF need not be ticked.")
        form.addRow(self._anh_vci)
        self._anh_vci_quanta = _plain_spin(6, 1, 30)
        self._anh_vci_modes = _plain_spin(3, 1, 12)
        self._anh_vci_guess = QComboBox()
        self._anh_vci_guess.addItems(["Harmonic guess", "VSCF guess (VCI@VSCF)"])
        self._anh_vci_guess.setCurrentIndex(1)
        form.addRow("Max quanta", self._anh_vci_quanta)
        form.addRow("Max coupled modes", self._anh_vci_modes)
        form.addRow("Initial guess", self._anh_vci_guess)

        self._anh_extra = _extra_box("Extra FREQCALC keywords, one per line")
        form.addRow("Extra", self._anh_extra)
        return _page(form)

    def _anharm_page(self) -> QWidget:
        form = _form()
        hint = QLabel(
            "Anharmonic X–H (or X–D) stretching. The chosen hydrogen moves along "
            "the direction to its first neighbour; no harmonic run is needed."
        )
        hint.setWordWrap(True)
        form.addRow(hint)

        self._anharm_atom = _plain_spin(1, 1, 100000)
        self._anharm_atom.setToolTip(
            "Sequence number of the hydrogen (or deuterium) atom, as CRYSTAL "
            "numbers atoms after reading the geometry."
        )
        form.addRow("Atom label", self._anharm_atom)

        self._anharm_keepsymm = QCheckBox("Stretch equivalent bonds together (KEEPSYMM)")
        self._anharm_points26 = QCheckBox("26 points instead of 7 (POINTS26)")
        self._anharm_noguess = QCheckBox("Atomic-density SCF guess (NOGUESS)")
        self._anharm_print = QCheckBox("Extended printing (PRINT)")
        self._anharm_test = QCheckBox("Test the neighbour only (TESTANHA)")
        for box in (self._anharm_keepsymm, self._anharm_points26, self._anharm_noguess,
                    self._anharm_print, self._anharm_test):
            form.addRow(box)

        self._anharm_iso = QCheckBox("Change an atomic mass (ISOTOPES)")
        form.addRow(self._anharm_iso)
        self._anharm_iso_atom = _plain_spin(1, 1, 100000)
        self._anharm_iso_mass = _plain_double(2.0, 0.1, 3, minimum=0.1, maximum=300.0)
        form.addRow("Isotope atom", self._anharm_iso_atom)
        form.addRow("Mass (amu)", self._anharm_iso_mass)

        self._anharm_extra = _extra_box("Extra ANHARM keywords, one per line")
        form.addRow("Extra", self._anharm_extra)
        return _page(form)

    def _cphf_page(self) -> QWidget:
        layout = QVBoxLayout()
        self._cphf = _CphfGroup("CPHF", with_order=True)
        layout.addWidget(self._cphf)
        layout.addStretch(1)
        return _wrap(layout)

    # ── signals ─────────────────────────────────────────────────────────
    def _connect_signals(self) -> None:
        self._method.currentIndexChanged.connect(self._on_form_changed)
        self._functional_mode.currentIndexChanged.connect(self._on_form_changed)
        self._nonlocal.toggled.connect(self._on_form_changed)
        self._soc_guess.currentIndexChanged.connect(self._on_form_changed)
        self._soc_2nd.toggled.connect(self._on_form_changed)
        self._soc_rot.valueChanged.connect(self._on_form_changed)
        self._soc_lock.toggled.connect(self._on_form_changed)
        self._twocompon.toggled.connect(self._refresh_preview)
        self._supercel.toggled.connect(self._refresh_preview)
        self._task.currentIndexChanged.connect(self._on_form_changed)
        self._freq_ir.toggled.connect(self._on_form_changed)
        self._freq_raman.toggled.connect(self._on_form_changed)
        self._freq_ramanexp.toggled.connect(self._on_form_changed)
        self._eos_range.toggled.connect(self._on_form_changed)
        self._eos_prange.toggled.connect(self._on_form_changed)
        for check in (self._disp_interp, self._disp_wang, self._disp_bands,
                      self._disp_pdos, self._disp_ins, self._qha_temp, self._qha_vrange,
                      self._anh_vscf, self._anh_vci, self._anharm_iso):
            check.toggled.connect(self._on_form_changed)
        for group in (self._disp_cell, self._qha_cell):
            group.toggled.connect(self._refresh_preview)
        if self._cphf.order is not None:
            self._cphf.order.currentIndexChanged.connect(self._on_form_changed)

        self._freq_technique.currentIndexChanged.connect(self._on_form_changed)
        combos = [
            self._functional, self._grid, self._opt_scope, self._basis,
            self._freq_numderiv, self._qha_points, self._disp_ins_type,
            self._anh_scheme, self._anh_vci_guess, self._noncollinear,
        ]
        for combo in combos:
            combo.currentIndexChanged.connect(self._refresh_preview)
        for combo in (self._functional, self._exchange, self._correlation):
            combo.editTextChanged.connect(self._refresh_preview)
            combo.currentIndexChanged.connect(self._refresh_preview)

        # the spin sub-controls gate each other, so they go through
        # _on_form_changed (which re-runs the enable rules) rather than a plain
        # preview refresh
        for widget in (self._spin, self._spinlock):
            widget.toggled.connect(self._on_form_changed)
        self._atomspin.textChanged.connect(self._refresh_preview)
        for box in (self._spinlock_nspin, self._spinlock_ncyc):
            box.valueChanged.connect(self._refresh_preview)

        checks = [
            self._d3, self._symmetry, self._preopt,
            self._freq_irspec, self._freq_ramspec, self._freq_analysis,
            self._freq_print, self._freq_restart, self._el_clampion,
            self._disp_noksym, self._disp_interp_print, self._disp_pdos_proj,
            self._qha_restart, self._qha_restart2,
            self._anh_restart, self._anh_restpes,
            self._anharm_keepsymm, self._anharm_points26, self._anharm_noguess,
            self._anharm_print, self._anharm_test,
            self._soc, self._soc_print, self._supercel_noshift,
        ]
        for check in checks:
            check.toggled.connect(self._refresh_preview)
        for row in self._disp_wang_rows:
            row.textChanged.connect(self._refresh_preview)

        self._title.textChanged.connect(self._refresh_preview)
        spins = [
            self._shrink, self._maxcycle, self._toldee,
            self._opt_toldeg, self._opt_toldex, self._opt_maxcycle,
            self._freq_stepsize, self._freq_temp, self._freq_laser,
            self._eos_vmin, self._eos_vmax, self._eos_vn,
            self._eos_pmin, self._eos_pmax, self._eos_pn,
            self._el_numderiv, self._el_stepsize,
            self._disp_bands_points,
            self._disp_pdos_max, self._disp_pdos_bins,
            self._disp_ins_max, self._disp_ins_bins,
            self._qha_step, self._qha_nt, self._qha_t1, self._qha_t2,
            self._qha_vmin, self._qha_vmax, self._qha_vn,
            self._anh_step, self._anh_vscftol, self._anh_vscfmix,
            self._anh_vci_quanta, self._anh_vci_modes,
            self._anharm_atom, self._anharm_iso_atom, self._anharm_iso_mass,
            self._hybrid, self._nonlocal_b, self._nonlocal_c,
            self._soc_theta, self._soc_phi, self._soc_rot_theta, self._soc_rot_phi,
            self._soc_nspin, self._soc_ncyc,
            *self._disp_interp_l, *self._disp_cell.cells, *self._qha_cell.cells,
            *self._supercel.cells,
            *self._tolinteg, *self._cphf.widgets(), *self._freq_cphf.widgets(),
        ]
        for spin in spins:
            signal = getattr(spin, "valueChanged", None) or spin.currentIndexChanged
            signal.connect(self._refresh_preview)

        for editor in (self._extra, self._opt_extra, self._freq_extra,
                       self._eos_extra, self._el_extra,
                       self._disp_extra, self._qha_extra,
                       self._anh_modes, self._anh_extra, self._anharm_extra, self._soc_extra,
                       self._cphf.extra, self._freq_cphf.extra):
            editor.textChanged.connect(self._refresh_preview)

    # ── reactivity ──────────────────────────────────────────────────────
    def _phonon_bands_shrink(self) -> int:
        """The ISS the phonon path is written over — the editor's, never a
        separate control that could drift out of step with the numbers."""
        return self._disp_bands_editor.effective_shrink() or 1

    def _phonon_bands_path(self) -> str:
        """The path as whole numbers over that factor, one segment per record.

        A factor that cannot express the path raises out of the editor; the
        preview shows the complaint rather than a rounded path.
        """
        from crystalline.core.properties_input import PropertiesInputError

        try:
            return "\n".join(self._disp_bands_editor.integer_rows())
        except PropertiesInputError:
            return ""

    def _on_form_changed(self) -> None:
        """A change that alters which rows apply, then refreshes the preview."""
        self._sync_enabled()
        self._refresh_preview()

    def _set_technique(self, index: int) -> None:
        """Change the IR technique without re-entering :meth:`_sync_enabled`."""
        self._syncing = True
        try:
            self._freq_technique.setCurrentIndex(index)
        finally:
            self._syncing = False

    def _sync_enabled(self) -> None:
        if self._syncing:
            return
        is_dft = self._method.currentText() == "DFT"
        split = self._functional_mode.currentIndex() == 1
        for widget in (self._functional_mode, self._d3, self._grid, self._hybrid, self._nonlocal):
            widget.setEnabled(is_dft)
        self._functional.setEnabled(is_dft and not split)
        for widget in (self._exchange, self._correlation):
            widget.setEnabled(is_dft and split)
        for widget in (self._nonlocal_b, self._nonlocal_c):
            widget.setEnabled(is_dft and self._nonlocal.isChecked())

        # The guess angles only mean anything for the rotated core-Hamiltonian guess.
        spin = self._spin.isChecked()
        for widget in (self._spinlock, self._atomspin):
            widget.setEnabled(spin)
        for widget in (self._spinlock_nspin, self._spinlock_ncyc):
            widget.setEnabled(spin and self._spinlock.isChecked())

        rotated_guess = self._soc_guess.currentText() == "GCOREROT"
        for widget in (self._soc_theta, self._soc_phi):
            widget.setEnabled(rotated_guess)
        self._soc_rot.setEnabled(self._soc_2nd.isChecked())
        for widget in (self._soc_rot_theta, self._soc_rot_phi):
            widget.setEnabled(self._soc_2nd.isChecked() and self._soc_rot.value() != 0)
        for widget in (self._soc_nspin, self._soc_ncyc):
            widget.setEnabled(self._soc_lock.isChecked())

        kind = self._task_kind()
        self._task_pages.setCurrentIndex(self._task.currentIndex())
        self._preopt.setEnabled(kind in _EQUILIBRIUM_TASKS)
        self._shrink.setEnabled(self._structure.is_periodic)

        # Raman implies IR through CPHF, so those controls stop being free choices.
        raman = self._freq_raman.isChecked()
        self._freq_ir.setEnabled(not raman)
        self._freq_technique.setEnabled(self._freq_ir.isChecked() and not raman)
        # Force INTCPHF while Raman is on, restoring the previous choice after, so
        # turning Raman off returns the form to the state the user left it in.
        if raman and self._freq_technique.currentIndex() != _INTCPHF_INDEX:
            self._technique_before_raman = self._freq_technique.currentIndex()
            self._set_technique(_INTCPHF_INDEX)
        elif not raman and self._technique_before_raman is not None:
            self._set_technique(self._technique_before_raman)
            self._technique_before_raman = None

        for widget in (self._freq_ramanexp, self._freq_ramspec):
            widget.setEnabled(raman)
        for widget in (self._freq_temp, self._freq_laser):
            widget.setEnabled(raman and self._freq_ramanexp.isChecked())
        self._freq_irspec.setEnabled(self._freq_ir.isChecked() or raman)
        # The INTCPHF block is only written when an intensity calculation actually
        # asks for it, so the group follows that rather than the combo alone.
        writes_intcphf = (raman or self._freq_ir.isChecked()) and (
            self._freq_technique.currentIndex() == _INTCPHF_INDEX
        )
        self._freq_cphf.setEnabled(writes_intcphf)
        self._freq_cphf.set_second_order_enabled(writes_intcphf and raman)

        if self._cphf.order is not None:  # CPHF2 keywords need fourth order
            self._cphf.set_second_order_enabled(self._cphf.order.currentIndex() == 2)

        for widget in (*self._disp_interp_l, self._disp_interp_print):
            widget.setEnabled(self._disp_interp.isChecked())
        for row in self._disp_wang_rows:
            row.setEnabled(self._disp_wang.isChecked())
        for widget in (self._disp_bands_points, self._disp_bands_editor):
            widget.setEnabled(self._disp_bands.isChecked())
        # BANDS already implies NOKSYMDISP, so the separate switch stops applying.
        self._disp_noksym.setEnabled(not self._disp_bands.isChecked())
        for widget in (self._disp_pdos_max, self._disp_pdos_bins, self._disp_pdos_proj):
            widget.setEnabled(self._disp_pdos.isChecked())
        for widget in (self._disp_ins_max, self._disp_ins_bins, self._disp_ins_type):
            widget.setEnabled(self._disp_ins.isChecked())

        # VCI runs its own VSCF, so the VSCF tuning applies to either request.
        solves = self._anh_vscf.isChecked() or self._anh_vci.isChecked()
        for widget in (self._anh_vscftol, self._anh_vscfmix):
            widget.setEnabled(solves)
        for widget in (self._anh_vci_quanta, self._anh_vci_modes, self._anh_vci_guess):
            widget.setEnabled(self._anh_vci.isChecked())
        for widget in (self._anharm_iso_atom, self._anharm_iso_mass):
            widget.setEnabled(self._anharm_iso.isChecked())

        for widget in (self._qha_nt, self._qha_t1, self._qha_t2):
            widget.setEnabled(self._qha_temp.isChecked())
        for widget in (self._qha_vmin, self._qha_vmax, self._qha_vn):
            widget.setEnabled(self._qha_vrange.isChecked())

        for widget in (self._eos_vmin, self._eos_vmax, self._eos_vn):
            widget.setEnabled(self._eos_range.isChecked())
        for widget in (self._eos_pmin, self._eos_pmax, self._eos_pn):
            widget.setEnabled(self._eos_prange.isChecked())

    def _task_kind(self) -> str:
        return _TASKS[self._task.currentIndex()][1]

    def _supercell(self):
        """The SCELPHONO matrix of whichever task page owns one."""
        kind = self._task_kind()
        if kind == "DISPERSION":
            return self._disp_cell.matrix()
        if kind == "QHA":
            return self._qha_cell.matrix()
        return None

    def _wang_tensor(self):
        """The nine dielectric-tensor elements, or ``None`` if not nine numbers.

        Returning ``None`` lets the preview show the builder's own complaint
        rather than raising out of the middle of a keystroke.
        """
        text = " ".join(row.text() for row in self._disp_wang_rows)
        try:
            values = [float(v) for v in text.replace(",", " ").split()]
        except ValueError:
            return None
        return values or None

    def _tolinteg_values(self) -> tuple:
        return tuple(box.value() for box in self._tolinteg)

    def _explain_tolinteg(self) -> None:
        """Spell out what the five TOLINTEG thresholds control."""
        rows = "".join(
            f"<p><b>{name}</b> — {short}<br>{detail}</p>"
            for name, short, detail in TOLINTEG_LABELS
        )
        box = QMessageBox(self)
        box.setWindowTitle("TOLINTEG")
        box.setTextFormat(Qt.RichText)
        # ITOL4/ITOL5 truncate the lattice summations, so they do nothing for a
        # molecule — but the card reads all five values together and ITOL1–ITOL3
        # still screen integrals within the cell, so it is written regardless.
        molecular_note = (
            ""
            if self._structure.is_periodic
            else "<p><i>This is a molecule: ITOL4 and ITOL5 truncate lattice "
            "summations and have no effect here, while ITOL1–ITOL3 still screen "
            "the integrals. All five are written, as CRYSTAL reads them as one "
            "record.</i></p>"
        )
        box.setText(
            "<p>Truncation thresholds for the bielectronic Coulomb and HF exchange "
            "series. Each value <i>n</i> means a cut-off of 10⁻ⁿ, so <b>larger "
            "numbers are tighter</b> — more accurate and more expensive. "
            "CRYSTAL's default is <tt>7 7 7 7 14</tt>.</p>" + molecular_note + rows
        )
        box.exec()

    def _spec(self) -> CrystalInputSpec:
        grid = self._grid.currentText()
        numderiv_index = self._freq_numderiv.currentIndex()
        return CrystalInputSpec(
            geometry=GeometryOptions(
                title=self._title.text() or "Generated by CRYSTALLine",
                use_symmetry=self._symmetry.isChecked(),
                # A reduction the user has applied to the crystal is part of
                # the crystal, not a setting of this dialog: it rides on the
                # structure, so the deck gets it without being told.
                kept_rotations=tuple(self._structure.reduced_symmetry),
                supercell=self._supercel.matrix(),
                supercell_noshift=self._supercel_noshift.isChecked(),
            ),
            basis=BasisOptions(name=self._basis.currentText()),
            method=MethodOptions(
                kind="DFT" if self._method.currentText() == "DFT" else "HF",
                functional_mode="SPLIT" if self._functional_mode.currentIndex() else "COMBINED",
                functional=functional_keyword(self._functional),
                exchange=_chosen(self._exchange, _HF_EXCHANGE),
                correlation=_chosen(self._correlation, _NO_CORRELATION),
                hybrid_percent=_int_of(self._hybrid),
                nonlocal_weights=(
                    (self._nonlocal_b.value(), self._nonlocal_c.value())
                    if self._nonlocal.isChecked()
                    else None
                ),
                dispersion_d3=self._d3.isChecked(),
                grid=None if grid == _GRID_DEFAULT_LABEL else grid,
            ),
            two_component=TwoComponentOptions(
                enabled=self._twocompon.isChecked(),
                soc=self._soc.isChecked(),
                guess=self._soc_guess.currentText(),
                guess_angles=(self._soc_theta.value(), self._soc_phi.value()),
                second_variational=self._soc_2nd.isChecked(),
                second_variational_rot=self._soc_rot.value(),
                second_variational_angles=(
                    self._soc_rot_theta.value(), self._soc_rot_phi.value()
                ),
                print_energies=self._soc_print.isChecked(),
                spinorlock=(
                    (self._soc_nspin.value(), self._soc_ncyc.value())
                    if self._soc_lock.isChecked()
                    else None
                ),
                noncollinear=self._noncollinear.currentText(),
                extra_keywords=self._soc_extra.toPlainText(),
            ),
            scf=ScfOptions(
                shrink=self._shrink.value(),
                tolinteg=self._tolinteg_values(),
                toldee=_int_of(self._toldee),
                maxcycle=self._maxcycle.value(),
                spin_polarized=self._spin.isChecked(),
                spinlock=(
                    (self._spinlock_nspin.value(), self._spinlock_ncyc.value())
                    if self._spin.isChecked() and self._spinlock.isChecked() else None
                ),
                atomspin=(
                    parse_atomspin(self._atomspin.text()) if self._spin.isChecked() else ()
                ),
            ),
            task=TaskOptions(
                kind=self._task_kind(),
                supercell=self._supercell(),
                optimize_cell=self._opt_scope.currentIndex() == 1,
                preoptimize=self._preopt.isChecked(),
                optgeom=OptGeomOptions(
                    toldeg=_value_of(self._opt_toldeg),
                    toldex=_value_of(self._opt_toldex),
                    maxcycle=_int_of(self._opt_maxcycle),
                    extra_keywords=self._opt_extra.toPlainText(),
                ),
                freq=FreqOptions(
                    numderiv=numderiv_index if numderiv_index else None,
                    stepsize=_value_of(self._freq_stepsize),
                    ir_intensities=self._freq_ir.isChecked(),
                    ir_technique=_IR_TECHNIQUES[self._freq_technique.currentIndex()][1],
                    raman_intensities=self._freq_raman.isChecked(),
                    ir_spectrum=self._freq_irspec.isChecked(),
                    raman_spectrum=self._freq_ramspec.isChecked(),
                    raman_experiment=(
                        (self._freq_temp.value(), self._freq_laser.value())
                        if self._freq_ramanexp.isChecked()
                        else None
                    ),
                    analysis=self._freq_analysis.isChecked(),
                    print_hessian=self._freq_print.isChecked(),
                    restart=self._freq_restart.isChecked(),
                    cphf=self._freq_cphf.options(),
                    extra_keywords=self._freq_extra.toPlainText(),
                ),
                cphf=self._cphf.options(),
                eos=EosOptions(
                    volume_range=(
                        (self._eos_vmin.value(), self._eos_vmax.value(), self._eos_vn.value())
                        if self._eos_range.isChecked()
                        else None
                    ),
                    pressure_range=(
                        (self._eos_pmin.value(), self._eos_pmax.value(), self._eos_pn.value())
                        if self._eos_prange.isChecked()
                        else None
                    ),
                    extra_keywords=self._eos_extra.toPlainText(),
                ),
                anhapes=AnhapesOptions(
                    modes=self._anh_modes.toPlainText(),
                    scheme=self._anh_scheme.currentIndex() + 1,
                    step=self._anh_step.value(),
                    restart=self._anh_restart.isChecked(),
                    restart_pes=self._anh_restpes.isChecked(),
                    vscf=self._anh_vscf.isChecked(),
                    vscf_tol=_int_of(self._anh_vscftol),
                    vscf_mix=_int_of(self._anh_vscfmix),
                    vci=self._anh_vci.isChecked(),
                    vci_quanta=self._anh_vci_quanta.value(),
                    vci_modes=self._anh_vci_modes.value(),
                    vci_guess=self._anh_vci_guess.currentIndex(),
                    extra_keywords=self._anh_extra.toPlainText(),
                ),
                anharm=AnharmOptions(
                    atom_label=self._anharm_atom.value(),
                    keepsymm=self._anharm_keepsymm.isChecked(),
                    points26=self._anharm_points26.isChecked(),
                    noguess=self._anharm_noguess.isChecked(),
                    print_extended=self._anharm_print.isChecked(),
                    test_only=self._anharm_test.isChecked(),
                    isotopes=(
                        [(self._anharm_iso_atom.value(), self._anharm_iso_mass.value())]
                        if self._anharm_iso.isChecked()
                        else []
                    ),
                    extra_keywords=self._anharm_extra.toPlainText(),
                ),
                dispersion=DispersionOptions(
                    noksymdisp=self._disp_noksym.isChecked(),
                    bands=self._disp_bands.isChecked(),
                    bands_shrink=self._phonon_bands_shrink(),
                    bands_points=self._disp_bands_points.value(),
                    bands_path=self._phonon_bands_path(),
                    interphess=(
                        (*[b.value() for b in self._disp_interp_l],
                         int(self._disp_interp_print.isChecked()))
                        if self._disp_interp.isChecked()
                        else None
                    ),
                    wang=self._wang_tensor() if self._disp_wang.isChecked() else None,
                    pdos=(
                        (self._disp_pdos_max.value(), self._disp_pdos_bins.value(),
                         int(self._disp_pdos_proj.isChecked()))
                        if self._disp_pdos.isChecked()
                        else None
                    ),
                    ins=(
                        (self._disp_ins_max.value(), self._disp_ins_bins.value(),
                         self._disp_ins_type.currentIndex())
                        if self._disp_ins.isChecked()
                        else None
                    ),
                    extra_keywords=self._disp_extra.toPlainText(),
                ),
                qha=QhaOptions(
                    step=_value_of(self._qha_step),
                    points=(
                        int(self._qha_points.currentText())
                        if self._qha_points.currentIndex()
                        else None
                    ),
                    temperature=(
                        (self._qha_nt.value(), self._qha_t1.value(), self._qha_t2.value())
                        if self._qha_temp.isChecked()
                        else None
                    ),
                    volume_range=(
                        (self._qha_vmin.value(), self._qha_vmax.value(), self._qha_vn.value())
                        if self._qha_vrange.isChecked()
                        else None
                    ),
                    restart=self._qha_restart.isChecked(),
                    restart2=self._qha_restart2.isChecked(),
                    extra_keywords=self._qha_extra.toPlainText(),
                ),
                elastic=ElasticOptions(
                    numderiv=_int_of(self._el_numderiv),
                    stepsize=_value_of(self._el_stepsize),
                    clamped_ion=self._el_clampion.isChecked(),
                    extra_keywords=self._el_extra.toPlainText(),
                ),
            ),
            extra_keywords=self._extra.toPlainText(),
        )

    def _refresh_preview(self) -> None:
        """Rebuild the deck; on an unsupported case show why and block saving."""
        warning = "\n".join(tolinteg_warnings(self._tolinteg_values()))
        self._tolinteg_warning.setText(warning)
        self._tolinteg_warning.setVisible(bool(warning))
        try:
            self._preview.setPlainText(build_input(self._structure, self._spec()))
            self._save_btn.setEnabled(True)
        except CrystalInputError as exc:
            self._preview.setPlainText(f"⚠  {exc}")
            self._save_btn.setEnabled(False)

    # ── save ────────────────────────────────────────────────────────────
    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CRYSTAL input", "input.d12", "CRYSTAL input (*.d12);;All files (*)"
        )
        if not path:
            return
        try:
            write_input(self._structure, path, self._spec())
        except Exception as exc:  # noqa: BLE001 - surface any build/write error
            QMessageBox.critical(
                self, "Save failed", f"Could not write the input file:\n{exc or type(exc).__name__}"
            )
            return
        self.accept()


# ── module-level widget helpers ────────────────────────────────────────────
def _wrap(layout) -> QWidget:
    """A plain container widget owning ``layout`` (for tabs and stack pages)."""
    widget = QWidget()
    widget.setLayout(layout)
    return widget


def _page(form: QFormLayout) -> QWidget:
    """A tab/stack page holding ``form``, packed to the top.

    A bare form stretched by its scroll area spreads the spare height between
    rows, which reads as erratic spacing; the trailing stretch absorbs it.
    """
    box = QVBoxLayout()
    box.setContentsMargins(0, 0, 0, 0)
    box.addLayout(form)
    box.addStretch(1)
    return _wrap(box)


def _form() -> QFormLayout:
    """A form layout with consistent, compact spacing.

    Qt's per-platform defaults leave generous gaps that read as erratic once
    rows alternate between labelled fields and full-width check boxes, so every
    form here is built through this factory.
    """
    form = QFormLayout()
    form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    form.setHorizontalSpacing(10)
    form.setVerticalSpacing(6)
    form.setContentsMargins(10, 10, 10, 10)
    return form


def _row(layout) -> QWidget:
    """A container for an inline row of widgets used as a single form field.

    Without the fixed vertical policy the container stretches to fill the form's
    spare height, which drags its contents away from their own label.
    """
    widget = _wrap(layout)
    layout.setContentsMargins(0, 0, 0, 0)
    widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    return widget


def _scrollable(widget: QWidget) -> QScrollArea:
    """Put ``widget`` in a frameless vertical scroll area that fills its tab."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    # As-needed rather than off: when the pane is dragged narrow, a clipped label
    # would otherwise be unreachable.
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    area.setWidget(widget)
    return area


def _chosen(combo: QComboBox, sentinel: str) -> Optional[str]:
    """The combo's text, or ``None`` when it is parked on its "unset" entry."""
    text = combo.currentText().strip()
    return None if not text or text == sentinel else text


def _gap(height: int = 8) -> QWidget:
    """A short vertical spacer for separating a group box from the row above."""
    spacer = QWidget()
    spacer.setFixedHeight(height)
    return spacer


def _extra_box(placeholder: str) -> QPlainTextEdit:
    box = QPlainTextEdit()
    box.setPlaceholderText(placeholder)
    box.setFixedHeight(48)
    return box


def _plain_double(
    value: float, step: float, decimals: int, minimum: float = 0.0, maximum: float = 10000.0
) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setValue(value)
    return box


def _row_widget(layout) -> QWidget:
    """Wrap a layout so it can sit in a QFormLayout row."""
    holder = QWidget()
    holder.setLayout(layout)
    layout.setContentsMargins(0, 0, 0, 0)
    return holder


def _fill_functionals(combo: QComboBox) -> None:
    """Populate a combo with the grouped functionals.

    Just the keyword on each row — spelling out what each functional is made the
    list long and hard to scan. The grouping stays: it is what turns fifty
    keywords into five short lists. Group headers are inserted as disabled rows,
    since a QComboBox has no real section header and a bare separator would not
    say what the section is.
    """
    from PySide6.QtGui import QStandardItem

    model = combo.model()
    for group, entries in FUNCTIONAL_GROUPS:
        header = QStandardItem(f"— {group} —")
        header.setFlags(Qt.NoItemFlags)  # a label, not a choice
        model.appendRow(header)
        for keyword, _description in entries:
            combo.addItem(keyword, keyword)


def functional_keyword(combo: QComboBox) -> str:
    """The CRYSTAL keyword a functional combo is currently naming.

    A listed row carries its keyword as data; anything typed is taken at face
    value, after the alias table has had a look at it. Without that, choosing a
    row would write "PBEXC — GGA — PBE (Perdew-Burke-Ernzerhof)" into the deck.
    """
    index = combo.findText(combo.currentText())
    if index >= 0 and combo.itemData(index):
        return str(combo.itemData(index))
    typed = combo.currentText().strip()
    # a typed alias, matched without regard to case: PBE -> PBEXC
    for alias, keyword in FUNCTIONAL_ALIASES.items():
        if typed.upper() == alias.upper():
            return keyword
    return typed


def parse_atomspin(text: str) -> tuple:
    """Read an ATOMSPIN entry into ``((label, spin), ...)``.

    Deliberately forgiving about separators — commas, newlines, semicolons or
    plain spaces all work — because the alternative is a table widget for what
    is usually two numbers. Raises ValueError with something a user can act on.
    """
    import re

    tokens = [t for t in re.split(r"[\s,;]+", text.strip()) if t]
    if not tokens:
        return ()
    if len(tokens) % 2:
        raise ValueError(
            "ATOMSPIN needs an atom number and a spin for each atom, so an even "
            f"number of values — got {len(tokens)}."
        )
    pairs = []
    for raw_label, raw_spin in zip(tokens[::2], tokens[1::2]):
        try:
            label, spin = int(raw_label), int(raw_spin)
        except ValueError:
            raise ValueError(
                f"ATOMSPIN takes whole numbers; could not read {raw_label!r} {raw_spin!r}."
            ) from None
        pairs.append((label, spin))
    return tuple(pairs)


def _plain_spin(value: int, minimum: int, maximum: int) -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    return box


__all__ = ["InputBuilderDialog"]
