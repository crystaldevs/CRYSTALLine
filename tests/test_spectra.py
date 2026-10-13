"""Enumerating the vibrational spectra a CRYSTAL output holds, and plotting them.

Discovery is driven by attribute *name* rather than a hard-coded list, so the
parsing is what needs pinning: it decides how a curve is labelled, grouped and
ordered, and it is what lets a level or component a future CRYSTALClear adds be
picked up without a change here.
"""

import glob
import os

import numpy as np
import pytest

import sample_data as sample  # noqa: E402
from crystalline.crystalio.spectra import (
    SpectrumKind,
    _parse,
    available_spectra,
    load_spectra,
)


@pytest.mark.parametrize(
    "attribute, kind, level, temperature, polarisation",
    [
        ("IR_HO_0K", "IR", "harmonic", "0 K", ""),
        ("IR_VSCF_T", "IR", "VSCF", "T", ""),
        ("IR_VCI_0K", "IR", "VCI", "0 K", ""),
        ("Ram_HO_0K_tot", "Raman", "harmonic", "0 K", "total"),
        ("Ram_HO_T_par", "Raman", "harmonic", "T", "parallel"),
        ("Ram_VSCF_0K_per", "Raman", "VSCF", "0 K", "perpendicular"),
        ("Ram_VPT2_T_comp_xy", "Raman", "VPT2", "T", "xy"),
        ("Ram_VCI_0K_comp_zz", "Raman", "VCI", "0 K", "zz"),
    ],
)
def test_attribute_names_are_read_into_what_they_mean(
    attribute, kind, level, temperature, polarisation
):
    parsed = _parse(attribute)

    assert parsed == SpectrumKind(attribute, kind, level, temperature, polarisation)


@pytest.mark.parametrize(
    "attribute",
    ["IR_HO_0K_tot", "Ram_HO_0K", "Ram_HO_0K_comp_qq", "IR_XYZ_0K", "eigenvector", "IR_"],
)
def test_names_that_are_not_spectra_are_ignored(attribute):
    """``dir()`` on a Crystal_output turns up plenty that isn't a spectrum."""
    assert _parse(attribute) is None


def test_labels_and_groups_read_the_way_a_legend_needs():
    ir = _parse("IR_HO_0K")
    raman = _parse("Ram_VCI_T_comp_xz")

    assert ir.label == "IR (harmonic, 0 K)"
    assert raman.label == "Raman xz (VCI, T)"
    # Sections: every IR curve in one, Raman split by level and temperature
    # (each of those carries nine polarisations).
    assert ir.group == "IR"
    assert raman.group == "Raman (VCI, T)"
    # ...and within a section, what tells the curves apart
    assert ir.leaf_label == "harmonic, 0 K"
    assert raman.leaf_label == "xz"


def test_menu_order_is_ir_then_raman_harmonic_then_anharmonic():
    names = [
        "Ram_VCI_T_comp_zz", "Ram_HO_0K_tot", "IR_VCI_0K", "IR_HO_0K",
        "Ram_HO_0K_comp_xx", "Ram_HO_0K_par", "IR_HO_T",
    ]
    kinds = sorted((_parse(n) for n in names), key=lambda k: k.sort_key)

    assert [k.attribute for k in kinds] == [
        "IR_HO_0K",          # IR first, harmonic first, 0 K first
        "IR_HO_T",
        "IR_VCI_0K",         # then the anharmonic levels
        "Ram_HO_0K_tot",     # then Raman, powder averages before components
        "Ram_HO_0K_par",
        "Ram_HO_0K_comp_xx",
        "Ram_VCI_T_comp_zz",
    ]


# A real ANHARM run, if one is on this machine: the stub below covers the
# parsing, but only a genuine output confirms CRYSTALClear populates the
# VSCF/VCI attributes the way its docstring says.
# Narrowed to the CRYSTAL output itself: the same directories hold scheduler
# logs, which are also ``.out`` and carry no spectra at all.
_ANHARM = sample.matches("co2/*/co2_anh_*.out")


@pytest.mark.skipif(not _ANHARM, reason="no sample anharmonic .out available")
def test_a_real_anharmonic_run_yields_every_level():
    pytest.importorskip("CRYSTALClear")

    kinds = available_spectra(_ANHARM[0])

    assert {k.level for k in kinds} == {"harmonic", "VSCF", "VCI"}
    assert {k.temperature for k in kinds} == {"0 K", "T"}
    assert {k.kind for k in kinds} == {"IR", "Raman"}
    # every Raman polarisation, at every level and temperature
    assert len([k for k in kinds if k.kind == "Raman"]) == 3 * 2 * 9


@pytest.mark.skipif(not _ANHARM, reason="no sample anharmonic .out available")
def test_the_anharmonic_correction_moves_the_co2_stretch_down():
    """A physics check on the parse: anharmonicity red-shifts the CO2
    asymmetric stretch from its harmonic value towards the observed ~2349 cm^-1.
    Reading the wrong block, or crossing levels up, would not reproduce that."""
    pytest.importorskip("CRYSTALClear")
    data = load_spectra(_ANHARM[0])
    by_label = {kind.label: array for kind, array in data.items()}

    def strongest(label):
        curve = by_label[label]
        return curve[np.argmax(curve[:, 1]), 0]

    harmonic = strongest("IR (harmonic, 0 K)")
    vscf = strongest("IR (VSCF, 0 K)")
    vci = strongest("IR (VCI, 0 K)")

    assert 2300 < vci < harmonic < 2500
    assert abs(vscf - vci) < 50          # the two anharmonic levels roughly agree
    assert harmonic - vci > 20           # and both sit well below the harmonic one
    # VCI adds overtones and combination bands, so it carries more transitions
    assert len(by_label["IR (VCI, 0 K)"]) > len(by_label["IR (harmonic, 0 K)"])


class _StubOutput:
    """A Crystal_output carrying spectra, without a file behind it.

    Neither of the outputs to hand has an ANHARM run, so the VSCF/VPT2/VCI path
    is exercised here rather than left untested until someone produces one.
    """

    def __init__(self, **arrays) -> None:
        self.__dict__.update(arrays)

    def get_IR(self):
        return self

    def get_Raman(self):
        return self

    def get_anh_spectra(self):
        return self


def _stub_spectra(monkeypatch, **arrays):
    """Point ``load_spectra`` at a stub instead of a real output file."""
    import CRYSTALClear.crystal_io as crystal_io

    monkeypatch.setattr(crystal_io, "Crystal_output", lambda _path: _StubOutput(**arrays))
    return load_spectra("anywhere.out")


def test_anharmonic_levels_are_discovered_alongside_the_harmonic_ones(monkeypatch):
    pytest.importorskip("CRYSTALClear")
    curve = np.array([[1000.0, 1.0], [2000.0, 2.0]])

    found = _stub_spectra(
        monkeypatch,
        IR_HO_0K=curve, IR_VSCF_0K=curve, IR_VCI_T=curve,
        Ram_VCI_0K_tot=curve, Ram_VPT2_T_comp_yz=curve,
    )

    assert {k.label for k in found} == {
        "IR (harmonic, 0 K)",
        "IR (VSCF, 0 K)",
        "IR (VCI, T)",
        "Raman total (VCI, 0 K)",
        "Raman yz (VPT2, T)",
    }
    assert all(v.shape == (2, 2) for v in found.values())


def test_empty_and_malformed_arrays_are_not_offered(monkeypatch):
    pytest.importorskip("CRYSTALClear")

    found = _stub_spectra(
        monkeypatch,
        IR_HO_0K=np.array([[100.0, 1.0]]),   # real, if short
        IR_VCI_0K=np.empty((0, 2)),          # the run didn't compute it
        Ram_HO_0K_tot=np.zeros(5),           # not (N, 2): not a spectrum
    )

    assert [k.attribute for k in found] == ["IR_HO_0K"]


def test_a_file_with_no_spectra_yields_nothing(tmp_path):
    pytest.importorskip("CRYSTALClear")
    plain = tmp_path / "plain.out"
    plain.write_text(" SOME OUTPUT\n EEEEEEEEEE TERMINATION\n")

    assert load_spectra(str(plain)) == {}
    assert available_spectra(str(plain)) == []
    assert available_spectra(None) == []


def test_plot_spectra_overlays_every_curve_with_a_legend():
    pytest.importorskip("CRYSTALClear")
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from crystalline.crystalio.spectra import plot_spectra

    curve = np.array([[500.0, 1.0], [1500.0, 3.0]])
    figure = plot_spectra(
        [("xx", curve), ("yy", curve * 1.1), ("zz", curve * 0.5)],
        lineshape="lorentz",
        hwhm=8.0,
        frequency_range=(0.0, 2000.0),
    )

    axes = figure.axes[0]
    assert len(axes.lines) >= 3
    assert [t.get_text() for t in axes.get_legend().get_texts()] == ["xx", "yy", "zz"]
    assert "Wavenumber" in axes.get_xlabel()


def test_plotting_a_single_curve_needs_no_legend():
    pytest.importorskip("CRYSTALClear")
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from crystalline.crystalio.spectra import plot_spectra

    figure = plot_spectra([("IR", np.array([[500.0, 1.0]]))], title="IR (harmonic, 0 K)")

    assert figure.axes[0].get_legend() is None
    assert figure.axes[0].get_title() == "IR (harmonic, 0 K)"


def test_plotting_nothing_is_an_error_not_an_empty_figure():
    pytest.importorskip("CRYSTALClear")
    from crystalline.crystalio.spectra import plot_spectra

    with pytest.raises(ValueError):
        plot_spectra([])


@pytest.mark.parametrize("eta", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_pseudo_voigt_mixes_the_two_profiles_by_eta(eta):
    """``eta*Lorentzian(hwhm) + (1-eta)*Gaussian(stdev)`` — checked against the
    two pure profiles drawn with the same widths, so the three parameters are
    known to reach the right places rather than being assumed to."""
    pytest.importorskip("CRYSTALClear")
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from crystalline.crystalio.spectra import plot_spectra

    peak = np.array([[1000.0, 1.0]])

    def profile(**kwargs):
        figure = plot_spectra([("x", peak)], frequency_range=(900.0, 1100.0), **kwargs)
        y = np.asarray(figure.axes[0].lines[-1].get_ydata())
        plt.close(figure)
        return y

    lorentzian = profile(lineshape="lorentz", hwhm=10.0)
    gaussian = profile(lineshape="gauss", stdev=10.0)
    voigt = profile(lineshape="pvoigt", hwhm=10.0, stdev=10.0, eta=eta)

    assert np.allclose(voigt, eta * lorentzian + (1.0 - eta) * gaussian)


def test_the_two_pseudo_voigt_widths_are_independent():
    """They were one shared value before, which silently tied them together."""
    pytest.importorskip("CRYSTALClear")
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from crystalline.crystalio.spectra import plot_spectra

    peak = np.array([[1000.0, 1.0]])

    def profile(hwhm, stdev, eta=0.5):
        figure = plot_spectra(
            [("x", peak)], lineshape="pvoigt", hwhm=hwhm, stdev=stdev, eta=eta,
            frequency_range=(900.0, 1100.0),
        )
        y = np.asarray(figure.axes[0].lines[-1].get_ydata())
        plt.close(figure)
        return y

    assert not np.allclose(profile(2.0, 30.0), profile(30.0, 2.0))
    # at eta = 1 the Gaussian width is out of the mixture entirely
    assert np.allclose(profile(10.0, 1.0, eta=1.0), profile(10.0, 99.0, eta=1.0))


def test_the_legend_can_be_turned_off():
    """Several curves are named by default; on a crowded spectrum the box
    hides the peaks it names, so it comes off."""
    pytest.importorskip("CRYSTALClear")
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from crystalline.crystalio.spectra import plot_spectra

    curve = np.array([[500.0, 1.0], [1500.0, 3.0]])
    datasets = [("xx", curve), ("yy", curve * 1.1)]

    assert plot_spectra(datasets).axes[0].get_legend() is not None
    assert plot_spectra(datasets, legend=False).axes[0].get_legend() is None


def _stack(datasets, **options):
    """Each curve's lines, and the gap the offset put between the first two."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from crystalline.crystalio.spectra import plot_spectra

    figure = plot_spectra(datasets, **options)
    lines = [np.asarray(line.get_ydata()) for line in figure.axes[0].lines]
    plt.close(figure)
    # The gap between the two, rather than either baseline: a Lorentzian's
    # tails never quite reach zero.
    return lines, lines[1].min() - lines[0].min()


def test_an_offset_stacks_the_curves_by_a_fraction_of_the_tallest_peak_drawn():
    """Drawn, not reported: a Lorentzian of HWHM 5 peaks at a sixteenth of its
    transition, so an offset measured against the transitions lifted each curve
    by many times anything on screen, and all but the first steps of the
    slider were wasted."""
    pytest.importorskip("CRYSTALClear")
    curve = np.array([[500.0, 1.0], [1500.0, 4.0]])
    datasets = [("first", curve), ("second", curve)]

    (flat, other), no_gap = _stack(datasets)
    _lines, gap = _stack(datasets, offset=0.5)

    assert no_gap == pytest.approx(0.0, abs=1e-12)
    assert gap == pytest.approx(0.5 * flat.max(), rel=1e-3)     # half the peak on screen
    assert gap < 0.1 * 4.0                                      # nowhere near half the transition


def test_wider_lines_keep_the_stack_in_proportion():
    """Broadening lowers every peak, and the gaps go down with them: the stack
    looks the same at any width, rather than drifting apart as the lines are
    widened, which is what a gap fixed in intensity would do."""
    pytest.importorskip("CRYSTALClear")
    curve = np.array([[500.0, 1.0], [1500.0, 4.0]])
    datasets = [("first", curve), ("second", curve)]

    for hwhm in (2.0, 5.0, 20.0):
        (flat, _other), _ = _stack(datasets, hwhm=hwhm)
        _lines, gap = _stack(datasets, hwhm=hwhm, offset=1.0)
        assert gap / flat.max() == pytest.approx(1.0, rel=2e-3), hwhm


def test_coincident_lines_are_measured_together():
    """A degenerate mode is listed once per partner at one frequency, and its
    peak is their sum — what is drawn, so what the offset is measured against."""
    pytest.importorskip("CRYSTALClear")
    pair = np.array([[800.0, 2.0], [800.0, 2.0], [1500.0, 3.0]])   # 4 at 800 beats 3
    datasets = [("first", pair), ("second", pair)]

    (flat, _other), _ = _stack(datasets)
    _lines, gap = _stack(datasets, offset=1.0)

    assert gap == pytest.approx(flat.max(), rel=1e-3)


def test_sticks_keep_to_the_frequency_window():
    """plot_cry_spec draws every stick it is handed whatever fmin and fmax say,
    so a stick spectrum asked for 200-1200 cm-1 ran on to 3800."""
    pytest.importorskip("CRYSTALClear")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from crystalline.crystalio.spectra import plot_spectra

    curve = np.array([[500.0, 1.0], [900.0, 2.0], [3800.0, 50.0]])
    figure = plot_spectra([("IR", curve)], lineshape="bars", frequency_range=(200.0, 1200.0))
    line = figure.axes[0].lines[0]
    frequencies, intensities = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
    plt.close(figure)

    assert frequencies.max() <= 1200.0
    assert intensities.max() == 2.0                            # the 3800 stick is not there
