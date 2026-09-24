# Phonons

The vibrational modes are read from the CRYSTAL output when it contains a
frequency calculation, and are listed in the **Phonons** panel.

## The list of modes

Each mode is listed with its frequency and, when the output reports the
selection rules, its infrared and Raman activity. The list can be restricted to
the active modes.

## Animation

A mode selected in the list is animated with the **Play** button. Bonds,
coordination polyhedra and hydrogen bonds follow the displacements.

The **Amplitude** is defined as the largest displacement of any atom, so that
one value is appropriate for structures of different size, and the **Speed**
sets the rate of the animation.

```{image} mode-mof5.gif
:alt: The linkers of MOF-5 twisting in a low-frequency mode
:width: 340px
:align: center
:class: only-light
```

```{image} mode-mof5-dark.gif
:alt: The linkers of MOF-5 twisting in a low-frequency mode
:width: 340px
:align: center
:class: only-dark
```

Above: MOF-5 seen down a cube axis, in the lowest optical mode at 17 cm⁻¹ —
the terephthalate linkers twisting between the Zn₄O clusters. The
crystallographic cell is shown: 424 atoms, animated live.

## Exporting an animation

**File → Export phonon animation…** writes the animation of the selected mode.

| Format | Requirement |
| --- | --- |
| GIF, or a sequence of numbered frames | none |
| MP4, MOV, WebM | `pip install CRYSTALLine[video]` |

The resolution, the number of frames and the frame rate are set in the dialog.

## Modes away from Γ

A dispersion calculation (`DISPERSI`) computes modes at several points of the
Brillouin zone. The q points are listed above the mode list, and a mode chosen
at one of them is animated as a travelling wave: each cell drawn carries its
own phase, in the crystallographic cell, in a supercell and after completion of
the molecules at the boundary alike.

The **Tile** button repeats the cell over one period of the wave, and restores
the original cell when pressed again.

```{image} wave-diamond.gif
:alt: A transverse acoustic wave travelling along c through six cells of diamond
:width: 520px
:align: center
:class: only-light
```

```{image} wave-diamond-dark.gif
:alt: A transverse acoustic wave travelling along c through six cells of diamond
:width: 520px
:align: center
:class: only-dark
```

Above: diamond at q = (0, 0, ½), the transverse acoustic mode at 537 cm⁻¹,
with the cell repeated six times along **c** — three wavelengths of the wave.

In a static image the displacement arrows can be scaled by the magnitude of the
displacement and coloured by the phase of the cell in which they are drawn,
which repeats once per wavelength. The amplitude is the same in every cell, and
the phase is what distinguishes them.

## Phonon band structures

Phonon band structures and densities of states are plotted from the **Plot**
menu; see [Property plots](plots.md).
