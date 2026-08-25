# Brillouin Sample Simulator

Paint a synthetic cell out of named materials, choose a scan resolution, and
generate CSV files in the same 13-column format as the rest of this project —
with noise derived from Oscar Nieves, *Noise in Stimulated Brillouin
Scattering* (PhD thesis, 2021), and with the true answer saved alongside.

The point is to have data where the right answer is known, so an algorithm can
be scored rather than eyeballed, and so the noise level can be swept until it
breaks.

```bash
python -m simulator
```

## What it does

**Paint.** A canvas finer than the scan grid (8 sub-cells per scan pixel by
default) holds one material index per cell. Pencil, line, rectangle, ellipse,
polygon, flood fill, eraser and picker, with zoom, pan and undo. An overlay
shows where the scan pixel boundaries fall.

**Mix.** Downsampling to the scan grid uses exact area overlap, so a pixel
straddling a boundary reports the true proportion of each material inside its
footprint. Each material contributes a Lorentzian normalised to unit area,
scaled by its fraction — so a mixed pixel keeps the same total peak area, the
peak just moves, broadens or splits. The confocal PSF blurs composition before
downsampling, so pixels mix optically as well as geometrically.

**Simulate.** Thermal and laser noise follow the thesis; shot noise, dark
counts and read noise are added on top. Details in *Physics* below.

**Generate.** Sweep exposure (and optionally step size), repeat each setting,
and write every scan plus a ground-truth sidecar and a shared `manifest.csv`.

Noise is different on every run -- there is no seed to set. A seed *is* drawn
and recorded, in the manifest and in each sidecar, so any individual scan can
be reproduced later by passing that number back through
`ScanSettings(seed=...)`. Different by default, reproducible when you need it.

## Output

For each generated scan:

| File | Contents |
|---|---|
| `<stem>.csv` | The 13-column instrument table. Indistinguishable in structure from `data/csv/*.csv`. |
| `<stem>_truth.npz` | Ground truth: per-pixel material fractions, true shift, linewidth and area, and what the same fitter returns with no noise. |
| `manifest.csv` | One row per file with every setting, the seed used and summary error metrics. |

The fitted columns in the CSV come from genuinely fitting the noisy spectrum
with this project's own `brillouin_fitting.fit_lorentzian_iterative`, so a
generated file contains exactly the information a real one does and nothing
more. Ground truth is kept strictly out of it.

Reading the truth back:

```python
from simulator.core.writer import load_truth_sidecar

truth = load_truth_sidecar("sample_..._truth.npz")
fractions = truth["fractions"]              # (n_pixels, n_materials)
names     = truth["meta"]["material_names"]
true_bfs  = truth["bfs_ghz"]                # what the materials actually are
ref_bfs   = truth["noisefree_bfs_ghz"]      # what a perfect fit would return
true_lw   = truth["linewidth_ghz"]          # FWHM, not the CSV's half width
```

Row order is X-fastest then Y, matching the CSV, so any per-pixel array
reshapes with `.reshape(grid_h, grid_w)`.

**Which truth to score against.** `bfs_ghz` is the area-weighted true shift.
`noisefree_bfs_ghz` is what the same Lorentzian fitter returns on the noise-free
spectrum. Score against the second to measure *noise tolerance*; score against
the first to also include the error a single Lorentzian makes on a Voigt-shaped,
multi-material peak. On a typical sample those differ by well under 1 MHz, but
for widely separated material mixtures the gap is the interesting part.

## Physics

Equation numbers are from the thesis.

**Peak position** — Eq. (1.1), `nu_B = 2 n v_a sin(theta/2) / lambda`.
Materials store the shift directly by default, since that is how the values in
this project are actually known; the calculator is there if you want it.

**Mixing** — each material contributes `area_fraction * efficiency` of a
unit-area Lorentzian at `+-nu_B`. Verified: pure, 50/50 and three-way mixtures
of the same photon budget integrate to the same area to within 0.3 parts in
2000. Turn on *Force identical total peak area* to make that exact regardless
of per-material brightness.

**Linewidth** — every linewidth in this package is a **full width at half
maximum**, matching the `Linewidth` column in your paclitaxel CSVs and
`LW_Lorentz` in the `*_Fitted.xlsx` workbooks. The instrument's own `Half Width`
column is a *half* width, so the writer halves the value on the way out — that
is the only place the two conventions meet.

Lorentzians convolve by adding widths, so the intrinsic Brillouin linewidth,
the laser line (Eq. 3.20) and the Lorentzian part of the spectrometer response
simply sum. The Gaussian part of the response is applied
by convolution, making each line a Voigt. That matters: real peaks here fall
off faster than a Lorentzian, and a pure-Lorentzian sample would be subtly
easier to fit than real data.

**Thermal noise** — Eq. (3.12) gives `sigma = k_B * T * alpha_ac`, so signal is
proportional to absolute temperature. Eq. (3.18) and the spectrum that follows
it establish that the thermal acoustic background is an Ornstein-Uhlenbeck
process with a Lorentzian power spectrum of FWHM `Gamma`.

A detector integrating for `t_exp` averages about `t_exp / tau_coh` independent
realisations of that process, which makes each spectral bin Gamma-distributed
with relative fluctuation `1/sqrt(M)`. For a 0.6 GHz line the coherence time is
about 0.5 ps, so a 1 second exposure averages of order 1e12 realisations and
thermal fluctuation washes out to parts per million. **That is the physically
correct answer**: thermal noise sets the signal level and the linewidth, but the
visible pixel-to-pixel scatter in real data is photon shot noise. The speckle
term is still exposed as a knob for short-exposure or few-mode regimes.

`simulator.core.noise.ou_thermal_series` implements Eq. (5.13)/(5.14) verbatim
for anyone who wants to check that shortcut against an explicit realisation.

**Laser phase noise** — Eq. (3.23)/(3.27), `phi(t) = sqrt(dOmega) W(t)`. On a
time-averaged spectrum this is exactly a Lorentzian broadening, applied
analytically; `noise.laser_phase_walk` synthesises the random walk explicitly.

**Detector** — not from the thesis, which models a waveguide rather than a
photon-counting microscope. Dark current scaled by exposure, Poisson shot
noise, optional EMCCD excess noise, Gaussian read noise, then rounding and
clipping at zero.

The elastic peak height is calibrated at the instrument's
`elastic_reference_exposure_s` (15 s by default) and scales linearly with
exposure. Read noise remains a per-acquisition term, while dark counts and both
elastic and Brillouin photons accumulate with integration time.

## How close is it to real data?

Matched conditions -- 31x41 px, 1 um step, 15 s exposure -- against
`Cell_Col Healthy_xy_scanning30um_40um_step1um_15s.csv`:

| Metric | Simulated | Real | Ratio |
|---|---|---|---|
| Elastic peak height | 30 329 | 29 935 | 1.01 |
| Brillouin peak counts | 777 | 710 | 1.09 |
| `Half Width` (CSV, a half width) | 0.389 | 0.392 | 0.99 |
| `Indensity` | 929 | 922 | 1.01 |
| `Frequency` | 5.516 | 5.493 | 1.00 |
| `Base` | -3.0 | -28.5 | 0.11 |

The `Base` column is the one clear mismatch. Real spectra have even cleaner
wings than a Voigt, so the vendor's Lorentzian fit lands on a strongly negative
baseline. Nothing in this project reads that column -- every loader refits from
the raw spectra -- so it is left as is rather than reverse-engineered.

## Where the default numbers come from

Worth knowing, because the tiers differ a lot in how much they should be
trusted.

**Measured from `data/csv` (27 317 pixels across 14 files).** The Brillouin
shift runs 5.475-5.767 GHz (median 5.577) and `Half Width` runs 0.246-0.436 GHz,
i.e. a linewidth of 0.49-0.87 GHz FWHM. The absolute brightness scale is
calibrated against those files too: `photon_rate` and `elastic_peak_counts` are
set so a matched 15 s simulation reproduces the table above.

Two material shifts are measured rather than assumed. Plexiglass sits at
**11.75 GHz**, read straight off `water_plexiglass.csv`, where that peak's
intensity anti-correlates with the water peak at r = -0.96 and varies 563-fold
across the field. Polystyrene at **10.38 GHz** matches the weak, near-uniform
10.4-10.6 GHz peak present in the Plate2/Plate3 scans, which are cells cultured
in plastic plates.

**Derived so the model reproduces the measurements.** Intrinsic linewidths are
back-solved: target observed linewidths were chosen to span the measured range,
then the Voigt width relation was inverted given the 0.40 GHz Gaussian
instrument response. Be aware the split between "material" and "instrument"
width is *not* identifiable from the data -- only the total is -- so the
0.40 GHz figure is a modelling choice, picked because it puts the peak wings and
the fitted baseline closest to real spectra. Acoustic velocities are likewise
back-solved from each stored shift via Eq. (1.1), so the shift calculator agrees
with the stored value to better than 10 MHz for every material. They land in
sensible ranges (water 1487 m/s, PMMA 2835 m/s, fused silica 5900 m/s), which is
a reassurance rather than a validation.

**Assumed, with no measurement behind them.** Which material gets which shift,
apart from plexiglass and polystyrene: the cytoplasm / nucleus / nucleolus
ordering is the conventional stiffness ordering placed across the measured
range, not data. Refractive indices are literature-typical values. And
`brillouin_efficiency` / `elastic_efficiency` are invented outright -- only
their *relative* ordering carries a claim (denser material scatters more), and
the overall scale is what was calibrated. If per-material brightness matters to
your conclusions, set these from your own measurements.

## Layout

```
simulator/
  core/          headless, no Qt, unit-testable
    physics.py     line shapes, Eq. (1.1), fluctuation-dissipation
    materials.py   Material, MaterialLibrary, shipped defaults
    instrument.py  axis, response, detector; presets
    drawing.py     CellDrawing: the painted index map
    raster.py      area-exact downsampling, PSF blur
    spectrum.py    composing a pixel spectrum from fractions
    noise.py       thermal (OU), laser phase, detector chain
    scan.py        ScanSettings, simulate_scan, fitting
    writer.py      CSV, truth sidecar, manifest
    batch.py       sweep expansion and the batch runner
  ui/            Qt shell over the above
  resources/     default_materials.json, and your saved library
```

The core never imports Qt, so it can be driven from a script:

```python
from simulator.core import materials, instrument, drawing, scan, writer

lib  = materials.MaterialLibrary.load()
inst = instrument.preset_narrow()
d    = drawing.CellDrawing.load("sample.bcell.npz")
res  = scan.simulate_scan(d, lib, inst,
                          scan.ScanSettings(grid_w=40, grid_h=40, exposure_s=10.0))
print(res.settings.seed)      # the seed that was drawn, for replaying this run
writer.write_scan_csv("out.csv", res, inst)
writer.write_truth_sidecar("out_truth.npz", res, inst)
```

## Notes and caveats

**Laser wavelength.** Not recorded anywhere in this repo. Reproducing the
measured ~5.51 GHz water shift through Eq. (1.1) with `n = 1.333` and
`v_a = 1483 m/s` in backscattering implies about 720 nm, which is the default.
Materials store their shift directly so nothing depends on it, but the Eq. (1.1)
calculator will be wrong until the real value is set in the Instrument tab.

**Material order matters.** Drawings store material *indices*. Reordering the
library reinterprets existing drawings; saved drawings carry their material
names and are remapped on load, warning about anything that has gone missing.

**Axis geometry.** The frequency axis must stay symmetric about zero, because
`brillouin_preprocessing.fold_brillouin_spectrum` folds about its midpoint. The
Instrument tab refuses to generate otherwise.
