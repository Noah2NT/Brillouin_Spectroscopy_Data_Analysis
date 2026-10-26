# Spatial-spectral reconstruction

This reconstructs the **expected measured spectrum at each pixel**, using neighbouring
spectra, without fitting Lorentzians. It does not identify a medium-free material
curve. Keep that distinction when interpreting output. The simulator saves the
medium-free curves separately, but they are never used to train or tune denoisers.

## Run the experiment

From the repository root (Python with numpy, scipy, scikit-learn, matplotlib,
threadpoolctl; pandas additionally for vendor CSVs):

The tested versions are recorded in `analysis/reconstruction_requirements.txt`.

```powershell
python -m analysis.reconstruction_benchmark
```

Default: three custom 28 x 28 scenes, 512 bins, 16/4/1/0.25/0.0625 s exposures,
three independent evaluation seeds. This is a 1/2/4/8/16x relative shot-noise
sweep. Truth is unchanged across exposures. Noise increases by reducing photon
counts using the simulator's detector, not by adding Gaussian noise to an
already noisy measurement. Outputs go to `outputs/reconstruction_benchmark`.

Small smoke experiment:

```powershell
python -m analysis.reconstruction_benchmark --size 16 --bins 128 --repeats 2 --iterations 12 --scenes cell_domains --exposures 16 1 0.0625 --output outputs/reconstruction_smoke
```

The implementation is in `brillouin_denoising.py`; custom scene generation is in
`simulator/generate_reconstruction_scenes.py`. Existing simulator and analysis
files are not modified.

## What was implemented from the paper

Salmon, Harmany, Deledalle and Willett (2014), *Poisson noise reduction with
non-local PCA*, [arXiv:1206.0338v4](https://arxiv.org/abs/1206.0338),
DOI 10.1007/s10851-013-0435-6.

- Spatial-spectral patch extraction and non-local Poisson/Bregman clustering.
- Low rank of the **log intensity**: `expected_counts = exp(U @ V)`.
- Poisson likelihood, full block Newton updates (Eqs. 13-14), Hessian damping.
- Averaging overlapping reconstructed patches back onto the original raster.
- An Anscombe-transformed patch-PCA comparator.
- A `poisson_spectral` control: one spectrum per patch, independent per-pixel
  coefficients, shared non-local dictionary. This avoids explicit spatial patch
  averaging and was added after the initial narrow-feature test showed attenuation.

This is the non-sparse NLPCA algorithm, not the optional L1-penalised NLSPCA.
Adaptations: full-spectrum patches, coarse spectral sums for clustering,
deterministic SVD initialization, and line searches. Full Hessian blocks retain
cross-component terms; the apparent diagonal-Hessian statements in the paper
are not used to discard these terms. No claims of reproducing its numerical tables.

## Success metrics and selection

An independent calibration acquisition is photon-split 80/20. Configuration
selection uses only held-out Poisson deviance. Each selected configuration then
fits fresh evaluation acquisitions; truth is used exclusively for scoring.
The high-exposure-configuration control is refitted at every noise level to test
whether one fixed parameter choice transfers. It does not borrow high-SNR data.

`summary.csv` reports mean and SD across independent noise seeds, plus the number
of individual realizations passing all criteria (not only their average):

| Metric | What failure it detects |
|---|---|
| Full-curve shape error inside cell/gel | Wrong spectral reconstruction, independently of brightness |
| Boundary and small-feature shape error | Smoothing across interfaces or deleting small structures |
| Centroid / IQR errors | Spectral displacement and broadening without relying on a peak fit |
| Signed feature contrast relative to truth | Producing stable images by erasing biological variation |
| Difference from high-exposure reconstruction | Exposure-dependent reconstruction changes |
| Pairwise differences across independent noise seeds | Unstable reconstructions at a fixed exposure |
| Independent excess Poisson deviance | Predictive accuracy on new photons |
| Acquisition-noise-scaled truth error / independent residual | Whether discrepancies exceed shot-noise expectations |
| Runtime | Practical cost per scan |

Centroid is a finite-window first moment, **not BFS**. IQR is central-50%-area
width, **not FWHM**. The same window is used for truth and every reconstruction.
The independent residual mean square should average `1 + shot_noise_error`,
not exactly 1, because reconstruction is itself uncertain. `poisson_variance`
in output files is acquisition variance, not an uncertainty interval for the
reconstructed curve. Rank is not interpreted as material count.

Thresholds are explicit in the script and report. They are provisional engineering
targets set before the main run, not established biological equivalence margins.
Failure at extreme noise is reported, not fixed by changing the acceptance criteria.
Three seeds are useful for an initial study, not enough for tight tail probabilities.

## Reconstruct an existing scan

NPZ input: `counts` with shape `(height, width, bins)`, optionally `frequency`,
`x`, `y`. Outputs preserve original count/exposure units.

```powershell
python -m analysis.reconstruct_scan outputs/reconstruction_benchmark/data/cell_domains_1s_seed0.npz --output outputs/cell_reconstructed.npz --select
```

Vendor CSV input: this project's `Raw Data`, `X`, `Y`, `startFrequency`,
`endFrequency` columns. Choose a single Brillouin branch in GHz:

```powershell
python -m analysis.reconstruct_scan "path/to/scan.csv" --window 3 8.5 --output outputs/scan_reconstructed.npz --select
```

The loader sorts spatial coordinates and rejects duplicate/missing pixels or
varying frequency axes. It crops bins but never folds, interpolates, subtracts
baseline or normalises before fitting. Adjust the window to your actual peak.
If your signal is camera ADU, convert using a validated detector model first;
integer-valued ADU is not necessarily Poisson photon counts. `--select` uses
exact Poisson thinning and is inappropriate for correlated/preprocessed data.

Use `--method poisson_spectral --select` to compare the variant preserving each
pixel's own coefficient vector. Larger spatial patches trade resolution for
noise reduction; the best full-image prediction need not preserve thin features.

Without `--select`, `--rank`, `--patch`, `--clusters` and `--iterations` control a
fixed configuration. All methods are deterministic given input and seed.

```python
from brillouin_denoising import DenoiseConfig, reconstruct
expected = reconstruct(raw_count_cube, DenoiseConfig(rank=3, patch=3, clusters=4))
curve = expected / expected.sum(axis=-1, keepdims=True)
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

Checks patch extraction/reprojection at edges, deterministic reconstruction,
monotonic Poisson optimisation, positive output, zero-count handling, photon-split
moments/independence, residual calibration and simulator reproducibility.

To refresh figures and tables from saved numeric results without rerunning fits:

```powershell
python -m analysis.reconstruction_benchmark --report-only
```

The initial recorded run was extended with the spectral-only control using
`python -m analysis.extend_reconstruction_benchmark`; new default runs already
include that control. The extension refuses to duplicate existing results.
## Detailed patch walkthrough

Run `python -m analysis.patch_walkthrough` to generate an offline interactive
walkthrough in `outputs/patch_walkthrough/index.html`. The default is a 28 × 28
synthetic cell, 512 bins, 1 s exposure, rank 3, 3 × 3 patches, four clusters,
12 Newton iterations and seed 7. No truth enters the reconstruction.

Every pixel is selectable at every coefficient/dictionary half-update. The guide
includes preparation maps, fixed-scale cell maps, full spectra, spectral evolution
heatmaps, downloadable per-pixel CSV, residual graphs, independent-photon scoring,
method comparisons, questions and explanations. Each numerical stage also has a
PNG. `walkthrough.npz` preserves full precision predictions and final patch proposals.
The HTML embeds all spectra as float32 for offline use (about 74 MB at defaults).
Tracing stores group predictions as well as whole cubes, so memory use scales with
iterations × patches × patch area × bins. Use smaller windows or fewer iterations
for very large scans. Normal `reconstruct` calls do not allocate trace history.

For a calibrated scan:

```powershell
python -m analysis.patch_walkthrough --input path/to/counts.npz --output outputs/my_scan_walkthrough
python -m analysis.patch_walkthrough --input path/to/scan.csv --window 3 8.5 --output outputs/my_scan_walkthrough
```

Input files must be in Poisson count units. For files without known truth or a
second acquisition, accuracy scores are unavailable. Coordinates are preserved.
For NPZ without frequency, the axis is a bin coordinate rather than GHz.

Programmatic instrumentation:

```python
trace = {}
expected = reconstruct(counts, DenoiseConfig(), trace=trace)
# trace['cubes']: overlap-averaged estimates at each half-update
# trace['groups']: per-cluster (name, patch predictions, objective) records
# trace['final_patches']: final individual patch proposals
```

Cluster traces are aligned by local iteration number for display, not wall-clock
execution order. Converged groups are frozen. The last trace cube equals the
returned fit to floating-point precision; gauge balancing may change rounding.


The default `index.html` now opens the interactive slide demo: preparation,
initialization, every coefficient/dictionary update, overlap averaging, residuals,
and results. Clicking a pixel preserves its selection between slides. The lower
plot displays the exact current-minus-previous curve, with optional magnification.
The long-form explanation remains in `tutorial.html`.

Refresh both pages from saved numerical results without refitting:

```powershell
python -m analysis.patch_walkthrough --render-only
```

For the real XY cell CSV, omit `--window` in `analysis.patch_walkthrough` to use its constant `marker1`/`marker2` fitting interval (about 2.92-7.98 GHz). The stronger 10.4 GHz peak is outside that interval and does not show the cell. Real-data reports provide modification diagnostics instead of unmeasurable truth-error tables. The corrected run is in `outputs/patch_walkthrough_real_xy_corrected`.
