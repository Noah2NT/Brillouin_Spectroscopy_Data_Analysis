# Brillouin Spectroscopy Data Analysis

Research code, experimental scans, simulations and saved analyses from Brillouin spectroscopy work. The main workflows cover peak fitting, sample/medium separation, spatial maps, line-scan comparisons and spectral denoising.

## Code

| Location | Contents |
|---|---|
| Root Python files | Shared preprocessing, Lorentzian/tabletop fitting, PCA/VCA decomposition, plotting, denoising and spatial feature extraction (`brillouin_*.py`, `feature_extraction.py`). |
| `analysis/` | Scripts for reconstruction benchmarks, interactive patch walkthroughs, synthetic-data comparisons, noise sweeps and edge detection. See [RECONSTRUCTION.md](analysis/RECONSTRUCTION.md) for the denoising workflow. |
| `analysis/clustering/` | Clustering demonstrations, smoothing and threshold-mask sweeps. |
| `analysis/line_scans/` | Control/treatment and dose comparisons, profile fitting, core/periphery analysis and per-scan feature statistics. |
| `analysis/mer/` | Maximum-entropy spectrum reconstruction and its demonstration/check script. |
| `analysis/misc/` | Exploratory scripts, background visualizations, a control/treated classifier and an interactive single-peak/two-peak/measured-background fitting demo. |
| `analysis/pca_vca/` | PCA/VCA, background-pixel selection and pooling, medium subtraction, single/two-Lorentzian fits, measured-reference-plus-peak fits and hydrogel spatial analyses. `parametric_fit.py` uses a measured medium reference with a parametric sample peak. |
| `simulator/` | Synthetic scan application and sweep/scene generators. See the [simulator guide](simulator/README.md). |
| `simulator/core/` | Materials, instrument response, spatial mixing, spectra, noise, scan generation and CSV/truth-file writing. |
| `simulator/ui/` | Drawing canvas, material editor, scan settings, previews and batch-generation interface. |
| `simulator/resources/` | Default and locally saved material libraries. |
| `simulator/samples/` | Currently empty. |
| `tests/` | Automated numerical checks for denoising, reconstruction, photon thinning, simulator behavior and data loading. |

## Input data

| Folder under `data/` | Contents |
|---|---|
| `csv/` | Experimental cell and hydrogel scans with raw spectra and instrument fit columns. `Exclude because they suck/` holds the excluded healthy-cell and water–plexiglass scans. |
| `paclitaxel_line_scans/` | CSV/XLSX line scans grouped into `Control`, `1.56 nM`, `3.125 nM`, `6.25 nM` and `12.5 nM`. `Control/` is the untreated-spheroid group shared by both drug series. |
| `carboplatin_1.5.26_20x/` | CSV/XLSX scans grouped into `control`, `50um`, `100um`, `200um` and `400um`. |
| `2D scans pacli 6.9.26/` | Five dated acquisition folders containing control and four paclitaxel-dose 2D scans. |
| `line_scans_old/` | Earlier `Control`, `Treated` and `Control 2D` scans; `Temp (scans i dont like)/` contains a set-aside scan. |
| `FAKE_DATA/` | Three repeated simulated scans, corresponding `_truth.npz` files and a generation manifest. |

## Saved results

`outputs/` contains generated figures, tables, arrays and reports rather than the analysis source. Its root also contains line-scan and separation-summary workbooks.

| Folder under `outputs/` | Contents |
|---|---|
| `Ashutosh research/` | `spheroid corrections 01.10.2026/` contains `Carboplatin/` and `Paclitaxel/` profile, dose and core/periphery figures. |
| `background_endmember/` | Pooled reference spectra, summaries and figures. `bfs_histograms/`, `bfs_vs_mask/` and `bfs_vs_scree/` hold diagnostics; `step_by_step/` and `walkthrough/` explain selection/extraction stages. |
| `cell_mask_pipeline/` | Cell masks, smoothing/threshold sweeps and in-cell BFS/VCA figures. |
| `lorentzian_plus_background/` | Per-pixel fits, maps and summaries comparing single Lorentzians with measured-reference-assisted fits. |
| `parametric/` | Hydrogel and simulated-water/cytoplasm results for a variable sample Lorentzian plus fixed measured reference. |
| `hydrogel_background_comparison/` | Matched least-squares single/reference-assisted fits, reconstructions, residual maps and report. |
| `hydrogel_intensity_gate/` | Results after a minimum sample-component intensity gate. `lorentzian_parameter_maps/` contains fitted parameter maps; `column_reconstruction/` contains contribution-column visualizations. |
| `hydrogel_edge_to_centre/` | Background-subtracted spectra and summaries grouped by distance inside the selected region. |
| `hydrogel_vca2/` | Anchored two-endmember VCA spectra, abundances, interactive views and smoothed displays. |
| `simulated_data/` | `simulated_water_cytoplasm/`, `simulated_water_nucleus/` and `simulated_water_plexiglass/` contain generated scans, truth sidecars, drawings and manifests. `noise_tolerance_water_plexiglass/` contains noise-modified measured scans and repeated-run metrics. |
| `hydrogel_sphere/`, `hydrogel_sphere_*` | Spherical-blob fits of the hydrogel scan (radial BFS/linewidth profiles, fit parameters, diagnostics). The suffixed folders are variants: `_model` (forward model), `_free_intensity`, `_pixel_scaled` and `_tapered_rim` (alternative boundary/intensity assumptions). |
| `sphere_fit_images/`, `sphere_fit_images_sharp_boundary/` | Numbered presentation figures for the sphere fit, with a smooth and a sharp boundary. |
| `hydrogel_flexible_mixture/` | Flexible two-component mixture fit of the hydrogel scan. |

### Thesis

| Folder under `outputs/` | Contents |
|---|---|
| `Thesis_Overleaf/` | Current thesis source (`main.tex`, `BrillouinMicroscopy.bib`, `figures/`) ready to upload to Overleaf, plus a compiled `Thesis.pdf` and revision notes. |
| `Thesis_Figure_Corrections/` | Corrected Figures 9, 12 and 13 that replace the matching files in `Thesis_Overleaf/figures/`. |
| `final_thesis_figures/` | Figure-generation scripts (`generate_all_figures.py`, etc.), approved figures, drafts, style options and reference material. |
| `thesis_figure_batch/` | Batch build of all 15 figures, a review page (`REVIEW.html`) and a compiled draft. |
| `thesis_revision/` | An earlier revised draft with its own figure script and data. |
| `Honors_Thesis_Revised.tex` | Standalone revised thesis source. |

## Not committed

`.gitignore` keeps these local-only: AI assistant working folders (`.claude/`, `.codex_tmp/`, `.chart-data-*/`, `.hydrogel-deck-build/`, `*.inspect.ndjson`), IDE settings (`.idea/`), Python caches, and the `outputs/*.zip` bundles (zipped copies of the thesis folders above).

## Starting points

- Peak/reference fitting: `analysis/pca_vca/lorentzian_plus_background_fit.py`, `two_lorentzian_fit.py` and `hydrogel_background_comparison.py`.
- Interactive pixel inspection: `analysis/misc/lorentzian_demo.py`.
- Synthetic generation: `simulator/generate_two_material_sweep.py` or the simulator UI (`python -m simulator`).
- Spatial denoising: `analysis/reconstruct_scan.py`; numerical checks: `python -m unittest discover -s tests -v`.

Some exploratory scripts retain older input/output paths; check their settings before running. Saved results document individual runs, not validation of every method or current script version. Folder descriptions are based on inspecting the tree, source and selected artifacts, not rereading every data row or generated file.
