# Revised thesis draft and Overleaf figures

## Use in Overleaf

Upload `main.tex`, `BrillouinMicroscopy.bib`, and the `figures` folder together. Set `main.tex` as the main document. The file uses pdfLaTeX and BibTeX (`natbib` with `apalike`). In an existing project, preserve a copy of your old source and bibliography before replacing them.

The nine-entry bibliography supplied here is sufficient for this revised text. If you prefer your existing bibliography, merge these entries by citation key and resolve duplicate entries. The rewritten text does not use every paper cited in the earlier draft.

The source was compiled locally with Tectonic (XeTeX and BibTeX) and checked for unresolved references and overfull text. The two final figure PDFs were rendered for visual inspection. Layout will change when the original placeholder artwork is restored or replaced, so check the final Overleaf output after that pass.

Your old `images` folder can remain alongside these files. The source uses the original artwork filenames where retained. If an image is missing, a labelled placeholder box is shown, allowing a fresh project to compile without those original assets. Existing cell and optical-layout artwork still needs the corrections listed in `Supervisor_feedback_summary.md`. The final captions can be shortened when that artwork is replaced.

## Downloadable figures

- `figures/concentration_parameters.pdf` / `.png`: BFS, intrinsic FWHM and integrated intensity against glycerol mass percentage. The dashed intensity curves are sensitivity cases, not measurements. The PDF is preferred for Overleaf.
- `figures/noiseless_simulated_dataset.pdf` / `.png`: fixed and varying samples, their central spectral line scans and selected spectra. Each underlying cube is 51 × 51 × 401. The concentration maps are prescribed midplane truth, not fitted BFS maps. No noise is added, but the assumed instrument broadening is included.

Captions and figure references are already inserted into `main.tex`. To add the figures to another source:

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{figures/concentration_parameters.pdf}
  \caption{Prescribed concentration dependence of BFS, intrinsic linewidth and integrated intensity. Solid curves are the baseline; dashed intensity curves are sensitivity cases. These curves are provisional simulation inputs, not measured calibrations.}
  \label{fig:concentration_parameters}
\end{figure}
```

Use the corresponding path and caption from `main.tex` for the dataset figure.

## What is provisional

The draft keeps the spectral coefficients from the pasted source: water BFS 6.05 GHz, water intrinsic linewidth 0.20 GHz, a=0.85 and b=2.0. Their concentration dependence is prescribed, not fitted to published 660 nm data. The baseline area is constant; ±20% area curves test sensitivity only. The 0–10% range is explicitly open to revision after pilot assessment.

The illustrative 20 µm sphere, 51 × 51 scan, 100 µm top-hat axial weighting and 0.276 GHz Lorentzian instrument FWHM are stated modelling assumptions. The reported microscope resolution motivates the chosen width but does not establish a Lorentzian measured response. Numerical frequency sampling is finer than spectral resolution, as it can be in a forward model.

Photon budgets, repetitions, test geometries, regularisation choices, stopping rules and success tolerances in the rewritten methodology are proposed study conditions. They fill the missing protocol as a reviewable draft without claiming they are measured, supervisor-approved, implemented or already validated. No laser power or acquisition time in seconds is invented; counts cannot be mapped to seconds without a photon-rate calibration.

## Data and reproducibility

`generate_figures.py` requires NumPy and Matplotlib and writes the two figures beside itself. Run it with `python generate_figures.py` in an environment with those packages. It contains self-checks for endpoints, positivity, the water-only ray, and agreement between depth integration and the analytic fixed mixture. It is a figure-generation script, not the implementation of the complete inverse-method benchmark.

`figure_parameters.json` records the illustrated values. `concentration_parameters.csv` contains the concentration curves. The two `*_central_line_scan.csv` files have one position per row and one frequency sample per column. `noiseless_simulated_dataset.npz` stores both full cubes in `(y, x, frequency)` order, coordinates in µm/GHz, and midplane mass-fraction maps. Spectra are point-sampled densities in arbitrary area units per GHz; Poisson count generation and integrated frequency bins are described in the methodology but are not used for these noiseless illustrations. The NPZ opens with `numpy.load`.

`Supervisor_feedback_summary.md` summarises both annotated PDFs and the in-person advice separately, and records which artwork corrections remain for the later figure pass.
