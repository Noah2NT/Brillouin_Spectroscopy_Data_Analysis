# Thesis figure batch

This folder is ready to upload as an Overleaf project or to use as a source of individual figures. It contains all 15 figures, a separately updated thesis source and a compiled reading copy. Figure 1 preserves the version Noah approved. The other figures are batch candidates for review.

## Open and review

- Open `REVIEW.html` in a browser for every figure at full size with its caption and PDF/SVG links.
- `review/overview_1.png` through `overview_4.png` are contact sheets.
- `figure_manifest.json` maps the review IDs from the conversation to the paper's automatic figure numbers. Filenames preserve review IDs to make comments easy to track.
- `CAPTIONS.md` contains the full explanatory captions; the paper uses shorter versions.
- `CHANGELOG.md` records how the requested changes were applied.

## Use in Overleaf

Upload this folder as a ZIP project, select `main.tex`, and compile with pdfLaTeX and BibTeX. The paper already references the PDF figures and includes the supplied nine-entry bibliography. All figure files are supplied as vector PDF, editable SVG and 300 dpi PNG. Preserve your own previous source and bibliography when merging into an existing project.

The narrative order is: cell mechanics → microscope acquisition → spectral definitions → material contrast → mixed collection → shot noise → single-peak limitations → unmixing → study questions → 3D sample models → concentration laws → fixed mixture → varying mixture → noiseless dataset → noisy dataset. Review IDs 13, 11 and 12 therefore appear as paper figures 11, 12 and 13 respectively.

## Model choices

The concentration model keeps the provisional BFS and linewidth laws from the earlier draft. Integrated intensity now rises linearly by 20% from 0% to 10% glycerol, as an explicitly chosen simulation assumption. The new intensity law is applied consistently to the local spectra, mixture contributions, full dataset and paper. It is not an experimentally measured calibration.

The exposure illustration assumes a photon rate of 1000 expected water-reference photons per second in the displayed 5.25–7.35 GHz band. Counts are independent Poisson samples; shorter exposures produce stronger relative fluctuations. The labelled seconds are illustrative and are not a calibration of the microscope. The main proposed benchmark remains specified by photon budgets.

The 3D cutaway depicts the prescribed radial field. It is not a reconstruction or a physical diffusion calculation. Red cylinders show the 30 µm axial collection range from z=0 to 30 µm; their drawn thickness is a visibility aid, not a model of the lateral point-spread function. Spectral maps are displayed over a cropped 5.65–7.00 GHz range, while stored spectra retain the full 4.50–8.50 GHz axis.

The single-peak-fit illustration uses a numerical least-squares search over centre and linewidth with the optimal area calculated at each candidate. Its error is recorded in `data/single_peak_fit.json`; it illustrates a chosen two-component example rather than asserting that every mixture can be visually distinguished.

## Reproduce

`source/generate_all_figures.py` requires NumPy, Matplotlib and Pillow. The current font paths use Windows Arial. Run it from any directory with those packages installed. It keeps the approved Figure 1, regenerates Figures 2–15 and saves their model data. `source/generate_figure_02.py` supplies the microscope-diagram template used by the batch generator. The original source used to assemble the paper lives in the workspace; the exported `main.tex` is already assembled and editable.

The generator checks that all component sums equal their corresponding mixed spectra, that the central pixel agrees with the full simulated cube, that water-only rays are correct, and that the response-law endpoints match the stated values. These checks verify the forward illustrations. They do not constitute implementation or validation of the inverse-method benchmark proposed in the paper.

## Latest feedback round

The shortened one-sided collection interval is applied to all quantitative illustrations and the editable paper, including integration limits and 120 depth samples. The fixed central pixel contains 33.3% water and 66.7% solution by collection weight; corresponding full-band signal fractions are 29.4% and 70.6%. These quantities differ because the prescribed solution response area is 20% higher. Figure 6 uses exactly 1000, 10 and 1 s. Figure 7 has no residual panel. Figure 11 displays its equations, where w is mass fraction (percentage divided by 100). Figure 13 uses visual vertical offsets and individually scaled local spectra; its caption explains these, and a colour bar identifies the concentration of the contributions in panel c. Figure 15 uses a 10000-water-photon reference budget and independent Poisson samples, converted to spectral density for comparison. Vector exports have transparent backgrounds; PNG previews retain white backgrounds.

## Third feedback round

The spectral axes now use Intensity, retaining photon counts for the exposure illustration and physical units for frequency, BFS and linewidth. Captions define intensity as spectral density where appropriate. Figure 9 is a visual overview of example inputs and recovery targets; its outputs are schematic and are not analysis results. Figure 10 shows 25 representative collection regions per sample, not simultaneous independent measurements or a change to the forward model. Figure 12 shows the full 4.5–8.5 GHz range. Figure 15 now uses 10,000 expected water-reference photons per pixel, reducing relative shot noise without smoothing the samples.

## Latest Figure 9 revision

The final panel shows prescribed BFS (blue dashed, left axis) and intrinsic linewidth (orange solid, right axis) against Glycerol (% by mass), over 0–10%. These lines illustrate assumed model relationships and are not recovered calibration results. The panel title retains Recover relationship & contribution, while the contribution distribution is not repeated. Grey arrows are positioned in the measured clear gaps between each input plot and its target labels. The figure caption explains all four study configurations. Model data are unchanged.
