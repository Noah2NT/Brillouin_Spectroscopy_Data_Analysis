# Hydrogel sphere spectral model

Run from the repository root (existing NumPy, SciPy, pandas and Matplotlib dependencies):

```powershell
python tests/test_hydrogel_sphere.py
python analysis/pca_vca/hydrogel_sphere.py
```

If the evaluation budget is exhausted, continue without discarding the fitted
parameters: `python analysis/pca_vca/hydrogel_sphere.py --initial outputs/hydrogel_sphere/fit.json --max-nfev 400`.

The default input is the existing Hydrogl1 scan and its saved background endmember.
Results go to `outputs/hydrogel_sphere/`: `fit.json` (parameters, convergence and
numerical diagnostics), `spectra.npz` (observations, reconstructions, a Y-X-frequency
prediction grid and residuals), `radial_profiles.csv` (the recovered functions),
`pixels.csv` (per-pixel MSE) and `diagnostics.png`. A failed convergence flag is
saved honestly even when the last iterate improves the loss.

## Model and conventions

Let rho = distance from the sphere centre / sphere radius. Thus rho=0 is the
centre and rho=1 is the boundary, rather than distance measured inward from the
surface. The relative concentration is C=max(1-rho^2,0). This is a proposed
relative concentration field, not a measured absolute HEC mass fraction: the
paper's preparation used 20% w/v HEC.

Inside the sphere:

```
BFS(rho) = BFS_center + (BFS_edge - BFS_center) * rho**bfs_power
LW(rho)  = LW_center  + (LW_edge  - LW_center)  * rho**lw_power
INT(C)   = INT_edge  + (INT_center - INT_edge) * C**int_power
L(f,rho) = (LW(rho)/2) / (pi * ((f-BFS(rho))**2 + (LW(rho)/2)**2))
```

This corrects the missing squared frequency difference in the original sketch.
BFS and LW depend directly on radius; only intensity is evaluated through C.
In the default hard-boundary model each local concentration contributes one local
line. It is not additionally mixed with water using C. The optional smooth-rim
model below adds an explicit hydrogel/water spectral mixture near the edge.
Outside the sphere, use the measured water endmember, scaled by water_intensity.
Without `--fit-rim`, boundary spectral continuity is not forced; INT_edge is the limiting interior
intensity. All line shapes are normalized to sum to one over the chosen 4–8 GHz
window, so INT means counts integrated over that window, not peak height or
infinite-band Lorentzian area.

Each predicted spectrum integrates this local spectrum over a normalized 3D
Gaussian point spread function centred on the scan position. Transverse
Gauss-Hermite quadrature and axial Gauss-Legendre quadrature integrate each sphere
chord, with the analytic Gaussian mass supplying the exterior water weight.
Axial integration is truncated at eight standard deviations. Intensity is evaluated
directly at each integration point; only the unit-area line shape uses linear
interpolation on a radial spectral grid. This avoids interpolating a sharp
intensity cusp near the boundary. The final prediction is checked
against finer transverse, axial and radial grids; `finer_quadrature_rmse` should
be small relative to the fit RMSE before interpreting a result.

`SphereModel.fit` uses SciPy bounded least squares on the entire measured spectral
array, minimizing mean squared residuals with equal pixel/channel weights. Raw
folded intensities are retained; per-pixel normalization would discard the INT
information. One constant residual scale conditions optimization without changing
the objective. This is still nonlinear model fitting, but it fits a shared 3D
continuum, rather than one or two independent Lorentzians at every pixel.

## Instrument and spatial scale

Source: attached *Principal Component Analysis in Application to Brillouin
Microscopy Data*, section 2.3, PDF page 4; spatial scale from Figure 5, PDF page 10.

- 660 nm Torus laser; confocal CM1 microscope; six-pass tandem scanning
  Fabry-Perot TFP1 spectrometer (TableStable); 512 acquisition channels.
- User-specified hydrogel objective: Mitutoyo Plan Apo 20X, NA 0.42, WD 20 mm.
- Paper-estimated 20X spatial resolution: 2 x 2 x 100 micrometres.
- Reported spectral resolution: approximately 276 MHz.
- Figure 5 labels X and Y in millimetres; the existing scan has 35 x 33 pixels
  and filename step500um. Together these support 500 micrometre scan spacing.
  The plotted outer blob suggests roughly 4–5 mm radius; 5 mm is an initial
  estimate, not a radius stated in the paper.

The Gaussian PSF shape and interpreting the quoted resolutions as FWHM are
explicit assumptions, not measured PSF calibration. Override with
`--psf-fwhm-um X Y Z`. WD is lens clearance, not sampling depth; the calculation
does not use 20 mm as the integration thickness. NA and wavelength identify the
setup; the empirical resolution estimate supplies the PSF instead of an ideal
diffraction-only estimate. LW parameters are effective observed local widths.
We do not broaden the measured background again or infer an intrinsic linewidth
from a resolution number without a calibrated instrument response.

Default reference radius is 5000 um, with fixed steps of 500 um (initially 0.1
radius). `radius_scale` fits the actual radius as `radius_scale * reference_radius`.
Centre coordinates and dx/dy are in reference-radius units, with the scan's
bounding-box midpoint as XY origin and the scan plane as z=0. The JSON also
reports centre, radius and steps in um. For a different dataset pass `--scan`,
`--background`, `--radius-um`, and `--step-um X Y` explicitly.

`--fit-steps` instead fixes the radius and fits dx/dy, starting at 0.1 radius as
in the original sketch. Fitting radius and both steps simultaneously would add
an avoidable scale ambiguity. A single plane also cannot distinguish +z0 from
-z0; the CLI fits nonnegative z0 and reports it as absolute depth. Depth,
radius and radial profiles may remain strongly confounded even with that choice.
At millimetre radius, 100 um axial resolution gives limited mixing of a smooth
radial gradient through most of the interior. Large measured broadening alone
does not establish that depth mixing explains it.

## Limits and checks

The sphere, power laws and Gaussian PSF are hypotheses. No drift correction,
arbitrary baseline, nonspherical geometry or intrinsic linewidth deconvolution
is added here. Per-pixel gains are optional, as described below. The existing folding convention is reused
so the measured background and scan align. It does not itself correct spectral
drift. If the scan has drift or illumination fluctuations, residuals can reflect
those effects. A small residual does not establish unique material properties.
Review optimizer status, active bounds, Jacobian singular values, quadrature
error, spatial residuals and fits from different starting conditions before
claiming a physical reconstruction. Especially small singular values flag weakly
constrained parameter combinations; they are not confidence intervals.

The runnable numerical check covers pure-water and uniform-material limits,
nonnegative spectra, concentration endpoints, depth reflection symmetry,
quadrature refinement, selected spectral-parameter recovery and invalid inputs.
It does not claim recovery of all geometry and spectral parameters from a single
plane. Numerical methods use existing SciPy routines:
[Gaussian quadrature](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_hermitenorm.html)
and [bounded least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

## Initial experimental run

The supplied 1155-pixel scan was fitted using the defaults, then resumed for
24 evaluations after the first 200-evaluation budget was exhausted. SciPy
reported xtol convergence. Final MSE was 7025.21 counts squared (RMSE 83.82
counts/channel), compared with 60191.61 at the original initialization; this
comparison is against the starting guess, not against conventional curve fitting.
The inferred radius was 4911.99 um. Finer quadrature changed predictions by
0.0745 counts/channel RMS. The largest spatial residual is in a bright region
outside the main blob; a uniform water intensity cannot reproduce it. Depth and
some radial profile parameters shifted during continuation with little change
in loss, so these fitted values must not be treated as uniquely recovered physics.

## Optional intensity scaling per pixel

```powershell
python analysis/pca_vca/hydrogel_sphere.py --pixel-scaling --initial outputs/hydrogel_sphere/fit.json --max-nfev 300
```

This writes a separate `outputs/hydrogel_sphere_pixel_scaled/` result by default.
For the integrated predicted spectrum m_i and observed spectrum y_i at pixel i:

```
gain_i = max(0, dot(m_i, y_i) / dot(m_i, m_i))
prediction_i = gain_i * m_i
```

This is the exact nonnegative least-squares amplitude at fixed shared parameters;
a zero predicted spectrum is assigned gain zero. The gain scales the entire
spectrum, including water, and is recomputed inside every geometry/BFS/LW/intensity
fitting step. It adds no per-channel scaling or additive baseline. Only water
intensity is held fixed to anchor the otherwise arbitrary common brightness
scale. Centre intensity, edge intensity and intensity curvature remain free.
Their relative values may still be weakly constrained where a pixel samples
nearly one radial location. Gains need not average to one.

`fit.json` records original, gain-only and refitted overall RMSE; `pixels.csv`
records before/after pixel RMSE and gains. `spectra.npz` saves the final scaled
prediction, gains and initial unscaled prediction. `rmse_comparison.png` compares
the error maps on one colour scale and displays the gain map. Overall RMSE is
sqrt(mean(residual**2)) over all pixels and channels, not mean(pixel_RMSE).
These are in-sample errors: 1155 additional amplitudes can lower RMSE without
establishing a more accurate physical sphere. Refinement error holds fitted gains
fixed to measure the effect of changing only numerical integration.

The numerical check additionally covers known per-pixel gain recovery, zero and
nonnegative-bound cases, spectral-parameter recovery and adjustment of radial
intensity with gains enabled, and independence from spectral-grid resolution for
a sharp radial intensity profile with spatially uniform spectral shape.

In the first pixel-scaling run (which held radial intensity fixed), the original
saved fit gave RMSE 83.8165 counts/channel;
gains alone reduced this to 23.2525, and the joint shape/geometry refit to 22.3122
(73.38% below the original). The refit converged in 70 evaluations; quadrature
refinement changed scaled predictions by 0.0185 counts/channel RMS. However,
centre depth moved from 87.7 um to 2955.1 um and centre BFS/linewidth reached
upper bounds. This exposes the ambiguity of extrapolating the unobserved sphere
interior after pixel brightness is free; it is not evidence for that new depth.
The earlier sphere diagrams remain illustrations of the original unscaled fit.

To check whether a local solution misses a larger sphere while also optimizing
radial intensity:

```powershell
python analysis/pca_vca/hydrogel_sphere.py --pixel-scaling --initial outputs/hydrogel_sphere_pixel_scaled/fit.json --radius-start-factors 1 1.05 1.12 --max-nfev 300 --output outputs/hydrogel_sphere_free_intensity
```

Every start refits the radius, centre, BFS, linewidth and radial intensity, with
pixel gains recomputed at each evaluation. The lowest-MSE result is retained;
individual start errors and convergence status are recorded in `fit.json`.
This searches several local basins, not every possible geometry or spectral model.

With direct intensity quadrature, all three starts converged. Best overall RMSE
was 22.3061 versus 22.3122 for the previous fixed-intensity fit. On the same 123
pixels within 0.5 mm of the old projected boundary, RMSE was 34.4765 versus
34.6886. The projected radius changed from 4.8160 to 4.7798 mm, and the residual
ring remained. Finer integration changed the new prediction by 0.0061
counts/channel RMS. Radial intensity and centre BFS/LW remain poorly constrained,
with several parameters at bounds; the tiny error improvement is not evidence
for their physical values. Reproduce the matched comparison with
`python analysis/pca_vca/compare_hydrogel_boundary.py`.

## Optional smooth rim

`--fit-rim` fits a gradual spectral transition into water rather than a flattened
underside. It adds one parameter, `rim_width`, the shell thickness divided by the
outer sphere radius. The starting value is 0.2; bounds are 0 to 1. Zero recovers
the original hard boundary. For positive width w:

```
t = clip((1-rho)/w, 0, 1)
h = t*t*(3-2*t)
local_spectrum = h * INT(C) * normalized_L(f,rho)
                 + (1-h) * water_intensity * measured_water_EM(f)
```

The hydrogel weight h is one in the inner core and smoothly decreases to zero
at the outer radius, with zero slope at both ends of the shell. The water
contribution fills the complementary weight, so fading the gel does not create
an artificial dark edge. This local mixture is integrated over the same 3D PSF;
per-pixel intensity gains are still solved at every optimizer evaluation. Radius,
centre, linewidth/BFS profiles and relative radial intensity also remain free.

This is an empirical spectral transition model, not a measurement of surface
height or an absolute concentration fraction. It does not establish a flattened
3D shape. The old C=1-rho^2 remains the input to the hydrogel branch's intensity.
The water intensity stays fixed only to anchor the common brightness scale.

```powershell
python analysis/pca_vca/hydrogel_sphere.py --pixel-scaling --fit-rim --initial outputs/hydrogel_sphere_pixel_scaled/fit.json --radius-start-factors 1 1.08 --max-nfev 250 --output outputs/hydrogel_sphere_tapered_rim
python analysis/pca_vca/compare_hydrogel_boundary.py --before outputs/hydrogel_sphere_free_intensity --after outputs/hydrogel_sphere_tapered_rim
```

`radial_profiles.csv` includes the fitted `hydrogel_weight`; JSON reports the
transition thickness as `rim_width_um`. Older fit files without rim_width load
as zero. Tests cover the zero-width limit, taper endpoints, preservation of a
uniform spectrum, and synthetic recovery of rim width with pixel scaling.

For this scan both starting radii converged to RMSE 19.7323 counts/channel,
versus 22.3061 for the sharp-boundary/free-intensity model. On the same 119
pixels within 0.5 mm of that model's projected boundary, RMSE fell from 35.0880
to 23.4915. Sphere radius changed from 5.6478 to 5.7504 mm; the scan-plane
intersection radius changed from 4.7798 to 5.2465 mm. Integration refinement
changed predictions by 0.00147 counts/channel RMS.

The fitted transition is broad (0.5897 of the radius, or 3.3909 mm in this
parameterization), not an established physical boundary thickness. Depth,
linewidth and intensity remain coupled. This is an in-sample improvement of an
empirical spectral mixture, not held-out validation or proof of surface geometry.

## Free background/hydrogel percentages per pixel

```powershell
python analysis/pca_vca/hydrogel_flexible_mixture.py
```

This isolates the effect of freeing mixture proportions while keeping the saved
smooth-rim geometry, intensity profile and BFS/LW profiles fixed. The integrated
hydrogel component is normalized to a unit-sum spectrum H_i for each pixel;
the measured water endmember B is also unit-sum. SciPy nonnegative least squares
solves `y_i = a_i * H_i + b_i * B`, with independent nonnegative a_i and b_i.
Hydrogel spectral percentage is `100*a_i/(a_i+b_i)` and background is its
complement. Total intensity is `a_i+b_i`. These describe signal within the fitted
frequency band, not volume or concentration. No additive baseline is introduced.

Where the saved sphere has zero hydrogel support, only background is fitted
(793 of 1155 pixels in this run). Zero total signal or exactly identical component
shapes receive undefined/NaN percentages rather than an arbitrary fraction.
Nearly identical spectra can still make the percentage poorly determined.

The result is saved separately under `outputs/hydrogel_flexible_mixture/`, including
spectra, a per-pixel percentages/intensities/RMSE table, summary and comparison map.
Overall RMSE fell from 19.7323 to 18.4622 counts/channel (6.44%). Every pixel's
error was checked against the previous fit; none worsened. This is an in-sample
comparison with additional local flexibility, not a refit of shared parameters or
an independent validation of physical composition. The numerical check covers
component reconstruction, known-fraction recovery, pure-water/pure-hydrogel
limits and unidentifiable fractions.
