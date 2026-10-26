# Anchored VCA(2)

Saved background endmember is fixed as EM1 (unit-sum normalization only).
EM2 is selected by the existing vca_anchored algorithm with seed 42 from unit-sum measured spectra.
No Lorentzian fits, baseline subtraction, spectral smoothing, or background hard-setting are used.
Abundances use exact constrained least squares on the line segment between the two endmembers,
with nonnegative coefficients summing to one. These are spectral abundances, not calibrated volume fractions.
The selected measured endpoint is not independently verified as pure hydrogel.

The 12% filter and eight-neighbour isolated-single-pixel removal affect display only.
The all-pixels button shows unfiltered abundances. Columns are centred at Z=0.
Reproduce: `python analysis/pca_vca/hydrogel_vca2_interactive.py`

{
  "pixels": 1155,
  "selected_pixel": 470,
  "selected_x": 15,
  "selected_y": 13,
  "retained_12pct": 264,
  "median_normalized_rmse": 0.00036204262475007505
}