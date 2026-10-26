# Hydrogel sphere-cap forward model

`python analysis/pca_vca/hydrogel_sphere_model.py [--scan-height MM]`

One global least-squares fit of every pixel and channel of `Hydrogl1_XYscan_20000_20000_step500um_10s (1)`.
The blob is a spherical cap with centre (x0, y0) and radius of curvature R.
The scan plane sits dz = dz_frac*R from the sphere centre.
Inside the gel (r = distance/R <= 1), each point emits INT(r) * Lorentzian(BFS(r), LW(r)).
Each profile is `edge + (centre - edge) * (1 - r^k)`.
Outside the gel, points emit the scaled water background endmember.
Each pixel sums its points over the 20X PSF: 2 um lateral, 100 um axial (paper), plus one fitted `blur`.

Limits:
- R and dz are partly degenerate. One XY slice shows a disk of radius sqrt(R^2 - dz^2),
  so curvature is pinned only by the shape of the radial profiles. Check the error bars in
  params.json. An XZ line scan, or 2-3 XY scans at known focus steps, would fix R directly.
- Without --scan-height, the dish plane is invisible, so cap height and contact angle are unknown.
- The profiles are only observed for r >= dz_frac (orange line in profiles.png).
  Values closer to the centre are extrapolated.
- The grey scatter in profiles.png comes from independent per-pixel fits.
  It is a sanity reference only, not the fit target.
