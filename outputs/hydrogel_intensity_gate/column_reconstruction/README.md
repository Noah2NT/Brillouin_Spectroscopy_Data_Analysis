# Centred hydrogel-percentage column plot

Source: outputs/hydrogel_background_comparison/{pixel_results.csv,reconstructions.npz}. This replaces the previous gated Lorentzian-share plot. The viewer is hydrogel_percent_centred.html.

Model per pixel: S_i(nu) = b_i B(nu) + L(nu; A_i, centre_i, gamma_i) + c_i. B is the shared background endmember; its scale and the Lorentzian amplitude, centre, width and constant baseline are fitted independently per pixel. No fixed pure-hydrogel spectrum is used.

Height = hydrogel percentage = 100 × sum(L_i) / [sum(L_i) + sum(b_i B)], excluding the constant baseline. Values are recomputed from the ungated fit parameters and checked against the saved percentages and reconstructed spectra. X and Y are original zero-based pixel indices, not inferred physical units.

All columns have midpoint Z = 0: bottom = -height/2, top = +height/2. Total column extent and colour encode the hydrogel percentage. Negative Z denotes position relative to the midpoint, not a negative contribution.

301 retained | 842 below 12% removed | 12 isolated pixels removed | Z midpoint = 0 | No smoothing

Spectral share; constant baseline excluded. 4 saved fits flagged as not converged (see CSV).

Filtering: remove hydrogel shares strictly below 12% (exactly 12% is retained). Then remove single pixels with no retained neighbour among the eight adjacent positions, including diagonals. Multi-pixel islands are not removed. Removed pixels are omitted from the 3D view and shown in grey on the top view. Raw fitted percentages remain in the CSV; height_pct is zero for removed pixels, with retained and removal_reason columns. Retained heights are unchanged and no spectral refitting or smoothing is applied.

The HTML is interactive and self-contained. The PNG includes a matching top view. column_heights.csv preserves the source coordinates and the plotted height for every pixel.

Reproduce: `python analysis/pca_vca/hydrogel_columns.py`

## Remaining sphere visualization work

1. Done: 12% threshold followed by isolated single-pixel removal (8-neighbour connectivity).
2. Common Z midpoint is already applied to the hydrogel columns.
3. Smooth the resulting shape.

A physical thickness interpretation requires calibration or the assumption that spectral share tracks path-length fraction. Centring imposes symmetry rather than measuring Z position.
