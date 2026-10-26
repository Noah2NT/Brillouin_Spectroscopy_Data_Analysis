# Hydrogel reconstruction comparison

Both models minimize unweighted squared residuals on the same unsmoothed, folded spectra (Stokes + anti-Stokes; existing pipeline band up to 8 GHz). Single: A L(nu; center, width) + c. Mixture: A L(nu; center, width) + B EM_bg(nu) + c. Amplitudes are nonnegative; the constant baseline is free in both models. Four versus five fitted parameters; multiple starts and the nested single solution are compared.

Background source: `outputs\background_endmember\Hydrogl1_XYscan_20000_20000_step500um_10s (1)_background_endmember.csv`. No new spatial smoothing is applied.

Percentages are each component spectral sum divided by the sum of the two component sums, excluding baseline, within the fitted frequency band. They sum to 100%; they are not volume fractions.

RMSE = sqrt(mean((measured - reconstructed)^2)) across frequency channels. The first two maps use an identical colour range. Pixel labels link to the six example panels.

Lower training residual is expected for an extra component and does not establish recovery of a noise-free spectrum. The endmember comes from this sample, so this is not independent validation. Overlapping component peaks can make fitted percentages and centers poorly identifiable.

```json
{
  "pixels": 1155,
  "channels": 112,
  "grid": [
    33,
    35
  ],
  "median_single_rmse": 80.00535594921661,
  "median_mixture_rmse": 25.51517045814276,
  "median_pixel_rmse_reduction_pct": 67.57503464679301,
  "global_single_rmse": 75.46212209353178,
  "global_mixture_rmse": 26.24535529816727,
  "median_background_pct": 96.03013543892008,
  "optimizer_converged": [
    1155,
    1151
  ]
}
```

## Interpretation and fit status

Most improvement occurs in the background-dominated exterior. Among 207 pixels with fitted background contribution below 50%, median per-pixel RMSE reduction is 6.75%. This subset is defined by the fitted model, not an independent hydrogel mask.

Four mixed fits (pixels 417, 796, 961, 983) did not report optimizer convergence. Their fitted background shares exceed 99.7%; component parameters are poorly constrained there. Their best feasible fits are retained and flagged in pixel_results.csv. All single fits converged; no mixture has higher SSE than its single fit. Synthetic noiseless mixture recovery passed.

