# Hydrogel minimum-intensity gate

Lorentzian integrated intensity must be at least 20% of Lorentzian + background intensity, excluding constant baseline. The threshold is a configurable modelling choice, not a calibrated detection limit.

Starting from the saved unrestricted fits, components below threshold are removed. Background amplitude (nonnegative) and constant baseline (free) are then refitted by exact linear least squares. Subthreshold pixels remain background-only even if a single Lorentzian could fit better. Components at or above threshold retain their original fit. No component is forced up to the threshold. Absent Lorentzian centres and widths are NaN in the CSV and explicitly masked in the NPZ.

The full background amplitude and constant offset may still adapt to each pixel. The gate stops an extra weak Lorentzian from polishing background residuals. It can sacrifice some in-sample fit quality and is not independent evidence of denoising. It is a post-fit model-selection rule, not a globally reoptimised fit under a disjoint amplitude constraint.

Source: outputs/hydrogel_background_comparison/reconstructions.npz and pixel_results.csv.

```json
{
  "pixels": 1155,
  "minimum_lorentzian_fraction": 0.2,
  "decision_counts": {
    "background_only": 876,
    "mixture_retained": 279
  },
  "median_single_rmse": 80.00535594921661,
  "median_previous_mixture_rmse": 25.51517045814276,
  "median_gated_rmse": 29.12613303655385,
  "median_reduction_pct": 62.15295725262543,
  "median_rmse_increase_on_suppressed": 3.6573993077875677,
  "active_subchannel_lorentzians": 0,
  "nonconverged": 0
}
```
