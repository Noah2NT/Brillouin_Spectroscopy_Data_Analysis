# Extra Lorentzian parameter maps

These maps describe the flexible Lorentzian added to the fixed background endmember, after the existing intensity gate. They do not fit a second independent Lorentzian to the background.

279 pixels retain the Lorentzian and 876 background-only pixels are N/A. N/A is grey in plots, literal N/A in the CSV and NaN in the map arrays. No spatial smoothing or new spectral fit is applied.

- Centre: fitted Lorentzian peak position in GHz.
- Integrated intensity: sum of the Lorentzian component counts over the fitted channels, excluding the background and constant baseline. This is a channel sum, not multiplied by frequency spacing.
- Linewidth: full width at half maximum, FWHM = 2 gamma, in GHz.
- Additional intensity figure: fitted peak-height amplitude and integrated spectral share.

All panels use the full valid data range, with no percentile clipping. Coordinates are zero-based pixel indices. Retained low-share peaks can still have uncertain parameters; the intensity gate is not an uncertainty estimate.

Source: outputs\hydrogel_intensity_gate

```json
{
  "center_ghz": {
    "min": 5.66648454634689,
    "median": 5.830818142178948,
    "max": 6.0112673284959515
  },
  "integrated_intensity_counts": {
    "min": 12418.245268497787,
    "median": 22671.416722988666,
    "max": 39877.96707811143
  },
  "fwhm_ghz": {
    "min": 0.5301665527636843,
    "median": 0.691206855540352,
    "max": 1.001450958206759
  },
  "peak_height_counts": {
    "min": 667.3374769391996,
    "median": 1310.493012325194,
    "max": 2212.005010686633
  },
  "spectral_intensity_pct": {
    "min": 20.17504533425148,
    "median": 68.04375456423077,
    "max": 100.0
  }
}
```
