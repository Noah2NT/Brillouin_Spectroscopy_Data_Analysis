"""
MER (Maximum Entropy Reconstruction) test on Brillouin data.

Stages:
  1. Lambda sweep on a single spectrum — find the right balance between
     entropy smoothing and data fidelity.
  2. Batch MER on all spectra — visualize the cube before/after, and report
     convergence stats.
  3. Lorentzian fitting on raw vs MER-denoised — frequency shift and
     linewidth maps side-by-side. This is what the paper actually validates.

(Note: VCA on MER-denoised spectra was tested previously and produces
degenerate endmembers because MER reduces inter-pixel diversity. MER is
purpose-built for parameter estimation via fitting, not for unmixing.)
"""

import numpy as np
import matplotlib.pyplot as plt

import brillouin_preprocessing as bp
import brillouin_fitting as bf
import mer_reconstruction as mer


# ===================
# Data Loading + Folding
# ===================

spectra, freq, df = bp.load_brillouin_csv(
    "XY_scanning_40um_60um_step1um_10s_fixed.csv"
)
x_vals, y_vals = bp.build_coordinate_arrays(df)
folded_spectra, shift_freq = bp.fold_brillouin_spectrum(spectra, freq)


keep = shift_freq <= 8
folded_spectra = folded_spectra[:, keep]
shift_freq = shift_freq[keep]


print(f"Loaded {folded_spectra.shape[0]} spectra "
      f"of {folded_spectra.shape[1]} bins each")


# ============================================================
# STAGE 1 — Lambda sweep on a single spectrum
# ============================================================

idx = len(folded_spectra) // 2
single_spectrum = folded_spectra[idx]

lam_values = [0.1, 1.0, 10.0, 100.0, 1000.0]
sweep_results = []

for lam in lam_values:
    f, info = mer.maximum_entropy_reconstruction(
        single_spectrum, lam=lam, max_iter=1000, tol=0.05,
    )
    sweep_results.append((lam, f, info))
    print(f"  lam={lam:>7.1f}  iters={info['iterations']:>4}  "
          f"converged={info['converged']!s:>5}  "
          f"chi2={info['final_chi2']:.4f}  "
          f"balance={info['final_balance']:.4f}")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

# First panel: just the raw
axes[0].plot(shift_freq, single_spectrum, 'k-', linewidth=1.2)
axes[0].set_title('Raw spectrum (no MER)')
axes[0].set_xlabel('Frequency Shift (GHz)')
axes[0].set_ylabel('Intensity')
axes[0].grid(alpha=0.3)

# Then one panel per lambda
for i, (lam, f, info) in enumerate(sweep_results):
    ax = axes[i + 1]
    ax.plot(shift_freq, single_spectrum, 'b-', alpha=0.4,
            linewidth=1.0, label='Raw')
    ax.plot(shift_freq, f, 'r-', linewidth=2.0, label='MER')
    ax.set_title(f'lam = {lam}  (conv={info["converged"]}, '
                 f'iters={info["iterations"]})')
    ax.set_xlabel('Frequency Shift (GHz)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle('Stage 1 — Lambda sweep on a single spectrum\n'
             '(low lam = over-smoothing, high lam = closer to data)',
             fontsize=13)
plt.tight_layout()
plt.show()

# Pick the best lambda for the rest of the script
BEST_LAM = 1000.0
print(f"\nUsing lam={BEST_LAM} for batch processing.")


# ============================================================
# STAGE 2 — Batch MER on all spectra
# ============================================================

print("\n[Stage 2] Running MER on all spectra...")
denoised_spectra, n_converged = mer.mer_batch(
    folded_spectra, lam=BEST_LAM, max_iter=1000, tol=0.05, verbose=True,
)
total = len(folded_spectra)
print(f"MER converged on {n_converged}/{total} spectra "
      f"({100 * n_converged / total:.1f}%)")

vmin, vmax = np.percentile(folded_spectra, [2, 98])

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

im0 = axes[0].imshow(folded_spectra, aspect='auto', origin='lower',
                     extent=[shift_freq[0], shift_freq[-1],
                             0, folded_spectra.shape[0]],
                     cmap='inferno', vmin=vmin, vmax=vmax)
axes[0].set_xlabel('Frequency Shift (GHz)')
axes[0].set_ylabel('Spectrum index (pixel)')
axes[0].set_title('Raw Folded Spectra')
plt.colorbar(im0, ax=axes[0], label='Intensity')

im1 = axes[1].imshow(denoised_spectra, aspect='auto', origin='lower',
                     extent=[shift_freq[0], shift_freq[-1],
                             0, denoised_spectra.shape[0]],
                     cmap='inferno', vmin=vmin, vmax=vmax)
axes[1].set_xlabel('Frequency Shift (GHz)')
axes[1].set_title(f'MER Denoised (lam={BEST_LAM})')
plt.colorbar(im1, ax=axes[1], label='Intensity')

plt.suptitle('Stage 2 — Hyperspectral cube before/after MER', fontsize=13)
plt.tight_layout()
plt.show()


# ============================================================
# STAGE 3 — Lorentzian fitting on raw vs MER  (the paper's use case)
# ============================================================

print("\n[Stage 3] Fitting raw spectra...")
_, raw_params = bf.fit_all_spectra(shift_freq, folded_spectra)
raw_amps, raw_shifts, raw_widths, raw_bases = bf.extract_fit_parameters(raw_params)

print("[Stage 3] Fitting MER-denoised spectra...")
_, mer_params = bf.fit_all_spectra(shift_freq, denoised_spectra)
mer_amps, mer_shifts, mer_widths, mer_bases = bf.extract_fit_parameters(mer_params)

raw_shift_map = bp.reshape_to_map(raw_shifts, y_vals, x_vals)
mer_shift_map = bp.reshape_to_map(mer_shifts, y_vals, x_vals)
raw_width_map = bp.reshape_to_map(raw_widths, y_vals, x_vals)
mer_width_map = bp.reshape_to_map(mer_widths, y_vals, x_vals)

# Use shared color limits (robust percentiles) to make raw/MER directly comparable
shift_vmin, shift_vmax = np.nanpercentile(
    np.concatenate([raw_shifts, mer_shifts]), [2, 98]
)
width_vmin, width_vmax = np.nanpercentile(
    np.concatenate([raw_widths, mer_widths]), [2, 98]
)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

im0 = axes[0, 0].imshow(raw_shift_map, cmap='inferno', origin='lower',
                        vmin=shift_vmin, vmax=shift_vmax)
axes[0, 0].set_title('Frequency Shift — Raw')
axes[0, 0].axis('off')
plt.colorbar(im0, ax=axes[0, 0], label='GHz')

im1 = axes[0, 1].imshow(mer_shift_map, cmap='inferno', origin='lower',
                        vmin=shift_vmin, vmax=shift_vmax)
axes[0, 1].set_title('Frequency Shift — MER')
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1], label='GHz')

im2 = axes[1, 0].imshow(raw_width_map, cmap='magma', origin='lower',
                        vmin=width_vmin, vmax=width_vmax)
axes[1, 0].set_title('Linewidth — Raw')
axes[1, 0].axis('off')
plt.colorbar(im2, ax=axes[1, 0], label='GHz')

im3 = axes[1, 1].imshow(mer_width_map, cmap='magma', origin='lower',
                        vmin=width_vmin, vmax=width_vmax)
axes[1, 1].set_title('Linewidth — MER')
axes[1, 1].axis('off')
plt.colorbar(im3, ax=axes[1, 1], label='GHz')

plt.suptitle('Stage 3 — Lorentzian fitting: Raw vs MER-denoised',
             fontsize=13)
plt.tight_layout()
plt.show()


# ===================
# Summary
# ===================

print("\n=== Summary ===")
print(f"Spectra MER-converged: {n_converged}/{total} "
      f"({100 * n_converged / total:.1f}%)")
print()
print(f"{'':<25} {'mean':>10}  {'std':>10}")
print(f"{'Frequency shift (raw)':<25} {np.nanmean(raw_shifts):>10.4f}  "
      f"{np.nanstd(raw_shifts):>10.4f}")
print(f"{'Frequency shift (MER)':<25} {np.nanmean(mer_shifts):>10.4f}  "
      f"{np.nanstd(mer_shifts):>10.4f}")
print(f"{'Linewidth (raw)':<25} {np.nanmean(raw_widths):>10.4f}  "
      f"{np.nanstd(raw_widths):>10.4f}")
print(f"{'Linewidth (MER)':<25} {np.nanmean(mer_widths):>10.4f}  "
      f"{np.nanstd(mer_widths):>10.4f}")
print()
print("Note: lower std on linewidth/shift typically indicates MER is "
      "successfully reducing pixel-to-pixel noise in the fitted parameters.")
