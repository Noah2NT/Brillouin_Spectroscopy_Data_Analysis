import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#width of curve, map using gaussian/lorentzian curves & research fourier transform

df = pd.read_csv("Cell_Col Healthy_xy_scanning30um_40um_step1um_15s.csv")
spectra = np.vstack(
    df["Raw Data"].apply(lambda s: np.array([float(v) for v in s.split(",")]))
)
start = df["startFrequency"].iloc[0]
end = df["endFrequency"].iloc[0]
freq = np.linspace(start, end, spectra.shape[1])



x_vals = np.sort(df["X"].unique())
y_vals = np.sort(df["Y"].unique())

# visualise the fist pixel frequency shift spectrum

spectrum = spectra[10]
rayleigh_centre = (freq[0] + freq[-1]) / 2.0
plt.figure()
plt.plot(freq, spectrum)
plt.axvline(rayleigh_centre, color="red", linestyle="--", linewidth=1.2)
plt.xlabel("Frequency Shift (GHz)")
plt.ylabel("Intensity")
plt.legend()
plt.show()


# Folded spectrum — Stokes + anti-Stokes added together about the Rayleigh peak
center_idx = np.argmin(np.abs(freq - rayleigh_centre))
bin_width = abs(freq[1] - freq[0])
stokes      = spectrum[:center_idx][::-1]   # reversed so index 0 = closest to Rayleigh
anti_stokes = spectrum[center_idx:]
w = min(len(stokes), len(anti_stokes))
folded = stokes[:w] + anti_stokes[:w]
shift_axis = np.arange(0, w) * bin_width    # 0 GHz = Rayleigh centre

plt.figure()
plt.plot(shift_axis, folded)
plt.xlabel("Frequency shift from Rayleigh (GHz)")
plt.ylabel("Stokes + anti-Stokes intensity")
plt.title("Folded spectrum (including Rayleigh peak + background)")
plt.show()


# Isolated Brillouin peak — narrow window around the peak with baseline subtracted
# (search for the Brillouin peak only beyond the Rayleigh tail)
search_start_ghz = 2.0
search_mask = shift_axis >= search_start_ghz
peak_idx_in_search = int(np.argmax(folded[search_mask]))
peak_freq = shift_axis[search_mask][peak_idx_in_search]

window_ghz = 2.0
mask = np.abs(shift_axis - peak_freq) < window_ghz
peak_x = shift_axis[mask]
peak_y = folded[mask]

# Estimate background from the wings of the window (first / last 10% of bins)
n_edge = max(3, len(peak_y) // 10)
baseline = np.mean(np.concatenate([peak_y[:n_edge], peak_y[-n_edge:]]))
peak_clean = peak_y - baseline

plt.figure()
plt.plot(peak_x, peak_clean)
plt.axhline(0, color="grey", linewidth=0.8)
plt.xlabel("Frequency shift from Rayleigh (GHz)")
plt.ylabel("Intensity (background-subtracted)")
plt.title("Isolated Brillouin peak")
plt.show()


# Gaussian vs Lorentzian fit comparison on the isolated Brillouin peak
from scipy.optimize import curve_fit

def lorentzian(x, amp, mean, gamma, c):
    return amp / (1 + ((x - mean) / gamma) ** 2) + c

def gaussian(x, amp, mean, std, c):
    return amp * np.exp(-((x - mean) ** 2) / (2 * std ** 2)) + c

# Initial guesses
amp0   = float(peak_clean.max())
mean0  = float(peak_x[np.argmax(peak_clean)])
width0 = 0.4   # GHz, a sensible Brillouin half-width starting point
c0     = 0.0

popt_l, _ = curve_fit(lorentzian, peak_x, peak_clean,
                      p0=[amp0, mean0, width0, c0], maxfev=5000)
popt_g, _ = curve_fit(gaussian,   peak_x, peak_clean,
                      p0=[amp0, mean0, width0, c0], maxfev=5000)

# Evaluate on a fine grid for smooth curves
x_fine = np.linspace(peak_x.min(), peak_x.max(), 400)
y_lor = lorentzian(x_fine, *popt_l)
y_gau = gaussian(x_fine,   *popt_g)

# Residual sum of squares for a quality comparison
rss_l = float(np.sum((peak_clean - lorentzian(peak_x, *popt_l)) ** 2))
rss_g = float(np.sum((peak_clean - gaussian(peak_x,   *popt_g)) ** 2))

plt.figure()
plt.plot(peak_x, peak_clean, 'k-', linewidth=1.2)
plt.plot(x_fine, y_lor, 'r-', linewidth=1.6)
plt.axhline(0, color='grey', linewidth=0.8)
plt.xlabel("Frequency shift from Rayleigh (GHz)")
plt.ylabel("Intensity (background-subtracted)")
plt.title(f"Lorentzian fit  (RSS = {rss_l:.2e})")
plt.show()

plt.figure()
plt.plot(peak_x, peak_clean, 'k-', linewidth=1.2)
plt.plot(x_fine, y_gau, 'b--', linewidth=1.6)
plt.axhline(0, color='grey', linewidth=0.8)
plt.xlabel("Frequency shift from Rayleigh (GHz)")
plt.ylabel("Intensity (background-subtracted)")
plt.title(f"Gaussian fit  (RSS = {rss_g:.2e})")
plt.show()

print(f"Lorentzian: amp={popt_l[0]:.2f}, mean={popt_l[1]:.3f} GHz, "
      f"gamma={popt_l[2]:.3f} GHz, c={popt_l[3]:.2f}  RSS={rss_l:.3e}")
print(f"Gaussian  : amp={popt_g[0]:.2f}, mean={popt_g[1]:.3f} GHz, "
      f"std={popt_g[2]:.3f} GHz, c={popt_g[3]:.2f}  RSS={rss_g:.3e}")



# visualise brillouin peak
'''
plt.figure()
for spectrum in spectra[200:250]:
    plt.plot(freq[320:380], spectrum[320:380])
plt.xlabel("Frequency Shift (GHz)")
plt.ylabel("Intensity")
plt.ylim(0, 1000)
plt.show()
'''

# heatmap of the brillouin peak maximum for all pixels

peak_intensity = spectra[:, 320:400].max(axis=1)
heatmap = np.full((len(y_vals), len(x_vals)), np.nan) #empty heatmap
for row_i, row in df.iterrows():
    x_i = int(row["X"])
    y_i = int(row["Y"])
    heatmap[y_i, x_i] = peak_intensity[row_i]
plt.figure()
plt.imshow(heatmap, origin="lower", cmap="inferno")
plt.colorbar(label="Intensity of brillouin peak")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Brillouin Peak Intensity Map")
plt.show()



peak_pos = freq[spectra[:, 320:400].argmax(axis=1)+320]
heatmap = np.full((len(y_vals), len(x_vals)), np.nan) #empty heatmap
for row_i, row in df.iterrows():
    x_i = int(row["X"])
    y_i = int(row["Y"])
    heatmap[y_i, x_i] = peak_pos[row_i]
plt.figure()
plt.imshow(heatmap, origin="lower", cmap="inferno")
plt.colorbar(label="Frequency shift of brillouin peak (GHz)")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Frequency Shift Heatmap")
plt.show()


# ─────────────────────────────────────────────
# VCA on the whole dataset — same workflow as VCA implementation.py
# ─────────────────────────────────────────────

import brillouin_preprocessing as bp
import brillouin_decomposition as bd
import brillouin_plotting as bplt
import brillouin_fitting as bf

# Extract the Brillouin peak window (bins 305-405) and trim shifts > 8 GHz
peakSpectraarr, peakFreq = bp.extract_frequency_window(spectra, freq, 305, 405)
keep = peakFreq <= 8
peakSpectraarr = peakSpectraarr[:, keep]
peakFreq = peakFreq[keep]

# VCA with 3 endmembers
num_endmembers = 3
endmembers, endmember_indices = bd.vca(peakSpectraarr, num_endmembers, seed=42)

print(f"Endmember pixel indices: {endmember_indices}")
for i, idx in enumerate(endmember_indices):
    row = df.iloc[idx]
    print(f"  Endmember {i+1}: X={row['X']}, Y={row['Y']}")

# ---- Graph 1: endmember spectra ----
plt.figure(figsize=(10, 6))
for i in range(num_endmembers):
    plt.plot(peakFreq, endmembers[i],
             label=f'Endmember {i+1} (pixel {endmember_indices[i]})',
             linewidth=1.5)
plt.xlabel("Frequency Shift (GHz)")
plt.ylabel("Intensity")
plt.title("VCA Extracted Endmember Spectra")
plt.legend()
plt.tight_layout()
plt.show()

# ---- Graph 2: abundance maps via plot_component_maps ----
abundances = bd.estimate_abundances(peakSpectraarr, endmembers)
abundance_arrays = [abundances[:, i] for i in range(num_endmembers)]
labels = [f"Endmember {i+1} Abundance" for i in range(num_endmembers)]
centres = [(int(df.iloc[idx]["X"]), int(df.iloc[idx]["Y"]))
           for idx in endmember_indices]
bplt.plot_component_maps(abundance_arrays, x_vals, y_vals,
                         labels=labels,
                         suptitle="VCA Abundance Maps",
                         cmap="inferno",
                         colorbar_label="Abundance",
                         centres=centres)


# ─────────────────────────────────────────────
# Brillouin Frequency Shift heatmap — 2D cell scan
# ─────────────────────────────────────────────



# ─────────────────────────────────────────────
# Sobel gradient of the BFS heatmap (cell sample)
# Original BFS | X-Sobel | Y-Sobel | gradient magnitude
# ─────────────────────────────────────────────

from scipy.ndimage import sobel

# Lorentzian-fit the Brillouin shift for every pixel in the cell sample
peakSpec_cell, peakFreq_cell = bp.extract_frequency_window(spectra, freq, 305, 405)
keep_cell = peakFreq_cell <= 8
peakSpec_cell = peakSpec_cell[:, keep_cell]
peakFreq_cell = peakFreq_cell[keep_cell]

print(f"Fitting Lorentzian to {peakSpec_cell.shape[0]} cell pixels...")
_, fit_params_cell = bf.fit_all_spectra(peakFreq_cell, peakSpec_cell)
_, shifts_cell, _, _ = bf.extract_fit_parameters(fit_params_cell)

# Reshape to a 2D BFS map (using each row's explicit X/Y)
bfs_map = np.full((len(y_vals), len(x_vals)), np.nan)
for row_i, row in df.iterrows():
    bfs_map[int(row["Y"]), int(row["X"])] = shifts_cell[row_i]

# Sobel gradients (X = along columns, Y = along rows) + magnitude
gx = sobel(bfs_map, axis=1)
gy = sobel(bfs_map, axis=0)
mag = np.sqrt(gx ** 2 + gy ** 2)

# 4-panel side-by-side plot
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
im0 = axes[0].imshow(bfs_map, origin="lower", cmap="inferno")
axes[0].set_title("Original BFS")
plt.colorbar(im0, ax=axes[0], label="GHz")

# Use symmetric color limits for the signed gradients so 0 = white
gmax_x = np.nanmax(np.abs(gx))
im1 = axes[1].imshow(gx, origin="lower", cmap="seismic",
                     vmin=-gmax_x, vmax=gmax_x)
axes[1].set_title("Sobel X gradient")
plt.colorbar(im1, ax=axes[1], label="dGHz/dx")

gmax_y = np.nanmax(np.abs(gy))
im2 = axes[2].imshow(gy, origin="lower", cmap="seismic",
                     vmin=-gmax_y, vmax=gmax_y)
axes[2].set_title("Sobel Y gradient")
plt.colorbar(im2, ax=axes[2], label="dGHz/dy")

im3 = axes[3].imshow(mag, origin="lower", cmap="magma")
axes[3].set_title("Gradient magnitude")
plt.colorbar(im3, ax=axes[3], label="|grad|")

for ax in axes:
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
plt.suptitle("BFS Sobel gradient — cell sample", fontsize=14)
plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────
# Demo: [1, 0, -1] kernel across adjacent pixel Brillouin peaks
# Steps 1-3 — same as the earlier demo but Step 3 reports the
# SIGNED AREA of the gradient (no abs value)
# ─────────────────────────────────────────────

demo_pixel = 20
spec_left   = peakSpectraarr[demo_pixel - 1]
spec_center = peakSpectraarr[demo_pixel]
spec_right  = peakSpectraarr[demo_pixel + 1]
diff = spec_right - spec_left
area = float(np.sum(diff))    # net signed area (no abs)

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

# Step 1 — three adjacent Brillouin peaks
axes[0].plot(peakFreq, spec_left,   'r--', alpha=0.85, linewidth=1.2,
             label=f'pixel {demo_pixel - 1} (left)')
axes[0].plot(peakFreq, spec_center, 'k-',  linewidth=1.6,
             label=f'pixel {demo_pixel} (centre)')
axes[0].plot(peakFreq, spec_right,  'b--', alpha=0.85, linewidth=1.2,
             label=f'pixel {demo_pixel + 1} (right)')
axes[0].set_ylabel('Intensity')
axes[0].set_title('Step 1 — three adjacent Brillouin peaks')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# Step 2 — signed gradient (right − left)
axes[1].axhline(0, color='grey', linewidth=0.8)
axes[1].plot(peakFreq, diff, 'g-', linewidth=1.4,
             label=f'spectra[{demo_pixel + 1}] − spectra[{demo_pixel - 1}]')
axes[1].set_ylabel('Signed gradient')
axes[1].set_title('Step 2 — kernel [1, 0, -1] across pixels')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

# Step 3 — signed AREA (no abs). Positive area shaded green, negative red
axes[2].fill_between(peakFreq, 0, diff, where=diff > 0, color='green',
                      alpha=0.4, interpolate=True, label='positive area')
axes[2].fill_between(peakFreq, 0, diff, where=diff < 0, color='red',
                      alpha=0.4, interpolate=True, label='negative area')
axes[2].plot(peakFreq, diff, 'k-', linewidth=1.0)
axes[2].axhline(0, color='grey', linewidth=0.8)
axes[2].set_xlabel('Frequency Shift (GHz)')
axes[2].set_ylabel('Signed gradient')
axes[2].set_title(f"Step 3 — total signed area = {area:.1f}")
axes[2].legend(fontsize=9)
axes[2].grid(alpha=0.3)

plt.suptitle('[1, 0, -1] across adjacent pixels — signed-area version',
             fontsize=13)
plt.tight_layout()
plt.show()



