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

spectrum = spectra[0]
plt.figure()
plt.plot(freq, spectrum)
plt.xlabel("Frequency Shift (GHz)")
plt.ylabel("Intensity")
plt.show()



# visualise brillouin peak

plt.figure()
for spectrum in spectra:
    plt.plot(freq[320:400], spectrum[320:400])
plt.xlabel("Frequency Shift (GHz)")
plt.ylabel("Intensity")
plt.ylim(0, 2000)
plt.show()


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



