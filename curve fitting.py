import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import curvesMapping as cm

df = pd.read_csv("Hydrogl1_XYscan_20000_20000_step500um_10s (1).csv") # 3 clusters, azim 100
spectra = np.vstack(
    df["Raw Data"].apply(lambda s: np.array([float(v) for v in s.split(",")]))
)
start = df["startFrequency"].iloc[0]
end = df["endFrequency"].iloc[0]
freq = np.linspace(start, end, spectra.shape[1])

x_vals = np.sort(df["X"].unique())
y_vals = np.sort(df["Y"].unique())

peakSpectra = []

for spectrum in spectra:
    peakSpectra.append(spectrum[305:405])

peakFreq = freq[305:405]


# manual fitting practise
s1 = peakSpectra[0]

'''
mean = peakFreq[np.argmax(s1)]
amp = s1.max()
C = 0
gamma = 0.36


std = 0.4 


plt.figure()
plt.plot(peakFreq, s1, label = 'Raw')
plt.plot(peakFreq, cm.gaussian(peakFreq, amp, mean, std, C), label='Gaussian')
plt.plot(peakFreq, cm.lorentzian(peakFreq, amp, mean, gamma, C), label='Lorentzian')
plt.ylim(0, 1000)
plt.legend()
plt.show()
print(f"Gaussian Error: " + {cm.error_function(s1, cm.gaussian(peakFreq, amp, mean, std, C)))}
print(f"Lorentzian Error: " + {cm.error_function(s1, cm.lorentzian(peakFreq, amp, mean, gamma, C)))}
'''


# in conclusion gaussian curves suck ---- lorntzian time


# fits a lorentzian curve to 1 frequency spectra
"""
mean = peakFreq[np.argmax(s1)]
C = -13
amp = s1.max()-C
gamma = 0.36

params = np.array([amp, mean, gamma, C], dtype=float)
steps = np.array([0.01*amp, 0.01, 0.01*gamma, 0.1], dtype=float)

phase = 0

for iteration in range(50):
    params_prev = params
    params, err = cm.update_parameter(peakFreq, s1, params, 0, steps[0])  # amplitude
    params, err = cm.update_parameter(peakFreq, s1, params, 1, steps[1])  # mean
    params, err = cm.update_parameter(peakFreq, s1, params, 2, steps[2])  # gamma
    params, err = cm.update_parameter(peakFreq, s1, params, 3, steps[3])  # C
    
    if(np.array_equal(params, params_prev)):
        if(phase < 4):
            phase = phase + 1
            steps *= 0.5
        else:
            break
    
    #print(f"iteration {iteration}, phase {phase}: params={params}, error={err:.0f}")
    #plt.figure()
    #plt.plot(peakFreq, s1, label = 'Raw')
    #plt.plot(peakFreq, cm.lorentzian(peakFreq, *params), label='Lorentzian')
    #plt.ylim(0, 1000)
    #plt.legend()
    #plt.show()
"""


fittedData = [] # intensity, frequency shift, peak width, C
noises = []
for spectrum in peakSpectra:
    mean = peakFreq[np.argmax(spectrum)]
    C = -13
    amp = spectrum.max()-C
    gamma = 0.36

    params = np.array([amp, mean, gamma, C], dtype=float)
    steps = np.array([0.01*amp, 0.01, 0.01*gamma, 0.1], dtype=float)

    phase = 0

    for iteration in range(50):
        params_prev = params.copy()
        params, err = cm.update_parameter(peakFreq, spectrum, params, 0, steps[0])  # amplitude
        params, err = cm.update_parameter(peakFreq, spectrum, params, 1, steps[1])  # mean
        params, err = cm.update_parameter(peakFreq, spectrum, params, 2, steps[2])  # gamma
        params, err = cm.update_parameter(peakFreq, spectrum, params, 3, steps[3])  # C
    
        if(np.array_equal(params, params_prev)):
            if(phase < 4):
                phase = phase + 1
                steps *= 0.5
            else:
                break
    fittedData.append(params)
    noises.append(err)
    
# side by side of fitted data vs raw data
'''  
plt.figure()
plt.ylim(0, 1000)
for data in fittedData:
    plt.plot(peakFreq, cm.lorentzian(peakFreq, *data))
plt.show()


plt.figure()
plt.ylim(0, 1000)
for spectrum in peakSpectra:
    plt.plot(peakFreq, spectrum)
plt.show()
'''

# peak intensity
heatmap = np.full((len(y_vals)+1, len(x_vals)+1), np.nan) #empty heatmap
for row_i, row in df.iterrows():
    x_i = int(row["X"])
    y_i = int(row["Y"])
    heatmap[y_i, x_i] = (fittedData[row_i][0]-fittedData[row_i][3])
plt.figure()
plt.imshow(heatmap, origin="lower", cmap="inferno")
plt.colorbar(label="Intensity of Brillouin Peak")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Brillouin Peak Intensity Map")
plt.show()

# frequency shift
heatmap = np.full((len(y_vals)+1, len(x_vals)+1), np.nan) #empty heatmap
for row_i, row in df.iterrows():
    x_i = int(row["X"])
    y_i = int(row["Y"])
    heatmap[y_i, x_i] = fittedData[row_i][1]
plt.figure()
plt.imshow(heatmap, origin="lower", cmap="magma")
plt.colorbar(label="Frequency shift of brillouin peak (GHz)")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Frequency Shift Heatmap")
plt.show()

# peak width
heatmap = np.full((len(y_vals)+1, len(x_vals)+1), np.nan) #empty heatmap
for row_i, row in df.iterrows():
    x_i = int(row["X"])
    y_i = int(row["Y"])
    heatmap[y_i, x_i] = fittedData[row_i][2]
plt.figure()
plt.imshow(heatmap, origin="lower", cmap="viridis")
plt.colorbar(label="Peak Width")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Peak Width Heatmap")
plt.show()



# noise level
heatmap = np.full((len(y_vals), len(x_vals)), np.nan) #empty heatmap
for row_i, row in df.iterrows():
    x_i = int(row["X"])
    y_i = int(row["Y"])
    heatmap[y_i, x_i] = noises[row_i]
plt.figure()
plt.imshow(heatmap, origin="lower", cmap="magma")
plt.colorbar(label="noise level")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Noise Potency map")
plt.show()