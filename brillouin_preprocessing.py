import numpy as np
import pandas as pd


def load_brillouin_csv(filepath):
    """
    Load a Brillouin scan CSV file.

    Returns:
        spectra : ndarray (N, B) — raw intensity spectra
        freq    : ndarray (B,)   — frequency axis (GHz)
        df      : DataFrame      — original dataframe (contains X, Y columns)
    """
    df = pd.read_csv(filepath)
    spectra = np.vstack(
        df["Raw Data"].apply(lambda s: np.array([float(v) for v in s.split(",")]))
    )
    start = df["startFrequency"].iloc[0]
    end   = df["endFrequency"].iloc[0]
    freq  = np.linspace(start, end, spectra.shape[1])
    return spectra, freq, df


def extract_frequency_window(spectra, freq, start_idx, end_idx):
    """
    Slice spectra and frequency axis to the Brillouin peak window.

    Parameters:
        spectra   : ndarray (N, B)
        freq      : ndarray (B,)
        start_idx : int — index of window start
        end_idx   : int — index of window end (exclusive)

    Returns:
        peak_spectra : ndarray (N, W)
        peak_freq    : ndarray (W,)
    """
    return spectra[:, start_idx:end_idx], freq[start_idx:end_idx]


def build_coordinate_arrays(df):
    """
    Extract sorted unique X and Y scan positions from the dataframe.

    Returns:
        x_vals : ndarray — unique x positions in ascending order
        y_vals : ndarray — unique y positions in ascending order
    """
    x_vals = np.sort(df["X"].unique())
    y_vals = np.sort(df["Y"].unique())
    return x_vals, y_vals


def reshape_to_map(data, y_vals, x_vals):
    """
    Reshape a flat per-pixel array into a 2D spatial map.

    Parameters:
        data   : ndarray (N,) — one value per scan pixel
        y_vals : array-like   — unique y positions
        x_vals : array-like   — unique x positions

    Returns:
        ndarray (len(y_vals), len(x_vals))
    """
    return data.reshape(len(y_vals), len(x_vals))
