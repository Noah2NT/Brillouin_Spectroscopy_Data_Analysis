import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import (
    gaussian_filter,
    sobel,
    laplace,
    maximum_filter,
    binary_erosion,
    binary_dilation,
    binary_opening,
    binary_closing
)

def max_pool(arr, k=2):
    ny, nx = arr.shape
    out = np.zeros((ny // k, nx // k))
    for i in range(0, ny - ny % k, k):
        for j in range(0, nx - nx % k, k):
            out[i // k, j // k] = np.max(arr[i:i+k, j:j+k])
    return out


def avg_pool(arr, k=2):
    ny, nx = arr.shape
    out = np.zeros((ny // k, nx // k))
    for i in range(0, ny - ny % k, k):
        for j in range(0, nx - nx % k, k):
            out[i // k, j // k] = np.mean(arr[i:i+k, j:j+k])
    return out


def smooth_heatmap(arr, sigma=1):
    return gaussian_filter(arr, sigma=sigma)


def gradient_magnitude(arr):
    gy, gx = np.gradient(arr)
    grad = np.sqrt(gx**2 + gy**2)
    return grad


def sobel_edges(arr):
    gx = sobel(arr, axis=1)
    gy = sobel(arr, axis=0)
    edges = np.sqrt(gx**2 + gy**2)
    return edges


def laplacian_filter(arr):
    return laplace(arr)


def threshold_mask(arr, threshold):
    return (arr > threshold).astype(int)


def local_maxima(arr, size=3):
    max_arr = maximum_filter(arr, size=size)
    return (arr == max_arr).astype(int)


def erode_mask(mask, iterations=1):
    out = mask.astype(bool)
    for _ in range(iterations):
        out = binary_erosion(out)
    return out.astype(int)


def dilate_mask(mask, iterations=1):
    out = mask.astype(bool)
    for _ in range(iterations):
        out = binary_dilation(out)
    return out.astype(int)


def open_mask(mask):
    return binary_opening(mask.astype(bool)).astype(int)


def close_mask(mask):
    return binary_closing(mask.astype(bool)).astype(int)


def clean_mask(mask):
    mask = binary_opening(mask.astype(bool))
    mask = binary_closing(mask.astype(bool))
    return mask.astype(int)
