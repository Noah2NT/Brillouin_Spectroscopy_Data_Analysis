import numpy as np
import matplotlib.pyplot as plt


def plot_heatmap(data, x_vals, y_vals, title="Heatmap", cmap="magma",
                 colorbar_label="", ax=None):
    """
    Display a 2D spatial heatmap from a flat per-pixel array.

    Parameters:
        data            : ndarray (N,) or (len(y), len(x))
        x_vals          : array-like — unique x positions
        y_vals          : array-like — unique y positions
        title           : str
        cmap            : str — matplotlib colormap
        colorbar_label  : str
        ax              : matplotlib Axes or None (creates new figure)

    Returns:
        im : AxesImage
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))

    heatmap = np.asarray(data).reshape(len(y_vals), len(x_vals))
    im = ax.imshow(heatmap, origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    plt.colorbar(im, ax=ax, label=colorbar_label)

    if standalone:
        plt.tight_layout()
        plt.show()

    return im


def plot_spectrum(freq, spectrum, title="Spectrum",
                  xlabel="Frequency Shift (GHz)", ylabel="Intensity",
                  ax=None, **kwargs):
    """
    Plot a single spectrum.

    Parameters:
        freq     : ndarray (B,)
        spectrum : ndarray (B,)
        title    : str
        xlabel   : str
        ylabel   : str
        ax       : matplotlib Axes or None
        **kwargs : passed to plt.plot / ax.plot
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots()

    ax.plot(freq, spectrum, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_multiple_spectra(freq, spectra, labels=None, title="Spectra",
                           xlabel="Frequency Shift (GHz)", ylabel="Intensity",
                           figsize=(10, 6)):
    """
    Overlay multiple spectra on one axes.

    Parameters:
        freq    : ndarray (B,)
        spectra : ndarray (N, B) or list of ndarray
        labels  : list of str or None
        title   : str
    """
    fig, ax = plt.subplots(figsize=figsize)
    for i, s in enumerate(spectra):
        label = labels[i] if labels is not None else f"Spectrum {i+1}"
        ax.plot(freq, s, label=label, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_2d_colored_scatter(x, y, colors, xlabel="Component 1",
                             ylabel="Component 2", title="2D Scatter",
                             cmap="jet", centers=None, s=20):
    """
    2D scatter plot with per-point colour coding (e.g. cluster labels).

    Parameters:
        x, y    : ndarray (N,) — axis values
        colors  : ndarray (N,) — values mapped to colormap
        centers : ndarray (k, 2) or None — if provided, marks centroids with X
        s       : marker size
    """
    plt.figure()
    plt.scatter(x, y, c=colors, cmap=cmap, s=s)
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1],
                    s=200, c="red", marker="X", label="Centroids")
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_3d_colored_scatter(x, y, z, colors, xlabel="PC1", ylabel="PC2",
                             zlabel="PC3", title="3D Scatter", cmap="jet",
                             elev=40, azim=105, s=10, endmember_indices=None):
    """
    3D scatter plot with per-point colour coding.

    Parameters:
        x, y, z            : ndarray (N,)
        colors             : ndarray (N,)
        elev               : elevation angle for 3D view
        azim               : azimuth angle for 3D view
        s                  : marker size
        endmember_indices  : array-like of int or None — pixel indices to highlight as stars
    """
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")
    sc  = ax.scatter(x, y, z, c=colors, cmap=cmap, s=s)
    plt.colorbar(sc, ax=ax, label="Value")

    if endmember_indices is not None:
        for i, idx in enumerate(endmember_indices):
            ax.scatter(x[idx], y[idx], z[idx],
                       s=200, marker="*", edgecolors="black", linewidths=0.5,
                       zorder=5, label=f"Endmember {i+1}")
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.view_init(elev=elev, azim=azim)
    plt.show()


def plot_component_maps(arrays, x_vals, y_vals, labels=None,
                        suptitle="Component Maps", cmap="magma",
                        colorbar_label="Score", figsize=None, centres=None):
    """
    Render a row of heatmaps — one per component (PCs, endmember abundances, etc.).

    Parameters:
        arrays         : list of ndarray (N,) — one flat array per component
        x_vals         : array-like
        y_vals         : array-like
        labels         : list of str or None — subplot titles
        suptitle       : str — overall figure title
        cmap           : str
        colorbar_label : str
        figsize        : tuple or None (auto-sized if None)
        centres         : list of (x, y) tuples

    """
    n = len(arrays)
    if labels is None:
        labels = [f"Component {i+1}" for i in range(n)]
    if figsize is None:
        figsize = (5 * n, 5)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        ax = [axes]

    for i, (arr, label) in enumerate(zip(arrays, labels)):

        plot_heatmap(arr, x_vals, y_vals,
                     title=label, cmap=cmap,
                     colorbar_label=colorbar_label,
                     ax=axes[i])

        if centres is not None and i < len(centres):
            x, y = centres[i]
            axes[i].scatter(x, y, color='cyan', s=80,
                       edgecolor='black', zorder=10)

    plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    plt.show()
