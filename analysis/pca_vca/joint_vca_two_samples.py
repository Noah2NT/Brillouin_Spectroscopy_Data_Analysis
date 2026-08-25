# Joint VCA on two samples: one shared endmember set, abundance maps side by side.


from pathlib import Path
import argparse
import sys, os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import brillouin_preprocessing as bp
import brillouin_decomposition as bd


DATA_DIR = Path("data/csv")
OUTPUT_DIR = Path("outputs") / "joint_vca"

ALIASES = {"RawData": "Raw Data", "FittingData": "Fitting Data",
           "HalfWidth": "Half Width"}


# ─────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────

def resolve_path(name):
    """Accept a full path, a path relative to the repo root, or a bare filename
    living in data/csv."""
    for cand in (Path(name), DATA_DIR / name):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"{name}: not found (looked in . and {DATA_DIR})")


def load_sample(path, max_shift_ghz):
    """Fold, crop and return one scan's spectra.

    The returned dict carries the flat spectra, the shift axis, the scatter
    indices needed to paint a per-pixel value back onto the scan grid, and the
    grid shape.
    """
    df = pd.read_csv(path).rename(columns={k: v for k, v in ALIASES.items()})
    spectra = np.vstack(
        df["Raw Data"].apply(lambda s: np.fromstring(s, sep=",")))
    freq = np.linspace(df["startFrequency"].iloc[0],
                       df["endFrequency"].iloc[0], spectra.shape[1])

    folded, shift = bp.fold_brillouin_spectrum(spectra, freq)
    keep = shift <= max_shift_ghz
    folded, shift = folded[:, keep], shift[keep]

    x_vals, y_vals = bp.build_coordinate_arrays(df)
    xi = np.searchsorted(x_vals, df["X"].to_numpy())
    yi = np.searchsorted(y_vals, df["Y"].to_numpy())

    return dict(name=Path(path).stem, folded=folded, shift_freq=shift,
                xi=xi, yi=yi, shape=(len(y_vals), len(x_vals)))


def align_axes(a, b):
    """Put both samples on a common shift axis so their spectra can be pooled.

    Identical axes are left alone; otherwise both are resampled onto the
    overlapping frequency range at the coarser of the two bin counts.
    """
    fa, fb = a["shift_freq"], b["shift_freq"]
    if len(fa) == len(fb) and np.allclose(fa, fb):
        return fa

    lo, hi = max(fa[0], fb[0]), min(fa[-1], fb[-1])
    if hi <= lo:
        raise ValueError("The two scans share no overlapping frequency range.")
    common = np.linspace(lo, hi, min(len(fa), len(fb)))
    for s in (a, b):
        f = s["shift_freq"]
        s["folded"] = np.vstack([np.interp(common, f, row)
                                 for row in s["folded"]])
        s["shift_freq"] = common
    print(f"  [note] axes differed; resampled both onto {len(common)} bins "
          f"over {lo:.2f}-{hi:.2f} GHz")
    return common


def scatter_to_map(values, sample):
    """Paint a per-pixel array back onto the scan grid (NaN where unvisited)."""
    m = np.full(sample["shape"], np.nan)
    m[sample["yi"], sample["xi"]] = values
    return m


# ─────────────────────────────────────────────
# Joint decomposition
# ─────────────────────────────────────────────

def joint_vca(a, b, num_endmembers, seed):
    """Run VCA once on the pooled spectra, then unmix each sample separately."""
    pool = np.vstack([a["spectra"], b["spectra"]])
    n_a = a["spectra"].shape[0]

    endmembers, indices = bd.vca(pool, num_endmembers, seed=seed)

    # Trace each endmember back to the sample and pixel it came from.
    origins = []
    for idx in indices:
        if idx < n_a:
            origins.append((a["name"], int(idx)))
        else:
            origins.append((b["name"], int(idx - n_a)))

    for s in (a, b):
        s["abundances"] = bd.estimate_abundances(s["spectra"], endmembers)
        s["errors"] = bd.compute_reconstruction_error(
            s["spectra"], endmembers, s["abundances"])

    return endmembers, indices, origins


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def short(name, limit=34):
    """Trim long sample stems so panel titles do not run off the figure."""
    return name if len(name) <= limit else name[:limit - 1] + "…"


def plot_abundance_grid(a, b, endmembers, origins, shift_freq, out_path):
    """One row per endmember: sample A map, sample B map, endmember spectrum."""
    p = endmembers.shape[0]
    fig, axes = plt.subplots(p, 3, figsize=(13, 3.4 * p),
                             constrained_layout=True, squeeze=False)

    for k in range(p):
        for col, s in enumerate((a, b)):
            amap = scatter_to_map(s["abundances"][:, k], s)
            im = axes[k, col].imshow(amap, origin="lower", cmap="viridis",
                                     vmin=0, vmax=1)
            mean = s["abundances"][:, k].mean()
            axes[k, col].set_title(
                f"{short(s['name'])}\nEM {k+1} abundance (mean {mean:.2f})",
                fontsize=9)
            axes[k, col].set_xticks([]); axes[k, col].set_yticks([])
            if col == 1:
                fig.colorbar(im, ax=axes[k, col], shrink=0.8,
                             label="abundance")

        axes[k, 2].plot(shift_freq, endmembers[k], linewidth=1.5,
                        color=f"C{k}")
        src_name, src_px = origins[k]
        axes[k, 2].set_title(f"EM {k+1} spectrum\nfrom {short(src_name)} "
                             f"px {src_px}", fontsize=9)
        axes[k, 2].set_xlabel("Shift (GHz)")
        axes[k, 2].set_ylabel("Normalised intensity")
        axes[k, 2].grid(alpha=0.3)

    fig.suptitle(f"Joint VCA({p}) - shared endmembers across "
                 f"{short(a['name'])} and {short(b['name'])}", fontsize=11)
    fig.savefig(out_path, dpi=140)
    return fig


def plot_endmember_spectra(endmembers, origins, shift_freq, out_path):
    """All shared endmember spectra overlaid on one axis."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for k, em in enumerate(endmembers):
        src_name, src_px = origins[k]
        ax.plot(shift_freq, em, linewidth=1.5,
                label=f"EM {k+1}  ({short(src_name)} px {src_px})",
                color=f"C{k}")
    ax.set_xlabel("Shift (GHz)")
    ax.set_ylabel("Normalised intensity")
    ax.set_title("Shared endmember spectra")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=140)
    return fig


# ─────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────

def prompt_inputs():
    """Ask for both files and the endmember count when they were not given on
    the command line. Uses a file dialog if one is available, plain prompts
    otherwise."""
    paths = []
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        initial = str(DATA_DIR if DATA_DIR.is_dir() else Path.cwd())
        for label in ("first", "second"):
            chosen = filedialog.askopenfilename(
                title=f"Select the {label} scan CSV",
                initialdir=initial, filetypes=[("CSV files", "*.csv")])
            if not chosen:
                paths = []
                break
            paths.append(chosen)
        root.destroy()
    except Exception:
        pass

    if len(paths) != 2:
        paths = [input("Path to first CSV:  ").strip().strip('"'),
                 input("Path to second CSV: ").strip().strip('"')]

    while True:
        raw = input("Number of endmembers [3]: ").strip() or "3"
        try:
            n = int(raw)
        except ValueError:
            print("  enter a whole number")
            continue
        if n < 2:
            print("  VCA needs at least 2 endmembers")
            continue
        return paths[0], paths[1], n


def parse_args():
    ap = argparse.ArgumentParser(
        description="Joint VCA on two Brillouin scans with shared endmembers.")
    ap.add_argument("csv1", nargs="?", help="first scan CSV")
    ap.add_argument("csv2", nargs="?", help="second scan CSV")
    ap.add_argument("-n", "--endmembers", type=int,
                    help="number of shared endmembers (>= 2)")
    ap.add_argument("--max-shift", type=float, default=13.0,
                    help="crop folded spectra above this shift, GHz (default 8)")
    ap.add_argument("--seed", type=int, default=42, help="VCA random seed")
    ap.add_argument("--out", default=str(OUTPUT_DIR),
                    help="directory for the saved figures")
    ap.add_argument("--no-show", action="store_true",
                    help="save the figures without opening a window")
    args = ap.parse_args()

    if not (args.csv1 and args.csv2 and args.endmembers):
        args.csv1, args.csv2, args.endmembers = prompt_inputs()
    if args.endmembers < 2:
        ap.error("--endmembers must be at least 2")
    return args


# ─────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    a = load_sample(resolve_path(args.csv1), args.max_shift)
    b = load_sample(resolve_path(args.csv2), args.max_shift)
    for s in (a, b):
        print(f"loaded {s['name']}: {s['folded'].shape[0]} spectra, "
              f"grid {s['shape'][0]}x{s['shape'][1]}")

    shift_freq = align_axes(a, b)

    # Sum-normalise after alignment so overall brightness differences between
    # the two scans do not bias the shared endmembers towards one sample.
    for s in (a, b):
        s["spectra"] = bp.sum_normalize_spectra(s["folded"])

    n_pool = a["spectra"].shape[0] + b["spectra"].shape[0]
    if n_pool <= args.endmembers:
        raise ValueError(f"{n_pool} pooled spectra cannot yield "
                         f"{args.endmembers} endmembers")

    endmembers, _, origins = joint_vca(a, b, args.endmembers, args.seed)

    print(f"\nJoint VCA({args.endmembers}) on {n_pool} pooled spectra")
    for k, (src_name, src_px) in enumerate(origins):
        print(f"  EM {k+1}: from {src_name}, pixel {src_px}")
    print("\nMean abundance per sample:")
    for s in (a, b):
        means = ", ".join(f"EM{k+1}={s['abundances'][:, k].mean():.3f}"
                          for k in range(args.endmembers))
        print(f"  {s['name']}: {means}")
        print(f"    reconstruction error - mean {s['errors'].mean():.4f}, "
              f"max {s['errors'].max():.4f}")

    stem = f"{a['name']}__vs__{b['name']}_vca{args.endmembers}"
    grid_path = out_dir / f"{stem}_abundance_grid.png"
    spec_path = out_dir / f"{stem}_endmember_spectra.png"
    plot_abundance_grid(a, b, endmembers, origins, shift_freq, grid_path)
    plot_endmember_spectra(endmembers, origins, shift_freq, spec_path)
    print(f"\nsaved {grid_path}\nsaved {spec_path}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
