"""Reconstruct a vendor CSV or a raw-count NPZ cube without Lorentzian fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

from brillouin_denoising import DenoiseConfig, reconstruct, validate_counts
from analysis.reconstruction_benchmark import candidate_configs, select_configs


def load_input(path, window=None):
    """Sort actual coordinates; refuse missing/duplicate pixels rather than reshape."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        from brillouin_preprocessing import load_brillouin_csv
        spectra, frequency, frame = load_brillouin_csv(path)
        x, y = np.sort(frame.X.unique()), np.sort(frame.Y.unique())
        if len(frame) != len(x)*len(y) or frame.duplicated(["X", "Y"]).any():
            raise ValueError("CSV must contain one spectrum per pixel on a complete rectangular grid")
        if not np.all(np.isfinite(frame[["X", "Y"]].to_numpy())):
            raise ValueError("Nonfinite raster coordinates")
        if frame.startFrequency.nunique() != 1 or frame.endFrequency.nunique() != 1:
            raise ValueError("Per-pixel frequency axes differ; calibrate before using this loader")
        order = np.lexsort((frame.X.to_numpy(), frame.Y.to_numpy()))
        counts = spectra[order].reshape(len(y), len(x), -1)
        if window is None:
            raise ValueError("CSV requires --window LOW HIGH in GHz to select a Brillouin branch")
    else:
        with np.load(path, allow_pickle=False) as data:
            counts = data["counts"]
            frequency = data["frequency"] if "frequency" in data else np.arange(counts.shape[-1])
            x = data["x"] if "x" in data else np.arange(counts.shape[1])
            y = data["y"] if "y" in data else np.arange(counts.shape[0])
    counts = validate_counts(counts)
    if frequency.shape != (counts.shape[-1],) or not np.all(np.isfinite(frequency)):
        raise ValueError("Invalid frequency axis")
    if not np.all(np.diff(frequency) > 0):
        raise ValueError("Frequency axis must increase")
    if window is not None:
        keep = (frequency >= window[0]) & (frequency <= window[1])
        if keep.sum() < 4:
            raise ValueError("Window contains fewer than four spectral bins")
        counts, frequency = counts[..., keep], frequency[keep]
    return counts, frequency, x, y


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--window", nargs=2, type=float, metavar=("LOW", "HIGH"))
    p.add_argument("--method", choices=["poisson_nlpca", "poisson_spectral", "anscombe_nlpca", "global_pca", "spatial_mean"],
                   default="poisson_nlpca")
    p.add_argument("--rank", type=int, default=3)
    p.add_argument("--patch", type=int, default=3)
    p.add_argument("--clusters", type=int, default=4)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--select", action="store_true",
                   help="Choose the requested method's configuration by Poisson photon splitting; integer photon counts only")
    args = p.parse_args()
    if args.output.suffix.lower() != ".npz":
        p.error("Output must use the .npz extension")
    if args.output.resolve() == args.input.resolve():
        p.error("Output must differ from the source")
    counts, frequency, x, y = load_input(args.input, args.window)
    config = DenoiseConfig(args.method, args.rank, 1 if args.method == "poisson_spectral" else args.patch,
                          args.clusters, args.iterations, seed=args.seed)
    details = []
    with threadpool_limits(1):
        if args.select:
            configs = [c for c in candidate_configs(args.iterations) if c.method == args.method]
            selected, details = select_configs(counts, configs, args.seed)
            config = selected[args.method]
        estimate = reconstruct(counts, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        np.savez_compressed(f, counts=counts, expected_counts=estimate,
                            normalised_spectra=estimate/np.maximum(estimate.sum(axis=-1, keepdims=True), 1e-12),
                            residual=counts-estimate, poisson_variance=estimate,
                            frequency=frequency, x=x, y=y)
    args.output.with_suffix(".json").write_text(json.dumps(dict(
        source=str(args.input.resolve()), config=config.to_dict(), selection=details,
        estimand="Expected measured spectrum, including medium; not a pure material curve",
        noise_assumption="Independent Poisson counts; variance is acquisition variance, not uncertainty of the estimate",
        preprocessing="Frequency crop only; no normalisation, baseline subtraction or folding before fit"
    ), indent=2), encoding="utf8")
    print(args.output.resolve())
    print(json.dumps(config.to_dict()))


if __name__ == "__main__":
    main()
