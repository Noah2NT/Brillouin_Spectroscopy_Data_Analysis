"""Generate a known-truth half-water/half-plexiglass noise sweep.

This drives the same headless core used by the Cell Builder UI.  Noise is
increased by reducing exposure: a requested relative shot-noise multiplier
``m`` uses ``baseline_exposure / m**2`` seconds.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simulator.core.drawing import CellDrawing
from simulator.core.instrument import preset_narrow
from simulator.core.materials import MaterialLibrary
from simulator.core.scan import ScanSettings, simulate_scan
from simulator.core.writer import (MANIFEST_COLUMNS, manifest_row,
                                   write_scan_csv, write_truth_sidecar)


def levels_arg(text):
    values = [float(v) for v in text.split(",")]
    if not values or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError("all noise multipliers must be >= 1")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", type=Path,
                        default=REPO_ROOT / "simulator/resources/materials.json")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "outputs/simulated_water_plexiglass")
    parser.add_argument("--grid", type=int, default=30,
                        help="square scan width/height in pixels")
    parser.add_argument("--step-um", type=float, default=1.0)
    parser.add_argument("--baseline-exposure", type=float, default=10.0)
    parser.add_argument("--levels", type=levels_arg,
                        default=levels_arg("1,2,4,8,16,24,32,48,64"))
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    library = MaterialLibrary.load(str(args.materials))
    water = library.index_of("Water")
    plexiglass = library.index_of("Plexiglass")
    if water is None or plexiglass is None:
        raise ValueError("materials.json must contain Water and Plexiglass")

    drawing = CellDrawing.for_scan(args.grid, args.grid, args.step_um,
                                   material_names=library.names)
    split = drawing.height_px // 2
    drawing.index_map[:split, :] = water
    drawing.index_map[split:, :] = plexiglass
    drawing_path = Path(drawing.save(str(args.output / "half_water_half_plexiglass")))

    # The normal narrow preset stops its fitting window at 8.48 GHz.  Extend it
    # so the CSV's fitted BFS can represent the 11.75 GHz plexiglass peak too.
    instrument = replace(preset_narrow(), marker1_ghz=3.0, marker2_ghz=13.2)
    manifest_rows = []
    for i, multiplier in enumerate(args.levels):
        exposure = args.baseline_exposure / multiplier**2
        settings = ScanSettings(
            grid_w=args.grid, grid_h=args.grid, step_um=args.step_um,
            exposure_s=exposure, seed=args.seed + i,
            reference_fit=False,
            fit_max_iter=300 if exposure <= 0.03 else 2000,
        )
        print(f"Generating {multiplier:g}x noise ({exposure:.6g} s)...",
              flush=True)
        result = simulate_scan(drawing, library, instrument, settings)
        stem = f"half_water_half_plexiglass_noise_{multiplier:g}x"
        csv_path = args.output / f"{stem}.csv"
        truth_path = args.output / f"{stem}_truth.npz"
        write_scan_csv(str(csv_path), result, instrument)
        write_truth_sidecar(str(truth_path), result, instrument,
                            drawing_name=drawing_path.name)
        row = manifest_row(str(csv_path), str(truth_path), result,
                           drawing_name=drawing_path.name)
        row["noise_multiplier"] = multiplier
        row["photon_fraction"] = 1.0 / multiplier**2
        manifest_rows.append(row)

    fields = MANIFEST_COLUMNS + ["noise_multiplier", "photon_fraction"]
    with open(args.output / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Drawing: {drawing_path}")
    print(f"Generated {len(manifest_rows)} scans in {args.output}")


if __name__ == "__main__":
    main()
