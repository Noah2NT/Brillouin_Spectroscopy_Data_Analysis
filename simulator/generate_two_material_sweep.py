"""Generate a known-truth half-and-half material exposure sweep."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from simulator.core.drawing import CellDrawing
from simulator.core.instrument import preset_narrow
from simulator.core.materials import MaterialLibrary
from simulator.core.scan import ScanSettings, simulate_scan
from simulator.core.writer import MANIFEST_COLUMNS, manifest_row, write_scan_csv, write_truth_sidecar


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def exposures_arg(text: str):
    values = [float(v) for v in text.split(",")]
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("exposures must be positive")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", type=Path, default=REPO_ROOT / "simulator/resources/materials.json")
    parser.add_argument("--material-a", default="Water")
    parser.add_argument("--material-b", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--grid", type=int, default=30)
    parser.add_argument("--step-um", type=float, default=1.0)
    parser.add_argument("--exposures", type=exposures_arg,
                        default=exposures_arg("100,30,10,3,1,0.3,0.1,0.03,0.01"))
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    pair_slug = f"{slug(args.material_a)}_{slug(args.material_b)}"
    output = args.output or REPO_ROOT / f"outputs/simulated_{pair_slug}"
    output.mkdir(parents=True, exist_ok=True)
    library = MaterialLibrary.load(str(args.materials))
    a_i, b_i = library.index_of(args.material_a), library.index_of(args.material_b)
    if a_i is None or b_i is None:
        raise ValueError(f"Unknown material pair: {args.material_a}, {args.material_b}")

    drawing = CellDrawing.for_scan(args.grid, args.grid, args.step_um, material_names=library.names)
    drawing.index_map[:drawing.height_px // 2, :] = a_i
    drawing.index_map[drawing.height_px // 2:, :] = b_i
    drawing_path = Path(drawing.save(str(output / f"half_{slug(args.material_a)}_half_{slug(args.material_b)}")))

    instrument = replace(preset_narrow(), marker1_ghz=3.0, marker2_ghz=13.2)
    baseline = max(args.exposures)
    rows = []
    for i, exposure in enumerate(args.exposures):
        multiplier = (baseline / exposure) ** 0.5
        print(f"Generating {args.material_a}/{args.material_b}: {exposure:g} s ({multiplier:.4g}x noise)...", flush=True)
        settings = ScanSettings(grid_w=args.grid, grid_h=args.grid, step_um=args.step_um,
                                exposure_s=exposure, seed=args.seed + i, reference_fit=False,
                                fit_max_iter=300 if exposure <= 0.03 else 2000)
        result = simulate_scan(drawing, library, instrument, settings)
        stem = f"half_{slug(args.material_a)}_half_{slug(args.material_b)}_{exposure:g}s"
        csv_path, truth_path = output / f"{stem}.csv", output / f"{stem}_truth.npz"
        write_scan_csv(str(csv_path), result, instrument)
        write_truth_sidecar(str(truth_path), result, instrument, drawing_name=drawing_path.name)
        row = manifest_row(str(csv_path), str(truth_path), result, drawing_name=drawing_path.name)
        row.update(noise_multiplier=multiplier, photon_fraction=exposure / baseline,
                   material_a=args.material_a, material_b=args.material_b)
        rows.append(row)

    fields = MANIFEST_COLUMNS + ["noise_multiplier", "photon_fraction", "material_a", "material_b"]
    with open(output / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
