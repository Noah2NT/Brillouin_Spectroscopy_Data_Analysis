"""Evaluate the spectral-only Poisson control on a completed saved benchmark.

Reuses exact calibration/test observations and seed schedule, without rerunning
the original methods. Idempotent: refuses to duplicate an existing extension.
"""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

from brillouin_denoising import reconstruct
from analysis.reconstruction_benchmark import (candidate_configs, select_configs, metrics,
                                                relative_error, normalise, write_csv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/reconstruction_benchmark"))
    args = parser.parse_args()
    out = args.output
    manifest = json.loads((out / "manifest.json").read_text())
    settings = manifest["arguments"]
    def read(name):
        with open(out / name, newline="", encoding="utf8") as f:
            return list(csv.DictReader(f))
    rows, tuning, stability = read("metrics.csv"), read("tuning.csv"), read("repeat_stability.csv")
    if any(r["method"] == "poisson_spectral" for r in rows):
        raise ValueError("Spectral control already exists; use --report-only on the benchmark")
    chosen_all = json.loads((out / "selected_configs.json").read_text())
    configs = [c for c in candidate_configs(settings["iterations"]) if c.method == "poisson_spectral"]
    start = time.perf_counter()
    with threadpool_limits(1):
        for scene_id, name in enumerate(settings["scenes"]):
            with np.load(out / "data" / f"{name}_truth.npz") as data:
                scene = dict(data)
            references = {}
            for level, exposure in enumerate(settings["exposures"]):
                print(f"Spectral control: {name}, {exposure:g} s", flush=True)
                base_seed = settings["seed"] + scene_id*100000 + level*1000
                with np.load(out / "data" / f"{name}_{exposure:g}s_calibration.npz") as data:
                    chosen, detail = select_configs(data["counts"], configs, base_seed+1)
                tuning.extend(dict(scene=name, exposure=exposure, **r) for r in detail)
                config = chosen["poisson_spectral"]
                chosen_all[f"{name}/{exposure:g}"]["poisson_spectral"] = config.to_dict()
                curves = []
                for repeat in range(settings["repeats"]):
                    with np.load(out / "data" / f"{name}_{exposure:g}s_seed{repeat}.npz") as data:
                        counts, test = data["counts"], data["independent_test"]
                    before = time.perf_counter()
                    estimate = reconstruct(counts, config)
                    elapsed = time.perf_counter()-before
                    values = metrics(estimate, scene, exposure, test, references.get(repeat))
                    rows.append(dict(scene=name, exposure=exposure, repeat=repeat,
                                     method="poisson_spectral", seconds=elapsed, **values))
                    rate = estimate/exposure
                    curves.append(rate)
                    if level == 0:
                        references[repeat] = rate
                differences = [relative_error(normalise(curves[a]), normalise(curves[b]), scene["inside"])
                               for a in range(len(curves)) for b in range(a)]
                stability.append(dict(scene=name, exposure=exposure, method="poisson_spectral",
                                      pairwise_shape_delta_mean=float(np.mean(differences)),
                                      pairwise_shape_delta_max=float(np.max(differences))))
                path = out / "examples" / f"{name}_{exposure:g}s.npz"
                with np.load(path) as data:
                    arrays = dict(data)
                arrays["poisson_spectral"] = curves[0]
                np.savez_compressed(path, **arrays)
    write_csv(out / "metrics.csv", rows)
    write_csv(out / "tuning.csv", tuning)
    write_csv(out / "repeat_stability.csv", stability)
    (out / "selected_configs.json").write_text(json.dumps(chosen_all, indent=2))
    manifest["spectral_control_extension"] = dict(runtime_seconds=time.perf_counter()-start,
                                                configurations=[c.to_dict() for c in configs],
                                                motivation="Exploratory response to initial thin-feature attenuation")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Extension complete. Regenerate the report with --report-only.")


if __name__ == "__main__":
    main()
