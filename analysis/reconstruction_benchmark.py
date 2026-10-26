"""Reproducible photon-exposure benchmark with truth-blind model selection.

Run from the repository root: python -m analysis.reconstruction_benchmark
No clean truth or evaluation realization is used to select hyperparameters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from brillouin_denoising import DenoiseConfig, reconstruct, thin_counts, poisson_deviance
from simulator.generate_reconstruction_scenes import SCENES, make_scene, sample_counts


# Declared before running. These are engineering acceptance criteria, not
# universal biological accuracy requirements or calibrated confidence bounds.
THRESHOLDS = dict(shape_nrmse=0.08, boundary_shape_nrmse=0.12,
                  centroid_rmse_mhz=15.0, iqr_rmse_mhz=40.0,
                  contrast_min=0.70, contrast_max=1.30,
                  high_exposure_shape_delta=0.08, raw_error_ratio=0.70)


def candidate_configs(iterations=20):
    return [DenoiseConfig("raw"),
            *[DenoiseConfig("spatial_mean", patch=p) for p in (3, 5)],
            *[DenoiseConfig("global_pca", rank=r) for r in (2, 4)],
            *[DenoiseConfig("anscombe_nlpca", rank=r, patch=3, clusters=4) for r in (2, 4)],
            *[DenoiseConfig("poisson_nlpca", rank=r, patch=3, clusters=4,
                           iterations=iterations) for r in (2, 4)],
            DenoiseConfig("poisson_nlpca", rank=3, patch=1, clusters=4,
                          iterations=iterations),
            *[DenoiseConfig("poisson_spectral", rank=r, patch=1, clusters=4,
                            iterations=iterations) for r in (2, 3, 4)]]


def normalise(a):
    return a / np.maximum(a.sum(axis=-1, keepdims=True), 1e-12)


def spectral_features(q, frequency):
    """Fit-free finite-window centroid and central-50%-area width, NOT BFS/FWHM."""
    centroid = q @ frequency
    cdf = np.cumsum(q, axis=-1)
    widths = []
    for row in cdf.reshape(-1, len(frequency)):
        widths.append(np.interp(.75, row, frequency) - np.interp(.25, row, frequency))
    return centroid, np.asarray(widths).reshape(q.shape[:-1])


def relative_error(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    return float(np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-12))


def metrics(estimate, scene, exposure, independent, reference=None):
    truth = scene["rate"]
    rate = estimate / exposure
    p, q = normalise(rate), normalise(truth)
    centre, width = spectral_features(p, scene["frequency"])
    tc, tw = spectral_features(q, scene["frequency"])
    inside = scene["inside"]
    target_contrast = tc[scene["feature"]].mean() - tc[scene["control"]].mean()
    contrast = centre[scene["feature"]].mean() - centre[scene["control"]].mean()
    mean = truth * exposure
    # Independent residual = new photon noise + reconstruction error.
    # Exact expectation of this mean square is 1 + shot_noise_error.
    independent_r = (independent-estimate) / np.sqrt(np.maximum(mean, 1e-12))
    residual_mean_spectrum = independent_r.mean(axis=(0, 1))
    df = float(np.mean(np.diff(scene["frequency"])))
    return dict(
        rate_nrmse=relative_error(rate, truth),
        shape_nrmse=relative_error(p, q, inside),
        boundary_shape_nrmse=relative_error(p, q, scene["boundary"]),
        background_shape_nrmse=relative_error(p, q, ~inside),
        feature_shape_nrmse=relative_error(p, q, scene["feature"]),
        centroid_rmse_mhz=float(np.sqrt(np.mean((centre[inside]-tc[inside])**2))*1000),
        iqr_rmse_mhz=float(np.sqrt(np.mean((width[inside]-tw[inside])**2))*1000),
        wasserstein_mhz=float(np.mean(np.sum(abs(np.cumsum(p-q, axis=-1)), axis=-1)[inside])*df*1000),
        contrast_ratio=float(contrast/target_contrast),
        true_contrast_mhz=float(target_contrast*1000),
        shot_noise_error=float(np.mean((estimate-mean)**2 / mean)),
        independent_residual_ms=float(np.mean(independent_r**2)),
        residual_spectral_mean_rms=float(np.sqrt(np.mean(residual_mean_spectrum**2))),
        test_excess_deviance_per_photon=float(np.sum(poisson_deviance(independent, estimate)
                                                     - poisson_deviance(independent, mean))/mean.sum()),
        high_exposure_shape_delta=(relative_error(p, normalise(reference), inside)
                                   if reference is not None else 0.0),
    )


def select_configs(counts, configs, seed):
    train, validation = thin_counts(counts, .8, seed)
    rows, chosen = [], {}
    for c in configs:
        started = time.perf_counter()
        estimate = reconstruct(train, c) * .25
        score = float(np.mean(poisson_deviance(validation, estimate)))
        rows.append(dict(method=c.method, config=json.dumps(c.to_dict(), sort_keys=True),
                         validation_deviance=score, seconds=time.perf_counter()-started))
        if c.method not in chosen or score < chosen[c.method][0]:
            chosen[c.method] = (score, c)
    return {method: value[1] for method, value in chosen.items()}, rows


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = []
    keys = ("shape_nrmse", "rate_nrmse", "boundary_shape_nrmse", "feature_shape_nrmse",
            "background_shape_nrmse", "centroid_rmse_mhz", "iqr_rmse_mhz",
            "wasserstein_mhz", "contrast_ratio", "shot_noise_error", "independent_residual_ms",
            "residual_spectral_mean_rms", "test_excess_deviance_per_photon",
            "high_exposure_shape_delta", "seconds")
    for scene, exposure, method in dict.fromkeys((r["scene"], r["exposure"], r["method"]) for r in rows):
        group = [r for r in rows if (r["scene"], r["exposure"], r["method"]) == (scene, exposure, method)]
        item = dict(scene=scene, exposure=exposure, method=method, repeats=len(group))
        for key in keys:
            values = np.array([r[key] for r in group])
            item[key] = float(values.mean())
            item[key+"_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(item)
    for item in summary:
        raw = next(r for r in summary if r["scene"] == item["scene"] and
                   r["exposure"] == item["exposure"] and r["method"] == "raw")
        item["raw_error_ratio"] = item["shape_nrmse"]/raw["shape_nrmse"]
        failed = []
        for key in ("shape_nrmse", "boundary_shape_nrmse", "centroid_rmse_mhz",
                    "iqr_rmse_mhz", "high_exposure_shape_delta", "raw_error_ratio"):
            if item[key] > THRESHOLDS[key]:
                failed.append(key)
        if not THRESHOLDS["contrast_min"] <= item["contrast_ratio"] <= THRESHOLDS["contrast_max"]:
            failed.append("contrast_ratio")
        item["passes_mean_criteria"] = not failed
        item["failed_criteria"] = ",".join(failed)
        passing_repeats = 0
        for row in rows:
            if (row["scene"], row["exposure"], row["method"]) != (item["scene"], item["exposure"], item["method"]):
                continue
            raw_repeat = next(r for r in rows if r["scene"] == row["scene"] and
                              r["exposure"] == row["exposure"] and r["repeat"] == row["repeat"]
                              and r["method"] == "raw")
            passed = all(row[key] <= THRESHOLDS[key] for key in
                         ("shape_nrmse", "boundary_shape_nrmse", "centroid_rmse_mhz",
                          "iqr_rmse_mhz", "high_exposure_shape_delta"))
            passed &= row["shape_nrmse"]/raw_repeat["shape_nrmse"] <= THRESHOLDS["raw_error_ratio"]
            passed &= THRESHOLDS["contrast_min"] <= row["contrast_ratio"] <= THRESHOLDS["contrast_max"]
            passing_repeats += int(passed)
        item["passing_repeats"] = passing_repeats
        item["all_repeats_pass"] = passing_repeats == item["repeats"]
    return summary


COLORS = dict(raw="#a3a3a3", spatial_mean="#b57926", global_pca="#6b59a5",
              anscombe_nlpca="#268a73", poisson_nlpca="#1766b5", poisson_frozen="#bd4f63",
              poisson_spectral="#d35f14")


def plots(output, summary, scenes, examples, exposures):
    methods = list(COLORS)
    fig, axes = plt.subplots(len(scenes), 3, figsize=(14, 3.7*len(scenes)), squeeze=False)
    for iy, scene in enumerate(scenes):
        for ix, (metric, title) in enumerate((("shape_nrmse", "Cell spectral error"),
                                             ("centroid_rmse_mhz", "Centroid error (MHz)"),
                                             ("contrast_ratio", "Small-feature / domain contrast"))):
            ax = axes[iy, ix]
            for method in methods:
                data = [r for r in summary if r["scene"] == scene and r["method"] == method]
                noise = [np.sqrt(max(exposures)/r["exposure"]) for r in data]
                ax.errorbar(noise, [r[metric] for r in data], yerr=[r[metric+"_sd"] for r in data],
                            color=COLORS[method], label=method, marker="o", markersize=3, capsize=2)
            ax.set_xscale("log", base=2)
            ax.set_title(scene.replace("_", " ")+"\n"+title)
            ax.set_xlabel("Relative shot noise (shorter exposure)")
            ax.grid(alpha=.2)
            if metric != "contrast_ratio":
                ax.set_yscale("log")
                ax.axhline(THRESHOLDS[metric], color="black", ls=":", lw=1)
            else:
                ax.axhspan(.7, 1.3, color="#dddddd", alpha=.4)
                ax.axhline(1, color="black", ls=":", lw=1)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Independent evaluation: means and SD across noise seeds; dotted lines = declared targets", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(output / "noise_robustness.png", dpi=170)
    plt.close(fig)
    for scene_name, scene in scenes.items():
        # Same feature pixel and colour limits at every exposure.
        points = np.argwhere(scene["feature"])
        iy, ix = points[len(points)//2]
        q = normalise(scene["rate"])
        truth_centre, _ = spectral_features(q, scene["frequency"])
        fig, axes = plt.subplots(len(exposures), 4, figsize=(15, 2.8*len(exposures)), squeeze=False)
        for row, exposure in enumerate(exposures):
            ax = axes[row, 0]
            ax.plot(scene["frequency"], q[iy, ix], color="black", label="truth", lw=2)
            for method in ("raw", "spatial_mean", "poisson_nlpca", "poisson_spectral"):
                if (scene_name, exposure, method) not in examples:
                    continue
                p = normalise(examples[(scene_name, exposure, method)])
                ax.plot(scene["frequency"], p[iy, ix], color=COLORS[method], label=method, alpha=.8, lw=1)
            ax.set_xlim(4.7, 6.6)
            ax.set_ylim(0, q[iy, ix].max()*1.65)
            ax.set_title(f"{exposure:g} s; fixed feature pixel ({iy}, {ix})")
            ax.set_xlabel("Frequency (GHz)")
            if row == 0:
                ax.legend(fontsize=7)
            for col, method in enumerate(("truth", "spatial_mean", "poisson_nlpca"), 1):
                centre = truth_centre if method == "truth" else spectral_features(
                    normalise(examples[(scene_name, exposure, method)]), scene["frequency"])[0]
                im = axes[row, col].imshow(centre*1000, vmin=truth_centre.min()*1000,
                                            vmax=truth_centre.max()*1000, cmap="viridis")
                axes[row, col].set_title(method+" centroid (MHz)")
                axes[row, col].plot(ix, iy, "r+", ms=5)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
        fig.suptitle(scene_name.replace("_", " ")+": shared colour scale; centroid is not fitted BFS")
        fig.subplots_adjust(top=.92, bottom=.05, left=.045, right=.88, hspace=.5, wspace=.32)
        # Dedicated colourbar axis stays outside the maps after layout changes.
        cax = fig.add_axes((.91, .27, .012, .46))
        fig.colorbar(im, cax=cax)
        fig.savefig(output / f"{scene_name}_reconstructions.png", dpi=150)
        plt.close(fig)


def audit_noise_conditions(output, scenes, exposures):
    """Characterise photon budget and spectral-variation SNR in a stated convention."""
    rows = []
    for name, scene in scenes.items():
        rate = scene["rate"]
        q = normalise(rate)[scene["inside"]]
        signal = float(np.mean(np.sum((q-q.mean(axis=0))**2, axis=1)))
        totals = rate.sum(axis=-1)[scene["inside"]]
        for exposure in exposures:
            noise = float(np.mean((1-np.sum(q*q, axis=1))/(totals*exposure)))
            rows.append(dict(scene=name, exposure=exposure,
                             relative_shot_noise=float(np.sqrt(max(exposures)/exposure)),
                             median_cell_photons=float(np.median(totals*exposure)),
                             approximate_shape_variation_to_noise_rms=float(np.sqrt(signal/noise))))
    write_csv(output / "noise_conditions.csv", rows)


def repeatability_plot(output, summary, exposures):
    with open(output / "repeat_stability.csv", newline="", encoding="utf8") as f:
        stability = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for name in dict.fromkeys(r["scene"] for r in summary):
        group = [r for r in summary if r["scene"] == name and r["method"] == "poisson_nlpca"]
        noise = [np.sqrt(max(exposures)/r["exposure"]) for r in group]
        axes[0].plot(noise, [r["high_exposure_shape_delta"] for r in group], "o-", label=name)
        group2 = [r for r in stability if r["scene"] == name and r["method"] == "poisson_nlpca"]
        axes[1].plot([np.sqrt(max(exposures)/float(r["exposure"])) for r in group2],
                     [float(r["pairwise_shape_delta_mean"]) for r in group2], "o-", label=name)
    for ax, title in zip(axes, ("Change from high-exposure reconstruction", "Differences across independent noise seeds")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Relative shot noise")
        ax.set_ylabel("Relative spectral-shape difference (cell region)")
        ax.set_title(title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].axhline(THRESHOLDS["high_exposure_shape_delta"], color="black", ls=":")
    fig.suptitle("Poisson NLPCA repeatability: similarity is checked alongside accuracy and contrast")
    fig.tight_layout()
    fig.savefig(output / "repeatability.png", dpi=170)
    plt.close(fig)


def report(output, summary, args):
    lines = ["# Spatial-spectral reconstruction benchmark", "",
             "Estimates the latent measured spectrum, including medium. Does not establish material purity.", "",
             f"Grid: {args.size} x {args.size}; bins: {args.bins}; independent evaluation seeds: {args.repeats}.",
             f"Exposures (s): {args.exposures}. Photon rate and scene stay fixed; relative noise scales as 1/sqrt(exposure).", "",
             "## Method selection and evaluation", "",
             "An independent calibration acquisition is split 80/20 by binomial photon thinning. Each method family selects its configuration by validation Poisson deviance, then refits on each fresh evaluation acquisition. All clustering uses only its fit input. The frozen Poisson control uses the configuration selected at the highest exposure, but relearns its dictionary from each new input. Evaluation truth and evaluation photons never select configurations.", "",
             "Mean/SD are across independent acquisition noise realizations, not independent biological samples. Criteria below are exploratory engineering targets fixed before the main run; passing a mean is not an equivalence test or a confidence guarantee.", "",
             "## Declared acceptance criteria", ""]
    lines.extend(f"- {key}: {value}" for key, value in THRESHOLDS.items())
    lines += ["", "Shape error is relative L2 error of unit-area spectra inside the cell/gel. Boundary and feature masks come from simulation truth and are used only for scoring. Centroid and IQR are finite-window spectral summaries, not BFS and FWHM. Contrast is the signed feature-minus-control centroid difference divided by truth. The baseline is raw spectra, not a fitted curve.", "",
              "## Results by scene and exposure", "",
              "Best predictive method is selected here descriptively using average independent test excess deviance. It is not used to tune another model.", "",
              "| Scene | Exposure (s) | Best predictive method | Cell shape error | Centroid RMSE (MHz) | Contrast retained | Passing repeats |",
              "|---|---:|---|---:|---:|---:|---|"]
    for scene in args.scenes:
        for exposure in args.exposures:
            group = [r for r in summary if r["scene"] == scene and r["exposure"] == exposure]
            best = min(group, key=lambda r:r["test_excess_deviance_per_photon"])
            lines.append(f"| {scene} | {exposure:g} | {best['method']} | {best['shape_nrmse']:.3f} | {best['centroid_rmse_mhz']:.1f} | {best['contrast_ratio']:.2f} | {best['passing_repeats']}/{best['repeats']} |")
    lines += ["", "## Best method after accuracy and repeatability guards", "",
              "Among methods passing every criterion in every evaluation seed, choose the lowest independent test excess deviance. The frozen high-exposure control is excluded from this recommendation. These are descriptive choices for these simulations, not statistically established universal winners.", "",
              "| Scene | Exposure (s) | Best acceptable tested method |", "|---|---:|---|"]
    for scene in args.scenes:
        for exposure in args.exposures:
            eligible = [r for r in summary if r["scene"] == scene and r["exposure"] == exposure
                        and r["all_repeats_pass"] and r["method"] != "poisson_frozen"]
            best = min(eligible, key=lambda r:r["test_excess_deviance_per_photon"])["method"] if eligible else "None passes all guards"
            lines.append(f"| {scene} | {exposure:g} | {best} |")
    lines += ["", "## Poisson NLPCA acceptance range", ""]
    for scene in args.scenes:
        group = [r for r in summary if r["scene"] == scene and r["method"] == "poisson_nlpca"]
        passed = [r["exposure"] for r in group if r["all_repeats_pass"]]
        lines.append(f"- {scene}: tested exposures passing every repeat {passed or 'none'}.")
        for r in group:
            if not r["passes_mean_criteria"]:
                lines.append(f"  - {r['exposure']:g} s fails: {r['failed_criteria']}.")
    lines += ["", "## Residual interpretation", "",
              "shot_noise_error = mean((estimated counts - true expected counts)^2 / true expected counts). The independent residual mean square has expectation 1 + shot_noise_error, rather than exactly one. Residuals use simulation truth only to audit calibration. In-sample reduced chi-square is not used for selection. Independent excess deviance subtracts the true-model deviance from the same test counts; negative estimates are possible from finite test noise.", "",
              "## Paper correspondence and deliberate changes", "",
              "Salmon et al., Poisson noise reduction with non-local PCA, arXiv:1206.0338v4, DOI 10.1007/s10851-013-0435-6. Implements non-sparse Algorithm 1, exp(UV) likelihood (Eq. 10), full block Newton updates (Eqs. 13-14), Poisson clustering, and uniform overlap reprojection. This is an adaptation, not a reproduction of the paper's benchmark numbers.", "",
              "Uses full-spectrum spatial patches (1x1xB or 3x3xB), 16 summed spectral bands for clustering, deterministic log-pilot SVD initialization, backtracking and relative Hessian damping. poisson_spectral restricts patches to 1x1xB and gives every pixel its own coefficients while sharing the learned spectral dictionary. This separately reported variant was added after observing attenuation of thin features in the initial experiment; its results are exploratory. No arbitrary rank is interpreted as material count. Anscombe baseline uses an asymptotic unbiased inverse, not the exact tabulated inverse. The sparse NLSPCA and spatial binning variants are not implemented. The basis changes with each acquisition; only the selection rule is reproducible.", "",
              "## Limitations", "",
              "Simulated instrument and scenes are plausible stress tests, not calibrated replicas of this microscope. Detector is independent Poisson with no read noise, gain fluctuations or resampling. The small-feature scene intentionally challenges spatial pooling. At sufficiently low counts, similar-looking outputs can be biased: feature contrast and truth errors prevent that from counting as success. Real scans require detector/calibration checks and independent repeats. Medium-free recovery requires separate identifiable constraints; pure_material_rate is stored for future work and is never a denoising target here.", "",
              "## Files", "",
              "- metrics.csv: each independent evaluation realization.",
              "- summary.csv: means, SD, acceptance flags and failure reasons.",
              "- tuning.csv and selected_configs.json: complete truth-blind model selection.",
              "- repeat_stability.csv: pairwise curve differences across independent realizations.",
              "- data/: exact truth and input count cubes; examples/: seed-zero reconstructed rate cubes.",
              "- noise_robustness.png and scene reconstruction figures: common scales across exposure.",
              "- repeatability.png: independent-seed and cross-exposure spectral differences.",
              "- noise_conditions.csv: photon budgets and approximate RMS spectral-variation/noise ratio. This ratio uses centred unit-area spectra inside the cell/gel, and the multinomial noise trace evaluated at expected total counts. It is a stated convention, not automatically the same SNR as a previous analysis.",
              "- manifest.json: settings, source hashes, versions, seed schedule and runtime."]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/reconstruction_benchmark"))
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--bins", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--exposures", type=float, nargs="+", default=[16, 4, 1, .25, .0625])
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--report-only", action="store_true", help="Regenerate figures/report from a finished run")
    args = parser.parse_args()
    if args.report_only:
        manifest = json.loads((args.output / "manifest.json").read_text())
        saved = argparse.Namespace(**manifest["arguments"])
        saved.output = args.output
        with open(args.output / "metrics.csv", newline="", encoding="utf8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for key in row:
                if key not in {"scene", "method"}:
                    row[key] = float(row[key])
        summary = summarize(rows)
        write_csv(args.output / "summary.csv", summary)
        scenes, examples = {}, {}
        for name in saved.scenes:
            with np.load(args.output / "data" / f"{name}_truth.npz") as data:
                scenes[name] = dict(data)
            for exposure in saved.exposures:
                with np.load(args.output / "examples" / f"{name}_{exposure:g}s.npz") as data:
                    examples.update({(name, exposure, key):data[key] for key in data.files})
        plots(args.output, summary, scenes, examples, saved.exposures)
        audit_noise_conditions(args.output, scenes, saved.exposures)
        repeatability_plot(args.output, summary, saved.exposures)
        report(args.output, summary, saved)
        print("Refreshed figures and report without rerunning reconstruction")
        return
    if args.repeats < 2 or min(args.exposures) <= 0 or len(set(args.exposures)) != len(args.exposures):
        parser.error("Need >=2 repeats and distinct positive exposures")
    args.exposures = sorted(args.exposures, reverse=True)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(exist_ok=True)
    (output / "examples").mkdir(exist_ok=True)
    start = time.perf_counter()
    configs = candidate_configs(args.iterations)
    rows, tuning, stability, selected, scenes, examples = [], [], [], {}, {}, {}
    with threadpool_limits(limits=1):
        for scene_id, name in enumerate(args.scenes):
            print(f"Generating {name}", flush=True)
            scene = make_scene(name, args.size, args.bins)
            scenes[name] = scene
            np.savez_compressed(output / "data" / f"{name}_truth.npz",
                                **{k:v for k,v in scene.items() if k != "instrument"})
            high_references = {}
            frozen = None
            for level, exposure in enumerate(args.exposures):
                base_seed = args.seed + scene_id*100000 + level*1000
                print(f"  {exposure:g} s: selecting configurations", flush=True)
                calibration = sample_counts(scene["rate"], exposure, base_seed)
                chosen, details = select_configs(calibration, configs, base_seed+1)
                np.savez_compressed(output / "data" / f"{name}_{exposure:g}s_calibration.npz",
                                    counts=calibration, frequency=scene["frequency"], exposure=exposure)
                tuning.extend(dict(scene=name, exposure=exposure, **r) for r in details)
                if frozen is None:
                    frozen = chosen["poisson_nlpca"]
                chosen["poisson_frozen"] = frozen
                selected[f"{name}/{exposure:g}"] = {k:v.to_dict() for k,v in chosen.items()}
                estimates = {method:[] for method in chosen}
                for repeat in range(args.repeats):
                    counts = sample_counts(scene["rate"], exposure, base_seed+10+repeat*2)
                    test = sample_counts(scene["rate"], exposure, base_seed+11+repeat*2)
                    np.savez_compressed(output / "data" / f"{name}_{exposure:g}s_seed{repeat}.npz",
                                        counts=counts, independent_test=test, frequency=scene["frequency"], exposure=exposure)
                    for method, c in chosen.items():
                        before = time.perf_counter()
                        estimate = reconstruct(counts, c)
                        elapsed = time.perf_counter()-before
                        reference = high_references.get((method, repeat))
                        values = metrics(estimate, scene, exposure, test, reference)
                        rows.append(dict(scene=name, exposure=exposure, repeat=repeat, method=method,
                                         seconds=elapsed, **values))
                        rate = estimate / exposure
                        estimates[method].append(rate)
                        if level == 0:
                            high_references[method, repeat] = rate
                        if repeat == 0:
                            examples[name, exposure, method] = rate
                    print(f"    repeat {repeat+1}/{args.repeats} complete", flush=True)
                for method, curves in estimates.items():
                    differences = [relative_error(normalise(curves[a]), normalise(curves[b]), scene["inside"])
                                   for a in range(len(curves)) for b in range(a)]
                    stability.append(dict(scene=name, exposure=exposure, method=method,
                                          pairwise_shape_delta_mean=float(np.mean(differences)),
                                          pairwise_shape_delta_max=float(np.max(differences))))
                np.savez_compressed(output / "examples" / f"{name}_{exposure:g}s.npz",
                                    **{method:examples[name, exposure, method] for method in chosen})
                write_csv(output / "metrics.csv", rows)
                write_csv(output / "tuning.csv", tuning)
                (output / "selected_configs.json").write_text(json.dumps(selected, indent=2))
    summary = summarize(rows)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "repeat_stability.csv", stability)
    plots(output, summary, scenes, examples, args.exposures)
    audit_noise_conditions(output, scenes, args.exposures)
    repeatability_plot(output, summary, args.exposures)
    report(output, summary, args)
    sources = [Path("brillouin_denoising.py"), Path(__file__), Path("simulator/generate_reconstruction_scenes.py")]
    manifest = dict(arguments={**vars(args), "output":str(output)}, thresholds=THRESHOLDS,
                    candidates=[c.to_dict() for c in configs], python=platform.python_version(),
                    numpy=np.__version__, runtime_seconds=time.perf_counter()-start,
                    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    instruments={name:s["instrument"] for name,s in scenes.items()},
                    seed_schedule="base=seed+scene_index*100000+exposure_index*1000; calibration=base; split=base+1; fit=base+10+2*repeat; test=base+11+2*repeat")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(f"Completed in {manifest['runtime_seconds']:.1f}s: {output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
