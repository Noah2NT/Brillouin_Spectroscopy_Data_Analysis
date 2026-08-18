"""Generating many scans in one go.

The point of a batch is to sweep the noise level and repeat each setting
several times, so an algorithm's accuracy can be plotted against exposure with
error bars rather than judged from a single lucky realisation.

Every run writes three things: the CSV, a ground-truth sidecar, and a row in a
shared ``manifest.csv`` that records every setting and the seed.  The manifest
is what a noise-tolerance study should actually be driven from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

import numpy as np

from . import scan as scan_mod
from . import writer


@dataclass
class BatchSpec:
    """A sweep to generate.

    Attributes
    ----------
    exposures_s : list of float
        The noise axis.  Peak area scales with exposure, so shot-noise-limited
        shift precision improves as its square root.
    repeats : int
        Independent realisations per exposure, each with its own seed.
    step_sizes_um : list of float or None
        Optional second axis.  ``None`` keeps the base setting's step size.
    output_dir : str
    drawing_name : str
        Used to build filenames and recorded in the manifest.
    seed_base : int or None
        Seeds run ``seed_base``, ``seed_base + 1``, ...  ``None`` -- the
        default -- draws a fresh base per batch, so every batch is different
        noise.  The seed actually used for each file lands in the manifest,
        so any individual scan can be regenerated later.
    line_terminator : str
    """

    exposures_s: list = field(default_factory=lambda: [1.0, 2.0, 5.0, 10.0, 20.0])
    repeats: int = 3
    step_sizes_um: list | None = None
    output_dir: str = "outputs/simulated"
    drawing_name: str = "sample"
    seed_base: int | None = None
    line_terminator: str = "\n"

    def jobs(self, base_settings):
        """Expand the sweep into concrete :class:`ScanSettings`.

        Returns
        -------
        list of (ScanSettings, str)
            Each setting paired with the filename stem to write it under.
        """
        steps = self.step_sizes_um or [base_settings.step_um]
        out = []
        seed = (int(self.seed_base) if self.seed_base is not None
                else int(np.random.SeedSequence().generate_state(1)[0]) // 2)
        for step in steps:
            for exposure in self.exposures_s:
                for rep in range(int(self.repeats)):
                    settings = replace(
                        base_settings,
                        step_um=float(step),
                        exposure_s=float(exposure),
                        seed=seed,
                    )
                    stem = writer.default_stem(self.drawing_name, settings)
                    if self.repeats > 1:
                        stem = f"{stem}_rep{rep + 1}"
                    out.append((settings, stem))
                    seed += 1
        return out

    @property
    def n_jobs(self):
        steps = len(self.step_sizes_um) if self.step_sizes_um else 1
        return steps * len(self.exposures_s) * int(self.repeats)


def run_batch(drawing, library, instrument, spec, base_settings,
              progress=None, should_cancel=None):
    """Generate every scan in a sweep.

    Parameters
    ----------
    drawing : CellDrawing
    library : MaterialLibrary
    instrument : Instrument
    spec : BatchSpec
    base_settings : ScanSettings
        Everything the sweep does not vary.
    progress : callable, optional
        ``progress(job_index, n_jobs, fraction_within_job, message)``.
    should_cancel : callable, optional
        Polled between and during jobs; return true to stop early.

    Returns
    -------
    dict
        ``{"files": [...], "manifest": path, "cancelled": bool}``
    """
    jobs = spec.jobs(base_settings)
    manifest_path = os.path.join(spec.output_dir, "manifest.csv")
    written = []

    def cancelled():
        return should_cancel is not None and should_cancel()

    for job_index, (settings, stem) in enumerate(jobs):
        if cancelled():
            return {"files": written, "manifest": manifest_path, "cancelled": True}

        def inner_progress(done, total, message, _i=job_index):
            if progress is not None:
                progress(_i, len(jobs), done / max(total, 1), message)
            return not cancelled()

        if progress is not None:
            progress(job_index, len(jobs), 0.0, f"Simulating {stem}")

        result = scan_mod.simulate_scan(drawing, library, instrument, settings,
                                        progress=inner_progress)
        if result is None:
            return {"files": written, "manifest": manifest_path, "cancelled": True}

        csv_path = os.path.join(spec.output_dir, stem + ".csv")
        truth_path = os.path.join(spec.output_dir, stem + "_truth.npz")

        writer.write_scan_csv(csv_path, result, instrument,
                              line_terminator=spec.line_terminator)
        writer.write_truth_sidecar(truth_path, result, instrument,
                                   drawing_name=spec.drawing_name)
        writer.append_manifest(
            manifest_path,
            writer.manifest_row(csv_path, truth_path, result,
                                drawing_name=spec.drawing_name))
        written.append(csv_path)

    if progress is not None:
        progress(len(jobs), len(jobs), 1.0, "Done")
    return {"files": written, "manifest": manifest_path, "cancelled": False}


def estimate_runtime(spec, base_settings, seconds_per_1000_pixels=2.5):
    """Rough wall-clock estimate, so the UI can warn before a long run."""
    n_pixels = base_settings.n_pixels
    return spec.n_jobs * n_pixels / 1000.0 * seconds_per_1000_pixels
