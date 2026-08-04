"""
Rank every line-scan SIGNAL_KIND by how strongly it separates the dose groups.

Runs both drugs in one go, independently of whichever config block happens to
be uncommented in carboplatin_line_scans.py - the dataset definitions below
are applied to that module for the duration of each pass.

Feature per scan
----------------
The mean of the signal over the cell interior (the shift tabletop's 'centre'
region). Passes from the SAME FILE are averaged first: the two sweeps in a
carboplatin CSV are one cell measured twice, not two cells, so treating them
as independent replicates would inflate n and understate every p-value.

Statistics
----------
eta^2   fraction of the total variance explained by dose - "is there any
        difference between the groups at all"
p_perm  permutation p-value for eta^2; dose labels are shuffled N_PERM times
        and the null distribution built empirically. Chosen over an F-test
        because the groups are tiny (2-8 files) and non-normal.
rho     Spearman correlation with numeric dose - "is the trend monotonic",
        which is what a dose-response argument actually needs.

Multiple comparisons
--------------------
Testing ~17 modes at 0.05 each would be expected to throw up roughly one
false positive per drug, so two corrections are reported:

  Bonferroni  threshold alpha/m; controls the chance of ANY false positive.
              Assumes independent tests, which these are emphatically not -
              bfs, bfs_0, bfs_grad, bfs_sobel and bfs_lap all derive from the
              same fitted BFS profile - so it is over-conservative here.
  BH (FDR)    Benjamini-Hochberg q-values; controls the expected PROPORTION
              of false positives among the hits. The fairer of the two for a
              correlated feature set like this one.

Run from the repo root (the data paths are relative).
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")          # analysis only, no figures are drawn
from scipy import stats

import carboplatin_line_scans as C


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

ALPHA = 0.05
N_PERM = 20000
SEED = 0

MODES = ["bfs", "lw", "bfs_0", "lw_0",
         "simple", "sobel", "lap",
         "fit_grad", "fit_sobel", "fit_lap",
         "bfs_grad", "bfs_sobel", "bfs_lap",
         "lw_grad", "lw_sobel", "lw_lap",
         "vca", "vca_grad", "vca_sobel", "vca_lap"]

DRUGS = [
    dict(drug="Carboplatin",
         base="1.5.26 20x Carboplatin",
         folders=["control", "100um", "200um", "400um"],
         doses=[0.0, 100.0, 200.0, 400.0],
         colors={"control": "green", "100um": "blue",
                 "200um": "orange", "400um": "red"},
         labels={"control": "Control", "100um": "100 uM",
                 "200um": "200 uM", "400um": "400 uM"}),
    dict(drug="Paclitaxel",
         base="Paclitaxel final line scans",
         folders=["Control", "1.52 nM", "3.125 nM", "6.25 nM", "12.5 nM"],
         doses=[0.0, 1.52, 3.125, 6.25, 12.5],
         colors={"Control": "green", "1.52 nM": "blue", "3.125 nM": "orange",
                 "6.25 nM": "red", "12.5 nM": "purple"},
         labels={}),
]


# ─────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────

def eta_squared(groups):
    """Between-group variance as a fraction of total variance."""
    allv = np.concatenate(groups)
    if allv.size < 3:
        return np.nan
    gm = allv.mean()
    ss_between = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    ss_total = ((allv - gm) ** 2).sum()
    return np.nan if ss_total <= 0 else ss_between / ss_total


def permutation_p(groups, rng, n_perm=N_PERM):
    """How often shuffled dose labels match or beat the observed eta^2.

    The (hits + 1) / (n_perm + 1) form keeps the p-value strictly positive:
    with n_perm draws the smallest reportable value is 1/(n_perm+1), so a
    printed 0.00005 means "at most this", never exactly zero.
    """
    obs = eta_squared(groups)
    if not np.isfinite(obs):
        return np.nan
    sizes = [len(g) for g in groups]
    allv = np.concatenate(groups)
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(allv)
        split, i = [], 0
        for s in sizes:
            split.append(perm[i:i + s])
            i += s
        if eta_squared(split) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def benjamini_hochberg(pvals):
    """BH FDR q-values, same order as the input.

    Sort ascending, scale each by m/rank, then sweep back from the largest
    enforcing monotonicity so a q-value can never exceed a larger one.
    """
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    if not ok.any():
        return q
    idx = np.argsort(p[ok])
    ps = p[ok][idx]
    m = ps.size
    scaled = ps * m / np.arange(1, m + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    qs = np.empty(m)
    qs[idx] = np.clip(scaled, 0, 1)
    q[ok] = qs
    return q


# ─────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────

def per_file_feature(results, mode):
    """Cell-interior mean of `mode`, one value per FILE.

    Gradient modes are already restricted to the centre region upstream, so
    they are used as-is; everything else is masked to region == 'center' here.
    """
    by_file = {}
    for r in results:
        sig = r.get(mode)
        if sig is None:
            continue
        v = np.asarray(sig, dtype=float)
        region = r["region"]
        if region is not None and mode not in C.GRADIENT_SIGNALS:
            v = v[region == "center"]
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        by_file.setdefault(r["scan"]["filename"], []).append(v.mean())
    return np.array([np.mean(vals) for vals in by_file.values()])


def analyse(cfg, rng):
    """Load one drug and score every mode. Returns a list of result dicts."""
    C.DRUG, C.BASE = cfg["drug"], cfg["base"]
    C.FOLDERS, C.FOLDER_COLORS = cfg["folders"], cfg["colors"]
    C.FOLDER_LABELS = cfg["labels"]
    all_results = {f: C.load_and_analyse(os.path.join(cfg["base"], f))
                   for f in cfg["folders"]}

    rows = []
    for mode in MODES:
        groups, dose_of, means = [], [], []
        for folder, dose in zip(cfg["folders"], cfg["doses"]):
            g = per_file_feature(all_results[folder], mode)
            means.append(g.mean() if g.size else np.nan)
            if g.size:
                groups.append(g)
                dose_of += [dose] * g.size
        if len(groups) < 2:
            continue
        flat = np.concatenate(groups)
        rho, p_rho = stats.spearmanr(dose_of, flat)
        rows.append(dict(mode=mode,
                         eta2=eta_squared(groups),
                         p_perm=permutation_p(groups, rng),
                         rho=rho, p_rho=p_rho,
                         means=means,
                         n=[len(g) for g in groups]))
    return rows, all_results


def report(cfg, rows, all_results):
    m = len(rows)
    bonf = ALPHA / m
    qvals = benjamini_hochberg([r["p_perm"] for r in rows])
    for r, q in zip(rows, qvals):
        r["q_bh"] = q
    rows = sorted(rows, key=lambda r: -(r["eta2"] if np.isfinite(r["eta2"])
                                        else -1))

    n_files = [len(per_file_feature(all_results[f], "bfs"))
               for f in cfg["folders"]]
    print(f"\n{'=' * 104}")
    print(f"{cfg['drug']}   files per dose = {n_files}   "
          f"({m} modes tested)")
    print(f"Bonferroni threshold = {ALPHA}/{m} = {bonf:.5f}    "
          f"BH controls FDR at {ALPHA}")
    print("=" * 104)
    print(f"{'mode':<11}{'eta^2':>7}{'p_perm':>9}{'q_BH':>9}"
          f"{'rho':>7}{'p_rho':>8}  {'sig':<6} dose means")
    print("-" * 104)
    for r in rows:
        p, q = r["p_perm"], r["q_bh"]
        flags = ""
        if np.isfinite(p) and p < bonf:
            flags += "B"
        if np.isfinite(q) and q < ALPHA:
            flags += "F"
        means = "  ".join(C._fmt_val(v) if np.isfinite(v) else "n/a"
                          for v in r["means"])
        print(f"{r['mode']:<11}{r['eta2']:>7.3f}{p:>9.5f}{q:>9.5f}"
              f"{r['rho']:>7.2f}{r['p_rho']:>8.3f}  {flags:<6} {means}")
    print("-" * 104)
    print("  sig: B = passes Bonferroni, F = passes Benjamini-Hochberg (FDR).")
    print("  BH is the fairer test here - the modes are strongly correlated,")
    print("  so Bonferroni's independence assumption makes it too strict.")
    n_b = sum(1 for r in rows if np.isfinite(r["p_perm"])
              and r["p_perm"] < bonf)
    n_f = sum(1 for r in rows if np.isfinite(r["q_bh"])
              and r["q_bh"] < ALPHA)
    print(f"  {n_b}/{m} pass Bonferroni, {n_f}/{m} pass BH.")
    if min(n_files) < 3:
        print(f"  [warn] smallest group has {min(n_files)} file(s): effect "
              f"sizes here are very unstable, read them with caution.")


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    for cfg in DRUGS:
        rows, all_results = analyse(cfg, rng)
        report(cfg, rows, all_results)
