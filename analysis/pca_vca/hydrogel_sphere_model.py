"""Global forward model: hydrogel blob as a spherical cap on the dish.

Every pixel is predicted from one shared geometry + three radial profiles,
instead of fitting each pixel independently.

    r      = |point - centre| / R          (0 = sphere centre, 1 = gel surface)
    inside : INT(r) * L(nu; BFS(r), LW(r))
    outside: A_water * background endmember
    pixel  = PSF-weighted sum of points + constant baseline

The scan plane height is unknown, so dz = (scan plane - sphere centre) is
fitted. R and dz are only partly separable from one XY slice (see README).

Run:  python analysis/pca_vca/hydrogel_sphere_model.py [--scan-height MM]
Test: python analysis/pca_vca/hydrogel_sphere_model.py --selftest
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import background_endmember as bge

ROOT = Path(__file__).resolve().parents[2]
STEM = 'Hydrogl1_XYscan_20000_20000_step500um_10s (1)'
OUT = ROOT / 'outputs/hydrogel_sphere_model'
STEP_MM = 0.5                      # scan step (500 um)
FWHM = 2.3548
SIGMA_XY = 0.002 / FWHM            # 20X / NA 0.42: ~2 um lateral (paper)
SIGMA_Z = 0.100 / FWHM             # ~100 um axial (paper)
N_AXIAL = 9                        # Gauss-Hermite points along the optical axis

NAMES = ['x0', 'y0', 'R', 'dz_frac', 'blur',
         'bfs_centre', 'bfs_edge', 'k_bfs',
         'lw_centre', 'lw_edge', 'k_lw',
         'int_centre', 'int_edge', 'k_int',
         'A_water', 'baseline']


def profile(r, centre, edge, k):
    """Value at r in [0, 1]: centre at r=0, edge at r=1, shape set by k (k=2 -> 1-r^2)."""
    return edge + (centre - edge) * (1 - r**k)


def lorentzian(nu, bfs, lw):
    """Unit-height Lorentzian; lw is FWHM."""
    half = lw / 2
    return half**2 / ((nu - bfs)**2 + half**2)


def remodel(p, px, py, nu, bg, scan_height=None):
    """Predicted spectra (pixels x channels) for parameter vector p."""
    q = dict(zip(NAMES, p))
    R, dz = q['R'], q['dz_frac'] * q['R']
    sig_xy = np.hypot(SIGMA_XY, q['blur'])
    sig_z = np.hypot(SIGMA_Z, q['blur'])
    t, w = np.polynomial.hermite.hermgauss(N_AXIAL)
    z = np.sqrt(2) * sig_z * t                      # axial offsets from scan plane
    w = w / np.sqrt(np.pi)
    rho2 = ((px - q['x0'])**2 + (py - q['y0'])**2)[:, None]
    d = np.sqrt(rho2 + (z[None, :] + dz)**2)        # (pixels, axial) distance to centre
    # ponytail: lateral PSF handled as a soft step across the surface (half-space
    # approximation, fine while sig_xy << R); full 3D quadrature if blur ~ R.
    inside = ndtr((R - d) / sig_xy)
    if scan_height is not None:                     # points below the dish are glass
        w = np.where(scan_height + z >= 0, w, 0.0)
    r = np.clip(d / R, 0, 1)
    bfs = profile(r, q['bfs_centre'], q['bfs_edge'], q['k_bfs'])
    lw = profile(r, q['lw_centre'], q['lw_edge'], q['k_lw'])
    amp = profile(r, q['int_centre'], q['int_edge'], q['k_int'])
    gel = amp[..., None] * lorentzian(nu, bfs[..., None], lw[..., None])
    water = q['A_water'] * bg
    spectra = inside[..., None] * gel + (1 - inside[..., None]) * water
    return np.einsum('a,pac->pc', w, spectra) + q['baseline']


def loss(p, y, *args):
    """Residual vector; least_squares minimises its sum of squares (= MSE x N)."""
    return (remodel(p, *args) - y).ravel()


def bounds(nu):
    lo = [-np.inf, -np.inf, 0.1, 0.0, 0.0,
          nu.min(), nu.min(), 0.3, 0.02, 0.02, 0.3, 0, 0, 0.3, 0, -np.inf]
    hi = [np.inf, np.inf, 50, 0.98, 2.0,
          nu.max(), nu.max(), 8, 3, 3, 8, np.inf, np.inf, 8, np.inf, np.inf]
    return lo, hi


def fit(y, p0, px, py, nu, bg, scan_height=None):
    lo, hi = bounds(nu)
    p0 = np.clip(p0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
    res = least_squares(loss, p0, args=(y, px, py, nu, bg, scan_height), bounds=(lo, hi),
                        x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=3000)
    dof = max(res.fun.size - len(p0), 1)
    s2 = np.sum(res.fun**2) / dof
    err = np.sqrt(np.abs(np.diag(np.linalg.pinv(res.jac.T @ res.jac))) * s2)
    return res, err


def initial_guess(y, px, py, nu, bg, mask, results):
    """Starting point from the VCA mask and the existing per-pixel mixture fits."""
    area = mask.sum() * STEP_MM**2
    R = np.sqrt(area / np.pi) / np.sqrt(1 - 0.5**2)   # dz_frac starts at 0.5
    base = np.median(np.percentile(y, 5, axis=1))
    peak = y.max(axis=1) - base
    c, f = results.mixture_center_ghz.to_numpy(), results.mixture_fwhm_ghz.to_numpy()
    return np.array([px[mask].mean(), py[mask].mean(), R, 0.5, 0.05,
                     np.percentile(c[mask], 90), np.percentile(c[mask], 10), 2,
                     np.percentile(f[mask], 50), np.percentile(f[mask], 50), 2,
                     np.percentile(peak[mask], 90), np.percentile(peak[mask], 30), 2,
                     np.median(peak[~mask]) / bg.max(), base])


def cap_geometry(p, scan_height):
    """Cap height and contact angle, only defined when the scan height is known."""
    q = dict(zip(NAMES, p))
    zc = scan_height - q['dz_frac'] * q['R']        # sphere centre height above dish
    zc = float(np.clip(zc, -q['R'], q['R']))
    return dict(centre_height_mm=zc, cap_height_mm=q['R'] + zc,
                footprint_radius_mm=float(np.sqrt(q['R']**2 - zc**2)),
                contact_angle_deg=float(np.degrees(np.arccos(-zc / q['R']))))


def selftest():
    rng = np.random.default_rng(0)
    nu = np.linspace(1.5, 8, 112)
    bg = lorentzian(nu, 5.55, 0.55)
    gx, gy = np.meshgrid(np.arange(30) * STEP_MM, np.arange(30) * STEP_MM)
    px, py = gx.ravel(), gy.ravel()
    true = np.array([7.3, 7.1, 6.0, 0.4, 0.05, 6.4, 5.7, 2, 0.9, 0.6, 2,
                     2500, 1200, 2, 1500, 20])
    y = remodel(true, px, py, nu, bg) + rng.normal(0, 25, (len(px), len(nu)))
    p0 = true * np.array([1.06, .94, 1.2, 1.25, 1, 1.03, .97, 1.2, 1.2, .8, .8,
                          .8, 1.2, 1.2, .9, 1])
    res, _ = fit(y, p0, px, py, nu, bg)
    got = dict(zip(NAMES, res.x))
    print({k: round(v, 3) for k, v in got.items()})
    assert abs(got['x0'] - 7.3) < 0.1 and abs(got['y0'] - 7.1) < 0.1
    assert abs(got['R'] - 6.0) / 6.0 < 0.15, got['R']
    assert abs(got['bfs_centre'] - 6.4) < 0.05 and abs(got['bfs_edge'] - 5.7) < 0.05
    print('selftest passed')


def plots(y, pred, p, px, py, nu, data, results):
    shape = data['shape']
    yi, xi = data['yi'], data['xi']
    q = dict(zip(NAMES, p))
    rmse = np.sqrt(np.mean((y - pred)**2, axis=1))
    r_slice = np.hypot(np.hypot(px - q['x0'], py - q['y0']), q['dz_frac'] * q['R']) / q['R']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')
    vmax = y.max(axis=1).max()
    for ax, v, title, cmap, lim in zip(axes.flat,
            [y.max(axis=1), pred.max(axis=1), rmse, r_slice],
            ['Measured peak counts', 'Model peak counts', 'Residual RMSE (counts)',
             'Normalised radius r in scan plane'],
            ['viridis', 'viridis', 'magma', 'cividis'],
            [(0, vmax), (0, vmax), (None, None), (0, 1.5)]):
        im = ax.imshow(bge.scatter_map(v, yi, xi, shape), origin='lower', cmap=cmap,
                       vmin=lim[0], vmax=lim[1])
        ax.set(title=title, xlabel='X pixel', ylabel='Y pixel')
        fig.colorbar(im, ax=ax, shrink=.85)
    axes.flat[3].contour(bge.scatter_map(r_slice, yi, xi, shape), levels=[1], colors='w')
    fig.suptitle('Hydrogel sphere-cap model | one global fit, %d pixels' % len(y))
    fig.savefig(OUT / 'maps.png', dpi=150); plt.close(fig)

    rr = np.linspace(0, 1, 200)
    inside = r_slice < 1
    ok = results.mixture_converged.to_numpy() & inside
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained')
    peak = y.max(axis=1) - q['baseline']
    for ax, key, scatter, label in zip(axes, ['bfs', 'lw', 'int'],
            [results.mixture_center_ghz, results.mixture_fwhm_ghz, peak],
            ['BFS (GHz)', 'Linewidth FWHM (GHz)', 'Peak intensity (counts)']):
        ax.scatter(r_slice[ok], np.asarray(scatter)[ok], s=6, alpha=.3, color='.4',
                   label='per-pixel fits (reference)')
        ax.plot(rr, profile(rr, q[key + '_centre'], q[key + '_edge'], q['k_' + key]),
                color='#0072b2', lw=2.5, label='fitted profile')
        ax.set(xlabel='r (0 = sphere centre, 1 = gel surface)', ylabel=label)
        ax.axvline(q['dz_frac'], color='#d55e00', ls=':', label='smallest r reached by scan')
        ax.legend(fontsize=8)
    fig.suptitle('Radial profiles. Grey: independent per-pixel fits (not the fit target)')
    fig.savefig(OUT / 'profiles.png', dpi=150); plt.close(fig)

    order = np.argsort(r_slice)
    rs = r_slice[order]
    picks = [('centre', order[0]),
             ('mid', order[min(np.searchsorted(rs, .5 * (1 + q['dz_frac'])), len(rs) - 1)]),
             ('edge', order[min(np.searchsorted(rs, .97), len(rs) - 1)]),
             ('water', order[-1])]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), layout='constrained')
    for ax, (name, i) in zip(axes, picks):
        ax.plot(nu, y[i], '.', ms=3, color='.5', label='measured')
        ax.plot(nu, pred[i], color='#0072b2', label='model')
        ax.set(title=f'{name}: pixel {i}, r={r_slice[i]:.2f}', xlabel='Brillouin shift (GHz)')
        ax.legend(fontsize=8)
    fig.savefig(OUT / 'example_spectra.png', dpi=150); plt.close(fig)
    return rmse


def main(scan_height=None):
    OUT.mkdir(parents=True, exist_ok=True)
    data = bge.load_cube(ROOT / 'data/csv' / f'{STEM}.csv')
    nu, y = data['shift_freq'], 2 * data['folded_raw']
    saved = pd.read_csv(ROOT / 'outputs/background_endmember' / f'{STEM}_background_endmember.csv')
    np.testing.assert_allclose(nu, saved.shift_ghz, atol=1e-10)
    bg = saved.endmember_norm.to_numpy(copy=True)
    bg /= bg.max()
    px, py = data['x_vals'][data['xi']] * STEP_MM, data['y_vals'][data['yi']] * STEP_MM
    vca = pd.read_csv(ROOT / 'outputs/hydrogel_vca2/pixel_abundances.csv')
    results = pd.read_csv(ROOT / 'outputs/hydrogel_background_comparison/pixel_results.csv')
    np.testing.assert_array_equal(vca.xi, data['xi'])
    np.testing.assert_array_equal(results.xi, data['xi'])
    mask = vca.background_pct.to_numpy() < 97

    p0 = initial_guess(y, px, py, nu, bg, mask, results)
    print('Fitting %d pixels x %d channels, %d parameters' % (*y.shape, len(p0)), flush=True)
    res, err = fit(y, p0, px, py, nu, bg, scan_height)
    pred = remodel(res.x, px, py, nu, bg, scan_height)
    rmse = plots(y, pred, res.x, px, py, nu, data, results)
    q = dict(zip(NAMES, res.x))
    out = dict(
        params={k: dict(value=float(v), err=float(e)) for k, v, e in zip(NAMES, res.x, err)},
        curvature_per_mm=dict(value=1 / q['R'], err=float(err[2] / q['R']**2)),
        dz_mm=q['dz_frac'] * q['R'],
        global_rmse=float(np.sqrt(np.mean((y - pred)**2))),
        median_pixel_rmse=float(np.median(rmse)),
        per_pixel_mixture_global_rmse=json.loads(
            (ROOT / 'outputs/hydrogel_background_comparison/summary.json').read_text())['global_mixture_rmse'],
        optimizer=dict(success=bool(res.success), nfev=int(res.nfev), message=res.message),
        units='x0,y0,R,blur,dz in mm; bfs,lw in GHz; int,A_water,baseline in counts',
    )
    if scan_height is not None:
        out['scan_height_mm'] = scan_height
        out['cap'] = cap_geometry(res.x, scan_height)
    (OUT / 'params.json').write_text(json.dumps(out, indent=2))
    (OUT / 'README.md').write_text(f"""# Hydrogel sphere-cap forward model

`python analysis/pca_vca/hydrogel_sphere_model.py [--scan-height MM]`

One global least-squares fit of every pixel and channel of `{STEM}`.
The blob is a spherical cap with centre (x0, y0) and radius of curvature R.
The scan plane sits dz = dz_frac*R from the sphere centre.
Inside the gel (r = distance/R <= 1), each point emits INT(r) * Lorentzian(BFS(r), LW(r)).
Each profile is `edge + (centre - edge) * (1 - r^k)`.
Outside the gel, points emit the scaled water background endmember.
Each pixel sums its points over the 20X PSF: 2 um lateral, 100 um axial (paper), plus one fitted `blur`.

Limits:
- R and dz are partly degenerate. One XY slice shows a disk of radius sqrt(R^2 - dz^2),
  so curvature is pinned only by the shape of the radial profiles. Check the error bars in
  params.json. An XZ line scan, or 2-3 XY scans at known focus steps, would fix R directly.
- Without --scan-height, the dish plane is invisible, so cap height and contact angle are unknown.
- The profiles are only observed for r >= dz_frac (orange line in profiles.png).
  Values closer to the centre are extrapolated.
- The grey scatter in profiles.png comes from independent per-pixel fits.
  It is a sanity reference only, not the fit target.
""", encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--scan-height', type=float, default=None,
                    help='focus height above the dish in mm, if known')
    a = ap.parse_args()
    selftest() if a.selftest else main(a.scan_height)
