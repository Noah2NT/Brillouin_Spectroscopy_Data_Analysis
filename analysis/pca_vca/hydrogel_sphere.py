"""Global sphere/PSF spectral inversion. See HYDROGEL_SPHERE.md for assumptions."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import ndtr, roots_hermitenorm, roots_legendre

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import brillouin_preprocessing as bp
from simulator.core.physics import lorentzian_unit_area

STEM = 'Hydrogl1_XYscan_20000_20000_step500um_10s (1)'
NAMES = ('x0', 'y0', 'z0', 'dx', 'dy', 'bfs_center', 'bfs_edge', 'bfs_power',
         'lw_center', 'lw_edge', 'lw_power', 'int_center', 'int_edge', 'int_power',
         'water_intensity', 'radius_scale', 'rim_width')


def concentration(rho):
    """Relative hydrogel concentration: rho=0 at centre, rho=1 at boundary."""
    return np.maximum(1.0 - np.asarray(rho)**2, 0.0)


def radial_curve(rho, center, edge, power):
    return center + (edge - center) * np.clip(rho, 0, 1)**power


def intensity(c, center, edge, power):
    return edge + (center - edge) * np.asarray(c)**power


def rim_fraction(rho, width):
    """Hydrogel weight: smoothstep to pure water over the outer width * radius."""
    if not np.isfinite(width) or not 0 <= width <= 1:
        raise ValueError('Rim width must be between zero and one radius.')
    rho = np.asarray(rho)
    if width == 0:
        return (rho < 1).astype(float)
    t = np.clip((1-rho)/width, 0, 1)
    return t*t*(3-2*t)


def scale_pixels(predicted, observed):
    """Exact nonnegative least-squares gain for each complete predicted spectrum."""
    predicted, observed = np.asarray(predicted, dtype=float), np.asarray(observed, dtype=float)
    if (predicted.ndim != 2 or observed.shape != predicted.shape
            or not np.isfinite(predicted).all() or not np.isfinite(observed).all()):
        raise ValueError('Predicted and observed spectra must be matching finite 2D arrays.')
    denominator = np.sum(predicted**2, axis=1)
    gain = np.maximum(np.divide(np.sum(predicted*observed, axis=1), denominator,
                                out=np.zeros_like(denominator), where=denominator > 0), 0)
    return predicted * gain[:, None], gain


class SphereModel:
    """PSF-weighted continuum of local lines; pure measured water outside.

    Coordinates and PSF widths are in reference-radius units. Spectra are counts/channel;
    intensity parameters are integrated counts over the fitted frequency window.
    LW is effective observed FWHM, not deconvolved intrinsic linewidth.
    """

    def __init__(self, frequency, background, pixel_xy, radius_um,
                 psf_fwhm_um=(2., 2., 100.), xy_order=5, z_order=48,
                 radial_nodes=129):
        self.nu = np.asarray(frequency, dtype=float)
        self.bg = np.asarray(background, dtype=float)
        self.xy = np.asarray(pixel_xy, dtype=float)
        widths = np.asarray(psf_fwhm_um, dtype=float)
        if (self.nu.ndim != 1 or len(self.nu) < 3 or
                not np.isfinite(self.nu).all() or np.any(np.diff(self.nu) <= 0)):
            raise ValueError('Frequency must be finite and strictly increasing.')
        if (self.bg.shape != self.nu.shape or not np.isfinite(self.bg).all()
                or np.any(self.bg < 0) or self.bg.sum() <= 0):
            raise ValueError('Background must be finite, nonnegative and nonzero.')
        if (self.xy.ndim != 2 or self.xy.shape[1] != 2 or len(self.xy) == 0
                or not np.isfinite(self.xy).all()):
            raise ValueError('pixel_xy must be a finite nonempty N x 2 array.')
        if (not np.isfinite(radius_um) or radius_um <= 0 or widths.shape != (3,)
                or not np.isfinite(widths).all() or np.any(widths <= 0)):
            raise ValueError('Radius and all three PSF widths must be positive.')
        if any(int(v) != v or v < 2 for v in (xy_order, z_order, radial_nodes)):
            raise ValueError('Quadrature orders and radial_nodes must be integers >= 2.')
        self.bg = self.bg / self.bg.sum()
        self.radius_um = float(radius_um)
        self.widths = widths
        self.sigma = widths / (np.sqrt(8 * np.log(2)) * radius_um)
        q, w = roots_hermitenorm(xy_order)
        a, b = np.meshgrid(q * self.sigma[0], q * self.sigma[1])
        self.offsets = np.column_stack((a.ravel(), b.ravel()))
        self.xy_weights = np.outer(w, w).ravel() / (2 * np.pi)
        self.z_nodes, self.z_weights = roots_legendre(z_order)
        self.rho = np.linspace(0., 1., radial_nodes)

    def remodel(self, p, return_components=False):
        """Return the total spectrum, or separate hydrogel and water contributions."""
        p = np.asarray(p, dtype=float)
        if p.shape != (len(NAMES),) or not np.isfinite(p).all():
            raise ValueError(f'Expected {len(NAMES)} finite parameters: {NAMES}')
        if (np.any(p[[3, 4, 7, 8, 9, 10, 13, 15]] <= 0)
                or np.any(p[[11, 12, 14]] < 0)):
            raise ValueError('Steps, widths and powers must be positive; intensities nonnegative.')
        if not 0 <= p[16] <= 1:
            raise ValueError('Rim width must be between zero and one radius.')
        x0, y0, z0, dx, dy = p[:5]
        radius = p[15]
        positions = self.xy * [dx, dy] - [x0, y0]
        d2 = np.sum((positions[:, None, :] + self.offsets)**2, axis=-1)
        prediction = np.broadcast_to(p[14] * self.bg, (len(self.xy), len(self.nu))).copy()
        active = np.any(d2 < radius**2, axis=1)
        if not np.any(active):
            return (np.zeros_like(prediction), prediction) if return_components else prediction
        d2 = d2[active]
        half_chord = np.sqrt(np.maximum(radius**2 - d2, 0))
        sz = self.sigma[2]
        # Integrate only the sphere's chord; exterior weight belongs to measured EM.
        lo = np.maximum(z0 - half_chord, -8 * sz)
        hi = np.minimum(z0 + half_chord, 8 * sz)
        span = np.maximum(hi - lo, 0)
        mass = np.where((d2 < radius**2) & (span > 0), ndtr(hi / sz) - ndtr(lo / sz), 0)
        z = lo[..., None] + span[..., None] * (self.z_nodes + 1) / 2
        weights = np.exp(-0.5 * (z / sz)**2) * self.z_weights
        weights *= np.divide(mass, weights.sum(axis=-1),
                             out=np.zeros_like(mass), where=weights.sum(axis=-1) > 0)[..., None]
        weights *= self.xy_weights[None, :, None]
        rho = np.sqrt(d2[..., None] + (z - z0)**2) / radius
        weights *= rim_fraction(rho, p[16])
        water = np.clip(1-weights.sum(axis=(1, 2)), 0, 1)
        # Evaluate intensity at integration points: interpolating an edge cusp
        # together with the spectral shape biases boundary brightness.
        weights *= intensity(concentration(rho), *p[11:14])
        # ponytail: linear radial interpolation; increase radial_nodes for narrow lines.
        coord = np.clip(rho, 0, 1) * (len(self.rho) - 1)
        left = np.minimum(coord.astype(int), len(self.rho) - 2)
        fraction = coord - left
        base = np.arange(len(d2))[:, None, None] * len(self.rho)
        size = len(d2) * len(self.rho)
        radial_weights = (np.bincount((base + left).ravel(),
                                     weights=(weights * (1 - fraction)).ravel(), minlength=size)
                          + np.bincount((base + left + 1).ravel(),
                                        weights=(weights * fraction).ravel(), minlength=size))
        radial_weights = radial_weights.reshape(len(d2), -1)
        bfs = radial_curve(self.rho, *p[5:8])
        lw = radial_curve(self.rho, *p[8:11])
        lines = np.array([lorentzian_unit_area(self.nu, f, width / 2)
                          for f, width in zip(bfs, lw)])
        lines /= lines.sum(axis=1, keepdims=True)
        gel = radial_weights @ lines
        prediction[active] = gel + water[:, None] * p[14] * self.bg
        if return_components:
            hydrogel = np.zeros_like(prediction)
            hydrogel[active] = gel
            return hydrogel, prediction-hydrogel
        return prediction

    def loss(self, p, observed):
        observed = self.validate_observed(observed)
        return float(np.mean((self.remodel(p) - observed)**2))

    def validate_observed(self, observed):
        observed = np.asarray(observed, dtype=float)
        if observed.shape != (len(self.xy), len(self.nu)) or not np.isfinite(observed).all():
            raise ValueError('Observed spectra must be finite N_pixels x N_channels.')
        return observed

    def fit(self, observed, initial, bounds, max_nfev=200, fit_names=NAMES, pixel_scaling=False):
        observed = self.validate_observed(observed)
        if pixel_scaling:
            # Anchor only the common brightness scale; relative radial intensity stays free.
            fit_names = [name for name in fit_names if name != 'water_intensity']
        if not fit_names or len(set(fit_names)) != len(fit_names):
            raise ValueError('Select distinct parameter names to fit.')
        indices = np.array([NAMES.index(name) for name in fit_names])
        initial = np.asarray(initial, dtype=float)
        self.remodel(initial)  # Validate fixed parameters too.
        lower, upper = np.broadcast_to(np.asarray(bounds, dtype=float), (2, len(NAMES)))
        scale = max(float(np.sqrt(np.mean(observed**2))), 1e-12)

        def unpack(values):
            full = initial.copy()
            full[indices] = values
            return full

        def residual(values):
            predicted = self.remodel(unpack(values))
            if pixel_scaling:
                predicted, _ = scale_pixels(predicted, observed)
            return ((predicted-observed) / scale).ravel()

        result = least_squares(residual,
                               initial[indices], bounds=(lower[indices], upper[indices]),
                               x_scale='jac', max_nfev=max_nfev)
        result.parameters = unpack(result.x)
        result.predicted = self.remodel(result.parameters)
        result.pixel_scale = np.ones(len(observed))
        if pixel_scaling:
            result.predicted, result.pixel_scale = scale_pixels(result.predicted, observed)
        result.mse = float(np.mean((result.predicted-observed)**2))
        result.fit_names = list(fit_names)
        return result


def load_scan(scan, endmember, low=4., high=8.):
    raw, frequency, table = bp.load_brillouin_csv(scan)
    folded, nu = bp.fold_brillouin_spectrum(raw, frequency)
    keep = (nu >= low) & (nu <= high)
    saved = pd.read_csv(endmember)
    indices = np.abs(saved.shift_ghz.to_numpy()[:, None] - nu[keep]).argmin(axis=0)
    if not np.allclose(saved.shift_ghz.to_numpy()[indices], nu[keep], atol=1e-9, rtol=0):
        raise ValueError('Saved background frequency grid does not match the folded scan.')
    x = np.searchsorted(np.sort(table.X.unique()), table.X)
    y = np.searchsorted(np.sort(table.Y.unique()), table.Y)
    xy = np.column_stack((x - x.max()/2, y - y.max()/2))
    return nu[keep], folded[:, keep], saved.endmember_norm.to_numpy()[indices], xy, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=Path, default=ROOT/'data/csv'/f'{STEM}.csv')
    parser.add_argument('--background', type=Path,
                        default=ROOT/'outputs/background_endmember'/f'{STEM}_background_endmember.csv')
    parser.add_argument('--radius-um', type=float, default=5000.,
                        help='Initial/reference radius; default 5 mm estimated from paper Fig. 5.')
    parser.add_argument('--step-um', type=float, nargs=2, default=(500., 500.), metavar=('X', 'Y'),
                        help='Fixed steps, default 500 um consistent with Fig. 5 and scan filename.')
    parser.add_argument('--fit-steps', action='store_true',
                        help='Instead fix radius and fit dx/dy starting at 0.1 reference radius.')
    parser.add_argument('--psf-fwhm-um', type=float, nargs=3, default=(2., 2., 100.))
    parser.add_argument('--z-order', type=int, default=48)
    parser.add_argument('--max-nfev', type=int, default=200)
    parser.add_argument('--initial', type=Path, help='Resume parameters from a previous fit.json.')
    parser.add_argument('--pixel-scaling', action='store_true',
                        help='Fit pixel gains and radial intensity; fix only water intensity as the scale reference.')
    parser.add_argument('--radius-start-factors', type=float, nargs='+', default=[1.],
                        help='Refit from each initial-radius multiplier and retain the lowest MSE.')
    parser.add_argument('--fit-rim', action='store_true',
                        help='Fit a smooth hydrogel-to-water rim; initial width is 20 percent of radius.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT/'outputs'/('hydrogel_sphere_tapered_rim' if args.fit_rim else
                                     'hydrogel_sphere_pixel_scaled' if args.pixel_scaling else 'hydrogel_sphere')
    nu, observed, bg, xy, table = load_scan(args.scan, args.background)
    model = SphereModel(nu, bg, xy, args.radius_um, args.psf_fwhm_um, z_order=args.z_order)
    water_area = float(np.median(observed.sum(axis=1)))
    water_bfs = float(nu[np.argmax(bg)])
    initial = np.array([0, 0, .1, .1, .1, water_bfs+.3, water_bfs, 2,
                        .9, .615, 2, water_area, water_area, 1, water_area, 1, 0])
    lower = [-5, -5, 0, .01, .01, nu.min(), nu.min(), .25, .03, .03, .25, 0, 0, .25, 0, .3, 0]
    upper = [5, 5, 1.5, 1, 1, nu.max(), nu.max(), 8, 3, 3, 8,
             20*water_area, 20*water_area, 8, 20*water_area, 3, 1]
    names = list(NAMES)
    if not args.fit_steps:
        initial[3:5] = np.asarray(args.step_um) / args.radius_um
        names = [name for name in names if name not in ('dx', 'dy')]
    else:
        names.remove('radius_scale')
    excess = np.maximum(nu[observed.argmax(axis=1)] - water_bfs, 0)
    if excess.sum() > 0:
        initial[:2] = np.average(xy, axis=0, weights=excess) * initial[3:5]
    if args.initial:
        saved = json.loads(args.initial.read_text(encoding='utf-8'))
        if saved['reference_radius_um'] != args.radius_um:
            raise ValueError('Initial fit must use the same reference radius.')
        initial = np.array([saved['parameters'].get(name, 0.) if name == 'rim_width'
                            else saved['parameters'][name] for name in NAMES])
        if not args.fit_steps:
            initial[3:5] = np.asarray(args.step_um) / args.radius_um
        else:
            initial[15] = 1.
    if args.fit_rim:
        if initial[16] == 0:
            initial[16] = .2
    else:
        names.remove('rim_width')
    initial_unscaled = model.remodel(initial)
    initial_mse = float(np.mean((initial_unscaled-observed)**2))
    scaled_only, _ = scale_pixels(initial_unscaled, observed)
    scaling_only_mse = float(np.mean((scaled_only-observed)**2))
    print(f'Fitting {len(observed)} spectra; initial unscaled MSE {initial_mse:.6g}', flush=True)
    if args.pixel_scaling:
        print(f'Gain-only RMSE {np.sqrt(scaling_only_mse):.6g}; refitting geometry/BFS/LW/intensity.', flush=True)
    if any(not np.isfinite(f) or f <= 0 for f in args.radius_start_factors):
        parser.error('Radius start factors must be finite and positive.')
    if args.fit_steps and args.radius_start_factors != [1.]:
        parser.error('Radius start factors require a fitted radius, not --fit-steps.')
    candidates, starts = [], []
    for factor in args.radius_start_factors:
        start = initial.copy()
        start[15] *= factor
        print(f'Starting radius factor {factor:g}', flush=True)
        candidate = model.fit(observed, start, (lower, upper), args.max_nfev, names, args.pixel_scaling)
        candidates.append(candidate)
        starts.append(dict(radius_factor=factor, mse=candidate.mse, success=bool(candidate.success), nfev=candidate.nfev))
        print(f'  RMSE {np.sqrt(candidate.mse):.6g}, converged={candidate.success}', flush=True)
    result = min(candidates, key=lambda item: item.mse)
    predicted = result.predicted
    finer = SphereModel(nu, bg, xy, args.radius_um, args.psf_fwhm_um,
                        xy_order=9, z_order=2*args.z_order, radial_nodes=257)
    numerical_rmse = float(np.sqrt(np.mean((finer.remodel(result.parameters)*result.pixel_scale[:, None]-predicted)**2)))
    args.output.mkdir(parents=True, exist_ok=True)
    ix, iy = (xy - xy.min(axis=0)).astype(int).T
    predicted_grid = np.full((iy.max()+1, ix.max()+1, len(nu)), np.nan)
    predicted_grid[iy, ix] = predicted
    np.savez_compressed(args.output/'spectra.npz', frequency_ghz=nu, observed=observed,
                        predicted=predicted, predicted_grid=predicted_grid,
                        pixel_scale=result.pixel_scale, initial_unscaled=initial_unscaled,
                        residual=observed-predicted, pixel_xy=xy,
                        background=model.bg)
    p = result.parameters
    pd.DataFrame(dict(rho=model.rho, concentration=concentration(model.rho),
                      bfs_ghz=radial_curve(model.rho, *p[5:8]),
                      linewidth_fwhm_ghz=radial_curve(model.rho, *p[8:11]),
                      hydrogel_weight=rim_fraction(model.rho, p[16]),
                      intensity_counts=intensity(concentration(model.rho), *p[11:14]))).to_csv(
                          args.output/'radial_profiles.csv', index=False)
    table[['X', 'Y']].assign(mse=np.mean((observed-predicted)**2, axis=1),
                             rmse=np.sqrt(np.mean((observed-predicted)**2, axis=1)),
                             initial_rmse=np.sqrt(np.mean((observed-initial_unscaled)**2, axis=1)),
                             pixel_scale=result.pixel_scale).to_csv(
        args.output/'pixels.csv', index=False)
    singular = np.linalg.svd(result.jac, compute_uv=False)
    summary = dict(parameters=dict(zip(NAMES, result.parameters.tolist())),
                   fit_names=result.fit_names, reference_radius_um=args.radius_um,
                   fitted_radius_um=float(result.parameters[15]*args.radius_um),
                   rim_width_um=float(result.parameters[16]*result.parameters[15]*args.radius_um),
                   center_um=(result.parameters[:3]*args.radius_um).tolist(),
                   step_um=(result.parameters[3:5]*args.radius_um).tolist(),
                   psf_fwhm_um=list(args.psf_fwhm_um),
                   initial_mse=initial_mse, final_mse=result.mse, success=bool(result.success),
                   initial_rmse=float(np.sqrt(initial_mse)), final_rmse=float(np.sqrt(result.mse)),
                   pixel_scaling=args.pixel_scaling,
                   scaling_only_rmse=float(np.sqrt(scaling_only_mse)) if args.pixel_scaling else None,
                   initial_file=str(args.initial) if args.initial else None,
                   starts=starts,
                   message=result.message, nfev=result.nfev, finer_quadrature_rmse=numerical_rmse,
                   jacobian_singular_values=singular.tolist(),
                   active_bounds=result.active_mask.tolist(),
                   scan=str(args.scan), background=str(args.background),
                   limitations=['Gaussian FWHM interpretation of reported spatial resolution is assumed.',
                                'Single-plane depth sign is unidentifiable; z0 represents absolute depth.',
                                'LW is effective observed FWHM; no intrinsic linewidth claim.',
                                'Local optimization, not proof of unique parameters or spherical geometry.'])
    if args.pixel_scaling:
        summary['limitations'].append('Pixel gains scale both hydrogel and water. Radial intensity is fitted relative to fixed water intensity; it can remain weakly constrained. RMSE is in-sample.')
    if result.parameters[16] > 0:
        summary['limitations'].append('The rim is an empirical local spectral mixture with measured water, not a measured surface height or concentration fraction.')
    (args.output/'fit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, values, title in zip(axes[:2], (observed.sum(1), np.sqrt(np.mean((observed-predicted)**2, axis=1))),
                                  ('Measured integrated counts', 'Reconstruction RMSE')):
        artist = ax.scatter(xy[:, 0], xy[:, 1], c=values, marker='s')
        fig.colorbar(artist, ax=ax)
        ax.set(xlabel='X pixel index (centred)', ylabel='Y pixel index (centred)', title=title, aspect='equal')
    i = int(np.argmax(np.mean((observed-predicted)**2, axis=1)))
    axes[2].plot(nu, observed[i], label='Measured')
    axes[2].plot(nu, predicted[i], label='Sphere prediction')
    axes[2].set(title=f'Largest residual: pixel {i}', xlabel='Frequency shift (GHz)', ylabel='Counts/channel')
    axes[2].legend()
    fig.savefig(args.output/'diagnostics.png', dpi=160)
    plt.close(fig)
    if args.pixel_scaling:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.8))
        fig.subplots_adjust(left=.06, right=.96, top=.76, bottom=.22, wspace=.35)
        before = np.sqrt(np.mean((observed-initial_unscaled)**2, axis=1))
        after = np.sqrt(np.mean((observed-predicted)**2, axis=1))
        limit = max(float(before.max()), float(after.max()))
        for ax, values, title, cmap, limits in zip(axes, (before, after, result.pixel_scale),
                ('Original sphere: pixel RMSE', 'With pixel scaling + refit: RMSE', 'Fitted intensity multiplier'),
                ('magma', 'magma', 'viridis'), ((0, limit), (0, limit), (None, None))):
            grid = np.full(predicted_grid.shape[:2], np.nan)
            grid[iy, ix] = values
            artist = ax.imshow(grid, origin='lower', interpolation='nearest', cmap=cmap,
                               vmin=limits[0], vmax=limits[1],
                               extent=[xy[:, 0].min()-.5, xy[:, 0].max()+.5, xy[:, 1].min()-.5, xy[:, 1].max()+.5])
            fig.colorbar(artist, ax=ax, shrink=.85)
            ax.set(title=title, xlabel='X pixel index (centred)', ylabel='Y pixel index (centred)')
        fig.suptitle('Reconstruction with one intensity scale per pixel', x=.06, ha='left', fontsize=19, weight='bold')
        fig.text(.06, .86, f'Overall RMSE: {np.sqrt(initial_mse):.2f} original  →  '
                 f'{np.sqrt(scaling_only_mse):.2f} scaling only  →  {np.sqrt(result.mse):.2f} after refit', fontsize=13)
        fig.text(.06, .06, 'RMSE units: counts/channel; both error maps share the same colour scale.\n'
                 'One multiplier scales each pixel spectrum. Radial intensity is fitted relative to fixed water intensity; errors are in-sample.',
                 fontsize=10, color='#475569', linespacing=1.6)
        fig.savefig(args.output/'rmse_comparison.png', dpi=180, facecolor='white')
        plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
