"""Fit independent hydrogel/water spectral areas, holding the saved sphere fixed."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.pca_vca.hydrogel_sphere import SphereModel, NAMES


def fit_mixture(hydrogel, background, observed):
    """Two nonnegative areas per pixel; percentages refer to the fitted band."""
    hydrogel, background, observed = [np.asarray(a, dtype=float) for a in (hydrogel, background, observed)]
    if (hydrogel.ndim != 2 or observed.shape != hydrogel.shape or hydrogel.shape[0] == 0
            or hydrogel.shape[1] < 2 or background.shape != (hydrogel.shape[1],)
            or not all(np.isfinite(a).all() for a in (hydrogel, background, observed))
            or np.any(hydrogel < 0) or np.any(background < 0) or background.sum() <= 0):
        raise ValueError('Require finite matching spectra and nonnegative nonzero background.')
    area = hydrogel.sum(axis=1, keepdims=True)
    shape = np.divide(hydrogel, area, out=np.zeros_like(hydrogel), where=area > 0)
    background = background/background.sum()
    coefficients = np.array([nnls(np.column_stack((row, background)), y)[0]
                             for row, y in zip(shape, observed)])
    predicted = coefficients[:, :1]*shape + coefficients[:, 1:]*background
    total = coefficients.sum(axis=1)
    fraction = np.divide(coefficients[:, 0], total, out=np.full_like(total, np.nan), where=total > 0)
    # Exactly indistinguishable component shapes have no unique percentage.
    ambiguous = np.linalg.norm(shape-background, axis=1) <= 1e-10
    fraction[ambiguous] = np.nan
    return predicted, coefficients, fraction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'outputs/hydrogel_sphere_tapered_rim')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/hydrogel_flexible_mixture')
    args = parser.parse_args()
    fit = json.loads((args.source/'fit.json').read_text())
    data = np.load(args.source/'spectra.npz')
    p = np.array([fit['parameters'].get(n, 0.) if n == 'rim_width' else fit['parameters'][n] for n in NAMES])
    model = SphereModel(data['frequency_ghz'], data['background'], data['pixel_xy'],
                        fit['reference_radius_um'], fit['psf_fwhm_um'])
    gel, water = model.remodel(p, return_components=True)
    np.testing.assert_allclose((gel+water)*data['pixel_scale'][:, None], data['predicted'], rtol=1e-8, atol=1e-8)
    observed, before = data['observed'], data['predicted']
    predicted, areas, fraction = fit_mixture(gel, model.bg, observed)
    old_mse = np.mean((observed-before)**2, axis=1)
    new_mse = np.mean((observed-predicted)**2, axis=1)
    assert np.all(new_mse <= old_mse+1e-6), 'Free mixture must include the previous fit.'
    xy = data['pixel_xy']
    ix, iy = (xy-xy.min(axis=0)).astype(int).T
    grid = np.full((iy.max()+1, ix.max()+1, len(model.nu)), np.nan)
    grid[iy, ix] = predicted
    summary = dict(source=str(args.source), method='Independent nonnegative hydrogel and water spectral areas per pixel',
                   shared_parameters_refitted=False, n_pixels=len(observed), n_channels=observed.shape[1],
                   before_rmse=float(np.sqrt(old_mse.mean())), final_rmse=float(np.sqrt(new_mse.mean())),
                   percentage_units='Fraction of integrated spectral intensity within the saved frequency window',
                   undefined_fraction_pixels=int(np.isnan(fraction).sum()),
                   zero_hydrogel_support_pixels=int(np.sum(gel.sum(axis=1) == 0)),
                   limitations=['In-sample fit with extra per-pixel flexibility; not physical concentration.',
                                'Pixels with no hydrogel support in the saved sphere remain pure background.',
                                'Similar component spectra make fractions weakly constrained.'])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    table = pd.read_csv(args.source/'pixels.csv')[['X', 'Y']]
    table.assign(hydrogel_pct=100*fraction, background_pct=100*(1-fraction),
                 hydrogel_area=areas[:, 0], background_area=areas[:, 1], total_area=areas.sum(axis=1),
                 before_rmse=np.sqrt(old_mse), rmse=np.sqrt(new_mse)).to_csv(args.output/'pixels.csv', index=False)
    np.savez_compressed(args.output/'spectra.npz', frequency_ghz=model.nu, pixel_xy=xy,
                        observed=observed, predicted=predicted, predicted_grid=grid,
                        residual=observed-predicted, hydrogel_fraction=fraction,
                        hydrogel_area=areas[:, 0], background_area=areas[:, 1])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.8))
    fig.subplots_adjust(left=.05, right=.97, bottom=.22, top=.75, wspace=.42)
    maximum = max(float(np.sqrt(old_mse).max()), float(np.sqrt(new_mse).max()))
    for ax, values, title, cmap, vmin, vmax in zip(axes,
            (np.sqrt(old_mse), np.sqrt(new_mse), 100*fraction, 100*(1-fraction)),
            ('Smooth rim: RMSE', 'Flexible mixture: RMSE', 'Hydrogel spectral %', 'Background spectral %'),
            ('magma', 'magma', 'viridis', 'viridis'), (0, 0, 0, 0), (maximum, maximum, 100, 100)):
        image = np.full(grid.shape[:2], np.nan)
        image[iy, ix] = values
        artist = ax.imshow(image, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest',
                           extent=[xy[:, 0].min()-.5, xy[:, 0].max()+.5, xy[:, 1].min()-.5, xy[:, 1].max()+.5])
        fig.colorbar(artist, ax=ax, shrink=.78)
        ax.set(title=title, xlabel='X pixel index (centred)', ylabel='Y pixel index (centred)')
    fig.suptitle('Letting each pixel choose its background / hydrogel mixture', x=.05, ha='left', fontsize=19, weight='bold')
    fig.text(.05, .85, f'Overall RMSE: {summary["before_rmse"]:.3f} → {summary["final_rmse"]:.3f} counts/channel', fontsize=14)
    fig.text(.05, .055, 'Same sphere and spectral profiles; two nonnegative component intensities fitted independently at every pixel.\n'
             'Both error maps share one colour scale. Percentages are spectral contributions, not material concentrations; errors are in-sample.',
             fontsize=10, color='#475569', linespacing=1.6)
    fig.savefig(args.output/'mixture_comparison.png', dpi=180, facecolor='white')
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
