"""Compare two saved fits on the same pixels, including the original boundary ring."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, default=ROOT/'outputs/hydrogel_sphere_pixel_scaled')
    parser.add_argument('--after', type=Path, default=ROOT/'outputs/hydrogel_sphere_free_intensity')
    args = parser.parse_args()
    old, new = [np.load(path/'spectra.npz') for path in (args.before, args.after)]
    fits = [json.loads((path/'fit.json').read_text()) for path in (args.before, args.after)]
    tapered = fits[1]['parameters'].get('rim_width', 0) > 0
    new_label = 'Smooth rim' if tapered else 'Intensity free'
    np.testing.assert_allclose(old['observed'], new['observed'])
    np.testing.assert_allclose(old['frequency_ghz'], new['frequency_ghz'])
    np.testing.assert_allclose(old['pixel_xy'], new['pixel_xy'])
    np.testing.assert_allclose(fits[0]['step_um'], fits[1]['step_um'])
    xy = old['pixel_xy'] * (np.array(fits[0]['step_um'])/1000)
    centers = [np.array(f['center_um'])/1000 for f in fits]
    radii = [np.sqrt(max((f['fitted_radius_um']/1000)**2-c[2]**2, 0)) for f, c in zip(fits, centers)]
    distance = np.linalg.norm(xy-centers[0][:2], axis=1)
    ring = np.abs(distance-radii[0]) <= .5
    errors = [np.mean((d['observed']-d['predicted'])**2, axis=1) for d in (old, new)]
    metrics = dict(overall_rmse=[float(np.sqrt(e.mean())) for e in errors],
                   old_boundary_ring_rmse=[float(np.sqrt(e[ring].mean())) for e in errors],
                   ring_pixels=int(ring.sum()), ring_half_width_mm=.5,
                   projected_radii_mm=radii, sphere_radii_mm=[f['fitted_radius_um']/1000 for f in fits])
    (args.after/'boundary_comparison.json').write_text(json.dumps(metrics, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8))
    fig.subplots_adjust(left=.06, right=.97, bottom=.22, top=.76, wspace=.35)
    fig.suptitle('Does a smooth rim reduce the boundary error?' if tapered else
                 'Does freeing radial intensity remove the boundary ring?', x=.06, ha='left', fontsize=18, weight='bold')
    fig.text(.06, .86, f'Overall RMSE {metrics["overall_rmse"][0]:.3f} → {metrics["overall_rmse"][1]:.3f}   |   '
             f'Same boundary pixels {metrics["old_boundary_ring_rmse"][0]:.3f} → {metrics["old_boundary_ring_rmse"][1]:.3f} counts/channel', fontsize=12)
    x, y = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    limit = max(float(np.sqrt(e).max()) for e in errors)
    for ax, e, title in zip(axes[:2], errors, ('Previous fit', f'Best new fit: {new_label.lower()}')):
        grid = np.full((len(y), len(x)), np.nan)
        grid[np.searchsorted(y, xy[:, 1]), np.searchsorted(x, xy[:, 0])] = np.sqrt(e)
        artist = ax.imshow(grid, origin='lower', extent=[x.min()-.25, x.max()+.25, y.min()-.25, y.max()+.25],
                           interpolation='nearest', cmap='magma', vmin=0, vmax=limit)
        ax.add_patch(Circle(centers[0][:2], radii[0], fill=False, color='white', lw=1.2, linestyle='--'))
        fig.colorbar(artist, ax=ax, shrink=.8)
        ax.set(title=title, xlabel='X (mm)', ylabel='Y (mm)')
    edges = np.arange(0, distance.max()+.5, .5)
    bins = np.digitize(distance, edges)-1
    for e, label in zip(errors, ('Previous', new_label)):
        means = [np.sqrt(e[bins == i].mean()) if np.any(bins == i) else np.nan for i in range(len(edges)-1)]
        axes[2].plot((edges[:-1]+edges[1:])/2, means, marker='o', ms=3, label=label)
    axes[2].axvline(radii[0], ls='--', color='#64748b', label='Previous boundary')
    axes[2].set(title='Error by distance from previous centre', xlabel='Radial distance in scan plane (mm)', ylabel='RMSE (counts/channel)')
    axes[2].grid(alpha=.2)
    axes[2].legend(fontsize=9)
    fig.text(.06, .06, 'Dashed circles mark the previous sphere / scan-plane intersection. Error maps use one colour scale.\n'
             'Boundary comparison uses the same pixels within 0.5 mm of the old circle; pixel gains are optimized in both fits.',
             fontsize=10, color='#475569', linespacing=1.6)
    fig.savefig(args.after/'boundary_comparison.png', dpi=180, facecolor='white')
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
