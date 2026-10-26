"""Plot the saved sphere fit without rerunning the optimization."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hydrogel_sphere import ROOT, radial_curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, default=ROOT/'outputs/hydrogel_sphere/fit.json')
    args = parser.parse_args()
    fit = json.loads(args.fit.read_text(encoding='utf-8'))
    p = fit['parameters']
    radius_mm = fit['fitted_radius_um'] / 1000
    rho = np.linspace(0, 1, 501)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    fig.subplots_adjust(left=.08, right=.97, bottom=.23, top=.70, wspace=.28)
    fig.suptitle('How the fitted hydrogel changes from centre to boundary',
                 x=.08, y=.98, ha='left', fontsize=19, fontweight='bold')
    fig.text(.08, .91, f'Spherical model  |  Fitted radius R = {radius_mm:.3f} mm  |  '
             'Local spectral properties before 3D averaging', color='#4b5563')
    for ax, key, title, label, color, symbol in zip(
            axes, ('bfs', 'lw'), ('Brillouin frequency shift', 'Linewidth'),
            ('BFS (GHz)', 'Observed FWHM (GHz)'), ('#176a9a', '#ac5130'), ('BFS', 'LW')):
        center, edge, power = (p[f'{key}_{suffix}'] for suffix in ('center', 'edge', 'power'))
        values = radial_curve(rho, center, edge, power)
        np.testing.assert_allclose(values[[0, -1]], [center, edge])
        ax.plot(rho, values, color=color, lw=3)
        ticks = np.linspace(0, 1, 5)
        ax.scatter(ticks, radial_curve(ticks, center, edge, power), c=color, s=28, zorder=3)
        for x, value, offset, align in ((0, center, (8, 12), 'left'),
                                         (1, edge, (-8, -24), 'right')):
            ax.annotate(f'{value:.3f} GHz', (x, value), xytext=offset,
                        textcoords='offset points', ha=align, color=color, weight='bold')
        span = abs(center-edge)
        ax.set(xlim=(-.025, 1.025), ylim=(min(center, edge)-.25*span, max(center, edge)+.22*span),
               xlabel=r'Normalized radius $\rho=r/R$', ylabel=label, xticks=ticks)
        ax.grid(alpha=.2)
        ax.set_axisbelow(True)
        ax.set_title(title, loc='left', fontweight='bold', pad=56)
        secondary = ax.secondary_xaxis('top', functions=(lambda x: x*radius_mm, lambda x: x/radius_mm))
        secondary.set_xlabel('Distance from centre (mm)', labelpad=8)
        secondary.set_xticks([0, 1, 2, 3, 4, radius_mm], labels=['0', '1', '2', '3', '4', f'{radius_mm:.2f}'])
        equation = rf'${symbol}(\rho)={center:.3f}{edge-center:+.3f}\,\rho^{{{power:.3f}}}$'
        ax.text(.5, -.25, equation, transform=ax.transAxes, ha='center', fontsize=13, color=color)
    fig.text(.08, .035, '0 = sphere centre; 1 = boundary. Beyond 1 the model uses the measured water endmember.\n'
             'These are fitted model curves, not direct radial measurements; uncertainty bands have not been estimated.',
             color='#4b5563', fontsize=10, linespacing=1.6)
    for suffix in ('png', 'svg'):
        path = args.fit.parent/f'radial_bfs_linewidth.{suffix}'
        fig.savefig(path, dpi=180, facecolor='white')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
