"""Plot hydrogel percentage from per-pixel mixture fits, centred at Z = 0.

Run with: python analysis/pca_vca/hydrogel_columns.py
Requires numpy, pandas, matplotlib and plotly.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = ROOT / 'outputs/hydrogel_background_comparison'
    # Keep the existing viewer URL while correcting its quantity and source.
    out = ROOT / 'outputs/hydrogel_intensity_gate/column_reconstruction'
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(source / 'pixel_results.csv')
    saved = np.load(source / 'reconstructions.npz')
    p = saved['mixture_parameters']
    nu = saved['frequency_ghz']
    bg = saved['background_peak_normalized']
    np.testing.assert_array_equal(saved['xi'], table.xi)
    np.testing.assert_array_equal(saved['yi'], table.yi)
    lorentzian = p[:, 0, None] / (1 + ((nu[None, :] - p[:, 1, None]) / p[:, 2, None]) ** 2)
    lorentzian_intensity = lorentzian.sum(axis=1)
    background_intensity = p[:, 4] * bg.sum()
    total = lorentzian_intensity + background_intensity
    assert np.all(total > 0)
    background_pct = 100 * background_intensity / total
    hydrogel_pct = 100 - background_pct
    heights = hydrogel_pct
    z_bottom, z_top = -heights / 2, heights / 2
    np.testing.assert_allclose((z_bottom + z_top) / 2, 0)
    np.testing.assert_allclose(z_top - z_bottom, heights)
    np.testing.assert_allclose(background_pct, table.background_pct, atol=1e-10)
    np.testing.assert_allclose(heights, table.lorentzian_pct, atol=1e-10)
    np.testing.assert_allclose(lorentzian + p[:, 4, None] * bg + p[:, 3, None], saved['mixture'])
    assert np.isfinite(heights).all() and ((heights >= 0) & (heights <= 100)).all()
    assert not table.duplicated(['xi', 'yi']).any()
    assert np.allclose(table.lorentzian_pct + table.background_pct, 100)
    x, y = table.xi.to_numpy(), table.yi.to_numpy()
    nx, ny = int(x.max()) + 1, int(y.max()) + 1
    assert len(table) == nx * ny
    above_threshold = hydrogel_pct >= 12.0
    mask = np.zeros((ny, nx), dtype=bool)
    mask[y, x] = above_threshold
    padded = np.pad(mask, 1)
    neighbours = np.zeros_like(mask, dtype=int)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                neighbours += padded[1+dy:1+dy+ny, 1+dx:1+dx+nx]
    isolated = above_threshold & (neighbours[y, x] == 0)
    active = above_threshold & ~isolated
    heights = np.where(active, hydrogel_pct, 0.)
    z_bottom, z_top = -heights / 2, heights / 2
    np.testing.assert_array_equal(heights[active], hydrogel_pct[active])
    assert np.all(heights[~active] == 0)
    assert np.all(neighbours[y[active], x[active]] > 0)
    nonconverged = int((~table.mixture_converged).sum())
    caption = (f'{active.sum()} retained | {(~above_threshold).sum()} below 12% removed | '
               f'{isolated.sum()} isolated pixels removed | Z midpoint = 0 | No smoothing')
    result = table[['pixel', 'xi', 'yi', 'x', 'y', 'mixture_converged']].copy()
    result['background_intensity'] = background_intensity
    result['lorentzian_intensity'] = lorentzian_intensity
    result['background_pct'] = background_pct
    result['lorentzian_pct'] = hydrogel_pct
    result['hydrogel_pct'] = hydrogel_pct
    result['lorentzian_amplitude'] = p[:, 0]
    result['lorentzian_center_ghz'] = p[:, 1]
    result['lorentzian_halfwidth_ghz'] = p[:, 2]
    result['background_scale'] = p[:, 4]
    result['constant_baseline'] = p[:, 3]
    result['height_pct'] = heights
    result['retained'] = active
    result['removal_reason'] = np.where(~above_threshold, 'below_12_percent',
                                       np.where(isolated, 'isolated_8_neighbour', 'retained'))
    result['z_bottom_pct'] = z_bottom
    result['z_top_pct'] = z_top
    result.to_csv(out / 'column_heights.csv', index=False)

    # Static, publication-friendly view plus a matching spatial map.
    fig = plt.figure(figsize=(15, 8), facecolor='white')
    ax = fig.add_subplot(121, projection='3d')
    norm = Normalize(0, 100)
    colors = plt.get_cmap('viridis')(norm(heights[active]))
    ax.bar3d(x[active] - .46, y[active] - .46, z_bottom[active],
             .92, .92, heights[active], color=colors, shade=True,
             edgecolor=(0, 0, 0, .15), linewidth=.15)
    ax.set(xlim=(-.5, nx-.5), ylim=(-.5, ny-.5), zlim=(-50, 50),
           xlabel='X pixel index', ylabel='Y pixel index',
           zlabel='Hydrogel %', title='Column midpoints aligned at Z = 0')
    ax.set_box_aspect((nx, ny, 25))
    ax.view_init(elev=30, azim=-55)
    flat = fig.add_subplot(122)
    grid = np.full((ny, nx), np.nan)
    grid[y[active], x[active]] = heights[active]
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#eeeeee')
    im = flat.imshow(grid, origin='lower', cmap=cmap, vmin=0, vmax=100,
                     interpolation='nearest')
    flat.set(xlabel='X pixel index', ylabel='Y pixel index', title='Same heights, viewed from above')
    fig.colorbar(im, ax=flat, shrink=.75, label='Hydrogel %')
    fig.suptitle('Hydrogel %: columns centred at Z = 0', fontsize=19, y=.95)
    fig.text(.5, .065, caption, ha='center', fontsize=11)
    note = ('Spectral share; constant baseline excluded. '
            f'{nonconverged} saved fits flagged as not converged (see CSV).')
    fig.text(.5, .03, note,
             ha='center', fontsize=10, color='#555555')
    fig.subplots_adjust(left=.03, right=.96, bottom=.15, top=.86, wspace=.25)
    fig.savefig(out / 'hydrogel_columns.png', dpi=180)
    plt.close(fig)

    # One indexed mesh combines all cuboids; the HTML embeds Plotly for offline use.
    vertices, triangles, intensity, labels = [], [], [], []
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
             (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
             (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    for xx, yy, h in zip(x[active], y[active], heights[active]):
        offset = len(vertices)
        vertices.extend([(xx+dx, yy+dy, zz) for zz in (-h/2, h/2)
                         for dx, dy in [(-.46, -.46), (.46, -.46), (.46, .46), (-.46, .46)]])
        triangles.extend([(a+offset, b+offset, c+offset) for a, b, c in faces])
        intensity.extend([h] * 8)
        labels.extend([f'X pixel: {xx}<br>Y pixel: {yy}<br>Hydrogel: {h:.2f}%<br>Background: {100-h:.2f}%<br>Z extent: {-h/2:.2f} to {h/2:.2f}'] * 8)
    figure = go.Figure()
    if vertices:
        vx, vy, vz = np.array(vertices).T
        ii, jj, kk = np.array(triangles).T
        figure.add_trace(go.Mesh3d(x=vx, y=vy, z=vz, i=ii, j=jj, k=kk,
            intensity=intensity, cmin=0, cmax=100, colorscale='Viridis',
            flatshading=True, text=labels, hovertemplate='%{text}<extra></extra>',
            colorbar=dict(title='Hydrogel %'), name='Hydrogel columns'))
    figure.update_layout(
        title=dict(text='Hydrogel %: columns centred at Z = 0<br>'
                   f'<sup>{caption}</sup>', x=.03),
        template='plotly_white', margin=dict(l=0, r=0, t=90, b=65),
        scene=dict(xaxis=dict(title='X pixel index', range=[-.5, nx-.5]),
                   yaxis=dict(title='Y pixel index', range=[-.5, ny-.5]),
                   zaxis=dict(title='Hydrogel %', range=[-50, 50],
                              zeroline=True, zerolinecolor='#334155', zerolinewidth=3),
                   aspectmode='manual', aspectratio=dict(x=nx/max(nx, ny), y=ny/max(nx, ny), z=.7),
                   camera=dict(eye=dict(x=1.5, y=-1.8, z=1.3))),
        annotations=[dict(text='Drag to rotate · Scroll to zoom<br>' + note,
                          x=.5, y=-.06, xref='paper', yref='paper', showarrow=False)],
        legend=dict(x=.01, y=.98))
    html = figure.to_html(include_plotlyjs=True,
                          config=dict(scrollZoom=True, displaylogo=False), full_html=True)
    for name in ('hydrogel_percent_centred.html', 'hydrogel_columns.html', 'hydrogel_percent_filtered_12pct.html'):
        (out / name).write_text(html, encoding='utf-8')
    (out / 'README.md').write_text(
        '# Centred hydrogel-percentage column plot\n\n'
        'Source: outputs/hydrogel_background_comparison/{pixel_results.csv,reconstructions.npz}. '
        'This replaces the previous gated Lorentzian-share plot. The viewer is hydrogel_percent_centred.html.\n\n'
        'Model per pixel: S_i(nu) = b_i B(nu) + L(nu; A_i, centre_i, gamma_i) + c_i. '
        'B is the shared background endmember; its scale and the Lorentzian amplitude, centre, width '
        'and constant baseline are fitted independently per pixel. No fixed pure-hydrogel spectrum is used.\n\n'
        'Height = hydrogel percentage = 100 × sum(L_i) / [sum(L_i) + sum(b_i B)], '
        'excluding the constant baseline. Values are recomputed from the ungated fit parameters '
        'and checked against the saved percentages and reconstructed spectra. '
        'X and Y are original zero-based pixel indices, not inferred physical units.\n\n'
        'All columns have midpoint Z = 0: bottom = -height/2, top = +height/2. '
        'Total column extent and colour encode the hydrogel percentage. '
        'Negative Z denotes position relative to the midpoint, not a negative contribution.\n\n'
        + caption + '\n\n'
        + note + '\n\n'
        'Filtering: remove hydrogel shares strictly below 12% (exactly 12% is retained). '
        'Then remove single pixels with no retained neighbour among the eight adjacent positions, '
        'including diagonals. Multi-pixel islands are not removed. Removed pixels are omitted '
        'from the 3D view and shown in grey on the top view. Raw fitted percentages remain in the CSV; '
        'height_pct is zero for removed pixels, with retained and removal_reason columns. '
        'Retained heights are unchanged and no spectral refitting or smoothing is applied.\n\n'
        'The HTML is interactive and self-contained. The PNG includes a matching top view. '
        'column_heights.csv preserves the source coordinates and the plotted height for every pixel.\n\n'
        'Reproduce: `python analysis/pca_vca/hydrogel_columns.py`\n\n'
        '## Remaining sphere visualization work\n\n'
        '1. Done: 12% threshold followed by isolated single-pixel removal (8-neighbour connectivity).\n'
        '2. Common Z midpoint is already applied to the hydrogel columns.\n'
        '3. Smooth the resulting shape.\n\n'
        'A physical thickness interpretation requires calibration or the assumption that spectral share '
        'tracks path-length fraction. Centring imposes symmetry rather than measuring Z position.\n',
        encoding='utf-8')
    print(json.dumps(dict(output=str(out), grid=[ny, nx], pixels=len(table),
                         quantity='hydrogel percentage', threshold_pct=12,
                         retained=int(active.sum()), below_threshold=int((~above_threshold).sum()),
                         isolated_removed=int(isolated.sum()),
                         nonconverged=int((~table.mixture_converged).sum()),
                         height_range_pct=[float(heights.min()), float(heights.max())]), indent=2))


if __name__ == '__main__':
    main()
