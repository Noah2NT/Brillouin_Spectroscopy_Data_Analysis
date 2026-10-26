"""Show the saved sphere centre and its intersection with the measured scan plane."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, default=ROOT/'outputs/hydrogel_sphere/fit.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--static-only', action='store_true')
    args = parser.parse_args()
    fit = json.loads(args.fit.read_text(encoding='utf-8'))
    out = args.output or args.fit.parent
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.fit.parent/'spectra.npz')
    scan = pd.read_csv(fit['scan'])
    pixels = pd.read_csv(args.fit.parent/'pixels.csv')
    np.testing.assert_allclose(scan[['X', 'Y']], pixels[['X', 'Y']])
    center = np.array(fit['center_um']) / 1000
    radius = fit['fitted_radius_um'] / 1000
    step = np.array(fit['step_um']) / 1000
    xy = data['pixel_xy'] * step
    x, y = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    xx, yy = np.meshgrid(x, y)
    bfs = np.full(xx.shape, np.nan)
    bfs[np.searchsorted(y, xy[:, 1]), np.searchsorted(x, xy[:, 0])] = scan.Frequency
    if not np.isfinite(bfs).all():
        raise ValueError('This plot requires a complete finite rectangular BFS map.')
    # Match each plotted pixel to the saved reconstruction's physical coordinates.
    assert bfs.shape == data['predicted_grid'].shape[:2]
    phi, theta = np.meshgrid(np.linspace(0, 2*np.pi, 81), np.linspace(0, np.pi, 41))
    sx = center[0] + radius*np.sin(theta)*np.cos(phi)
    sy = center[1] + radius*np.sin(theta)*np.sin(phi)
    sz = center[2] + radius*np.cos(theta)
    np.testing.assert_allclose((sx-center[0])**2+(sy-center[1])**2+(sz-center[2])**2, radius**2)
    angles = np.linspace(0, 2*np.pi, 241)
    intersects = abs(center[2]) <= radius
    section_radius = np.sqrt(max(radius**2-center[2]**2, 0))
    cx, cy = center[0]+section_radius*np.cos(angles), center[1]+section_radius*np.sin(angles)
    norm = Normalize(float(bfs.min()), float(bfs.max()))
    cmap = plt.get_cmap('viridis')
    extent = [x[0]-step[0]/2, x[-1]+step[0]/2, y[0]-step[1]/2, y[-1]+step[1]/2]
    zlim = [-radius-abs(center[2])-.4, radius+abs(center[2])+.4]
    caption = ('Map colours: original CSV fitted BFS, not the sphere prediction. XY origin: scan midpoint; scan plane: Z = 0.\n'
               'Sphere is a model estimate. Depth is weakly constrained; the equally valid reflected centre at -Z is not drawn.')

    fig = plt.figure(figsize=(14, 7.8))
    fig.suptitle('Where the fitted hydrogel sphere sits in the scan', x=.06, y=.98,
                 ha='left', fontsize=20, fontweight='bold')
    fig.text(.06, .92, f'Radius {radius:.3f} mm   |   Centre (X, Y, Z) = '
             f'({center[0]:+.3f}, {center[1]:+.3f}, {center[2]:+.3f}) mm   |   Equal physical units; no depth exaggeration',
             fontsize=11, color='#475569')
    ax = fig.add_axes([.055, .20, .38, .62])
    ax.imshow(bfs, origin='lower', extent=extent, cmap=cmap, norm=norm, interpolation='nearest')
    if intersects:
        ax.add_patch(Circle(center[:2], section_radius, fill=False, edgecolor='white', lw=2,
                            linestyle='--', label='Sphere / scan-plane intersection'))
    ax.scatter(*center[:2], s=150, c='#fa3970', edgecolors='white', linewidths=1.7,
               marker='o', zorder=5, label='Estimated centre (XY projection)')
    ax.annotate('Estimated centre', center[:2], xytext=(center[0]+1, center[1]+2.3), color='white',
                fontweight='bold', arrowprops=dict(arrowstyle='->', color='white', lw=1.7),
                bbox=dict(facecolor='#172033', alpha=.8, edgecolor='none', pad=5))
    ax.set(title='Measured 2D BFS map', xlabel='X from scan midpoint (mm)', ylabel='Y from scan midpoint (mm)')
    ax.legend(loc='lower left', fontsize=8, facecolor='#172033', labelcolor='white', framealpha=.85)
    ax3 = fig.add_axes([.46, .16, .49, .70], projection='3d', computed_zorder=False)
    # Explicit transparent overlay avoids mplot3d sorting the whole plane above the sphere.
    ax3.plot_surface(sx, sy, sz, color='#7bb8d1', alpha=.10, linewidth=0, shade=False, zorder=2)
    ax3.plot_wireframe(sx, sy, sz, rstride=5, cstride=8, color='#5089a6', alpha=.55, linewidth=.8, zorder=3)
    ax3.plot_surface(xx, yy, np.zeros_like(xx), facecolors=cmap(norm(bfs)),
                     rstride=1, cstride=1, shade=False, antialiased=False, alpha=.98, zorder=1)
    for row in range(4, len(y), 8):
        ax3.plot(x, np.full_like(x, y[row]), np.zeros_like(x), color='#ffbf47',
                 lw=1.1, alpha=.9, zorder=4)
    ax3.text2D(.02, .96, 'Gold lines: selected scan rows at Z = 0\nPink line: centre-to-plane depth',
               transform=ax3.transAxes, fontsize=9, va='top')
    if intersects:
        ax3.plot(cx, cy, np.zeros_like(cx), color='white', lw=2, linestyle='--', zorder=4)
    ax3.scatter(*center, c='#fa3970', edgecolor='white', s=100, depthshade=False, zorder=6)
    ax3.plot([center[0]]*2, [center[1]]*2, [0, center[2]], color='#fa3970', lw=3, zorder=5)
    ax3.set(xlim=extent[:2], ylim=extent[2:], zlim=zlim,
            xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)', title='Fitted sphere with measured map at Z = 0')
    ax3.set_box_aspect((extent[1]-extent[0], extent[3]-extent[2], zlim[1]-zlim[0]))
    ax3.view_init(elev=24, azim=-57)
    color_axis = fig.add_axes([.945, .29, .014, .43])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_axis, label='CSV fitted BFS (GHz)')
    fig.text(.06, .045, caption, fontsize=10, color='#475569', linespacing=1.7)
    fig.savefig(out/'sphere_geometry.png', dpi=180, facecolor='white')
    plt.close(fig)
    if args.static_only:
        print(out/'sphere_geometry.png')
        return

    interactive = make_subplots(rows=1, cols=2, specs=[[{'type': 'xy'}, {'type': 'scene'}]],
                               column_widths=[.43, .57], horizontal_spacing=.07,
                               subplot_titles=('Measured BFS and estimated centre', 'Drag to rotate the sphere and scan plane'))
    interactive.add_trace(go.Heatmap(x=x, y=y, z=bfs, coloraxis='coloraxis',
                                     hovertemplate='X %{x:.2f} mm<br>Y %{y:.2f} mm<br>BFS %{z:.4f} GHz<extra></extra>'), 1, 1)
    if intersects:
        interactive.add_trace(go.Scatter(x=cx, y=cy, mode='lines', name='Scan-plane intersection',
                                         line=dict(color='white', dash='dash', width=2)), 1, 1)
    interactive.add_trace(go.Scatter(x=[center[0]], y=[center[1]], mode='markers', name='Centre XY projection',
                                     marker=dict(color='#fa3970', size=13, line=dict(color='white', width=2))), 1, 1)
    interactive.add_trace(go.Surface(x=xx, y=yy, z=np.zeros_like(xx), surfacecolor=bfs, coloraxis='coloraxis',
                                     name='Measured scan', customdata=bfs,
                                     hovertemplate='X %{x:.2f} mm<br>Y %{y:.2f} mm<br>Z = 0<br>BFS %{customdata:.4f} GHz<extra></extra>'), 1, 2)
    interactive.add_trace(go.Surface(x=sx, y=sy, z=sz, opacity=.15, showscale=False,
                                     colorscale=[[0, '#8bbcd4'], [1, '#8bbcd4']], hoverinfo='skip', name='Sphere'), 1, 2)
    for row in range(0, len(theta), 5):
        interactive.add_trace(go.Scatter3d(x=sx[row], y=sy[row], z=sz[row], mode='lines',
                                           line=dict(color='#5488a2', width=1), showlegend=False, hoverinfo='skip'), 1, 2)
    for col in range(0, sx.shape[1]-1, 8):
        interactive.add_trace(go.Scatter3d(x=sx[:, col], y=sy[:, col], z=sz[:, col], mode='lines',
                                           line=dict(color='#5488a2', width=1), showlegend=False, hoverinfo='skip'), 1, 2)
    if intersects:
        interactive.add_trace(go.Scatter3d(x=cx, y=cy, z=np.zeros_like(cx), mode='lines', showlegend=False,
                                           line=dict(color='white', width=5, dash='dash'), hoverinfo='skip'), 1, 2)
    interactive.add_trace(go.Scatter3d(x=[center[0]], y=[center[1]], z=[center[2]], mode='markers', name='3D centre',
                                       marker=dict(size=6, color='#fa3970', line=dict(color='white', width=2)),
                                       hovertemplate='Centre<br>X %{x:.4f} mm<br>Y %{y:.4f} mm<br>Z %{z:.4f} mm<extra></extra>'), 1, 2)
    interactive.update_layout(template='plotly_white', height=780,
                              title=f'Fitted hydrogel sphere | R = {radius:.3f} mm | measured map at Z = 0',
                              margin=dict(l=65, r=90, t=100, b=145),
                              coloraxis=dict(colorscale='Viridis', cmin=norm.vmin, cmax=norm.vmax,
                                             colorbar=dict(title='BFS (GHz)', len=.6)),
                              scene=dict(aspectmode='data', xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
                                         xaxis=dict(range=extent[:2]), yaxis=dict(range=extent[2:]), zaxis=dict(range=zlim),
                                         camera=dict(eye=dict(x=1.4, y=-1.8, z=1.1))),
                              legend=dict(orientation='h', y=-.13, x=0))
    interactive.update_xaxes(title_text='X from scan midpoint (mm)', range=extent[:2])
    interactive.update_yaxes(title_text='Y from scan midpoint (mm)', scaleanchor='x', scaleratio=1, range=extent[2:])
    interactive.add_annotation(text=caption.replace('\n', '<br>'), x=0, y=-.25,
                               xref='paper', yref='paper', xanchor='left', showarrow=False, align='left', font=dict(size=11))
    interactive.write_html(out/'sphere_geometry.html', include_plotlyjs=True,
                           config=dict(scrollZoom=True, displaylogo=False))
    print(f'Created {out / "sphere_geometry.png"}\nCreated {out / "sphere_geometry.html"}')


if __name__ == '__main__':
    main()
