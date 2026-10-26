"""Export the original sharp-boundary sphere fit with whole-spectrum pixel gains."""
import json
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from PIL import Image, ImageOps, ImageDraw

from hydrogel_sphere import ROOT, concentration, intensity, radial_curve, rim_fraction, scale_pixels


def main():
    source = ROOT/'outputs/hydrogel_sphere_pixel_scaled'
    out = ROOT/'outputs/sphere_fit_images_sharp_boundary'
    out.mkdir(parents=True, exist_ok=True)
    fit = json.loads((source/'fit.json').read_text())
    data = np.load(source/'spectra.npz')
    p = fit['parameters']
    assert p.get('rim_width', 0) == 0 and fit['pixel_scaling']
    center = np.array(fit['center_um'])/1000
    radius = fit['fitted_radius_um']/1000
    step = np.array(fit['step_um'])/1000
    xy = data['pixel_xy']*step
    x, y = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    ix, iy = np.searchsorted(x, xy[:, 0]), np.searchsorted(y, xy[:, 1])
    obs, pred = data['observed'], data['predicted']
    residual = obs-pred
    rmse = np.sqrt(np.mean(residual**2, axis=1))
    overall = np.sqrt(np.mean(residual**2))
    np.testing.assert_allclose(overall, fit['final_rmse'])
    np.testing.assert_allclose(residual, data['residual'])
    assert len(set(zip(ix, iy))) == len(x)*len(y) == len(obs)
    extent = [x.min()-step[0]/2, x.max()+step[0]/2,
              y.min()-step[1]/2, y.max()+step[1]/2]
    section = np.sqrt(radius**2-center[2]**2)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    images = []

    def save(fig, name, caption):
        fig.text(.055, .025, caption, fontsize=10, color='#475569')
        fig.savefig(out/name, dpi=190, facecolor='white')
        plt.close(fig)
        images.append(out/name)

    def map_image(ax, values, title, label, **kwargs):
        grid = np.empty((len(y), len(x)))
        grid[iy, ix] = values
        im = ax.imshow(grid, origin='lower', extent=extent, interpolation='nearest', **kwargs)
        ax.add_patch(Circle(center[:2], section, fill=False, color='white', ls='--', lw=1.3))
        ax.scatter(*center[:2], c='#ff436f', marker='+', s=75, linewidths=2)
        ax.set(title=title, xlabel='X (mm)', ylabel='Y (mm)')
        plt.colorbar(im, ax=ax, shrink=.8, label=label)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.subplots_adjust(left=.055, right=.98, bottom=.18, top=.81, wspace=.35)
    fig.suptitle(f'Spectral reconstruction error | overall RMSE = {overall:.2f} counts/channel', fontsize=19)
    legacy_max = np.sqrt(np.mean((obs-data['initial_unscaled'])**2, axis=1)).max()
    map_image(axes[0], rmse, 'Per-pixel RMSE (original colour scale)', 'RMSE (counts/channel)', cmap='magma', vmin=0, vmax=legacy_max)
    bound = np.max(np.abs(residual.mean(axis=1)))
    map_image(axes[1], residual.mean(axis=1), 'Mean signed residual: observed − model', 'Mean residual (counts/channel)',
              cmap='RdBu_r', vmin=-bound, vmax=bound)
    axes[2].hist(rmse, bins=35, color='#197b9b')
    axes[2].axvline(overall, color='#c64542', ls='--', label=f'Overall RMSE {overall:.2f}')
    axes[2].set(title='Distribution across 1,155 pixels', xlabel='Pixel RMSE (counts/channel)', ylabel='Number of pixels')
    axes[2].legend()
    save(fig, '01_reconstruction_error.png', 'Sharp-boundary sphere with fitted whole-spectrum pixel gains; in-sample error over 69 frequency channels.\nDashed circle: sphere/scan-plane intersection. Pink cross: centre XY projection. XY origin: scan midpoint.')

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.subplots_adjust(left=.055, right=.98, bottom=.18, top=.83, wspace=.35)
    fig.suptitle('Measured and reconstructed spectral intensity', fontsize=19)
    vmax = max(obs.sum(axis=1).max(), pred.sum(axis=1).max())
    for ax, values, title in zip(axes[:2], (obs.sum(axis=1), pred.sum(axis=1)), ('Measured', 'Sphere reconstruction')):
        map_image(ax, values, title, 'Sum over fitted channels (counts)', cmap='viridis', vmin=0, vmax=vmax)
    map_image(axes[2], data['pixel_scale'], 'Optimized per-pixel intensity gain', 'Multiplicative gain', cmap='cividis')
    save(fig, '02_intensity_reconstruction.png', 'The two intensity maps share a colour scale. Each fitted gain multiplies the complete hydrogel + water spectrum.\nBrightness agreement alone does not establish spectral-shape agreement; see the RMSE image.')

    subprocess.run([sys.executable, str(ROOT/'analysis/pca_vca/plot_hydrogel_geometry.py'),
                    '--fit', str(source/'fit.json'), '--output', str(out), '--static-only'], check=True)
    geometry = out/'sphere_geometry.png'
    geometry.replace(out/'03_scan_plane_in_sphere.png')
    images.append(out/'03_scan_plane_in_sphere.png')

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={'width_ratios': [1.5, 1]})
    fig.subplots_adjust(left=.065, right=.97, bottom=.19, top=.84, wspace=.28)
    fig.suptitle('Where the scan samples the fitted sphere', fontsize=20)
    ax = axes[0]
    gx, gz = np.meshgrid(np.linspace(center[0]-radius, center[0]+radius, 401),
                         np.linspace(center[2]-radius, center[2]+radius, 401))
    rr = np.hypot(gx-center[0], gz-center[2])/radius
    h = np.ma.masked_where(rr > 1, rim_fraction(rr, 0))
    ax.imshow(h, origin='lower', extent=[gx.min(), gx.max(), gz.min(), gz.max()], cmap='Blues', vmin=0, vmax=1, alpha=.7)
    ax.add_patch(Circle((center[0], center[2]), radius, fill=False, color='#317e9d', lw=2))
    ax.axhline(0, color='#b87900', lw=2, label='Scan plane Z = 0')
    ax.axhline(center[2], color='#82909c', ls=':', label='Sphere equator')
    ax.scatter(x, np.zeros_like(x), s=13, color='#b87900', zorder=4)
    ax.scatter(center[0], center[2], s=90, color='#fa3970', label='Estimated centre', zorder=5)
    ax.annotate('', (center[0], 0), (center[0], center[2]), arrowprops=dict(arrowstyle='<->', color='#fa3970', lw=2))
    ax.text(center[0]+.3, center[2]/2, f'{center[2]:.3f} mm\ncentre depth', fontsize=10)
    ax.set(aspect='equal', xlim=extent[:2], ylim=(-4, 9), xlabel='X (mm)', ylabel='Z (mm)', title='Central X–Z section through sphere centre')
    ax.legend(loc='upper left', fontsize=9)
    z = np.linspace(-.18, .18, 500)
    sigma = fit['psf_fwhm_um'][2]/1000/np.sqrt(8*np.log(2))
    axes[1].plot(z*1000, np.exp(-.5*(z/sigma)**2), lw=3, color='#197b9b')
    axes[1].axvspan(-50, 50, alpha=.12, color='#197b9b', label='100 µm axial FWHM')
    axes[1].set(title='Magnified depth weighting at each pixel', xlabel='Z relative to scan plane (µm)', ylabel='Relative optical sampling weight', ylim=(0, 1.1))
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(alpha=.2)
    save(fig, '04_scan_depth_section.png', f'R = {radius:.3f} mm; scan-plane intersection radius = {section:.3f} mm. Blue region: sphere interior, with a sharp boundary.\nDots mark X sampling at 0.5 mm intervals in this central section; Gaussian PSF assumed (2 × 2 × 100 µm FWHM).\nDepth is weakly constrained: reflection to −Z gives the same fit. The PSF inset has its own magnified scale.')

    rho = np.linspace(0, 1, 501)
    r = rho*radius
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(left=.085, right=.965, bottom=.15, top=.88, hspace=.40, wspace=.28)
    fig.suptitle('Local model properties as you leave the sphere centre', fontsize=20)
    for ax, key, label, color in zip(axes[0], ('bfs', 'lw'), ('BFS (GHz)', 'Effective linewidth FWHM (GHz)'), ('#197b9b', '#b85c38')):
        values = radial_curve(rho, *(p[f'{key}_{s}'] for s in ('center', 'edge', 'power')))
        ax.plot(r, values, lw=3, color=color, label=f'Power exponent = {p[key+"_power"]:.3f}')
        ax.set(ylabel=label, title=f'{values[0]:.3f} at centre → {values[-1]:.3f} GHz at boundary')
        ax.legend(fontsize=9)
    local_int = intensity(concentration(rho), p['int_center'], p['int_edge'], p['int_power'])
    axes[1, 0].plot(r, local_int/1000, label='Hydrogel intensity inside sphere', lw=2)
    axes[1, 0].axhline(p['water_intensity']/1000, ls='--', color='#e47c36', label='Water intensity outside sphere')
    axes[1, 0].set(ylabel='Local intensity (thousand counts)', title='Intensity before PSF integration and pixel gain')
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot([0, radius, radius], [1, 1, 0], lw=2, label='Sharp hydrogel support; water outside R')
    axes[1, 1].plot(r, concentration(rho), color='#777777', ls=':', label='Assumed concentration 1 − (r/R)²')
    axes[1, 1].set(ylabel='Relative support / concentration', title='Sharp boundary into background water')
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.axvspan(0, abs(center[2]), color='#888888', alpha=.13)
        ax.set(xlabel='Distance from sphere centre r (mm)', xlim=(0, radius))
        ax.grid(alpha=.2)
    save(fig, '05_radial_properties.png', 'Grey region: inward of the closest scan-plane centre; these values rely strongly on extrapolation. No uncertainty bands.\nBFS/LW describe the local hydrogel branch before optical averaging. Centre linewidth reaches its upper fit bound (3 GHz).\nThis original scaled fit held the radial intensity profile fixed; whole-spectrum pixel gains were optimized. No soft rim.')

    distance = np.linalg.norm(xy-center[:2], axis=1)
    targets = [(0, 'Closest to centre projection'), (.55*section, 'Mid-radius'),
               (.95*section, 'Near outer intersection'), (1.2*section, 'Outside sphere')]
    indices = [int(np.argmin((distance-d)**2 + .02*(xy[:, 1]-center[1])**2)) for d, _ in targets]
    indices += [int(np.argmax(rmse)), int(np.argmin(abs(rmse-np.median(rmse))))]
    labels = [label for _, label in targets]+['Largest reconstruction error', 'Typical reconstruction error']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.subplots_adjust(left=.06, right=.975, bottom=.14, top=.85, wspace=.28, hspace=.5)
    fig.suptitle('Measured spectra compared with the sphere reconstruction', fontsize=20)
    for ax, i, label in zip(axes.flat, indices, labels):
        ax.plot(data['frequency_ghz'], obs[i], '.', color='#283f51', ms=4, label='Observed')
        ax.plot(data['frequency_ghz'], pred[i], color='#e47c36', lw=2, label='Reconstructed')
        ax.set(title=f'{label}\n(X, Y) = ({xy[i,0]:.1f}, {xy[i,1]:.1f}) mm; RMSE {rmse[i]:.2f}', xlabel='Frequency (GHz)', ylabel='Counts/channel')
        ax.grid(alpha=.15)
    axes.flat[0].legend(fontsize=9)
    save(fig, '06_example_spectra.png', 'Saved sharp-boundary sphere prediction, including fitted whole-spectrum pixel gains. The same 69 channels are used in every RMSE.\n“Centre projection” is the closest point in the scan plane, not the physical 3D sphere centre.')

    earlier = np.load(ROOT/'outputs/hydrogel_sphere_pixel_scaled/spectra.npz')
    np.testing.assert_allclose(earlier['observed'], obs)
    np.testing.assert_allclose(earlier['pixel_xy'], data['pixel_xy'])
    for saved in (earlier, data):
        assert np.all(saved['pixel_scale'] > 0)
        unscaled = saved['predicted']/saved['pixel_scale'][:, None]
        checked, gain = scale_pixels(unscaled, saved['observed'])
        np.testing.assert_allclose(checked, saved['predicted'])
        np.testing.assert_allclose(gain, saved['pixel_scale'])
    earlier_rmse = np.sqrt(np.mean((obs-earlier['initial_unscaled'])**2, axis=1))
    legacy_max = np.sqrt(np.mean((obs-earlier['initial_unscaled'])**2, axis=1)).max()
    zoom_max = rmse.max()
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    fig.subplots_adjust(left=.08, right=.96, bottom=.13, top=.90, wspace=.26, hspace=.30)
    fig.suptitle('Sharp-boundary sphere: whole-spectrum intensity scaling', fontsize=20)
    for row, limit in enumerate((legacy_max, zoom_max)):
        for col, (values, label) in enumerate(((earlier_rmse, 'Original unscaled prediction'), (rmse, 'With pixel scaling + refit'))):
            map_image(axes[row, col], values,
                      f'{label} | overall RMSE {np.sqrt(np.mean(values**2)):.2f}\n'
                      f'{"Original" if row == 0 else "Zoomed"} colour range: 0–{limit:.1f}',
                      'RMSE (counts/channel)', cmap='magma', vmin=0, vmax=limit)
            # Remove the current fit's geometry overlay: only error values are compared here.
            axes[row, col].patches[-1].remove()
            axes[row, col].collections[-1].remove()
    save(fig, '07_error_scale_comparison.png', 'Each row shares a colour scale. The top row matches the original before/after display.\nThe lower row increases contrast; unscaled errors above the colour maximum are saturated.\nSaved scaled predictions were checked against their optimal whole-spectrum pixel gains; no new refitting was performed.')

    # A contact sheet makes the exported folder quick to browse; full-size PNGs retain detail.
    sheet = Image.new('RGB', (1600, 1620), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((25, 12), f'SHARP-BOUNDARY SPHERE + PIXEL GAINS | RMSE {overall:.2f} counts/channel | full-size images in this folder', fill='#283f51')
    for k, path in enumerate(images[:6]):
        with Image.open(path) as im:
            thumb = ImageOps.contain(im.convert('RGB'), (780, 495))
        col, row = k % 2, k // 2
        sheet.paste(thumb, (10+col*800+(780-thumb.width)//2, 45+row*520))
        draw.text((25+col*800, 45+row*520+495), path.name, fill='#283f51')
    sheet.save(out/'00_overview.png')
    assert len(images) == 7 and all(path.is_file() for path in images)
    print(f'Exported 8 PNG images to {out}; verified RMSE = {overall:.8f}')


if __name__ == '__main__':
    main()
