"""Compare matched least-squares single and background-assisted reconstructions.

Run: python analysis/pca_vca/hydrogel_background_comparison.py
Fractions are integrated spectral contributions over the fitted band, excluding
the free constant baseline. They are not material volume fractions.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import background_endmember as bge

ROOT = Path(__file__).resolve().parents[2]
STEM = 'Hydrogl1_XYscan_20000_20000_step500um_10s (1)'
OUT = ROOT / 'outputs/hydrogel_background_comparison'


def components(nu, p, bg):
    peak = p[0] / (1 + ((nu - p[1]) / p[2]) ** 2)
    background = p[4] * bg if len(p) == 5 else np.zeros_like(nu)
    return peak, background, np.full_like(nu, p[3])


def predict(nu, p, bg):
    return sum(components(nu, p, bg))


def fit(y, nu, bg, starts, mixed):
    lower = [0, nu.min(), 1e-4, -np.inf] + ([0] if mixed else [])
    upper = [np.inf, nu.max(), np.ptp(nu), np.inf] + ([np.inf] if mixed else [])
    candidates = []
    for start in starts:
        # Retain the exact nested single solution as a feasible candidate.
        candidates.append((np.sum((predict(nu, start, bg) - y)**2), start, False))
        result = least_squares(lambda p: predict(nu, p, bg) - y, start,
                               bounds=(lower, upper), x_scale='jac',
                               ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=1200)
        candidates.append((np.sum(result.fun**2), result.x, result.success))
    return min(candidates, key=lambda item: (item[0], not item[2]))[1:]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = bge.load_cube(ROOT / 'data/csv' / f'{STEM}.csv')
    nu, y = data['shift_freq'], 2 * data['folded_raw']
    em_path = ROOT / 'outputs/background_endmember' / f'{STEM}_background_endmember.csv'
    saved = pd.read_csv(em_path)
    np.testing.assert_allclose(nu, saved.shift_ghz, atol=1e-10)
    bg = saved.endmember_norm.to_numpy(copy=True)
    assert np.isfinite(y).all() and np.isfinite(bg).all() and (bg >= 0).all()
    bg /= bg.max()  # Conditioning only; percentages below use spectral sums.
    p_single, p_mixed, status = [], [], []
    print(f'Fitting {len(y)} pixels, {len(nu)} channels, grid {data["shape"]}', flush=True)
    for i, yi in enumerate(y):
        scale = max(yi.max(), 1)
        ys = yi / scale
        baseline = np.percentile(ys, 5)
        center = nu[ys.argmax()]
        starts = [np.array([max(ys.max()-baseline, 1e-6), center, width, baseline])
                  for width in (0.15, 0.4, 0.8)]
        p1, ok1 = fit(ys, nu, bg, starts, False)
        starts = [np.r_[p1, 0.]]
        for offset in (-0.35, 0, 0.35):
            starts.append(np.array([p1[0]*0.5, np.clip(p1[1]+offset, nu.min(), nu.max()),
                                    p1[2], p1[3], 0.5]))
        p2, ok2 = fit(ys, nu, bg, starts, True)
        p1[[0, 3]] *= scale
        p2[[0, 3, 4]] *= scale
        p_single.append(p1); p_mixed.append(p2); status.append([ok1, ok2])
        if (i+1) % 200 == 0:
            print(f'  {i+1}/{len(y)} fitted', flush=True)
    p1, p2 = np.array(p_single), np.array(p_mixed)
    rec1 = np.array([predict(nu, p, bg) for p in p1])
    rec2 = np.array([predict(nu, p, bg) for p in p2])
    rmse1 = np.sqrt(np.mean((y-rec1)**2, axis=1))
    rmse2 = np.sqrt(np.mean((y-rec2)**2, axis=1))
    gain = 100*(1-rmse2/np.maximum(rmse1, 1e-12))
    assert np.all(rmse2 <= rmse1 + 1e-6), 'Nested fit regression'
    assert np.isfinite(rec2).all()
    peak_counts = np.array([components(nu, p, bg)[0].sum() for p in p2])
    bg_counts = p2[:, 4]*bg.sum()
    fraction = bg_counts/np.maximum(peak_counts+bg_counts, 1e-12)
    table = pd.DataFrame(dict(pixel=np.arange(len(y)), xi=data['xi'], yi=data['yi'],
                             x=data['x_vals'][data['xi']], y=data['y_vals'][data['yi']],
                             single_rmse=rmse1, mixture_rmse=rmse2, rmse_reduction_pct=gain,
                             lorentzian_pct=100*(1-fraction), background_pct=100*fraction,
                             single_center_ghz=p1[:,1], mixture_center_ghz=p2[:,1],
                             single_fwhm_ghz=2*p1[:,2], mixture_fwhm_ghz=2*p2[:,2],
                             mixture_baseline=p2[:,3], single_converged=np.array(status)[:,0],
                             mixture_converged=np.array(status)[:,1]))
    table.to_csv(OUT/'pixel_results.csv', index=False)
    # Deterministic examples include typical, extreme, and poor residual fits.
    choices = [('Typical improvement', int(np.argmin(abs(gain-np.median(gain))))),
               ('Largest improvement', int(np.argmax(gain))),
               ('Least improvement', int(np.argmin(gain))),
               ('Largest remaining error', int(np.argmax(rmse2))),
               ('Typical background share', int(np.argmin(abs(fraction-np.median(fraction))))),
               ('Smallest background share', int(np.argmin(fraction)))]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')
    vmax = max(rmse1.max(), rmse2.max())
    for ax, values, title, cmap, limits in zip(axes.flat,
            [rmse1, rmse2, gain, 100*fraction],
            ['Single Lorentzian: reconstruction RMSE', 'Lorentzian + background: reconstruction RMSE',
             'RMSE reduction (%)', 'Background contribution (%)'],
            ['magma', 'magma', 'viridis', 'cividis'], [(0,vmax),(0,vmax),(0,100),(0,100)]):
        im = ax.imshow(bge.scatter_map(values, data['yi'], data['xi'], data['shape']),
                       origin='lower', cmap=cmap, vmin=limits[0], vmax=limits[1])
        ax.set(title=title, xlabel='X pixel index', ylabel='Y pixel index')
        fig.colorbar(im, ax=ax, shrink=.85, label='Counts / channel' if 'RMSE' in title and '%' not in title else '%')
        for j, (_, idx) in enumerate(choices):
            ax.text(data['xi'][idx], data['yi'][idx], str(j+1), ha='center', va='center',
                    fontsize=9, bbox=dict(facecolor='white', alpha=.8, edgecolor='black', boxstyle='circle,pad=.15'))
    fig.suptitle('Hydrogel | matched least-squares fits\nError against measured spectra; shared RMSE colour scale', fontsize=15)
    fig.savefig(OUT/'error_maps.png', dpi=170); plt.close(fig)
    for page in range(2):
        fig, axes = plt.subplots(2, 3, figsize=(16, 7), height_ratios=[2,1], layout='constrained')
        for col, (label, idx) in enumerate(choices[page*3:page*3+3]):
            ax, residual = axes[:,col]
            peak, back, base = components(nu, p2[idx], bg)
            ax.plot(nu, y[idx], '.', color='.5', ms=3, label='Measured')
            ax.plot(nu, rec1[idx], color='#d55e00', lw=1.6, label='Single Lorentzian + baseline')
            ax.plot(nu, rec2[idx], color='#0072b2', lw=1.8, label='Lorentzian + background + baseline')
            ax.plot(nu, peak, '--', color='#009e73', label='Lorentzian component')
            ax.plot(nu, back, '--', color='#cc79a7', label='Background component')
            ax.plot(nu, base, ':', color='.3', label='Constant baseline')
            ax.set_title(f'{page*3+col+1}. {label} | pixel {idx}\n(x={data["xi"][idx]}, y={data["yi"][idx]}) | L {100*(1-fraction[idx]):.1f}% + BG {100*fraction[idx]:.1f}%\nRMSE {rmse1[idx]:.1f} → {rmse2[idx]:.1f} ({gain[idx]:.1f}% lower)', fontsize=10)
            ax.set_ylabel('Folded summed counts'); ax.legend(fontsize=7)
            residual.axhline(0, color='.5', lw=.7)
            residual.plot(nu, y[idx]-rec1[idx], color='#d55e00', lw=1, label='Single residual')
            residual.plot(nu, y[idx]-rec2[idx], color='#0072b2', lw=1, label='Mixture residual')
            residual.set(xlabel='Brillouin shift (GHz)', ylabel='Measured − fit')
            residual.legend(fontsize=7)
        fig.suptitle('Hydrogel | individual pixel fits and residuals', fontsize=15)
        fig.savefig(OUT/f'pixel_examples_{page+1}.png', dpi=170); plt.close(fig)
    np.savez_compressed(OUT/'reconstructions.npz', frequency_ghz=nu, measured=y,
                        single=rec1, mixture=rec2, single_parameters=p1, mixture_parameters=p2,
                        background_peak_normalized=bg, xi=data['xi'], yi=data['yi'])
    summary = dict(pixels=len(y), channels=len(nu), grid=list(data['shape']),
                   median_single_rmse=float(np.median(rmse1)), median_mixture_rmse=float(np.median(rmse2)),
                   median_pixel_rmse_reduction_pct=float(np.median(gain)),
                   global_single_rmse=float(np.sqrt(np.mean((y-rec1)**2))),
                   global_mixture_rmse=float(np.sqrt(np.mean((y-rec2)**2))),
                   median_background_pct=float(np.median(fraction)*100),
                   optimizer_converged=np.array(status).sum(axis=0).tolist())
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    table.iloc[[idx for _,idx in choices]].to_csv(OUT/'example_pixels.csv', index=False)
    (OUT/'REPORT.md').write_text(
        '# Hydrogel reconstruction comparison\n\n'
        'Both models minimize unweighted squared residuals on the same unsmoothed, folded spectra '
        '(Stokes + anti-Stokes; existing pipeline band up to 8 GHz). '
        'Single: A L(nu; center, width) + c. Mixture: A L(nu; center, width) + B EM_bg(nu) + c. '
        'Amplitudes are nonnegative; the constant baseline is free in both models. '
        'Four versus five fitted parameters; multiple starts and the nested single solution are compared.\n\n'
        f'Background source: `{em_path.relative_to(ROOT)}`. No new spatial smoothing is applied.\n\n'
        'Percentages are each component spectral sum divided by the sum of the two component sums, '
        'excluding baseline, within the fitted frequency band. They sum to 100%; they are not volume fractions.\n\n'
        'RMSE = sqrt(mean((measured - reconstructed)^2)) across frequency channels. '
        'The first two maps use an identical colour range. Pixel labels link to the six example panels.\n\n'
        'Lower training residual is expected for an extra component and does not establish recovery of a '
        'noise-free spectrum. The endmember comes from this sample, so this is not independent validation. '
        'Overlapping component peaks can make fitted percentages and centers poorly identifiable.\n\n'
        '```json\n'+json.dumps(summary, indent=2)+'\n```\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
