"""Generate a complete, offline patch-NLPCA walkthrough with every pixel and update."""
from pathlib import Path
import argparse, base64, csv, hashlib, json, platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from brillouin_denoising import DenoiseConfig, reconstruct, extract_patches, reproject_patches, poisson_deviance
from simulator.generate_reconstruction_scenes import make_scene, sample_counts
from analysis.reconstruct_scan import load_input


def shape(c):
    return c / np.maximum(c.sum(axis=-1, keepdims=True), 1e-12)


def score(c, truth, frequency, independent, mask, observed=None):
    result = {}
    if truth is None and observed is not None:
        q, raw = shape(c), shape(observed)
        movement = ((q-raw) @ frequency)*1000
        result.update(shape_change_pct=float(100*np.linalg.norm(q-raw)/np.linalg.norm(raw)),
                      median_centroid_change_mhz=float(np.median(np.abs(movement))),
                      residual_rms=float(np.sqrt(np.mean((c-observed)**2))),
                      total_intensity_change_pct=float(100*(c.sum()/observed.sum()-1)))
    if truth is not None:
        q, t = shape(c), shape(truth)
        result['shape_nrmse'] = float(np.linalg.norm((q-t)[mask])/np.linalg.norm(t[mask]))
        result['centroid_rmse_mhz'] = float(1000*np.sqrt(np.mean(((q-t) @ frequency)[mask]**2)))
        result['count_nrmse'] = float(np.linalg.norm((c-truth)[mask])/np.linalg.norm(truth[mask]))
    if independent is not None:
        result['independent_deviance_per_bin'] = float(poisson_deviance(independent, c).mean())
    return result


def run(args):
    out=args.output
    out.mkdir(parents=True, exist_ok=True)
    truth=independent=scene=None
    if args.input:
        if args.window is None and args.input.suffix.lower()=='.csv':
            import pandas as pd
            frame=pd.read_csv(args.input, usecols=['marker1','marker2'])
            if frame.marker1.nunique()!=1 or frame.marker2.nunique()!=1:
                raise ValueError('CSV fitting markers vary; choose an explicit consistent window')
            args.window=[float(frame.marker1.iloc[0]),float(frame.marker2.iloc[0])]
            if not np.isfinite(args.window).all() or args.window[0]>=args.window[1]:
                raise ValueError('Invalid CSV fitting markers; choose --window explicitly')
        counts, frequency, x, y = load_input(args.input, args.window)
        mask=np.ones(counts.shape[:2], bool)
        source=str(args.input.resolve())
    else:
        scene=make_scene('cell_domains', args.size, args.bins)
        truth=scene['rate']*args.exposure
        counts=sample_counts(scene['rate'], args.exposure, args.seed)
        independent=sample_counts(scene['rate'], args.exposure, args.seed+10000)
        frequency=scene['frequency']; mask=scene['inside']
        x=y=np.arange(args.size)
        source='Synthetic cell_domains; known expected measured spectra including medium and instrument response'
    config=DenoiseConfig(rank=args.rank, patch=args.patch, clusters=args.clusters, iterations=args.iterations, seed=args.seed)
    trace={}
    with threadpool_limits(1):
        final=reconstruct(counts, config, trace=trace)
        controls={name:reconstruct(counts, DenoiseConfig(method=name, rank=args.rank, patch=1 if name=='poisson_spectral' else args.patch, clusters=args.clusters, iterations=args.iterations, seed=args.seed)) for name in ('spatial_mean','global_pca','poisson_spectral')}
    cubes=np.stack([counts]+trace['cubes'])
    names=['Measured detector intensity' if args.input else 'Measured counts']+trace['names']
    h,w,b=counts.shape
    summaries=[]
    for name,c in zip(names,cubes):
        summaries.append(dict(stage=name, **score(c,truth,frequency,independent,mask,counts)))
    comparison=[dict(stage=name, **score(c,truth,frequency,independent,mask,counts)) for name,c in [('raw',counts),('patch_nlpca',final),*controls.items()]]
    for filename, rows in [('stages.csv',summaries),('comparison.csv',comparison)]:
        with (out/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    arrays=dict(counts=counts, frequency=frequency, x=x,y=y, predictions=cubes, stage_names=np.array(names), labels=trace['labels'], final_patches=trace['final_patches'], losses=np.array(trace['losses']))
    if truth is not None: arrays.update(truth=truth,independent=independent,inside=mask)
    np.savez_compressed(out/'walkthrough.npz',**arrays)
    centroids=shape(cubes)@frequency
    lo,hi=np.min(centroids),np.max(centroids)
    selected=[(h//2,w//2),(h//2,w//4),(h//4,w//2),(0,0)]
    for i,(name,c) in enumerate(zip(names,cubes)):
        fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        im=ax[0,0].imshow(centroids[i],vmin=lo,vmax=hi,cmap='viridis');fig.colorbar(im,ax=ax[0,0],label='Centroid (GHz)')
        ax[0,0].set_title('Whole raster: spectral centroid (not BFS)')
        delta=(shape(c)-shape(counts))@frequency*1000
        im=ax[0,1].imshow(delta,cmap='coolwarm',vmin=-150,vmax=150);fig.colorbar(im,ax=ax[0,1],label='Change from raw (MHz); clipped at +/-150')
        ax[0,1].set_title('Change in cell prediction')
        for a,(iy,ix) in zip([ax[0,2],*ax[1]],selected):
            a.plot(frequency,counts[iy,ix],color='#aaaaaa',lw=.6,label='Measured')
            if truth is not None:a.plot(frequency,truth[iy,ix],color='black',lw=1.3,label='Truth')
            a.plot(frequency,c[iy,ix],color='#167e83',lw=1.3,label='Current')
            a.set(xlabel='Frequency (GHz)',ylabel='Recorded intensity / bin' if args.input else 'Counts / bin',title=f'Pixel y={iy}, x={ix}',ylim=(0,max(cubes[:,iy,ix].max(),truth[iy,ix].max() if truth is not None else 0)*1.05))
            a.legend(fontsize=8)
        fig.suptitle(f'{i:02d} - {name}',fontsize=18)
        fig.savefig(out/f'stage_{i:02d}.png',dpi=110);plt.close(fig)
    coverage=np.zeros((h,w))
    for iy in range(h-args.patch+1):
        for ix in range(w-args.patch+1):coverage[iy:iy+args.patch,ix:ix+args.patch]+=1
    fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    ax[0].imshow(counts.sum(-1));ax[0].add_patch(plt.Rectangle((-.5,-.5),args.patch,args.patch,fill=False,edgecolor='red',lw=2));ax[0].set_title('Raw total counts; first patch outlined')
    ax[1].imshow(trace['labels'].reshape(h-args.patch+1,w-args.patch+1),cmap='tab10');ax[1].set_title('Group of each patch (top-left coordinate)')
    im=ax[2].imshow(coverage);fig.colorbar(im,ax=ax[2],label='Number of contributing patches');ax[2].set_title('Overlap averaging weights')
    fig.savefig(out/'preparation.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    ax[0].plot(trace['losses']);ax[0].set(title='Patch training objective',xlabel='Half-update index',ylabel='Poisson loss (constant omitted)')
    residual=(counts-final)/np.sqrt(np.maximum(final,1e-12))
    im=ax[1].imshow(residual.mean(-1),cmap='coolwarm');fig.colorbar(im,ax=ax[1]);ax[1].set_title('Mean residual / sqrt(prediction)' if args.input else 'Mean fitted Pearson residual')
    ax[2].plot(frequency,residual.mean((0,1)));ax[2].axhline(0,color='black',lw=.7);ax[2].set(title='Spectral residual structure',xlabel='Frequency (GHz)',ylabel='Mean residual / sqrt(prediction)')
    fig.savefig(out/'diagnostics.png',dpi=140);plt.close(fig)
    if truth is not None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
        ax[0].plot(range(1,len(names)), [100*r['shape_nrmse'] for r in summaries[1:]], marker='.')
        ax[0].set(xlabel='Recorded stage (raw omitted)', ylabel='Cell shape error (%)', title='Reconstruction error against known truth')
        ax[1].plot(range(1,len(names)), [r['independent_deviance_per_bin'] for r in summaries[1:]], marker='.')
        ax[1].set(xlabel='Recorded stage (raw omitted)', ylabel='Independent deviance / bin', title='Prediction of a new photon acquisition')
        fig.savefig(out/'stage_accuracy.png', dpi=140);plt.close(fig)
    manifest=dict(source=source,config=config.to_dict(),shape=list(counts.shape),exposure_s=args.exposure if scene else None,seed=args.seed,independent_seed=args.seed+10000 if scene else None,python=platform.python_version(),input_sha256=hashlib.sha256(counts.tobytes()).hexdigest(),comparison=comparison,notes='Fixed configuration, not tuned on truth or test photons. Independent acquisition only scored. Cluster iterations aligned for visualization; converged groups frozen. Trace uses full precision; browser uses float32. Rank is not material count.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    payload=dict(source=source, patch=args.patch, final_patches=base64.b64encode(trace['final_patches'].astype('<f4').tobytes()).decode(), names=names,shape=[len(cubes),h,w,b],frequency=frequency.tolist(),x=np.asarray(x).tolist(),y=np.asarray(y).tolist(),lo=float(lo),hi=float(hi),comparison=comparison,
                 cubes=base64.b64encode(cubes.astype('<f4').tobytes()).decode(),truth=base64.b64encode(truth.astype('<f4').tobytes()).decode() if truth is not None else None)
    report=['# Patch reconstruction walkthrough','',f'Source: {source}',f'Cube: {h} x {w} pixels x {b} spectral bins. {len(names)} recorded stages.','', 'Open index.html to select every pixel, move through every update, and inspect all overlapping patch contributions from walkthrough.npz. Each stage also has a standalone PNG.','', '## Results (cell mask for synthetic data; entire raster for input files)','', '| Method | Shape NRMSE | Centroid RMSE (MHz) | Independent deviance / bin |','|---|---:|---:|---:|']
    for row in comparison:
        report.append('| '+row['stage']+' | '+' | '.join(f'{row[k]:.5g}' if k in row else 'Unavailable' for k in ('shape_nrmse','centroid_rmse_mhz','independent_deviance_per_bin'))+' |')
    report+=['','Shape NRMSE is the relative Euclidean error after each spectrum is normalized to unit area. Centroid is a finite-window spectral mean, not a fitted Brillouin shift. Independent deviance is measured across the entire raster. Lower is better; the noise contribution prevents zero even for perfect truth. Raw zero predictions receive a numerical floor in deviance.','', '## Interpretation and room for improvement','', 'This is an instrument-blurred expected measured spectrum, including medium. It does not estimate medium-free spectra or biological constituent fractions. The synthetic result is one scene and one noise seed, not evidence of real-cell accuracy.','', 'The training objective should fall, but truth error and independent prediction need not improve at every update. Overlapping patches reuse photons, so the loss is a composite objective, not a likelihood for independent patches. Patch-to-patch disagreement is a sensitivity diagnostic, not a confidence interval.','', 'Next: choose patch size, rank and cluster count with a separate calibration acquisition and photon split; retain a third acquisition for final evaluation. Repeat across exposure levels and seeds. Check boundary and thin-feature errors separately because global improvement can hide local blurring. Compare spectral-only reconstruction where spatial detail is most important. Calibrate detector gain, offset and read noise before applying Poisson likelihood to real camera intensities. Use bootstrap or repeated acquisitions to estimate reconstruction uncertainty; do not call sqrt(prediction) a reconstruction confidence interval.','', '## Method provenance','', '[Salmon et al., Poisson noise reduction with non-local PCA](https://arxiv.org/abs/1206.0338): non-sparse Poisson log-low-rank patch model, alternating Newton optimization, clustering, overlap averaging. Adaptations here: full-spectrum spatial patches, 16 coarse spectral bands for clustering, log-smoothed SVD initialization, damping and line search. No sparse penalty or reproduction of the paper tables is claimed.','', 'See analysis/RECONSTRUCTION.md for noise assumptions and the earlier multi-scene benchmark.']
    report += ['', '## Optimization budget and predictive plateau', '',
               f'This run allowed {args.iterations} full Newton iterations per group. This is a finite iteration budget, not a claim of full convergence.']
    if independent is not None:
        best = min(summaries[1:], key=lambda r:r['independent_deviance_per_bin'])
        report += [f"The best observed independent score among fitted stages occurred at {best['stage']} ({best['independent_deviance_per_bin']:.6f}), versus {summaries[-1]['independent_deviance_per_bin']:.6f} at the final stage. This is a post-hoc diagnostic, not a tuned stopping rule. Differences need repeat-acquisition testing before treating them as significant."]
    if args.input:
        difference = shape(final)-shape(counts)
        movement = (difference @ frequency)*1000
        real = dict(window_ghz=[float(frequency[0]),float(frequency[-1])],
                    median_absolute_centroid_change_mhz=float(np.median(np.abs(movement))),
                    percentile95_absolute_centroid_change_mhz=float(np.percentile(np.abs(movement),95)),
                    relative_shape_change=float(np.linalg.norm(difference)/np.linalg.norm(shape(counts))),
                    relative_total_intensity_change=float(final.sum()/counts.sum()-1),
                    residual_rms_intensity=float(np.sqrt(np.mean((counts-final)**2))))
        manifest.update(real_data_diagnostics=real, noise_assumption='Exploratory Poisson-shaped objective on recorded detector intensity; gain and offset not verified. No photon thinning or calibrated residual significance.', source_file_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest())
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        report += ['', '## Real-data diagnostics (changes, not accuracy)', '',
                   f"Positive-branch frequency window: {frequency[0]:.4f} to {frequency[-1]:.4f} GHz on the CSV axis; crop only, no folding or baseline subtraction.",
                   f"Median absolute centroid movement: {real['median_absolute_centroid_change_mhz']:.3f} MHz; 95th percentile: {real['percentile95_absolute_centroid_change_mhz']:.3f} MHz.",
                   f"Relative normalized-spectrum change: {100*real['relative_shape_change']:.2f}%; total recorded intensity change: {100*real['relative_total_intensity_change']:.3f}%.",
                   'These quantify how much the reconstruction changed the observation; they do not quantify error reduction. No ground truth, repeated acquisition or validated gain/offset was provided. Integer detector values are not necessarily photon counts. The Poisson-shaped objective is exploratory, and residual/sqrt(prediction) is an uncalibrated display scale. No photon thinning or accuracy ranking is claimed.']
        report = ['# Real-cell patch reconstruction', '', f'Source: {source}',
                  f'Raster: {h} x {w} pixels; {b} original spectral bins; {len(names)} recorded stages.', '',
                  '## Spectral window', '',
                  f'Fitting window: {frequency[0]:.4f} to {frequency[-1]:.4f} GHz. Only original bins are retained; no folding, interpolation or baseline subtraction.',
                  ('For this CSV, the cell-bearing Brillouin peak is near 5.56 GHz. The earlier 8.5-12.5 GHz run selected a different peak and did not capture the cell contrast. Its reported modification statistics should not be interpreted as cell-denoising results.' if args.input.name=='XY_scanning_40um_60um_step1um_10s_fixed.csv' else 'The requested window or the CSV fitting markers select the analysed spectral branch.'), '',
                  '## Measured changes (not accuracy estimates)', '',
                  '| Method | Shape change (%) | Median centroid movement (MHz) | Residual RMS (intensity/bin) | Total intensity change (%) |',
                  '|---|---:|---:|---:|---:|']
        for row in comparison:
            report.append('| '+row['stage']+' | '+' | '.join(f'{row[k]:.5g}' for k in ['shape_change_pct','median_centroid_change_mhz','residual_rms','total_intensity_change_pct'])+' |')
        report += ['', 'Shape change compares unit-area reconstructed spectra with unit-area measured spectra. Centroid movement is the absolute change of the finite-window spectral mean; it is not a fitted BFS error. Residual RMS measures how much recorded intensity was removed or added. Raw has zero change by definition. Lower modification is not necessarily better denoising.', '',
                   '## What the fit can establish', '',
                   f"Patch reconstruction changes normalized spectra by {100*real['relative_shape_change']:.2f}%. Median centroid movement is {real['median_absolute_centroid_change_mhz']:.3f} MHz, with a 95th percentile of {real['percentile95_absolute_centroid_change_mhz']:.3f} MHz. Total intensity changes by {100*real['relative_total_intensity_change']:.4f}%.",
                   'The cell should already be visible in the raw centroid map and remain visible through the iterations. Compare it with window_diagnosis.png: the CSV vendor Frequency map is a contextual reference, not ground truth or an input to the denoiser.', '',
                   '## What remains untested', '',
                   'No clean reference or independent repeat is available, so reconstruction error and independent predictive accuracy cannot be measured. Detector gain and offset are not verified. The Poisson-shaped objective is exploratory; residual/sqrt(prediction) is an uncalibrated plotting scale, not a significance score. No binomial photon splitting was performed.',
                   'Next: validate detector calibration; collect repeat scans; select rank and patch size on separate validation data; quantify boundary and thin-feature preservation. Do not infer medium-free spectra or material fractions from denoising alone.', '',
                   '## Reproduce and inspect', '',
                   'Open index.html for clickable slides, tutorial.html for the full explanation, stages.csv for every update, comparison.csv for method modifications, and walkthrough.npz for original observations and all predictions.',
                   f'Configuration: rank {args.rank}, patch {args.patch} x {args.patch}, {args.clusters} clusters, at most {args.iterations} iterations per cluster, seed {args.seed}. Groups can stop early.', '',
                   'Method: [Salmon et al., Poisson noise reduction with non-local PCA](https://arxiv.org/abs/1206.0338). Full-spectrum patches, coarse spectral grouping, and safeguarded Newton updates are the documented adaptations.']
    (out/'REPORT.md').write_text('\n'.join(report),encoding='utf8')
    payload['diagnostics']=manifest.get('real_data_diagnostics')
    render_pages(out,payload)
    print(json.dumps(manifest,indent=2))


def render_pages(out, payload):
    for template, target in [('patch_walkthrough_template.html', 'tutorial.html'),
                             ('patch_slides_template.html', 'index.html')]:
        html = Path(__file__).with_name(template).read_text(encoding='utf8')
        if payload.get('truth') is None:
            for original,replacement in [('Counts / bin','Recorded intensity / bin'),('Counts per bin','Recorded intensity per bin'),('counts per bin','intensity units per bin'),('Raw total counts','Raw total intensity'),('Total counts','Total intensity')]:
                html=html.replace(original,replacement)
            html=html.replace(' · black: simulated truth', '')
            html=html.replace(' · black: known truth (simulation only)', '')
            html=html.replace('This one simulated cell does not establish real-cell accuracy.', 'This real scan has no known-truth reference or independent repeat.')
            html=html.replace('This is a supplied scan. Ground truth and independent test photons are unavailable; accuracy scores are deliberately omitted.', 'Real detector data: gain and offset are not verified. This is an exploratory Poisson-shaped fit. Ground truth and independent test photons are unavailable; accuracy scores are deliberately omitted.')
            html=html.replace('This is a supplied scan.', 'This is a real detector scan.')
        # Source is inserted as text, never interpreted as markup or JavaScript.
        html=html.replace('draw();\n</script>', "draw();\nif(D.source){const badge=document.createElement('p');badge.textContent=D.source;badge.className='legend';badge.style.overflowWrap='anywhere';document.getElementById('source').after(badge)}\n</script>")
        (out/target).write_text(html.replace('__DATA__', json.dumps(payload).replace('<','\\u003c')), encoding='utf8')


def render_saved(out):
    """Refresh the interactive pages without rerunning any numerical fitting."""
    manifest=json.loads((out/'manifest.json').read_text())
    with np.load(out/'walkthrough.npz', allow_pickle=False) as data:
        cubes=data['predictions']; frequency=data['frequency']
        centroids=shape(cubes) @ frequency
        encode=lambda a:base64.b64encode(a.astype('<f4').tobytes()).decode()
        payload=dict(diagnostics=manifest.get('real_data_diagnostics'), source=manifest['source'], patch=manifest['config']['patch'], final_patches=encode(data['final_patches']),
                     names=data['stage_names'].tolist(), shape=list(cubes.shape),
                     frequency=frequency.tolist(), x=data['x'].tolist(),y=data['y'].tolist(),
                     lo=float(centroids.min()),hi=float(centroids.max()),comparison=manifest['comparison'],
                     cubes=encode(cubes),truth=encode(data['truth']) if 'truth' in data else None)
        render_pages(out,payload)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--render-only',action='store_true',help='Refresh pages from saved walkthrough arrays without refitting')
    p.add_argument('--input',type=Path);p.add_argument('--window',type=float,nargs=2)
    p.add_argument('--output',type=Path,default=Path('outputs/patch_walkthrough'))
    for key,value in [('size',28),('bins',512),('rank',3),('patch',3),('clusters',4),('iterations',12),('seed',7)]:p.add_argument('--'+key,type=int,default=value)
    p.add_argument('--exposure',type=float,default=1)
    args=p.parse_args()
    if args.render_only:
        render_saved(args.output)
    else:
        with threadpool_limits(1):run(args)

if __name__=='__main__':main()
