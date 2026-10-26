"""Suppress negligible Lorentzian components in the saved hydrogel fits.

Run with --min-lorentzian-fraction 0.20 (default) to require 20% of the
integrated component signal, excluding the fitted constant baseline.
This is a model-selection gate, not a lower bound forcing peaks into existence.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hydrogel_background_comparison import components, predict

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/hydrogel_background_comparison'


def apply_intensity_gate(nu, y, bg, mixture_params, minimum=0.20):
    """Return selected parameters and gate diagnostics; never clamp a tiny peak up.

    An absent peak has amplitude zero. Its centre/width must be masked in
    scientific outputs because the placeholders in the parameter array carry
    no physical information. Every subthreshold component is removed even if
    a single Lorentzian could achieve a smaller residual.
    """
    if not 0 < minimum <= 1:
        raise ValueError('minimum must be in (0, 1]')
    peak, back, _ = components(nu, mixture_params, bg)
    fraction = peak.sum() / max(peak.sum() + back.sum(), 1e-12)
    if fraction >= minimum:
        return mixture_params.copy(), 'mixture_retained', fraction
    # Exact least-squares solution for B*bg+c with B >= 0 and c free.
    centered_bg = bg - bg.mean()
    variance = centered_bg @ centered_bg
    if variance <= 0:
        raise ValueError('background must not be constant')
    b = max(float(centered_bg @ (y-y.mean()) / variance), 0.0)
    c = float(y.mean()-b*bg.mean())
    pbg = np.array([0., nu.mean(), np.ptp(nu)/2, c, b])
    return pbg, 'background_only', fraction


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--min-lorentzian-fraction',type=float,default=.20)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/hydrogel_intensity_gate')
    args=parser.parse_args()
    if not 0 < args.min_lorentzian_fraction <= 1:
        parser.error('--min-lorentzian-fraction must be in (0, 1]')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    z=np.load(SOURCE/'reconstructions.npz')
    original=pd.read_csv(SOURCE/'pixel_results.csv')
    nu,y,bg=z['frequency_ghz'],z['measured'],z['background_peak_normalized']
    params=[];decisions=[]
    for i in range(len(y)):
        p,decision,_=apply_intensity_gate(nu,y[i],bg,z['mixture_parameters'][i],args.min_lorentzian_fraction)
        params.append(p);decisions.append(decision)
    params=np.array(params);decisions=np.array(decisions)
    rec=np.array([predict(nu,p,bg) for p in params])
    rmse=np.sqrt(np.mean((y-rec)**2,axis=1))
    old=np.sqrt(np.mean((y-z['mixture'])**2,axis=1))
    single=np.sqrt(np.mean((y-z['single'])**2,axis=1))
    signal=np.array([components(nu,p,bg)[0].sum() for p in params])
    background=params[:,4]*bg.sum()
    fraction=signal/np.maximum(signal+background,1e-12)
    active=params[:,0]>0
    assert np.all(~active | (fraction >= args.min_lorentzian_fraction-1e-12))
    assert np.isfinite(rmse).all()
    assert np.all(params[~active,0]==0)
    table=original.copy()
    table['previous_lorentzian_pct']=original.lorentzian_pct
    table['previous_mixture_rmse']=old
    table['decision']=decisions
    table['lorentzian_present']=active
    table['lorentzian_pct']=fraction*100
    table['background_pct']=(1-fraction)*100
    table['mixture_rmse']=rmse
    table['rmse_reduction_pct']=100*(1-rmse/single)
    table['rmse_increase_vs_ungated']=rmse-old
    table['mixture_center_ghz']=np.where(active,params[:,1],np.nan)
    table['mixture_fwhm_ghz']=np.where(active,2*params[:,2],np.nan)
    table['mixture_baseline']=params[:,3]
    table['mixture_converged']=np.where(decisions=='mixture_retained',original.mixture_converged,True)
    table.to_csv(out/'pixel_results.csv',index=False)
    examples=pd.read_csv(SOURCE/'example_pixels.csv').pixel.astype(int).tolist()
    table.iloc[examples].to_csv(out/'example_pixels.csv',index=False)
    np.savez_compressed(out/'reconstructions.npz',frequency_ghz=nu,measured=y,single=z['single'],
                        previous_mixture=z['mixture'],mixture=rec,single_parameters=z['single_parameters'],
                        mixture_parameters=params,lorentzian_present=active,
                        background_peak_normalized=bg,xi=z['xi'],yi=z['yi'])
    shape=(int(z['yi'].max()+1),int(z['xi'].max()+1))
    def grid(v):
        a=np.full(shape,np.nan);a[z['yi'],z['xi']]=v;return a
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    panels=[(single,'Single Lorentzian RMSE','magma',0,single.max()),
            (old,'Previous unrestricted mixture RMSE','magma',0,single.max()),
            (rmse,'With minimum intensity gate: RMSE','magma',0,single.max()),
            (100*fraction,'Retained Lorentzian intensity (%)','viridis',0,100),
            ((decisions=='background_only').astype(float),'Background-only model selected','Greys',0,1),
            (rmse-old,'RMSE change from previous mixture','coolwarm',-max(abs(rmse-old)),max(abs(rmse-old)))]
    for ax,(v,title,cmap,lo,hi) in zip(axes.flat,panels):
        im=ax.imshow(grid(v),origin='lower',cmap=cmap,vmin=lo,vmax=hi)
        ax.set(title=title,xlabel='X pixel index',ylabel='Y pixel index');fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle(f'Hydrogel: Lorentzian must contribute at least {100*args.min_lorentzian_fraction:g}% or be absent',fontsize=17)
    fig.savefig(out/'error_maps.png',dpi=170);plt.close(fig)
    for page in range(2):
        fig,axes=plt.subplots(2,3,figsize=(16,7),height_ratios=[2,1],layout='constrained')
        for col,i in enumerate(examples[page*3:page*3+3]):
            ax,res=axes[:,col]
            ax.plot(nu,y[i],'.',color='.5',ms=3,label='Measured')
            ax.plot(nu,z['mixture'][i],color='#d55e00',label='Previous mixture')
            ax.plot(nu,rec[i],color='#0072b2',label='After intensity gate')
            peak,back,base=components(nu,params[i],bg)
            ax.plot(nu,peak,'--',color='#009e73',label='Retained Lorentzian')
            ax.plot(nu,back,'--',color='#cc79a7',label='Background component')
            ax.set_title(f'Pixel {i}: {decisions[i]}\nLorentzian {100*fraction[i]:.2f}% | RMSE {old[i]:.2f} to {rmse[i]:.2f}',fontsize=10)
            ax.set_ylabel('Counts');ax.legend(fontsize=7)
            res.axhline(0,color='.7',lw=.7)
            res.plot(nu,y[i]-z['mixture'][i],color='#d55e00',label='Previous residual')
            res.plot(nu,y[i]-rec[i],color='#0072b2',label='Gated residual')
            res.set(xlabel='Brillouin shift (GHz)',ylabel='Measured - fit');res.legend(fontsize=7)
        fig.savefig(out/f'pixel_examples_{page+1}.png',dpi=170);plt.close(fig)
    summary=dict(pixels=len(y),minimum_lorentzian_fraction=args.min_lorentzian_fraction,
                 decision_counts=pd.Series(decisions).value_counts().to_dict(),
                 median_single_rmse=float(np.median(single)),median_previous_mixture_rmse=float(np.median(old)),
                 median_gated_rmse=float(np.median(rmse)),median_reduction_pct=float(np.median(100*(1-rmse/single))),
                 median_rmse_increase_on_suppressed=float(np.median((rmse-old)[decisions=='background_only'])) if np.any(decisions=='background_only') else None,
                 active_subchannel_lorentzians=int(np.sum(active & (2*params[:,2]<np.diff(nu).mean()))),
                 nonconverged=int((~table.mixture_converged).sum()))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'REPORT.md').write_text('# Hydrogel minimum-intensity gate\n\n'
        f'Lorentzian integrated intensity must be at least {100*args.min_lorentzian_fraction:g}% of Lorentzian + background intensity, excluding constant baseline. '
        'The threshold is a configurable modelling choice, not a calibrated detection limit.\n\n'
        'Starting from the saved unrestricted fits, components below threshold are removed. '
        'Background amplitude (nonnegative) and constant baseline (free) are then refitted by exact linear least squares. '
        'Subthreshold pixels remain background-only even if a single Lorentzian could fit better. '
        'Components at or above threshold retain their original fit. No component is forced up to the threshold. '
        'Absent Lorentzian centres and widths are NaN in the CSV and explicitly masked in the NPZ.\n\n'
        'The full background amplitude and constant offset may still adapt to each pixel. '
        'The gate stops an extra weak Lorentzian from polishing background residuals. '
        'It can sacrifice some in-sample fit quality and is not independent evidence of denoising. '
        'It is a post-fit model-selection rule, not a globally reoptimised fit under a disjoint amplitude constraint.\n\n'
        'Source: outputs/hydrogel_background_comparison/reconstructions.npz and pixel_results.csv.\n\n'
        '```json\n'+json.dumps(summary,indent=2)+'\n```\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
