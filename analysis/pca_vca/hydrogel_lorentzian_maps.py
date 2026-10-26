"""Map the extra Lorentzian retained by the hydrogel intensity gate.

Background-only pixels are N/A in every parameter map and output column.
Intensity is the sum of the fitted Lorentzian counts over retained channels,
excluding background and baseline. Peak height and spectral share are also saved.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=ROOT/'outputs/hydrogel_intensity_gate')
    args=parser.parse_args()
    source=args.input_dir
    out=source/'lorentzian_parameter_maps';out.mkdir(parents=True,exist_ok=True)
    z=np.load(source/'reconstructions.npz')
    table=pd.read_csv(source/'pixel_results.csv')
    summary=json.loads((source/'summary.json').read_text())
    p=z['mixture_parameters'];nu=z['frequency_ghz']
    present=z['lorentzian_present'].astype(bool)
    np.testing.assert_array_equal(present,table.lorentzian_present)
    np.testing.assert_array_equal(z['xi'],table.xi)
    np.testing.assert_array_equal(z['yi'],table.yi)
    assert not np.any(present & (table.decision.to_numpy()=='background_only'))
    peak=p[:,0,None]/(1+((nu[None,:]-p[:,1,None])/p[:,2,None])**2)
    values=dict(center_ghz=p[:,1],integrated_intensity_counts=peak.sum(axis=1),
                fwhm_ghz=2*p[:,2],peak_height_counts=p[:,0],
                spectral_intensity_pct=table.lorentzian_pct.to_numpy())
    values={name:np.where(present,v,np.nan) for name,v in values.items()}
    shape=(int(z['yi'].max())+1,int(z['xi'].max())+1)
    maps={}
    for name,v in values.items():
        a=np.full(shape,np.nan);a[z['yi'],z['xi']]=v;maps[name]=a
        assert np.isnan(v[~present]).all() and np.isfinite(v[present]).all()
    results=table[['pixel','xi','yi','x','y','decision','lorentzian_present']].copy()
    for name,v in values.items():results[name]=v
    results.to_csv(out/'lorentzian_parameters.csv',index=False,na_rep='N/A')
    np.savez_compressed(out/'parameter_maps.npz',**maps,lorentzian_present=present,
                        xi=z['xi'],yi=z['yi'],frequency_ghz=nu)
    absent=int((~present).sum());active=int(present.sum())
    na_color='#dedede'
    def render(specs,filename,caption):
        fig,axes=plt.subplots(1,len(specs),figsize=(5.5*len(specs),6.0),squeeze=False)
        for ax,(key,title,unit,cmap_name) in zip(axes.flat,specs):
            cmap=plt.get_cmap(cmap_name).copy();cmap.set_bad(na_color)
            finite=values[key][present]
            lo,hi=(float(finite.min()),float(finite.max())) if finite.size else (0.,1.)
            if lo==hi:hi=lo+1e-6
            if key=='spectral_intensity_pct':lo,hi=0.,100.
            im=ax.imshow(np.ma.masked_invalid(maps[key]),origin='lower',
                         interpolation='nearest',cmap=cmap,vmin=lo,vmax=hi)
            ax.set(title=title,xlabel='X pixel index',ylabel='Y pixel index')
            ax.title.set_fontsize(14)
            fig.colorbar(im,ax=ax,fraction=.047,pad=.035,label=unit)
        fig.suptitle('Hydrogel: extra fitted Lorentzian after the intensity gate',fontsize=17,y=.985)
        fig.legend(handles=[Patch(facecolor=na_color,edgecolor='#888888',label=f'N/A: background only ({absent} pixels)')],
                   loc='lower center',bbox_to_anchor=(.5,.085),frameon=False,fontsize=12)
        fig.text(.5,.045,caption,ha='center',fontsize=10)
        fig.subplots_adjust(top=.85,bottom=.21,left=.055,right=.97,wspace=.34)
        fig.savefig(out/f'{filename}.png',dpi=200,bbox_inches='tight',pad_inches=.15)
        fig.savefig(out/f'{filename}.svg',bbox_inches='tight',pad_inches=.15)
        plt.close(fig)
    render([('center_ghz','Centre','Brillouin shift (GHz)','viridis'),
            ('integrated_intensity_counts','Integrated intensity','Lorentzian counts summed over fitted channels','magma'),
            ('fwhm_ghz','Linewidth (FWHM)','FWHM (GHz)','cividis')],
           'lorentzian_parameter_maps',
           f'{active} retained pixels; minimum Lorentzian share {100*summary["minimum_lorentzian_fraction"]:g}%.  '
           'Full data ranges; no spatial smoothing. Intensity excludes background and constant baseline.')
    render([('peak_height_counts','Peak height','Fitted Lorentzian amplitude (counts/channel)','magma'),
            ('spectral_intensity_pct','Spectral intensity share','Lorentzian share of component signal (%)','viridis')],
           'lorentzian_intensity_definitions',
           'Peak height is A; spectral share is 100 × sum(L) / [sum(L) + sum(background)]. Baseline excluded.')
    stats={k:dict(min=float(np.nanmin(v)),median=float(np.nanmedian(v)),max=float(np.nanmax(v)))
           for k,v in values.items()} if active else {}
    (out/'README.md').write_text(
        '# Extra Lorentzian parameter maps\n\n'
        'These maps describe the flexible Lorentzian added to the fixed background endmember, '
        'after the existing intensity gate. They do not fit a second independent Lorentzian to the background.\n\n'
        f'{active} pixels retain the Lorentzian and {absent} background-only pixels are N/A. '
        'N/A is grey in plots, literal N/A in the CSV and NaN in the map arrays. '
        'No spatial smoothing or new spectral fit is applied.\n\n'
        '- Centre: fitted Lorentzian peak position in GHz.\n'
        '- Integrated intensity: sum of the Lorentzian component counts over the fitted channels, '
        'excluding the background and constant baseline. This is a channel sum, not multiplied by frequency spacing.\n'
        '- Linewidth: full width at half maximum, FWHM = 2 gamma, in GHz.\n'
        '- Additional intensity figure: fitted peak-height amplitude and integrated spectral share.\n\n'
        'All panels use the full valid data range, with no percentile clipping. Coordinates are zero-based pixel indices. '
        'Retained low-share peaks can still have uncertain parameters; the intensity gate is not an uncertainty estimate.\n\n'
        f'Source: {source.relative_to(ROOT) if source.is_relative_to(ROOT) else source}\n\n'
        '```json\n'+json.dumps(stats,indent=2)+'\n```\n',encoding='utf-8')
    print(json.dumps(dict(retained=active,background_na=absent,statistics=stats),indent=2))


if __name__=='__main__':main()
