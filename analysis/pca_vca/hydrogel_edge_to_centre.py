"""Spatially ordered spectral evolution; concentration is not inferred."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/hydrogel_edge_to_centre'
OUT.mkdir(parents=True,exist_ok=True)
source=ROOT/'outputs/hydrogel_background_comparison'
z=np.load(source/'reconstructions.npz')
table=pd.read_csv(source/'pixel_results.csv')
vca=pd.read_csv(ROOT/'outputs/hydrogel_vca2/pixel_abundances.csv')
np.testing.assert_array_equal(table.pixel,vca.pixel)
np.testing.assert_array_equal(table.xi,vca.xi)
np.testing.assert_array_equal(table.yi,vca.yi)
keep=vca.background_pct.to_numpy()<97
x,y=table.xi.to_numpy(),table.yi.to_numpy()
nx,ny=int(x.max()+1),int(y.max()+1)
mask=np.zeros((ny,nx),bool);mask[y,x]=keep
# Padding makes exterior distance explicit even if support meets a scan edge.
distance=distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]
d=distance[y,x]
nu=z['frequency_ghz']; measured=z['measured']
params=z['mixture_parameters']
background=params[:,4,None]*z['background_peak_normalized'][None,:]
baseline=params[:,3,None]
corrected=measured-background-baseline
lorentzian=params[:,0,None]/(1+((nu[None,:]-params[:,1,None])/params[:,2,None])**2)
area=lorentzian.sum(axis=1)
assert (area[keep]>0).all()
scale=np.maximum(area,1e-12)
normalized=corrected/scale[:,None]
single=lorentzian/scale[:,None]; mixture=single
err1=np.sqrt(np.mean((normalized-single)**2,axis=1));err2=err1
np.testing.assert_allclose(corrected-lorentzian,measured-z['mixture'],atol=1e-7)
edges=np.linspace(0,d.max(),7)
group=np.clip(np.searchsorted(edges,d,side='left')-1,0,5)
group[~keep]=-1
order=np.flatnonzero(keep)[np.argsort(d[keep],kind='stable')]
colors=sample_colorscale('Viridis',np.linspace(.08,.92,6))
fig=make_subplots(rows=4,cols=2,vertical_spacing=.075,horizontal_spacing=.13,
    subplot_titles=['Background-subtracted spectra: edge to centre','Mean spectra by distance band (offset for clarity)',
                    'Single Lorentzian centre after background subtraction','Single Lorentzian linewidth after background subtraction',
                    'Normalized fit error (training residual)','Distance inside selected boundary — click a pixel',
                    'Selected pixel: corrected spectrum and single Lorentzian','Selected pixel: residuals'])
fig.add_trace(go.Heatmap(x=nu,y=np.arange(len(order)),z=normalized[order],
    customdata=np.broadcast_to(np.column_stack([order,d[order]])[:,None,:],(len(order),len(nu),2)),
    colorscale='Viridis',colorbar=dict(title='Intensity',x=.44,y=.905,len=.17),
    hovertemplate='Pixel %{customdata[0]}<br>Distance %{customdata[1]:.2f} px<br>Shift %{x:.3f} GHz<br>Intensity %{z:.5f}<extra></extra>'),row=1,col=1)
rows=[];means=[]
for b in range(6):
    selection=keep & (group==b)
    if not selection.any():continue
    mean=normalized[selection].mean(axis=0);means.append(mean)
    shift=b*.018
    label=f'Band {b+1}: {edges[b]:.1f}–{edges[b+1]:.1f} px (n={selection.sum()})'
    fig.add_trace(go.Scatter(x=nu,y=mean+shift,mode='lines',name=label,line=dict(color=colors[b]),
        customdata=mean,hovertemplate=label+'<br>%{x:.3f} GHz<br>Mean intensity %{customdata:.5f}<extra></extra>'),row=1,col=2)
    record=dict(band=b+1,distance_lower=edges[b],distance_upper=edges[b+1],n=int(selection.sum()),
                mean_distance=float(d[selection].mean()))
    for key,values in [('single_center',table.single_center_ghz.to_numpy()),
                       ('mixture_center',table.mixture_center_ghz.to_numpy()),
                       ('single_fwhm',table.single_fwhm_ghz.to_numpy()),
                       ('mixture_fwhm',table.mixture_fwhm_ghz.to_numpy()),
                       ('single_normalized_rmse',err1),('mixture_normalized_rmse',err2)]:
        ok=table.single_converged.to_numpy() if key.startswith('single') else table.mixture_converged.to_numpy()
        vals=values[selection & ok]
        record[key+'_n']=len(vals)
        for name,q in [('q25',.25),('median',.5),('q75',.75)]:
            record[key+'_'+name]=float(np.quantile(vals,q)) if len(vals) else np.nan
    rows.append(record)
stats=pd.DataFrame(rows)
for name,color,prefix in [('Single Lorentzian after subtraction','#2563eb','mixture')]:
    ok=table.single_converged.to_numpy() if prefix=='single' else table.mixture_converged.to_numpy()
    for key,values,row,col in [('center',table[prefix+'_center_ghz'].to_numpy(),2,1),
                              ('fwhm',table[prefix+'_fwhm_ghz'].to_numpy(),2,2),
                              ('normalized_rmse',err1 if prefix=='single' else err2,3,1)]:
        use=keep & ok
        fig.add_trace(go.Scatter(x=d[use],y=values[use],mode='markers',marker=dict(color=color,size=4,opacity=.22),
            name=name+' individual pixels',showlegend=False,customdata=table.pixel.to_numpy()[use],
            hovertemplate='Pixel %{customdata}<br>Distance %{x:.2f} px<br>Value %{y:.5f}<extra></extra>'),row=row,col=col)
        k=prefix+'_'+key
        fig.add_trace(go.Scatter(x=stats.mean_distance,y=stats[k+'_median'],name=name,
            mode='lines+markers',line=dict(color=color),showlegend=(key=='center'),
            error_y=dict(type='data',symmetric=False,array=stats[k+'_q75']-stats[k+'_median'],
                         arrayminus=stats[k+'_median']-stats[k+'_q25']),
            hovertemplate='Distance %{x:.2f} px<br>Median %{y:.5f}<extra></extra>'),row=row,col=col)
spatial=np.full((ny,nx),np.nan);spatial[y[keep],x[keep]]=d[keep]
pixelgrid=np.full((ny,nx),-1);pixelgrid[y,x]=table.pixel
map_trace=len(fig.data)
fig.add_trace(go.Heatmap(z=spatial,customdata=pixelgrid,colorscale='Viridis',hoverongaps=False,
    colorbar=dict(title='Distance<br>(pixels)',x=1.0,y=.365,len=.16),
    hovertemplate='Pixel %{customdata}<br>X %{x}, Y %{y}<br>Inside distance %{z:.2f} px<extra></extra>'),row=3,col=2)
selected=int(order[len(order)//2]); dynamic=[]
for name,color,values,row,col in [('Background-subtracted data','#6b7280',normalized,4,1),
                                ('Single Lorentzian','#2563eb',single,4,1),
                                ('Residual','#2563eb',normalized-single,4,2)]:
    dynamic.append(len(fig.data))
    fig.add_trace(go.Scatter(x=nu,y=values[selected],name=name,mode='lines',line=dict(color=color),
                            showlegend=False),row=row,col=col)
for row,col in [(1,1),(1,2),(4,1),(4,2)]:fig.update_xaxes(title_text='Brillouin shift (GHz)',row=row,col=col)
fig.update_yaxes(title_text='Pixel rank (edge → centre)',row=1,col=1)
fig.update_yaxes(title_text='Normalized corrected intensity + offset',row=1,col=2)
for row,col in [(2,1),(2,2),(3,1)]:fig.update_xaxes(title_text='Distance inside boundary (pixels)',row=row,col=col)
fig.update_yaxes(title_text='Peak centre (GHz)',row=2,col=1)
fig.update_yaxes(title_text='FWHM (GHz)',row=2,col=2)
fig.update_yaxes(title_text='RMSE / fitted Lorentzian area',row=3,col=1)
fig.update_xaxes(title_text='X pixel index',row=3,col=2)
fig.update_yaxes(title_text='Y pixel index',scaleanchor='x6',scaleratio=1,row=3,col=2)
fig.update_yaxes(title_text='Normalized intensity',row=4,col=1)
fig.update_yaxes(title_text='Measured − fit',row=4,col=2)
fig.update_layout(title=f'Hydrogel spectral evolution: edge → centre | {keep.sum()} pixels',height=1650,
    template='plotly_white',margin=dict(l=80,r=95,t=145,b=100),legend=dict(orientation='h',y=1.07,font=dict(size=10)))
fig.add_annotation(text='Spatial distance is not polymer concentration. Mask: VCA background <97%; ordering is spatial.<br>'
    'Bands average unit-sum measured spectra, without baseline subtraction. Error bars show pixel quartiles, not confidence intervals.<br>'
    'Blue: single fit; red: mixture fit. Click the spatial map to inspect spectra and residuals; no smoothing applied.',
    x=.5,y=-.07,xref='paper',yref='paper',showarrow=False)
payload=dict(n=normalized.tolist(),s=single.tolist(),m=mixture.tolist(),d=d.tolist(),
             x=x.tolist(),y=y.tolist(),single_ok=table.single_converged.tolist(),mixed_ok=table.mixture_converged.tolist())
post="""const data=PAYLOAD; const gd=document.getElementById('{plot_id}');
gd.on('plotly_click',function(e){const p=e.points[0];if(p.curveNumber!==MAPTRACE)return;
const i=p.customdata;if(i<0)return;
Plotly.restyle(gd,{y:[data.n[i],data.s[i],data.n[i].map((v,j)=>v-data.s[i][j])]},TRACES);
Plotly.relayout(gd,{'annotations[6].text':'Pixel '+i+' ('+data.x[i]+', '+data.y[i]+'), distance '+data.d[i].toFixed(2)+' px',
'annotations[7].text':'Corrected data minus Lorentzian | fit converged: '+data.mixed_ok[i]});});
""".replace('PAYLOAD',json.dumps(payload)).replace('MAPTRACE',str(map_trace)).replace('TRACES',json.dumps(dynamic))
# Stack panels so the dashboard remains readable in a narrow browser pane.
# Axis and trace IDs stay unchanged, preserving click-to-inspect behavior.
panel_gap = .035
panel_height = (1 - 7*panel_gap) / 8
for panel in range(8):
    suffix = '' if panel == 0 else str(panel+1)
    top = 1 - panel*(panel_height+panel_gap)
    bottom = max(0., top-panel_height)
    fig.layout['xaxis'+suffix].update(domain=[0,.88],automargin=True,title_standoff=12)
    fig.layout['yaxis'+suffix].update(domain=[bottom,top],automargin=True,title_standoff=12)
    fig.layout.annotations[panel].update(x=0,y=top+.007,xanchor='left',yanchor='bottom',
                                          font=dict(size=14))
fig.layout.yaxis6.scaleanchor=None
fig.layout.yaxis6.scaleratio=None
fig.data[0].colorbar.update(x=.92,y=1-panel_height/2,len=panel_height,thickness=12,title=dict(text='Intensity',side='right'))
map_top = 1-5*(panel_height+panel_gap)
fig.data[map_trace].colorbar.update(x=.92,y=map_top-panel_height/2,len=panel_height,
                                   thickness=12,title=dict(text='Distance (px)',side='right'))
for trace in fig.data:
    if trace.name and trace.name.startswith('Band '):
        trace.showlegend=False
fig.update_layout(title=dict(text='Hydrogel spectra: edge to centre',x=.03,y=.995,font=dict(size=21)),
    height=3400,margin=dict(l=85,r=85,t=160,b=180),
    legend=dict(orientation='h',x=0,y=1.03,yanchor='bottom',font=dict(size=11)))
fig.layout.annotations[-1].update(y=-.035,x=0,xanchor='left',align='left',font=dict(size=11),
    text='Distance is spatial, not polymer concentration. Mask: VCA background <97%.<br>'
         'Bands: edge (purple) to centre (yellow). Normalize by fitted Lorentzian channel sum.<br>'
         'Background and constant baseline subtracted. Error bars: pixel quartiles.<br>'
         'Click the distance map to update the final two panels. No spectral smoothing.')
fig.layout.annotations[6].text=f'Pixel {selected} ({x[selected]}, {y[selected]}), distance {d[selected]:.2f} px'
fig.layout.annotations[7].text='Corrected data minus single Lorentzian'
assert int(stats.n.sum()) == int(keep.sum()) == 301
assert np.all(np.diff(d[order]) >= 0)
np.testing.assert_allclose(single[keep].sum(axis=1), 1)
# HTML header sits outside Plotly, so text and legend cannot overlap panels.
fig.update_layout(title=None,showlegend=False,margin=dict(l=85,r=85,t=45,b=180))
body=fig.to_html(full_html=False,include_plotlyjs=True,post_script=post,
                 config=dict(displaylogo=False,scrollZoom=True))
header="""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Background-subtracted hydrogel spectra</title><style>
body{margin:0;font-family:Arial,sans-serif;color:#172033;background:white}
header{padding:24px 30px 16px;border-bottom:1px solid #ddd}h1{font-size:24px;margin:0 0 12px}
p{line-height:1.5;margin:8px 0;max-width:1000px}.legend{display:flex;flex-wrap:wrap;gap:12px 28px;font-size:14px;padding:10px 0}
.key{display:inline-block;width:25px;height:3px;vertical-align:middle;margin-right:8px}
</style></head><body><header><h1>Hydrogel spectra after background subtraction</h1>
<p>301 pixels, ordered from edge to centre. The fitted background spectrum and constant baseline are subtracted from each measured spectrum. Only the remaining single Lorentzian is shown.</p>
<div class="legend"><span><i class="key" style="background:#6b7280"></i>Background-subtracted data</span>
<span><i class="key" style="background:#2563eb"></i>Single Lorentzian / residual</span>
<span><i class="key" style="background:linear-gradient(90deg,#482878,#21918c,#d8e219)"></i>Distance bands: edge → centre</span></div>
<p>Click a pixel in the distance map to update the final two plots. Spectra are normalized by the fitted Lorentzian area; negative residual noise is retained.</p></header>"""
html=header+body+'</body></html>'
for filename in ('edge_to_centre.html','edge_to_centre_subtracted.html'):
    (OUT/filename).write_text(html,encoding='utf-8')
table['retained_spatial_mask']=keep;table['distance_inside_pixels']=np.where(keep,d,np.nan)
table['distance_band']=np.where(keep,group+1,0);table['background_subtracted_normalized_rmse']=err2
display_stats=stats[[c for c in stats.columns if not c.startswith('single_')]].rename(columns=lambda c:c.replace('mixture_','corrected_lorentzian_'))
display_stats.to_csv(OUT/'distance_band_summary.csv',index=False);table.to_csv(OUT/'pixel_metrics.csv',index=False)
np.savez_compressed(OUT/'spectral_evolution.npz',frequency_ghz=nu,normalized_measured=normalized,
    selected_pixel_order=order,distance=d,mask=mask,band=group,band_mean_spectra=np.array(means))
(OUT/'README.md').write_text("""# Background-subtracted edge-to-centre spectra

301 pixels selected by VCA background <97%, ordered by Euclidean distance inside
the mask and grouped in six distance bands. Spatial distance is not concentration.

Corrected data = measured counts - fitted scaled background - fitted constant baseline.
The single displayed Lorentzian is the component from the saved joint fit, not an
independent refit: its amplitude, centre and width were fitted per pixel alongside
the background scale and constant baseline. No full mixture curves are displayed.
The corrected residual equals the original joint-fit residual (verified numerically).

Normalize corrected data and its Lorentzian by that Lorentzian's channel sum.
This preserves negative noise and avoids normalizing by a noisy residual sum.
Band means weight pixels equally. Peak and width summaries exclude nonconverged
fits; error bars show quartiles, not confidence intervals. No smoothing is applied.
The subtraction depends on the fit model, and cannot independently establish a
pure hydrogel spectrum or polymer concentration.

An external HTML header holds the legend, outside the plot title/axis layout.
Reproduce: python analysis/pca_vca/hydrogel_edge_to_centre.py
""",encoding='utf-8')
print(display_stats[['band','n','corrected_lorentzian_center_median','corrected_lorentzian_fwhm_median']].to_string(index=False))
print('Saved interactive dashboard:',OUT/'edge_to_centre.html')
