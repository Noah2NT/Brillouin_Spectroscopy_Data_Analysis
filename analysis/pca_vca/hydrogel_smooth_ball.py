"""Smooth Lorentzian-share heights on the VCA mask, preserving column footprints."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates
import plotly.graph_objects as go

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/hydrogel_vca2'
table=pd.read_csv(OUT/'hydrogel_lorentzian_3d_vca_mask.csv')
keep=table.retained_vca_background_below_97.to_numpy(dtype=bool)
x=table.xi.to_numpy(); y=table.yi.to_numpy()
nx,ny=int(x.max()+1),int(y.max()+1)
raw=np.zeros((ny,nx))
raw[y[keep],x[keep]]=table.lorentzian_pct.to_numpy()[keep]
# Zero exterior supplies a gentle taper. Only retained footprints are meshed.
sigma=1.2
smooth=gaussian_filter(raw,sigma=sigma,mode='constant',cval=0)
vertices=[]; triangles=[]; colors=[]; labels=[]
width=.92  # Exact width of the preceding bar graph, including its visual gaps.
subdivisions=4
for xx,yy,original in zip(x[keep],y[keep],table.lorentzian_pct.to_numpy()[keep]):
    axis=np.linspace(-width/2,width/2,subdivisions+1)
    gx,gy=np.meshgrid(xx+axis,yy+axis)
    heights=map_coordinates(smooth,[gy.ravel(),gx.ravel()],order=1,mode='constant',cval=0)
    base=len(vertices); n=(subdivisions+1)**2
    for sign in (-1,1):
        vertices.extend(zip(gx.ravel(),gy.ravel(),sign*heights/2))
        colors.extend(heights)
        labels.extend([f'Pixel ({xx}, {yy})<br>Original hydrogel: {original:.2f}%<br>Smoothed height: {hh:.2f}%' for hh in heights])
    for row in range(subdivisions):
        for col in range(subdivisions):
            a=row*(subdivisions+1)+col; b=a+1; c=a+subdivisions+2; d=c-1
            triangles.extend([(base+a,base+c,base+b),(base+a,base+d,base+c),
                              (base+n+a,base+n+b,base+n+c),(base+n+a,base+n+c,base+n+d)])
    # Close each original column footprint without stretching any pixel.
    edge=list(range(subdivisions+1))
    edge += [r*(subdivisions+1)+subdivisions for r in range(1,subdivisions+1)]
    edge += [subdivisions*(subdivisions+1)+c for c in range(subdivisions-1,-1,-1)]
    edge += [r*(subdivisions+1) for r in range(subdivisions-1,0,-1)]
    for a,b in zip(edge,edge[1:]+edge[:1]):
        triangles.extend([(base+a,base+b,base+n+b),(base+a,base+n+b,base+n+a)])
    area=(gx.max()-gx.min())*(gy.max()-gy.min())
    assert np.isclose(area,.92**2)
v=np.array(vertices); t=np.array(triangles)
assert np.isfinite(v).all()
assert len(v)==keep.sum()*2*(subdivisions+1)**2
mesh=go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=t[:,0],j=t[:,1],k=t[:,2],
    intensity=colors,cmin=0,cmax=100,colorscale='Viridis',flatshading=False,
    colorbar=dict(title='Hydrogel %'),text=labels,hovertemplate='%{text}<extra></extra>',
    lighting=dict(ambient=.6,diffuse=.8,roughness=.6),name='Smoothed hydrogel')
fig=go.Figure(mesh)
fig.update_layout(title='Smoothed hydrogel — original pixel footprints preserved',template='plotly_white',
    height=850,margin=dict(l=15,r=30,t=85,b=90),scene=dict(
        xaxis=dict(title='X pixel index',range=[-.5,nx-.5]),
        yaxis=dict(title='Y pixel index',range=[-.5,ny-.5]),
        zaxis=dict(title='Hydrogel %',range=[-50,50]),aspectmode='manual',
        aspectratio=dict(x=1,y=ny/nx,z=.7),camera=dict(eye=dict(x=1.5,y=-1.8,z=1.3))))
fig.add_annotation(text=f'{keep.sum()} VCA-selected pixels; Lorentzian-fit heights smoothed with Gaussian sigma {sigma} pixels.<br>'
    'Each footprint remains 0.92 × 0.92, matching the original columns; small inter-column gaps preserved. Midpoint Z = 0.',
    x=.5,y=-.08,xref='paper',yref='paper',showarrow=False)
fig.write_html(OUT/'hydrogel_smoothed_ball.html',include_plotlyjs=True,config=dict(scrollZoom=True,displaylogo=False))
table['smoothed_center_height_pct']=np.where(keep,smooth[y,x],np.nan)
table.to_csv(OUT/'hydrogel_smoothed_ball.csv',index=False)
(OUT/'SMOOTHING.md').write_text("""# Smoothed hydrogel visualization

Uses exactly the VCA background <97% mask (301 pixels), but heights are the
background-plus-individual-Lorentzian hydrogel shares, not VCA abundances.
Gaussian smoothing with sigma 1.2 pixels acts on the height grid with zero outside
the mask. It rounds local height changes and tapers edges; it does not fit a sphere
or conserve total height/volume. Original spectral results are unchanged.

Each retained pixel retains its original centre and exact 0.92 x 0.92 column
footprint (area 0.8464 pixel-index units squared). No mask expansion or pixel
resizing. The original 0.08 gaps therefore remain: this is a rounded tiled shape,
not a watertight continuous sphere. Each tile is subdivided into 4 x 4 patches;
heights are interpolated bilinearly from the smoothed grid. Upper and lower
surfaces are mirrored about Z=0. Z is spectral percentage, not physical distance.

Run: `python analysis/pca_vca/hydrogel_smooth_ball.py`
""",encoding='utf-8')
print(json.dumps(dict(retained=int(keep.sum()),footprint_area=width**2,sigma=sigma,
                     vertices=len(v),triangles=len(t)),indent=2))
