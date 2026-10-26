"""Anchored VCA(2), fixed saved background, interactive abundance columns."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from background_endmember import vca_anchored

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/hydrogel_vca2'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'outputs/hydrogel_background_comparison'
    z = np.load(source / 'reconstructions.npz')
    table = pd.read_csv(source / 'pixel_results.csv')
    nu = z['frequency_ghz']
    measured = z['measured']
    spectra = measured / measured.sum(axis=1, keepdims=True)
    saved = pd.read_csv(ROOT / 'outputs/background_endmember/Hydrogl1_XYscan_20000_20000_step500um_10s (1)_background_endmember.csv')
    np.testing.assert_allclose(nu, saved.shift_ghz)
    bg = saved.endmember_norm.to_numpy()
    bg = bg / bg.sum()
    em, indices = vca_anchored(spectra, 2, bg, seed=42)
    np.testing.assert_array_equal(em[0], bg)
    # Exact nonnegative, sum-to-one least squares for two endmembers.
    direction = em[1] - em[0]
    denominator = direction @ direction
    assert denominator > 1e-15
    fraction = np.clip((spectra - em[0]) @ direction / denominator, 0, 1)
    abundance = np.column_stack([1-fraction, fraction])
    reconstruction = abundance @ em
    rmse = np.sqrt(np.mean((spectra-reconstruction)**2, axis=1))
    np.testing.assert_allclose(abundance.sum(axis=1), 1)
    assert np.isfinite(abundance).all()
    x, y = z['xi'], z['yi']
    np.testing.assert_array_equal(x, table.xi)
    np.testing.assert_array_equal(y, table.yi)
    nx, ny = int(x.max()+1), int(y.max()+1)
    h = 100*fraction
    mask = np.zeros((ny,nx), dtype=bool)
    mask[y,x] = h >= 12
    pad = np.pad(mask,1)
    neighbours = sum(pad[1+dy:1+dy+ny,1+dx:1+dx+nx].astype(int)
                     for dy in (-1,0,1) for dx in (-1,0,1) if dx or dy)
    keep = mask[y,x] & (neighbours[y,x] > 0)
    result = table[['pixel','xi','yi','x','y']].copy()
    result['background_pct'] = 100-h
    result['hydrogel_vca_pct'] = h
    result['normalized_rmse'] = rmse
    result['retained_12pct'] = keep
    result.to_csv(OUT/'pixel_abundances.csv',index=False)
    pd.DataFrame({'frequency_ghz':nu,'fixed_background':em[0],'vca_second_endmember':em[1]}).to_csv(OUT/'endmembers.csv',index=False)
    np.savez_compressed(OUT/'vca2_results.npz',endmembers=em,indices=indices,abundances=abundance,
                        frequency_ghz=nu,xi=x,yi=y,rmse=rmse,retained_12pct=keep)
    faces = np.array([(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
                      (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)])
    def mesh(selection, visible, heights=None, label="Hydrogel VCA"):
        heights = h if heights is None else heights
        vertices, triangles, values, texts = [], [], [], []
        for xx,yy,hh in zip(x[selection],y[selection],heights[selection]):
            offset = len(vertices)
            vertices.extend([(xx+dx,yy+dy,zz) for zz in (-hh/2,hh/2)
                             for dx,dy in [(-.46,-.46),(.46,-.46),(.46,.46),(-.46,.46)]])
            triangles.extend(faces+offset)
            values.extend([hh]*8)
            texts.extend([f'X: {xx}, Y: {yy}<br>{label}: {hh:.2f}%<br>Background: {100-hh:.2f}%']*8)
        v = np.asarray(vertices).reshape(-1,3)
        t = np.asarray(triangles,dtype=int).reshape(-1,3)
        return go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=t[:,0],j=t[:,1],k=t[:,2],
                         intensity=values,cmin=0,cmax=100,colorscale='Viridis',flatshading=True,
                         colorbar=dict(title='Hydrogel %',x=.59,len=.75),text=texts,
                         hovertemplate='%{text}<extra></extra>',visible=visible,showlegend=False)
    fig = make_subplots(rows=2,cols=2,specs=[[{'type':'scene','rowspan':2},{'type':'xy'}],[None,{'type':'xy'}]],
                        column_widths=[.63,.37],horizontal_spacing=.12,vertical_spacing=.17,
                        subplot_titles=['Centred VCA abundance columns','Two endmember spectra','Hydrogel abundance map'])
    fig.add_trace(mesh(h>0,False),row=1,col=1)
    fig.add_trace(mesh(keep,True),row=1,col=1)
    for i,name in enumerate(['EM1: fixed background','EM2: VCA-selected hydrogel candidate']):
        fig.add_trace(go.Scatter(x=nu,y=em[i],name=name,mode='lines'),row=1,col=2)
    for selection,visible in [(np.ones(len(h),bool),False),(keep,True)]:
        grid = np.full((ny,nx),np.nan)
        grid[y[selection],x[selection]] = h[selection]
        fig.add_trace(go.Heatmap(z=grid,zmin=0,zmax=100,colorscale='Viridis',showscale=False,
                                visible=visible,hovertemplate='X: %{x}, Y: %{y}<br>Hydrogel VCA: %{z:.2f}%<extra></extra>'),row=2,col=2)
    fig.update_layout(title='Hydrogel: anchored VCA with 2 endmembers',template='plotly_white',
        height=850,margin=dict(l=20,r=25,t=135,b=80),legend=dict(orientation='h',x=.65,y=1.12),
        scene=dict(xaxis_title='X pixel index',yaxis_title='Y pixel index',
                   zaxis=dict(title='Hydrogel %',range=[-50,50]),
                   aspectmode='manual',aspectratio=dict(x=1,y=ny/nx,z=.7),
                   camera=dict(eye=dict(x=1.5,y=-1.8,z=1.3))),
        updatemenus=[dict(type='buttons',direction='right',x=0,y=1.12,buttons=[
            dict(label='All pixels',method='update',args=[{'visible':[True,False,True,True,True,False]}]),
            dict(label=f'12% + isolated removal ({keep.sum()} pixels)',method='update',args=[{'visible':[False,True,True,True,False,True]}])])])
    fig.update_xaxes(title_text='Frequency shift (GHz)',row=1,col=2)
    fig.update_yaxes(title_text='Unit-sum intensity',row=1,col=2)
    fig.update_xaxes(title_text='X pixel index',range=[-.5,nx-.5],row=2,col=2)
    fig.update_yaxes(title_text='Y pixel index',range=[-.5,ny-.5],row=2,col=2)
    fig.add_annotation(text='Fixed saved background + VCA-selected measured spectrum; sum-to-one constrained abundances.<br>'
                       'EM2 is a spectral endpoint, not proof of pure hydrogel. No Lorentzian fitting or smoothing.',
                       x=.5,y=-.1,xref='paper',yref='paper',showarrow=False)
    fig.write_html(OUT/'hydrogel_vca2.html',include_plotlyjs=True,config=dict(scrollZoom=True,displaylogo=False))

    # Separate full-map view: every measured pixel, no display filtering.
    full = make_subplots(rows=1, cols=2, subplot_titles=['EM1: fixed background (%)', 'EM2: hydrogel VCA abundance (%)'], horizontal_spacing=.14)
    for col, values in enumerate((100-h, h), start=1):
        grid = np.full((ny, nx), np.nan)
        grid[y, x] = values
        assert np.isfinite(grid).sum() == len(h)
        full.add_trace(go.Heatmap(z=grid, zmin=0, zmax=100, colorscale='Viridis',
            colorbar=dict(title='%', x=.43 if col == 1 else 1.0, len=.85),
            hovertemplate='X: %{x}, Y: %{y}<br>Abundance: %{z:.2f}%<extra></extra>'), row=1, col=col)
        full.update_xaxes(title_text='X pixel index', range=[-.5,nx-.5], row=1, col=col)
        full.update_yaxes(title_text='Y pixel index', range=[-.5,ny-.5],
                         scaleanchor='x' if col == 1 else 'x2', scaleratio=1, row=1, col=col)
    full.update_layout(title=f'Full anchored VCA(2) abundance maps — all {len(h)} pixels',
                       template='plotly_white', height=720, margin=dict(l=60,r=70,t=100,b=75))
    full.add_annotation(text='No pixel gating, threshold, isolated-pixel removal, or smoothing. Both abundances sum to 100% at each pixel.',
                        x=.5,y=-.13,xref='paper',yref='paper',showarrow=False)
    full.write_html(OUT/'hydrogel_vca2_full_map.html', include_plotlyjs=True,
                    config=dict(scrollZoom=True,displaylogo=False))

    background_below_97 = (100-h) < 97
    for trace, values in zip(full.data, (100-h, h)):
        filtered_grid = np.full((ny, nx), np.nan)
        filtered_grid[y[background_below_97], x[background_below_97]] = values[background_below_97]
        trace.z = filtered_grid
        trace.hoverongaps = False
        assert np.isfinite(filtered_grid).sum() == background_below_97.sum()
    full.update_layout(title=f'Anchored VCA(2): background <97% — {background_below_97.sum()} pixels retained')
    full.layout.annotations[-1].text = 'Pixels with background abundance ≥97% removed from both maps. No other filtering or smoothing.'
    full.write_html(OUT/'hydrogel_vca2_background_below_97.html', include_plotlyjs=True,
                    config=dict(scrollZoom=True,displaylogo=False))
    columns = go.Figure(mesh(background_below_97, True))
    columns.data[0].colorbar = dict(title='Hydrogel %', len=.8)
    columns.update_layout(
        title=f'VCA hydrogel 3D map — exactly {background_below_97.sum()} retained pixels (background <97%)',
        template='plotly_white', height=850, margin=dict(l=20,r=40,t=90,b=70),
        scene=dict(xaxis=dict(title='X pixel index',range=[-.5,nx-.5]),
                   yaxis=dict(title='Y pixel index',range=[-.5,ny-.5]),
                   zaxis=dict(title='Hydrogel %',range=[-50,50]),
                   aspectmode='manual',aspectratio=dict(x=1,y=ny/nx,z=.7),
                   camera=dict(eye=dict(x=1.5,y=-1.8,z=1.3))))
    columns.add_annotation(text='Same pixel mask as the 97% cutoff maps. Column height = VCA hydrogel abundance; midpoint = Z 0. No additional filtering or smoothing.',
                           x=.5,y=-.07,xref='paper',yref='paper',showarrow=False)
    assert len(columns.data[0].x) == 8 * background_below_97.sum()
    columns.write_html(OUT/'hydrogel_vca2_3d_background_below_97.html',include_plotlyjs=True,
                       config=dict(scrollZoom=True,displaylogo=False))
    # VCA selects spatial support only; heights use the independent flexible Lorentzian fits.
    fitted_hydrogel = table.lorentzian_pct.to_numpy()
    assert np.isfinite(fitted_hydrogel).all()
    hybrid = go.Figure(mesh(background_below_97, True, fitted_hydrogel, 'Hydrogel (Lorentzian fit)'))
    hybrid.update_layout(columns.layout)
    hybrid.data[0].colorbar = dict(title='Hydrogel %', len=.8)
    hybrid.update_layout(title=f'Hydrogel % from per-pixel Lorentzian fits — {background_below_97.sum()} pixels selected using VCA')
    hybrid.layout.annotations[0].text = 'VCA background <97% selects pixels only. Height = Lorentzian share from background + individual Lorentzian fits. Centred at Z = 0; no smoothing.'
    assert len(hybrid.data[0].x) == 8 * background_below_97.sum()
    np.testing.assert_allclose(np.asarray(hybrid.data[0].intensity).reshape(-1,8)[:,0], fitted_hydrogel[background_below_97])
    hybrid.write_html(OUT/'hydrogel_lorentzian_3d_vca_mask.html',include_plotlyjs=True,
                      config=dict(scrollZoom=True,displaylogo=False))
    hybrid_table = table[['pixel','xi','yi','x','y','lorentzian_pct','background_pct']].copy()
    hybrid_table['vca_background_pct'] = 100-h
    hybrid_table['retained_vca_background_below_97'] = background_below_97
    hybrid_table['column_height_pct'] = np.where(background_below_97, fitted_hydrogel, np.nan)
    hybrid_table['z_bottom'] = -hybrid_table.column_height_pct/2
    hybrid_table['z_top'] = hybrid_table.column_height_pct/2
    hybrid_table.to_csv(OUT/'hydrogel_lorentzian_3d_vca_mask.csv',index=False)
    result['retained_background_below_97'] = background_below_97
    result.to_csv(OUT/'pixel_abundances.csv', index=False)
    print(f'Background >=97% removed: {(~background_below_97).sum()}; retained: {background_below_97.sum()}')
    info=dict(pixels=len(h),selected_pixel=int(indices[1]),selected_x=int(x[indices[1]]),
              selected_y=int(y[indices[1]]),retained_12pct=int(keep.sum()),
              median_normalized_rmse=float(np.median(rmse)))
    (OUT/'README.md').write_text("""# Anchored VCA(2)

Saved background endmember is fixed as EM1 (unit-sum normalization only).
EM2 is selected by the existing vca_anchored algorithm with seed 42 from unit-sum measured spectra.
No Lorentzian fits, baseline subtraction, spectral smoothing, or background hard-setting are used.
Abundances use exact constrained least squares on the line segment between the two endmembers,
with nonnegative coefficients summing to one. These are spectral abundances, not calibrated volume fractions.
The selected measured endpoint is not independently verified as pure hydrogel.

The 12% filter and eight-neighbour isolated-single-pixel removal affect display only.
The all-pixels button shows unfiltered abundances. Columns are centred at Z=0.
Reproduce: `python analysis/pca_vca/hydrogel_vca2_interactive.py`

""" + json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2))

if __name__ == '__main__':
    main()
