"""Regenerate the thesis illustrations; these are prescribed simulations, not measurements.

Requires numpy and matplotlib. Run from any directory; outputs sit beside this script.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
BLUE, ORANGE, GREEN = '#24649A', '#BE5A26', '#2E7C69'
P = dict(w_max=0.10, bfs_water_GHz=6.05, bfs_slope=0.85,
         linewidth_water_GHz=0.20, linewidth_slope=2.0,
         instrument_fwhm_GHz=0.276, radius_um=20.0,
         collection_length_um=100.0, beta=1.0, depth_step_um=0.25,
         lateral_step_um=1.0, frequency_step_GHz=0.01,
         integrated_intensity=1.0, intensity_variation=0.0)

def parameters(w):
    bfs = P['bfs_water_GHz'] * (1 + P['bfs_slope'] * w)
    gamma = P['linewidth_water_GHz'] * ((bfs / P['bfs_water_GHz'])**2
                                             + P['linewidth_slope'] * w)
    return bfs, gamma

def lorentzian(f, w, instrument=True):
    bfs, gamma = parameters(w)
    # Lorentzian convolution is analytic: FWHMs add and integrated area is preserved.
    gamma = gamma + (P['instrument_fwhm_GHz'] if instrument else 0)
    return (gamma[..., None] / 2 / np.pi) / ((f - bfs[..., None])**2
                                                     + (gamma[..., None] / 2)**2)

def save(fig, stem):
    fig.savefig(FIG / (stem + '.pdf'), bbox_inches='tight')
    fig.savefig(FIG / (stem + '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

w = np.linspace(0, P['w_max'], 201)
bfs, gamma = parameters(w)
fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.25), layout='constrained')
for a, vals, col, title, ylabel in [
    (ax[0], bfs, BLUE, '(a) Brillouin frequency shift', 'BFS (GHz)'),
    (ax[1], gamma, ORANGE, '(b) Intrinsic linewidth', 'FWHM (GHz)')]:
    a.plot(w*100, vals, color=col, lw=2.2)
    a.set(title=title, xlabel='Glycerol (% by mass)', ylabel=ylabel,
          xlim=(0, 10))
    a.grid(alpha=.18)
for delta, style, col, label in [(0, '-', GREEN, 'Baseline: constant area'),
                                  (.2, '--', BLUE, 'Sensitivity: +20% at 10%'),
                                  (-.2, '--', ORANGE, 'Sensitivity: -20% at 10%')]:
    ax[2].plot(w*100, 1+delta*w/P['w_max'], style, color=col, lw=2, label=label)
ax[2].set(title='(c) Integrated intensity', xlabel='Glycerol (% by mass)',
          ylabel=r'Relative area $A/A_0$', xlim=(0,10), ylim=(.73,1.52))
ax[2].legend(frameon=False, fontsize=11, loc='upper left')
ax[2].grid(alpha=.18)
save(fig, 'concentration_parameters')
np.savetxt(ROOT/'concentration_parameters.csv',
           np.column_stack([w*100,bfs,gamma,np.ones_like(w),1+.2*w/.1,1-.2*w/.1]),
           delimiter=',', header='glycerol_mass_percent,bfs_GHz,intrinsic_FWHM_GHz,area_baseline,area_plus20percent,area_minus20percent',comments='')

x = np.arange(-25., 26., P['lateral_step_um'])
f = np.linspace(4.5, 8.5, 401)
z = np.arange(-50 + .125, 50, .25)  # midpoint integration; sum weights = 1
assert len(z) == 400
xx, yy = np.meshgrid(x, x)
cubes, fields = {}, {}
for kind in ['fixed', 'varying']:
    cube = np.empty((len(x),len(x),len(f)))
    radius_xy = np.hypot(xx,yy)
    inside = radius_xy < P['radius_um']
    fields[kind] = np.where(inside,
        .1 if kind == 'fixed' else .1*np.maximum(0,1-radius_xy/20),0)
    for iy,y in enumerate(x):
        r = np.sqrt(x[:,None]**2+y*y+z[None,:]**2)
        wlocal = np.where(r < 20, .1 if kind == 'fixed' else
                          .1*np.maximum(0,1-r/20)**P['beta'],0)
        cube[iy] = lorentzian(f,wlocal).mean(axis=1)
    cubes[kind] = cube

# Physical/numerical checks independent of the figure layout.
assert np.allclose(parameters(np.array([0,.1]))[0], [6.05,6.56425])
assert np.allclose(parameters(np.array([0,.1]))[1], [.2,.275445])
assert np.allclose(cubes['fixed'][0,0], lorentzian(f,np.array(0.)))
fraction = np.mean(z*z < 20*20)
assert np.isclose(fraction,.4)
assert np.allclose(cubes['fixed'][25,25],
                   (1-fraction)*lorentzian(f,np.array(0.))+
                    fraction*lorentzian(f,np.array(.1)))
assert all(np.isfinite(a).all() and (a>=0).all() for a in cubes.values())
assert not np.allclose(cubes['fixed'],cubes['varying'])

fig = plt.figure(figsize=(12.0,7.6),layout='constrained')
gs = fig.add_gridspec(2,3,width_ratios=[1,1.25,1.25])
vmax = max(v.max() for v in cubes.values())
colors = ['#777777',GREEN,ORANGE,BLUE]
positions = [-25,-18,-10,0]
for row,kind in enumerate(['fixed','varying']):
    a=fig.add_subplot(gs[row,0])
    im=a.imshow(100*fields[kind],origin='lower',extent=[-25.5,25.5,-25.5,25.5],
                vmin=0,vmax=10,cmap='viridis',interpolation='nearest')
    a.axhline(0,color='white',lw=1,ls='--')
    for xp,col in zip(positions,colors): a.plot(xp,0,'o',mec='white',mew=.6,color=col,ms=5)
    a.set(title=f'({chr(97+row*3)}) '+('Fixed composition' if row==0 else 'Varying concentration'),
          xlabel=r'$x$ ($\mu$m)',ylabel=r'$y$ ($\mu$m)')
    cb=fig.colorbar(im,ax=a,fraction=.05,pad=.03)
    cb.set_label('Glycerol (% by mass)',fontsize=12)
    a=fig.add_subplot(gs[row,1])
    im=a.imshow(cubes[kind][25],origin='lower',aspect='auto',
                extent=[f[0]-.005,f[-1]+.005,-25.5,25.5],vmin=0,vmax=vmax,
                cmap='magma',interpolation='nearest')
    a.set(title=f'({chr(98+row*3)}) Central spectral line scan',
          xlabel='Frequency shift (GHz)',ylabel=r'$x$ at $y=0$ ($\mu$m)')
    cb=fig.colorbar(im,ax=a,fraction=.05,pad=.03)
    cb.set_label('a.u./GHz',fontsize=12)
    a.set_xticks([5,6,7,8])
    a=fig.add_subplot(gs[row,2])
    for xp,col in zip(positions,colors):
        a.plot(f,cubes[kind][25,int(xp+25)],color=col,lw=1.8,
               label=fr'$x={xp}$ $\mu$m')
    a.set(title=f'({chr(99+row*3)}) Example noiseless spectra',
          xlabel='Frequency shift (GHz)',ylabel='Spectral density (a.u./GHz)',
          xlim=(5.3,7.35),ylim=(0,vmax*1.08))
    a.set_xticks([5.5,6,6.5,7])
    a.legend(frameon=False,fontsize=11)
    a.grid(alpha=.15)
save(fig,'noiseless_simulated_dataset')
np.savez_compressed(ROOT/'noiseless_simulated_dataset.npz',
    x_um=x,y_um=x,f_GHz=f,fixed=cubes['fixed'],varying=cubes['varying'],
    fixed_midplane_mass_fraction=fields['fixed'],varying_midplane_mass_fraction=fields['varying'])
for kind in cubes:
    np.savetxt(ROOT/f'{kind}_central_line_scan.csv',
               np.column_stack([x,cubes[kind][25]]),delimiter=',',
               header='x_um,'+','.join(f'f_{v:.2f}_GHz' for v in f),comments='')
(ROOT/'figure_parameters.json').write_text(json.dumps(P,indent=2)+'\n',encoding='utf-8')
print('Generated two figures in PDF and PNG, numerical data and parameters. All self-checks passed.')
