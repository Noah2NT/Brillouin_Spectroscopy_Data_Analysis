"""Thesis figure batch: original vector diagrams and reproducible simulation plots.

Requires NumPy, Matplotlib and Pillow. All concentration laws and photon budgets are

prescribed illustration inputs, not experimental calibrations.

"""

from pathlib import Path

import json
import csv

import shutil

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib import font_manager, colors

from matplotlib.patches import Rectangle, Ellipse, Polygon, ConnectionPatch, FancyArrowPatch

from matplotlib.ticker import MaxNLocator

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE=Path(__file__).resolve().parent

OUT=HERE.parent if HERE.name=='source' else HERE.parent/'thesis_figure_batch'

FIG=OUT/'figures';DATA=OUT/'data'

FIG.mkdir(parents=True,exist_ok=True);DATA.mkdir(exist_ok=True)

for name in ['arial.ttf','arialbd.ttf','ariali.ttf']:

    p=Path('C:/Windows/Fonts')/name

    if p.exists():font_manager.fontManager.addfont(str(p))

plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9,

    'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic',

    'mathtext.bf':'Arial:bold','pdf.fonttype':42,'svg.fonttype':'none',

    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.9,

    'axes.titlesize':9,'axes.labelsize':9,'legend.fontsize':8,

    'xtick.labelsize':8,'ytick.labelsize':8,'savefig.facecolor':'white'})

BLUE,ORANGE,INK,GREY,RED='#24649A','#BE5A26','#263238','#7A858B','#C43845'

CMAP=colors.LinearSegmentedColormap.from_list('concentration',[BLUE,'#CDD9DC',ORANGE])

CAPTIONS={}

P={'w_max':.1,'bfs_water_GHz':6.05,'bfs_slope':.85,'linewidth_water_GHz':.20,

   'linewidth_slope':2.,'intensity_increase_at_max':.20,'instrument_fwhm_GHz':.276,

   'radius_um':20.,'collection_length_um':30.,'collection_start_um':0.,'beta':1.,

   'area_reference_mass_fraction':.10,'area_slope':2.,'figure6_water_reference_photon_budgets':[100000,10000,1000],'seed':24948806}

def params(w):

    w=np.asarray(w);bfs=6.05*(1+.85*w)

    gamma=.2*((bfs/6.05)**2+2*w)

    area=1+P['area_slope']*w

    return bfs,gamma,area

def lor(f,mu,gamma,area=1.):

    return area/np.pi*(gamma/2)/((f-mu)**2+(gamma/2)**2)

def response(f,w,broaden=True):

    mu,gamma,area=params(w)

    if broaden:gamma=gamma+.276

    return lor(np.asarray(f),mu[...,None],gamma[...,None],area[...,None])

def plotaxes(a,xlabel=r'Frequency shift, $f$ (GHz)',ylabel='Intensity',xlim=None):

    a.set_xlabel(xlabel);a.set_ylabel(ylabel)

    if xlim:a.set_xlim(*xlim)

    a.set_ylim(bottom=0);a.grid(False)

def title(a,text):a.set_title(text,loc='left',fontweight='bold',pad=9)

def save(fig,n,stem,caption):

    filename=f'figure_{n:02}_{stem}'

    for ext in ['pdf','svg','png']:

        fig.savefig(FIG/(filename+'.'+ext),dpi=300,bbox_inches='tight',pad_inches=.10,transparent=(ext!='png'))

    plt.close(fig);CAPTIONS[n]={'stem':filename,'caption':caption}

    print(f'Figure {n}: {stem}',flush=True)

def arrow(a,start,end,col=GREY,lw=1.2,style='-|>'):

    a.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle=style,color=col,lw=lw,mutation_scale=10))

def insetcurve(fig,rect,f,y,col=INK,label=None):

    a=fig.add_axes(rect);a.plot(f,y,color=col,lw=1.5)

    a.set_xlim(f[0],f[-1]);a.set_ylim(0,np.max(y)*1.13)

    a.set_xticks([6.0,7.0]);a.set_yticks([]);a.tick_params(labelsize=8)

    if label:a.set_title(label,fontsize=8,pad=3)

    a.set_xlabel('')

    return a

def geometry(a,kind='fixed',radius=20,ray_x=0,points=False,labels=True,many=False):

    """Transparent 3D model and red axial collection volume, in micrometres."""

    u=np.linspace(0,2*np.pi,45);v=np.linspace(0,np.pi,30)

    sx=radius*np.outer(np.cos(u),np.sin(v));sy=radius*np.outer(np.sin(u),np.sin(v))

    sz=radius*np.outer(np.ones_like(u),np.cos(v))

    if kind=='fixed':

        a.plot_surface(sx,sy,sz,color=ORANGE,alpha=.20,linewidth=0,shade=True)

        a.plot_wireframe(sx,sy,sz,rstride=10,cstride=7,color=ORANGE,alpha=.25,lw=.5)

    else:

        # Back hemisphere plus the exposed radial cut plane reveals internal concentration.

        ub=np.linspace(0,np.pi,32)

        a.plot_surface(radius*np.outer(np.cos(ub),np.sin(v)),

            radius*np.outer(np.sin(ub),np.sin(v)),

            radius*np.outer(np.ones_like(ub),np.cos(v)),color=BLUE,alpha=.13,linewidth=0)

        rr,tt=np.meshgrid(np.linspace(0,radius,35),np.linspace(0,2*np.pi,65))

        a.plot_surface(rr*np.cos(tt),np.zeros_like(rr),rr*np.sin(tt),

            facecolors=CMAP(1-rr/radius),shade=False,alpha=.48,linewidth=0)

        a.plot_wireframe(sx,sy,sz,rstride=11,cstride=9,color=BLUE,alpha=.14,lw=.4)

    # Water surrounding the sample: sparse edges keep the sample visible.

    bottom=[[-24,-24,-23],[24,-24,-23],[24,24,-23],[-24,24,-23]]

    top=[[x,y,33] for x,y,_ in bottom]

    for layer in [bottom,top]:

        q=np.array(layer+[layer[0]]);a.plot(q[:,0],q[:,1],q[:,2],color='#9BAFB9',lw=.55,alpha=.5)

    for lo,hi in zip(bottom,top):a.plot(*zip(lo,hi),color='#9BAFB9',lw=.5,alpha=.4)

    zz,ang=np.meshgrid(np.linspace(0,30,8),np.linspace(0,2*np.pi,12))

    positions=[(xx,yy) for xx in [-12,-6,0,6,12] for yy in [-12,-6,0,6,12] if xx*xx+yy*yy<20**2] if many else [(ray_x,0)]

    for xx,yy in positions:

        r=.65 if many else 1.3

        a.plot_surface(xx+r*np.cos(ang),yy+r*np.sin(ang),zz,color=RED,alpha=.42,shade=False,linewidth=0)

        for zp in [0,30]:a.plot(xx+r*np.cos(u),yy+r*np.sin(u),np.full_like(u,zp),color=RED,lw=1.0 if many else 1.5)

        a.plot([xx,xx],[yy,yy],[0,30],color=RED,lw=.55 if many else 1.2,alpha=.65,ls='--')

    if points:

        pp=np.linspace(0,20,9);wc=.1*(1-pp/20)

        a.computed_zorder=False

        a.scatter(np.full_like(pp,ray_x),np.full_like(pp,-1.9),pp,c=wc,cmap=CMAP,

            vmin=0,vmax=.1,s=25,edgecolors='white',linewidths=.5,depthshade=False,zorder=30)

    a.set(xlim=(-25,25),ylim=(-25,25),zlim=(-23,34))

    a.set_box_aspect((1,1,1.1),zoom=1.12);a.view_init(elev=18,azim=-65)

    a.set_xticks([-20,0,20]);a.set_yticks([-20,0,20]);a.set_zticks([-20,0,20])

    a.tick_params(labelsize=8,pad=0)

    a.set_xlabel(r'$x$ ($\mu$m)',fontsize=7,labelpad=-3)

    a.set_ylabel(r'$y$ ($\mu$m)',fontsize=7,labelpad=-3)

    a.set_zlabel('');a.text2D(.83,.91,r'$z$ ($\mu$m)',transform=a.transAxes,fontsize=8)

    for axis in [a.xaxis,a.yaxis,a.zaxis]:axis.pane.fill=False;axis.pane.set_edgecolor('white')

    a.grid(False)

    if labels:a.text2D(.03,.89,'Collection volumes',transform=a.transAxes,color=RED,fontsize=8,fontweight='bold',bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))

f=np.linspace(4.5,8.5,401)
edges=np.r_[4.5,(f[:-1]+f[1:])/2,8.5]
bin_width=np.diff(edges)
assert len(f)==401 and np.allclose(np.diff(f),.01)
assert np.allclose(bin_width[[0,-1]],.005) and np.allclose(bin_width[1:-1],.01)

def response_bins(w):
    """Exact instrument-broadened Lorentzian integrals over retained bins."""
    mu,gamma,area=params(w);half=(gamma+.276)/2
    return area[...,None]/np.pi*np.diff(np.arctan((edges-mu[...,None])/half[...,None]),axis=-1)

water_bins=response_bins(np.array(0.));water_band=water_bins.sum()
assert np.isclose(water_band,response_bins(np.array(0.)).sum())

z=np.arange(.125,30,.25)

def localw(kind,x=0,y=0):

    r=np.sqrt(x*x+y*y+z*z)

    return np.where(r<20,.1 if kind=='fixed' else .1*np.maximum(0,1-r/20),0)

def decomposition(kind,x=0,y=0):

    w=localw(kind,x,y)

    if kind=='fixed':groups=[w==0,w>0]

    else:

        groups=[w==0]+[(w>lo)&(w<=hi) for lo,hi in zip(np.linspace(0,.1,10)[:-1],np.linspace(0,.1,10)[1:])]

    curves=[];centres=[]

    for mask in groups:

        if mask.any():

            curves.append(response(f,w[mask]).sum(axis=0)/len(z));centres.append(float(w[mask].mean()))

    parts=np.array(curves);total=response(f,w).mean(axis=0)

    assert np.allclose(parts.sum(axis=0),total)

    return parts,np.array(centres),total

# 1: retain the approved drawing and curves, with side-by-side panels.
script=HERE/'generate_figure_01.py'
one=script.read_text(encoding='utf-8-sig')
one=one.replace("OUT = ROOT / 'drafts'",f"OUT = Path({str(Path.cwd()/'outputs/thesis_figure_batch/figures')!r})")
one=one.replace("'font.family': 'Arial', 'font.size': 11", "'font.family': 'serif', 'font.serif':['DejaVu Serif'], 'font.size': 9")
one=one.replace("'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',\n    'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold'", "'mathtext.fontset': 'cm'")
one=one.replace("figsize=(6.8, 6.7)","figsize=(6.3, 2.65)")
one=one.replace("gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=.36,\n                      left=.15, right=.95, top=.96, bottom=.11)","gs = fig.add_gridspec(1, 2, width_ratios=[1,1.15],wspace=.24,left=.02,right=.98,top=.84,bottom=.20)")
one=one.replace("aspect='equal'","aspect='auto'")
one=one.replace("Different mechanical responses","Mechanical responses")
one=one.replace("ax.text(.72,.99", "ax.text(.76,1.02").replace("ax.text(.97,.47", "ax.text(.66,.22")
one=one.replace("'Higher apparent\\nstiffness'", "'Higher apparent\\nstiffness'")
one=one.replace('fontsize=11','fontsize=8').replace('fontsize=12','fontsize=9').replace('fontsize=14','fontsize=10').replace('fontsize=9.5','fontsize=7.5')
one=one.replace("labelpad=24", "labelpad=18").replace("'figure_01_cell_mechanics_v1'", "'figure_01_cell_mechanics'")
exec(compile(one,str(script),'exec'),{'__file__':str(script)})
CAPTIONS[1]={'stem':'figure_01_cell_mechanics','caption':'Mechanical differences can be measured despite similar appearances. (a) An AFM probe presses into a cell. (b) At the same applied force F, the higher-stiffness response has less indentation, delta_H < delta_L. The curves illustrate the contact response.'}

# 2: the acquisition diagram uses shifts relative to the incident frequency.
script=HERE/'generate_figure_02.py'
two=script.read_text(encoding='utf-8-sig')
two=two.replace("OUT=ROOT/'drafts'",f"OUT=Path({str(Path.cwd()/'outputs/thesis_figure_batch/figures')!r})")
two=two.replace("'figure_02_brillouin_scan_v2'", "'figure_02_brillouin_scan'")
exec(compile(two,str(script),'exec'),{'__file__':str(script)})
CAPTIONS[2]={'stem':'figure_02_brillouin_scan','caption':'Brillouin microscopy produces a spectrum at each scan position. (a) Incident light at f_inc is focused into the sample and backscattered light is directed to a separate spectrometer and detector. Acoustic-wave motifs indicate fluctuations within the sample. (b) Repeating the measurement produces a spatial scan. (c) Expanding guides connect one pixel to its spectrum. The incident-light, Stokes and anti-Stokes peaks occur at shifts 0, -f_B and +f_B.'}

# 3: consistent Lorentzian peaks, centre guides and intrinsic FWHM definition.
fig=plt.figure(figsize=(6.3,4.0));gs=fig.add_gridspec(2,2,height_ratios=[1.25,1],hspace=.70,wspace=.38)
a=fig.add_subplot(gs[0,:]);ff=np.linspace(-14,14,2000)
yy=.9*lor(ff,0,.50)/lor(0,0,.50)+.30*lor(ff,-7,1.4)/lor(-7,-7,1.4)+.30*lor(ff,7,1.4)/lor(7,7,1.4)
a.plot(ff,yy,color=INK,lw=1.6);a.set_ylim(0,1.15);a.set_yticks([])
plotaxes(a,xlim=(-14,14));title(a,'(a) Reading a Brillouin spectrum')
a.text(-.9,1.02,'Incident-light peak',ha='center',fontsize=8)
for centre in [-7,0,7]:a.vlines(centre,0,.79,color=GREY,lw=.8,ls=':')
for pos,label in [(-7,'Stokes'),(7,'Anti-Stokes')]:a.text(pos,.41,label,ha='center',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',pad=1))
arrow(a,(0,.73),(7,.73),BLUE,1.1,'<->');a.text(3.5,.81,r'BFS, $f_{\mathrm{B}}$',ha='center',color=BLUE,fontsize=8)
arrow(a,(7-.7,.15),(7+.7,.15),ORANGE,1.2,'<->')
a.annotate(r'FWHM, $\Gamma$',xy=(7.7,.15),xytext=(10.1,.32),fontsize=8,color=ORANGE,arrowprops=dict(arrowstyle='-',color=ORANGE,lw=.7))
for j,mode in enumerate(['centre','width']):
    a=fig.add_subplot(gs[1,j]);fx=np.linspace(4,10,350)
    a.plot(fx,lor(fx,6.1,.6),color=BLUE,lw=1.7,label='Initial')
    a.plot(fx,lor(fx,7.5 if mode=='centre' else 6.1,.6 if mode=='centre' else 1.2),color=ORANGE,lw=1.7,ls='--',label='Changed')
    plotaxes(a,xlim=(4,10));a.set_yticks([]);title(a,'(b) Centre shifts' if j==0 else '(c) Peak broadens');a.legend(frameon=False,fontsize=7)
save(fig,3,'spectral_features','Reading a Brillouin spectrum. (a) BFS is the centre-to-centre separation from the incident-light peak to a Brillouin peak; FWHM, Gamma, is the width at half peak height. All drawn peaks are Lorentzian. (b) A change in peak centre shifts BFS. (c) Increased linewidth broadens the peak at constant integrated intensity.')

# 4: composition/material identity creates distinguishable responses.

fig,a=plt.subplots(figsize=(6.5,2.8));fx=np.linspace(3,16,500)

for mu,gam,col,name in [(6.1,.7,BLUE,'Water'),(12.0,.85,ORANGE,'PMMA')]:

    y=lor(fx,mu,gam);a.plot(fx,y/y.max(),color=col,lw=2,label=name);a.fill_between(fx,0,y/y.max(),color=col,alpha=.10);a.axvline(mu,color=col,lw=.9,ls=':')

plotaxes(a,ylabel='Normalised intensity',xlim=(3,16));a.set_ylim(0,1.22)

a.legend(frameon=False);

save(fig,4,'material_responses','Schematic separated Brillouin responses illustrate how material identity can produce different BFS values. Water and PMMA are introductory examples, not the glycerol simulation calibration. Peak centres and widths are illustrative.')

# 5: three coherent stages, not an unlabelled sketch of many waves.

fig=plt.figure(figsize=(6.3,3.6));gs=fig.add_gridspec(1,3,width_ratios=[1,1,1.15],wspace=.5)

a=fig.add_subplot(gs[0]);a.set(xlim=(0,1),ylim=(0,1));a.axis('off');title(a,'(a) Collection\nvolume')

a.add_patch(Rectangle((.03,.08),.94,.76,facecolor='#EDF4F7',edgecolor='#A5B7C0',lw=1))

a.add_patch(Ellipse((.5,.43),.68,.45,facecolor='#F3DBCC',edgecolor=ORANGE,lw=1.2))

a.add_patch(Ellipse((.48,.43),.22,.15,facecolor='#D7B8A7',edgecolor='none'))

a.add_patch(Rectangle((.62,.28),.13,.47,facecolor='#FFFFFF',alpha=.60,edgecolor=RED,lw=1.6))

a.add_patch(Rectangle((.62,.28),.13,.47,facecolor='none',edgecolor=RED,lw=2.0))

a.annotate('Surrounding medium',xy=(.2,.73),xytext=(.04,.94),color=BLUE,fontsize=8,arrowprops=dict(arrowstyle='-',color=BLUE,lw=.8))

a.annotate('Sample',xy=(.4,.47),xytext=(.05,.15),color=ORANGE,fontsize=8.5,arrowprops=dict(arrowstyle='-',color=ORANGE,lw=.8))

a.annotate('Collection volume',xy=(.75,.60),xytext=(.98,.89),color=RED,ha='right',fontsize=8,arrowprops=dict(arrowstyle='-',color=RED,lw=.8))

a=fig.add_subplot(gs[1]);title(a,'(b) Local\ncontributions')

fx=np.linspace(5.1,7.8,400);p1=.62*lor(fx,6,.4);p2=.38*lor(fx,6.7,.45)

a.plot(fx,p1,color=BLUE,lw=2,label='Medium');a.plot(fx,p2,color=ORANGE,lw=2,label='Sample')

plotaxes(a,xlim=(5.1,7.8));a.legend(frameon=False,fontsize=8);a.set_yticks([]);a.set_ylim(0,(p1+p2).max()*1.35)

a=fig.add_subplot(gs[2]);title(a,'(c) Combined\nspectrum')

a.plot(fx,p1+p2,color=INK,lw=2.2,label='Combined\nspectrum')

a.fill_between(fx,p1,alpha=.1,color=BLUE);a.fill_between(fx,p1+p2,p1,alpha=.12,color=ORANGE)

plotaxes(a,xlim=(5.1,7.8));a.set_yticks([]);a.legend(frameon=False,fontsize=7,loc='upper right',handlelength=1);a.set_ylim(0,(p1+p2).max()*1.35)

save(fig,5,'collection_volume','A finite collection region can contain both sample and surrounding medium. Their collection-weighted local spectra combine into one measured spectrum. Panels (b) and (c) use the same axes and contributions; the example is schematic, not a depth reconstruction.')

# 6: exact bin-integrated photon budgets, shared with the benchmark definition.
rng=np.random.default_rng(P['seed'])
mixed_bins=water_bins/3+2*response_bins(np.array(.1))/3
fig,axes=plt.subplots(1,3,figsize=(6.3,2.6),sharex=True,layout='constrained')
noise_rows=[]
for j,(a,budget) in enumerate(zip(axes,P['figure6_water_reference_photon_budgets'])):
    expected=budget*mixed_bins/water_band;observed=rng.poisson(expected)
    a.plot(f,observed,color=ORANGE,lw=.75,label='Noisy realisation')
    a.plot(f,expected,color=GREY,lw=1.4,label='Noiseless expectation')
    title(a,r'$P=10^{'+str(int(np.log10(budget)))+r'}$');plotaxes(a,ylabel='Photon count' if j==0 else '',xlim=(5.2,7.5))
    a.set_xticks([5.5,6.0,6.5,7.0,7.5]);a.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    a.yaxis.set_major_locator(MaxNLocator(nbins=4,integer=True));noise_rows.append([budget,expected.sum(),observed.sum()])
    assert np.isclose((budget*water_bins/water_band).sum(),budget)
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False,fontsize=8)
save(fig,6,'exposure_noise','Fewer detected photons produce greater relative shot noise. Connected orange traces are Poisson realisations and grey traces are their noiseless expectations for the same fixed mixture. The pure-water reference budgets P are 100000, 10000 and 1000 expected photons in the retained 4.50–8.50 GHz band. All 401 bin centres are spaced by 0.01 GHz; endpoint bins are clipped to the retained band. Raw photon-count scales vary between panels.')
np.savetxt(DATA/'photon_budget_counts_summary.csv',noise_rows,delimiter=',',header='reference_photon_budget,expected_mixture_counts,realised_mixture_counts',comments='')

# 7: fit the true sum, and show the residual rather than misaligning components.

fx=np.linspace(5.2,7.5,420)

c1=.52*lor(fx,6.0,.32);c2=.48*lor(fx,6.6,.38);truth=c1+c2

best=(np.inf,None)

for mu in np.linspace(5.95,6.65,120):

    for gamma in np.linspace(.35,1.35,100):

        basis=lor(fx,mu,gamma);amp=np.dot(truth,basis)/np.dot(basis,basis)

        pred=amp*basis;err=np.sum((pred-truth)**2)

        if err<best[0]:best=(err,(mu,gamma,amp,pred))

mu,gamma,amp,pred=best[1]

assert np.sqrt(best[0]/np.sum(truth**2))>.05

fig=plt.figure(figsize=(6.3,2.7));gs=fig.add_gridspec(1,2,wspace=.35)

a=fig.add_subplot(gs[0,0]);a.plot(fx,c1,color=BLUE,label='Component 1');a.plot(fx,c2,color=ORANGE,label='Component 2');a.plot(fx,truth,color=INK,lw=2,label='Combined spectrum')

plotaxes(a,xlim=(5.2,7.5));title(a,'(a) Two true contributions');a.legend(frameon=False,fontsize=8)

b=fig.add_subplot(gs[0,1],sharex=a,sharey=a);b.plot(fx,truth,color=INK,lw=2,label='Combined spectrum');b.plot(fx,pred,color=RED,lw=1.8,ls='--',label='Best single-peak fit')

plotaxes(b,xlim=(5.2,7.5));title(b,'(b) One peak misses the shape');b.legend(frameon=False,fontsize=8)

save(fig,7,'single_peak_limit','A single Lorentzian cannot exactly reproduce this deliberately chosen two-component sum. The plotted single-peak curve is a numerical least-squares fit to the same noiseless sum shown in panel (a). This example illustrates model mismatch, not a universal claim that every mixture is visibly separable. Closely overlapping components can also produce deceptively small residuals.')

(DATA/'single_peak_fit.json').write_text(json.dumps({'centre_GHz':float(mu),'FWHM_GHz':float(gamma),'area':float(amp),'relative_rms_error':float(np.sqrt(best[0]/np.sum(truth**2)))},indent=2))

# 8: supplied reference spectra versus estimation of unknown component spectra.
fig=plt.figure(figsize=(6.3,4.4));canvas=fig.add_axes([0,0,1,1]);canvas.set(xlim=(0,1),ylim=(0,1));canvas.axis('off')
canvas.text(.035,.965,'(a) Known component spectra',fontweight='bold',fontsize=9)
canvas.text(.54,.965,'(b) Unknown component spectra',fontweight='bold',fontsize=9)
fx=np.linspace(5.2,7.5,220);r1=lor(fx,6,.35);r2=lor(fx,6.65,.40)
insetcurve(fig,[.06,.70,.17,.17],fx,r1,BLUE,'Reference 1')
insetcurve(fig,[.30,.70,.17,.17],fx,r2,ORANGE,'Reference 2')
insetcurve(fig,[.06,.36,.17,.17],fx,r1/3+2*r2/3,INK,'Mixed spectrum')
canvas.add_patch(Rectangle((.29,.37),.19,.16,facecolor='#EDF3F6',edgecolor=GREY,lw=.9));canvas.text(.385,.45,'Fit supplied\nspectra',ha='center',va='center',fontsize=8)
arrow(canvas,(.24,.45),(.28,.45));arrow(canvas,(.385,.61),(.385,.55));arrow(canvas,(.385,.36),(.385,.245))
canvas.plot([.145,.145,.385,.385],[.625,.61,.61,.625],color=GREY,lw=1)
bar=fig.add_axes([.31,.055,.16,.145]);bar.bar([0,1],[1/3,2/3],color=[BLUE,ORANGE],width=.5)
bar.set(ylim=(0,1),xticks=[0,1],xticklabels=['1','2'],yticks=[0,.5,1]);bar.tick_params(labelsize=7)
bar.set_ylabel('Fraction',fontsize=8,labelpad=2);bar.set_title('Estimated fractions',fontsize=8)
for j,pv in enumerate([.8,.5,.2]):insetcurve(fig,[.55+j*.145,.70,.125,.17],fx,pv*r1+(1-pv)*r2,INK,f'Mixture {j+1}')
canvas.add_patch(Rectangle((.61,.41),.31,.15,facecolor='#EDF3F6',edgecolor=GREY,lw=.9));canvas.text(.765,.485,'Estimate component spectra\nand fractions',ha='center',va='center',fontsize=8)
canvas.plot([.605,.605,.91,.91],[.625,.61,.61,.625],color=GREY,lw=1)
arrow(canvas,(.765,.60),(.765,.575));arrow(canvas,(.765,.40),(.765,.32))
insetcurve(fig,[.56,.10,.16,.17],fx,r1,BLUE,'Component 1');insetcurve(fig,[.78,.10,.16,.17],fx,r2,ORANGE,'Component 2')
canvas.plot([.51,.51],[.04,.93],color='#C7CED2',lw=.8,ls='--')
canvas.text(.15,.29,r'Frequency shift, $f$ (GHz)',ha='center',fontsize=8)
canvas.text(.75,.008,r'Frequency shift, $f$ (GHz)',ha='center',fontsize=8)
save(fig,8,'unmixing_workflow','Known and unknown component spectra lead to different recovery tasks. (a) Supplied reference spectra are fitted to estimate their normalised fractions. (b) Multiple mixed spectra provide variation for estimating unknown component spectra and their fractions. The displayed spectra and fractions illustrate the tasks rather than fitted benchmark results.')

# 9: the four analysis cases, with their inputs and targets.
fig=plt.figure(figsize=(6.3,5.8));gs=fig.add_gridspec(2,4,left=.08,right=.98,bottom=.10,top=.86,wspace=1.15,hspace=1.0)
fx=np.linspace(5.2,7.5,240);r1=response(fx,np.array(0));r2=response(fx,np.array(.1))
headings=['Fixed sample, known spectra','Fixed sample, unknown spectra','Varying sample, known spectra','Varying sample, unknown spectra']
for i in range(4):
    row,col=divmod(i,2);left=fig.add_subplot(gs[row,col*2]);right=fig.add_subplot(gs[row,col*2+1])
    fig.text(.07+col*.50,.95-row*.46,headings[i],fontweight='bold',fontsize=8)
    if i==0:
        left.plot(fx,r1,color=BLUE,lw=1.4,label='Water');left.plot(fx,r2,color=ORANGE,lw=1.4,label='10%')
        title(left,'Given references');right.bar([0,1],[1/3,2/3],color=[BLUE,ORANGE],width=.6)
        right.set(xticks=[0,1],xticklabels=['Water','Sample'],ylim=(0,1),ylabel='Fraction');title(right,'Estimate fractions')
    elif i==1:
        for frac in [.2,.5,.8]:left.plot(fx,frac*r1+(1-frac)*r2,color=INK,alpha=.35+frac/2,lw=1.1)
        title(left,'Mixed spectra');right.plot(fx,r1,color=BLUE,lw=1.4);right.plot(fx,r2,color=ORANGE,lw=1.4)
        plotaxes(right,xlim=(5.2,7.5));right.set_yticks([]);title(right,'Estimate spectra\nand fractions')
    elif i==2:
        for wv in np.linspace(0,.1,7):left.plot(fx,response(fx,np.array(wv)),color=CMAP(wv/.1),lw=1.1)
        title(left,'Given local spectra');vv=np.linspace(0,10,51)
        sample_mass=np.full(50,(2/3)/50)
        right.bar(vv[1:],sample_mass,width=.18,color=CMAP(vv[1:]/10),edgecolor='none')
        right.plot([0,0],[0,1/3],color=BLUE,lw=2.3);right.scatter([0],[1/3],color=BLUE,s=15,zorder=5)
        right.annotate('Water: 1/3',xy=(0,1/3),xytext=(1.5,.30),fontsize=8,color=BLUE)
        right.text(1.5,.10,'Sample total:\n2/3',fontsize=8,color=ORANGE)
        right.set(xlabel='Glycerol (% by mass)',ylabel='Fraction',xlim=(-.5,10.5),ylim=(0,.39),xticks=[0,5,10]);title(right,'Estimate distribution')
        assert len(sample_mass)+1==51 and np.isclose(sample_mass.sum()+1/3,1)
    else:
        for frac in [.25,.5,.75]:left.plot(fx,frac*r1+(1-frac)*r2,color=INK,lw=.8,alpha=.35)
        left.plot(fx,r1,color=BLUE,lw=1.5,label='Water');left.plot(fx,r2,color=ORANGE,lw=1.5,label='10%')
        title(left,'Mixtures + endpoints');left.legend(frameon=False,fontsize=8,loc='upper right',handlelength=.9)
        bf,ga,_=params(np.linspace(0,.1,100));right.plot(bf,ga,color=ORANGE,lw=1.7)
        right.scatter(bf[[0,-1]],ga[[0,-1]],color=[BLUE,ORANGE],s=18,zorder=5)
        right.set(xlabel='BFS (GHz)',ylabel='Intrinsic linewidth (GHz)',xlim=(6.0,6.65),ylim=(.19,.29),xticks=[6.0,6.3,6.6])
        title(right,'Recover relationship\n& contribution')
    plotaxes(left,xlim=(5.2,7.5));left.set_yticks([]);left.set_xticks([5.5,6.5,7.5])
    if i==3:left.set_ylim(0,max(r1.max(),r2.max())*1.65)
    left.set_xlabel(r'Frequency shift, $f$ (GHz)',fontsize=8);left.set_ylabel('Intensity',fontsize=8)
    if i==1:right.set_xlabel(r'Frequency shift, $f$ (GHz)',fontsize=8)
    for ax in [left,right]:
        ax.tick_params(labelsize=8);ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8);ax.title.set_size(8)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    start_x=left.bbox.x1+4;end_x=right.get_tightbbox(renderer).x0-4
    if end_x-start_x>8:
        ymid=(left.bbox.y0+left.bbox.y1)/2
        start,end=fig.transFigure.inverted().transform([(start_x,ymid),(end_x,ymid)])
        fig.add_artist(ConnectionPatch(start,end,coordsA=fig.transFigure,coordsB=fig.transFigure,arrowstyle='-|>',color=GREY,lw=.9,mutation_scale=8))
save(fig,9,'study_questions','Inputs and recovery targets for the four cases. Fixed samples use supplied spectra to fit collection fractions, or mixed observations to estimate spectra and signal fractions. Varying samples use a concentration grid when spectra are known; the 51 nodes include pure water and 50 positive concentrations. With unknown spectra, mixed observations and supplied endpoint spectra constrain linewidth versus BFS and signal contributions. The orange curve shows the relationship to be recovered.')

# 10: matched 3D sample definitions, with red collection cylinders.

fig=plt.figure(figsize=(6.3,3.9),layout='constrained')

for j,kind in enumerate(['fixed','varying']):

    a=fig.add_subplot(1,2,j+1,projection='3d');geometry(a,kind,many=True)

    title(a,'(a) Fixed sample' if j==0 else '(b) Varying sample')

sm=plt.cm.ScalarMappable(norm=colors.Normalize(0,10),cmap=CMAP)

cb=fig.colorbar(sm,ax=fig.axes,orientation='horizontal',shrink=.5,pad=.04,fraction=.035)

cb.set_label('Glycerol (% by mass)',fontsize=8);cb.ax.tick_params(labelsize=7)

save(fig,10,'sample_models_3d','Three-dimensional prescribed sample models in surrounding water. The uniform sphere has 10% glycerol by mass; the cutaway gradient sphere increases from 0% at the boundary to 10% at the centre. The red cylinder array illustrates collection volumes at multiple lateral scan positions, each with the finite axial collection interval (30 micrometres, from z=0 to 30). The cutaway reveals a radial model field, not a reconstructed volume. Concentration laws are test functions, not physical diffusion models; dimensions are in micrometres.')

# 11: fixed mixture, from 3D geometry to correctly weighted coloured contributions.

fig=plt.figure(figsize=(6.3,2.9));gs=fig.add_gridspec(1,3,width_ratios=[1.1,1,1],wspace=.75)

a=fig.add_subplot(gs[0],projection='3d');geometry(a,'fixed',labels=False);title(a,'(a) Collection volume')

parts,centres,total=decomposition('fixed')

for j in [1,2]:

    a=fig.add_subplot(gs[j]);title(a,'(b) Weighted spectra' if j==1 else '(c) Combined spectrum')

    for curve,wc in zip(parts,centres):a.plot(f,curve,color=CMAP(wc/.1),lw=1.6,label='Water: 33.3%' if wc==0 else 'Sample: 66.7%')

    if j==2:a.plot(f,total,color=INK,lw=2.1,label='Combined spectrum')

    plotaxes(a,xlim=(4.5,8.5),ylabel='Intensity')
    if j==1:a.legend(frameon=False,fontsize=7,loc='upper left',handlelength=1,borderaxespad=.3)
    else:
        a.legend([a.lines[-1]],['Combined\nspectrum'],frameon=False,fontsize=7,loc='upper right',handlelength=1,borderaxespad=.3)

    a.set_ylim(0,total.max()*1.4)

save(fig,11,'fixed_mixture_3d','A single axial collection region contains water and homogeneous 10% glycerol solution. The central ray has collection fractions 1/3 water and 2/3 solution (33.3% and 66.7%); their full-band signal fractions are 29.4% and 70.6%. The coloured curves already include those weights and the chosen concentration-dependent areas; their exact sum is the black curve. These coefficients are collection fractions, not glycerol mass fractions or necessarily equal signal fractions. Spectra include the assumed instrument response.')

# 12: many immediate depth points, each with a colour-coded local response.

fig=plt.figure(figsize=(6.3,3.2));gs=fig.add_gridspec(1,3,width_ratios=[1.1,1.1,1.1],wspace=.80)

a=fig.add_subplot(gs[0],projection='3d');geometry(a,'varying',points=True,labels=False);title(a,'(a) Sample along depth')

b=fig.add_subplot(gs[1]);pp=np.linspace(0,20,9);wc=.1*(1-pp/20)

for zp,wv in zip(pp,wc):

    yy=response(f,np.array(wv));b.plot(f,zp+6.5*yy/yy.max(),color=CMAP(wv/.1),lw=1.4)

b.set(xlim=(5.45,7.25),ylim=(-1,28),xlabel=r'Frequency shift, $f$ (GHz)',ylabel=r'Depth, $z$ ($\mu$m)');title(b,'(b) Local spectra')

b.set_yticks([0,10,20]);b.grid(alpha=.15)

c=fig.add_subplot(gs[2]);parts,centres,total=decomposition('varying')

for curve,wv in zip(parts,centres):c.plot(f,curve,color=CMAP(wv/.1),lw=1,alpha=.8)

c.plot(f,total,color=INK,lw=2.2,label='Combined spectrum');plotaxes(c,xlim=(5.45,7.25),ylabel='Intensity');title(c,'(c) Combined spectrum');c.legend([c.lines[-1]],['Combined\nspectrum'],frameon=False,fontsize=7,loc='upper right',handlelength=1);c.set_ylim(0,total.max()*1.35)

key=c.inset_axes([.88,.48,.04,.28])
cb=fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0,10),cmap=CMAP),cax=key,orientation='vertical')
cb.set_ticks([0,5,10]);cb.ax.tick_params(labelsize=7,pad=1,length=2);cb.set_label('Glycerol (% by mass)',fontsize=7,labelpad=2)

save(fig,12,'varying_mixture_depth','A varying sample produces different local spectra within one collection volume. (a) Coloured points mark depth positions in the varying sample. (b) Local spectra are individually scaled and vertically offset to show their shapes. (c) Depth-weighted concentration-bin and water contributions combine into the black spectrum. The colour bar gives glycerol concentration as percent by mass.')

# 13: one stated, increasing intensity law rather than three competing display curves.

ww=np.linspace(0,.1,201);bfs,gamma,area=params(ww)

fig,axes=plt.subplots(1,3,figsize=(6.3,2.5),layout='constrained')

for a,y,col,ttl,yl in zip(axes,[bfs,gamma,area],[INK,INK,INK],

    ['(a) BFS','(b) Intrinsic linewidth','(c) Integrated intensity'],

    ['BFS (GHz)','FWHM (GHz)',r'Relative area, $A/A_0$']):

    a.plot(ww*100,y,color=col,lw=2);a.set(xlabel='Glycerol (% by mass)',ylabel=yl,xlim=(0,10));title(a,ttl)

equations=[r'$f_{\mathrm{B}}=6.05(1+0.85w_g)$',r'$\Gamma=0.20[(1+0.85w_g)^2+2w_g]$',r'$A/A_0=1+2w_g$']

for a,eq in zip(axes,equations):

    lo,hi=a.get_ylim();a.set_ylim(lo,hi+(hi-lo)*.35);a.text(.03,.97,eq,transform=a.transAxes,va='top',fontsize=8)

save(fig,13,'concentration_parameters','Chosen local response laws over the initial 0–10% glycerol mass range. BFS and intrinsic linewidth use the provisional laws specified in the methodology. Integrated intensity is prescribed to rise linearly by 20% between 0% and 10% glycerol. This increasing intensity is an explicit modelling choice, not a measured concentration calibration. Peak area and peak height are distinct.')

np.savetxt(DATA/'concentration_parameters.csv',np.column_stack([ww*100,bfs,gamma,area]),delimiter=',',header='mass_percent,bfs_GHz,intrinsic_FWHM_GHz,relative_area',comments='')

# 14: same forward model as concentration and mixture figures, not unrelated illustrative spectra.

x=np.arange(-25.,26.);cubes={}

for kind in ['fixed','varying']:

    cube=np.empty((51,51,len(f)))

    for iy,y in enumerate(x):

        rr=np.sqrt(x[:,None]**2+y*y+z[None,:]**2)

        wloc=np.where(rr<20,.1 if kind=='fixed' else .1*np.maximum(0,1-rr/20),0)

        cube[iy]=response(f,wloc).mean(axis=1)

    cubes[kind]=cube

    assert np.allclose(cube[25,25],decomposition(kind)[2])

    assert np.allclose(cube[0,0],response(f,np.array(0)))

vmax=max(a.max() for a in cubes.values())
fig=plt.figure(figsize=(6.3,4.7),layout='constrained');gs=fig.add_gridspec(2,2,width_ratios=[1,1.1],wspace=.10)
for row,kind in enumerate(['fixed','varying']):
    a=fig.add_subplot(gs[row,0]);im=a.imshow(cubes[kind][25],origin='lower',aspect='auto',extent=[4.5,8.5,-25.5,25.5],cmap='magma',vmin=0,vmax=vmax)
    a.set(xlim=(5.65,7.0),xlabel=r'Frequency shift, $f$ (GHz)',ylabel=r'Scan position, $x$ ($\mu$m)')
    a.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    a.axhline(0,color='white',lw=.9,ls='--');title(a,f'({chr(97+row*2)}) '+('Fixed sample' if row==0 else 'Varying sample'))
    a=fig.add_subplot(gs[row,1]);parts,centres,total=decomposition(kind)
    for idx,(curve,wv) in enumerate(zip(parts,centres)):
        label='Water' if idx==0 else ('10% sample' if kind=='fixed' else ('Concentration bins' if idx==1 else None))
        a.plot(f,curve,color=BLUE if idx==0 else ORANGE,lw=1.2 if idx==0 else .8,label=label)
    a.plot(f,total,color=INK,lw=1.8,label='Combined spectrum');a.set_ylim(0,vmax*1.3)
    plotaxes(a,xlim=(5.65,7.0));a.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    title(a,f'({chr(98+row*2)}) Central mixed pixel');a.legend(frameon=False,fontsize=7,loc='upper right')
cb=fig.colorbar(im,ax=fig.axes,orientation='horizontal',shrink=.50,pad=.10,fraction=.035)
cb.set_label('Intensity',fontsize=8);cb.ax.tick_params(labelsize=7)
save(fig,14,'noiseless_dataset','Noiseless datasets from the forward model. (a,c) Spectra across the central lateral line of the fixed and varying samples; the dashed line marks x=y=0. (b,d) Coloured contributions for that selected scan position and their exact black combined spectrum. Water is blue and sample contributions are orange. The display window is 5.65–7.00 GHz, while the retained band is 4.50–8.50 GHz. Both rows share frequency and intensity scales and include instrumental broadening.')

# 15: independent Poisson counts from exact bin integrals, with P=10000.
P['figure15_water_reference_photons']=10000
count_scale=P['figure15_water_reference_photons']/water_band
noisy={}
for kind in ['fixed','varying']:
    expected_bins=np.empty_like(cubes[kind])
    for iy,y in enumerate(x):
        rr=np.sqrt(x[:,None]**2+y*y+z[None,:]**2)
        wloc=np.where(rr<20,.1 if kind=='fixed' else .1*np.maximum(0,1-rr/20),0)
        expected_bins[iy]=response_bins(wloc).mean(axis=1)*count_scale
    noisy[kind]=rng.poisson(expected_bins)
    assert np.isclose(expected_bins[0,0].sum(),P['figure15_water_reference_photons'])
fig=plt.figure(figsize=(6.3,4.7),layout='constrained');gs=fig.add_gridspec(2,2,width_ratios=[1,1.1],wspace=.10)
for row,kind in enumerate(['fixed','varying']):
    density=noisy[kind]/(count_scale*bin_width)
    a=fig.add_subplot(gs[row,0]);im=a.imshow(density[25],origin='lower',aspect='auto',extent=[4.5,8.5,-25.5,25.5],cmap='magma',vmin=0,vmax=vmax)
    a.set(xlim=(5.65,7.0),xlabel=r'Frequency shift, $f$ (GHz)',ylabel=r'Scan position, $x$ ($\mu$m)')
    a.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    a.axhline(0,color='white',lw=.9,ls='--');title(a,f'({chr(97+row*2)}) '+('Fixed sample' if row==0 else 'Varying sample'))
    a=fig.add_subplot(gs[row,1]);a.plot(f,density[25,25],color=ORANGE,lw=.75,label='Noisy realisation')
    a.plot(f,cubes[kind][25,25],color=GREY,lw=1.3,label='Noiseless expectation')
    plotaxes(a,xlim=(5.65,7.0));a.set_ylim(0,max(vmax*1.3,density[25,25].max()*1.3))
    a.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    title(a,f'({chr(98+row*2)}) Central noisy pixel');a.legend(frameon=False,fontsize=7,loc='upper right')
cb=fig.colorbar(im,ax=fig.axes,orientation='horizontal',shrink=.50,pad=.10,fraction=.035)
cb.set_label('Intensity',fontsize=8);cb.ax.tick_params(labelsize=7)
save(fig,15,'noisy_dataset','Poisson-noise datasets at a pure-water reference budget P=10000 expected photons in the retained 4.50–8.50 GHz band. (a,c) Noisy line scans corresponding to the fixed and varying samples. (b,d) The selected scan-position traces: orange is a noisy realisation and grey is its noiseless expectation. Counts are generated by exact integration over the same 401 frequency bins as the noise model, then divided by count scale and bin width for intensity display. Heatmap limits match the noiseless figure.')

np.savez_compressed(DATA/'noisy_dataset.npz',x_um=x,y_um=x,f_GHz=f,frequency_bin_edges_GHz=edges,fixed_counts=noisy['fixed'],varying_counts=noisy['varying'],count_scale=count_scale)

assert all(np.issubdtype(v.dtype,np.integer) and np.all(v>=0) for v in noisy.values())

np.savez_compressed(DATA/'noiseless_dataset.npz',x_um=x,y_um=x,f_GHz=f,fixed=cubes['fixed'],varying=cubes['varying'])

(DATA/'model_parameters.json').write_text(json.dumps(P,indent=2)+'\n')

assert np.allclose(params(np.array([0,.1]))[0],[6.05,6.56425])

assert np.allclose(params(np.array([0,.1]))[1],[.2,.275445])

assert np.allclose(params(np.array([0,.1]))[2],[1,1.2])

assert np.isclose(params(np.array(.025))[2],1.05)
assert np.isclose(params(np.array(.10))[2],1.20)
baseline_max=P['w_max']
for test_max in [.025,.05,.10]:
    P['w_max']=test_max
    assert np.isclose(params(np.array(.025))[2],1.05)
P['w_max']=baseline_max
assert len(CAPTIONS)==15

(OUT/'captions.json').write_text(json.dumps(CAPTIONS,indent=2),encoding='utf-8')

(OUT/'CAPTIONS.md').write_text('# Figure captions\n\n'+ '\n\n'.join(f'## Review figure {n}\n\n{CAPTIONS[n]["caption"]}' for n in sorted(CAPTIONS))+'\n',encoding='utf-8')

project=HERE.parent/'Thesis_Overleaf'
if (project/'figure_names.csv').exists():
    by_stem={v['stem']:v['caption'] for v in CAPTIONS.values()}
    final_captions={}
    with (project/'figure_names.csv').open(newline='',encoding='utf-8-sig') as mapping:
        for row in csv.DictReader(mapping):
            for ext in ['pdf','svg','png']:
                shutil.copy2(FIG/(row['old_stem']+'.'+ext),project/'figures'/(row['new_stem']+'.'+ext))
            final_captions[row['new_stem']]=by_stem[row['old_stem']]
    review=HERE.parent.parent/'.codex_tmp'/'collated_feedback_review'
    review.mkdir(parents=True,exist_ok=True)
    (review/'captions_by_filename.json').write_text(json.dumps(final_captions,indent=2),encoding='utf-8')

print('All 15 figures complete; forward-model consistency checks passed.',flush=True)

