"""Thesis figure batch: original vector diagrams and reproducible simulation plots.



Requires NumPy, Matplotlib and Pillow. All concentration laws and count rates are

prescribed illustration inputs, not experimental calibrations.

"""

from pathlib import Path

import json

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

plt.rcParams.update({'font.family':'Arial','font.size':10,

    'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic',

    'mathtext.bf':'Arial:bold','pdf.fonttype':42,'svg.fonttype':'none',

    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.9,

    'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,

    'xtick.labelsize':8,'ytick.labelsize':8,'savefig.facecolor':'white'})

BLUE,ORANGE,INK,GREY,RED='#24649A','#BE5A26','#263238','#7A858B','#C43845'

CMAP=colors.LinearSegmentedColormap.from_list('concentration',[BLUE,'#CDD9DC',ORANGE])

CAPTIONS={}

P={'w_max':.1,'bfs_water_GHz':6.05,'bfs_slope':.85,'linewidth_water_GHz':.20,

   'linewidth_slope':2.,'intensity_increase_at_max':.20,'instrument_fwhm_GHz':.276,

   'radius_um':20.,'collection_length_um':30.,'collection_start_um':0.,'beta':1.,

   'figure6_water_reference_photons_per_second':1000.,'seed':24948806}



def params(w):

    w=np.asarray(w);bfs=6.05*(1+.85*w)

    gamma=.2*((bfs/6.05)**2+2*w)

    area=1+.2*w/.1

    return bfs,gamma,area



def lor(f,mu,gamma,area=1.):

    return area/np.pi*(gamma/2)/((f-mu)**2+(gamma/2)**2)



def response(f,w,broaden=True):

    mu,gamma,area=params(w)

    if broaden:gamma=gamma+.276

    return lor(np.asarray(f),mu[...,None],gamma[...,None],area[...,None])



def plotaxes(a,xlabel='Frequency shift (GHz)',ylabel='Intensity',xlim=None):

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

    a.set_xticks([5.5,6.5,7.5]);a.set_yticks([]);a.tick_params(labelsize=8)

    if label:a.set_title(label,fontsize=9,pad=3)

    a.set_xlabel('GHz',fontsize=8,labelpad=1)

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

        a.plot_surface(xx+r*np.cos(ang),yy+r*np.sin(ang),zz,color=RED,alpha=.20,shade=False,linewidth=0)

        for zp in [0,30]:a.plot(xx+r*np.cos(u),yy+r*np.sin(u),np.full_like(u,zp),color=RED,lw=.7 if many else 1.25)

        a.plot([xx,xx],[yy,yy],[0,30],color=RED,lw=.55 if many else 1.2,alpha=.65,ls='--')

    if points:

        pp=np.linspace(0,20,9);wc=.1*(1-pp/20)

        a.computed_zorder=False

        a.scatter(np.full_like(pp,ray_x),np.full_like(pp,-1.9),pp,c=wc,cmap=CMAP,

            vmin=0,vmax=.1,s=25,edgecolors='white',linewidths=.5,depthshade=False,zorder=30)

    a.set(xlim=(-25,25),ylim=(-25,25),zlim=(-23,34))

    a.set_box_aspect((1,1,1.1),zoom=1.12);a.view_init(elev=18,azim=-65)

    a.set_xticks([-20,0,20]);a.set_yticks([-20,0,20]);a.set_zticks([-20,0,20,30])

    a.tick_params(labelsize=6,pad=-1)

    a.set_xlabel(r'$x$ ($\mu$m)',fontsize=7,labelpad=-3)

    a.set_ylabel(r'$y$ ($\mu$m)',fontsize=7,labelpad=-3)

    a.set_zlabel(r'$z$ ($\mu$m)',fontsize=7,labelpad=0)

    for axis in [a.xaxis,a.yaxis,a.zaxis]:axis.pane.fill=False;axis.pane.set_edgecolor('white')

    a.grid(False)

    if labels:a.text2D(.03,.89,'Collection regions',transform=a.transAxes,color=RED,fontsize=9,fontweight='bold',bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))



f=np.linspace(4.5,8.5,401)

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



# 1: preserve the approved illustration exactly.

source=HERE/'approved/figure_01_cell_mechanics'

for ext in ['pdf','svg','png']:

    if source.with_suffix('.'+ext).exists():shutil.copy2(source.with_suffix('.'+ext),FIG/f'figure_01_cell_mechanics.{ext}')

assert (FIG/'figure_01_cell_mechanics.pdf').exists()

CAPTIONS[1]={'stem':'figure_01_cell_mechanics','caption':

    'Similar cell appearances can conceal different mechanical responses. An AFM probe applies force to a cell; the same force produces less indentation in the higher-stiffness illustrative response. Schematic, not measured data.'}



# 2: update the reviewed microscope schematic while retaining its pixel enlargement.

script=HERE/'generate_figure_02.py'

if script.exists():

    s=script.read_text(encoding='utf-8-sig')

    s=s.replace("OUT=ROOT/'drafts'",f"OUT=Path({str(FIG)!r})")

    s=s.replace("'Acoustic waves\\n(along optical axis)'","'Acoustic\\nwaves'")

    s=s.replace("spec.set_ylabel('Scattered intensity (a.u.)',labelpad=10)","spec.set_ylabel('Photon count',labelpad=10)")

    s=s.replace('for a,b in [(optic,grid)]:','for a,b in []:')

    # Keep the microscope panels balanced and scan pixels square.

    s=s.replace('width_ratios=[1.55,.95,1.2]','width_ratios=[1.18,1,1.05]')

    s=s.replace('wspace=.28','wspace=.38')

    s=s.replace('arrow(grid,(.2,.18),(.94,.18),INK,1.2,scale=9)',"")

    s=s.replace('arrow(grid,(.12,.27),(.12,.98),INK,1.2,scale=9)',"")

    s=s.replace('figsize=(14.0,5.5)','figsize=(8.2,4.2)')

    s=s.replace("'font.size':12","'font.size':10")

    s=s.replace('fontsize=11','fontsize=9').replace('fontsize=10.5','fontsize=9.5').replace('fontsize=12','fontsize=10')

    s=s.replace('Rectangle((3.6,9.5),2.8,1.05','Rectangle((3.2,9.5),3.6,1.05')

    s=s.replace('frame_bottom=box.y0-.115','frame_bottom=box.y0-.155')


    s=s.replace("'figure_02_brillouin_scan_v2'","'figure_02_brillouin_scan'")

    exec(compile(s,str(script),'exec'),{'__file__':str(script)})

CAPTIONS[2]={'stem':'figure_02_brillouin_scan','caption':

    'Brillouin microscopy produces a spectrum at each scan position. (a) Laser light is focused through the objective; backscattered light is sent through a beam splitter to a separate spectrometer and detector. Vertical wave motifs symbolise acoustic fluctuations, not particle trajectories. (b) A spatial scan contains many measurement positions. (c) Expanding guides connect one selected pixel to its cartoon photon-count spectrum. The peaks are elastic Rayleigh scattering and frequency-shifted Stokes and anti-Stokes scattering. Optical geometry, peak shapes and counts are schematic.'}



# 3: definitions first, then two small examples of the parameters changing.

fig=plt.figure(figsize=(7.2,5.0));gs=fig.add_gridspec(2,2,height_ratios=[1.2,1],hspace=.7,wspace=.4)

a=fig.add_subplot(gs[0,:]);ff=np.linspace(-14,14,1000)

yy=.9*np.exp(-.5*(ff/.35)**2)+.3*np.exp(-.5*((ff-7)/.7)**2)+.3*np.exp(-.5*((ff+7)/.7)**2)

a.plot(ff,yy,color=INK,lw=1.8);a.set_ylim(0,1.14);a.set_yticks([])

plotaxes(a,xlabel='Frequency shift (GHz)',ylabel='Intensity',xlim=(-14,14))

title(a,'(a) Reading a Brillouin spectrum')

a.text(0,.98,'Rayleigh',ha='center',fontsize=9)

for pos,label in [(-7,'Stokes'),(7,'Anti-Stokes')]:a.text(pos,.4,label,ha='center',fontsize=9)

for centre in [-7,0,7]:a.vlines(centre,0,.9,color=GREY,lw=.9,ls=':')

arrow(a,(0,.7),(7,.7),BLUE,1.1,'<->');a.text(3.5,.76,r'BFS, $f_{\mathrm{B}}$',ha='center',color=BLUE,fontsize=9)

half=.7*np.sqrt(2*np.log(2));arrow(a,(7-half,.15),(7+half,.15),ORANGE,1.3,'<->')

a.annotate('FWHM',xy=(7+half,.15),xytext=(10,.33),fontsize=8.5,color=ORANGE,arrowprops=dict(arrowstyle='-',color=ORANGE,lw=.8))

for j,mode in enumerate(['centre','width']):

    a=fig.add_subplot(gs[1,j]);fx=np.linspace(4,10,350)

    a.plot(fx,lor(fx,6.1,.6),color=BLUE,lw=1.8,label='Initial')

    a.plot(fx,lor(fx,7.5 if mode=='centre' else 6.1,.6 if mode=='centre' else 1.2),color=ORANGE,lw=1.8,ls='--',label='Changed')

    plotaxes(a,xlim=(4,10));a.set_yticks([]);title(a,'(b) Centre shifts' if j==0 else '(c) Peak broadens')

    a.legend(frameon=False,fontsize=8)

save(fig,3,'spectral_features','Spectral features introduced independently of material mixing. (a) BFS is the separation of a Brillouin peak from the incident-light frequency; FWHM is the width at half peak height. (b) A shift changes peak centre. (c) Broadening changes linewidth, with constant integrated area in this illustration. Curves are schematic; instrumental broadening must be distinguished from intrinsic damping.')



# 4: composition/material identity creates distinguishable responses.

fig,a=plt.subplots(figsize=(6.5,2.8));fx=np.linspace(3,16,500)

for mu,gam,col,name in [(6.1,.7,BLUE,'Water'),(12.0,.85,ORANGE,'Plexiglass')]:

    y=lor(fx,mu,gam);a.plot(fx,y/y.max(),color=col,lw=2,label=name);a.fill_between(fx,0,y/y.max(),color=col,alpha=.10);a.axvline(mu,color=col,lw=.9,ls=':')

plotaxes(a,ylabel='Normalised intensity',xlim=(3,16));a.set_ylim(0,1.22)

a.legend(frameon=False);a.set_title('Different materials can have different BFS',loc='left',pad=12,fontsize=10)

save(fig,4,'material_responses','Schematic separated Brillouin responses illustrate how material identity can produce different BFS values. Water and Plexiglass are introductory examples, not the glycerol simulation calibration. Peak centres and widths are illustrative.')



# 5: three coherent stages, not an unlabelled sketch of many waves.

fig=plt.figure(figsize=(8,3.6));gs=fig.add_gridspec(1,3,width_ratios=[1,1,1.15],wspace=.5)

a=fig.add_subplot(gs[0]);a.set(xlim=(0,1),ylim=(0,1));a.axis('off');title(a,'(a) Finite collection volume')

a.add_patch(Rectangle((.03,.08),.94,.76,facecolor='#EDF4F7',edgecolor='#A5B7C0',lw=1))

a.add_patch(Ellipse((.5,.43),.68,.45,facecolor='#F3DBCC',edgecolor=ORANGE,lw=1.2))

a.add_patch(Ellipse((.48,.43),.22,.15,facecolor='#D7B8A7',edgecolor='none'))

a.add_patch(Rectangle((.62,.28),.13,.47,facecolor=RED,alpha=.11,edgecolor=RED,lw=1.6))

a.add_patch(Rectangle((.62,.28),.13,.47,facecolor='none',edgecolor=RED,lw=1.5))

a.annotate('Surrounding medium',xy=(.2,.73),xytext=(.04,.94),color=BLUE,fontsize=8,arrowprops=dict(arrowstyle='-',color=BLUE,lw=.8))

a.annotate('Sample',xy=(.4,.47),xytext=(.05,.15),color=ORANGE,fontsize=8.5,arrowprops=dict(arrowstyle='-',color=ORANGE,lw=.8))

a.annotate('Collection region',xy=(.75,.60),xytext=(.98,.89),color=RED,ha='right',fontsize=8,arrowprops=dict(arrowstyle='-',color=RED,lw=.8))

a=fig.add_subplot(gs[1]);title(a,'(b) Local contributions')

fx=np.linspace(5.1,7.8,400);p1=.62*lor(fx,6,.4);p2=.38*lor(fx,6.7,.45)

a.plot(fx,p1,color=BLUE,lw=2,label='Medium');a.plot(fx,p2,color=ORANGE,lw=2,label='Sample')

plotaxes(a,xlim=(5.1,7.8));a.legend(frameon=False,fontsize=8);a.set_yticks([]);a.set_ylim(0,(p1+p2).max()*1.35)

a=fig.add_subplot(gs[2]);title(a,'(c) Combined spectrum')

a.plot(fx,p1+p2,color=INK,lw=2.2,label='Combined signal')

a.fill_between(fx,p1,alpha=.1,color=BLUE);a.fill_between(fx,p1+p2,p1,alpha=.12,color=ORANGE)

plotaxes(a,xlim=(5.1,7.8));a.set_yticks([]);a.legend(frameon=False,fontsize=8);a.set_ylim(0,(p1+p2).max()*1.35)

save(fig,5,'collection_volume','A finite collection region can contain both sample and surrounding medium. Their collection-weighted local spectra combine into one measured spectrum. Panels (b) and (c) use the same axes and contributions; the example is schematic, not a depth reconstruction.')



# 6: raw count traces at decreasing exposure, with faint clean expectations.

rng=np.random.default_rng(P['seed']);fn=np.linspace(5.25,7.35,151);df=fn[1]-fn[0]

mix=(1/3)*response(fn,np.array(0))+(2/3)*response(fn,np.array(.1))

water_band=np.sum(response(fn,np.array(0)))*df

rate=mix*df/water_band*1000

times=[1000,10,1]

fig,axes=plt.subplots(1,3,figsize=(8.2,2.9),sharex=True,layout='constrained')

noise_rows=[]

for j,(a,t) in enumerate(zip(axes,times)):

    expected=t*rate;observed=rng.poisson(expected)

    a.plot(fn,expected,color=BLUE,alpha=.16,lw=2.2,label='Expected counts')

    a.plot(fn,observed,color=ORANGE,lw=.8,label='Poisson count trace')

    title(a,f'{t:g} s exposure');plotaxes(a,ylabel='Photon count' if j==0 else '',xlim=(5.25,7.35))

    a.set_xlabel('Shift (GHz)');a.yaxis.set_major_locator(MaxNLocator(nbins=4,integer=True));a.set_xticks([5.5,6.5,7.0])

    noise_rows.append([t,expected.sum(),observed.sum()])

fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False,fontsize=8)

save(fig,6,'exposure_noise','Shorter exposure increases relative photon shot noise. Each panel shows a connected Poisson-count realisation (orange) and a faint noiseless expected trace (blue), for the same mixed spectrum at fixed illumination and collection efficiency. The illustrative count scale is 1000 expected water-reference photons per second in the displayed 5.25–7.35 GHz band; these are simulated exposure times, not calibrated instrument times. Each vertical axis is in raw photon counts and therefore changes scale with exposure. No extra Gaussian noise is added.')

np.savetxt(DATA/'exposure_counts_summary.csv',noise_rows,delimiter=',',header='exposure_s,expected_counts,realised_counts',comments='')



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

fig=plt.figure(figsize=(7.2,2.9));gs=fig.add_gridspec(1,2,wspace=.35)

a=fig.add_subplot(gs[0,0]);a.plot(fx,c1,color=BLUE,label='Component 1');a.plot(fx,c2,color=ORANGE,label='Component 2');a.plot(fx,truth,color=INK,lw=2,label='True sum')

plotaxes(a,xlim=(5.2,7.5));title(a,'(a) Two true contributions');a.legend(frameon=False,fontsize=8)

b=fig.add_subplot(gs[0,1],sharex=a,sharey=a);b.plot(fx,truth,color=INK,lw=2,label='True sum');b.plot(fx,pred,color=RED,lw=1.8,ls='--',label='Best single-peak fit')

plotaxes(b,xlim=(5.2,7.5));title(b,'(b) One peak misses the shape');b.legend(frameon=False,fontsize=8)

save(fig,7,'single_peak_limit','A single Lorentzian cannot exactly reproduce this deliberately chosen two-component sum. The plotted single-peak curve is a numerical least-squares fit to the same noiseless sum shown in panel (a). This example illustrates model mismatch, not a universal claim that every mixture is visibly separable. Closely overlapping components can also produce deceptively small residuals.')

(DATA/'single_peak_fit.json').write_text(json.dumps({'centre_GHz':float(mu),'FWHM_GHz':float(gamma),'area':float(amp),'relative_rms_error':float(np.sqrt(best[0]/np.sum(truth**2)))},indent=2))



# 8: every measured or component spectrum is a single curve on its own axes.

fig=plt.figure(figsize=(8.0,4.9));canvas=fig.add_axes([0,0,1,1]);canvas.set(xlim=(0,1),ylim=(0,1));canvas.axis('off')

canvas.text(.04,.965,'(a) Known reference spectra',fontweight='bold',fontsize=11)

canvas.text(.55,.965,'(b) Unknown component spectra',fontweight='bold',fontsize=11)

fx=np.linspace(5.2,7.5,220);r1=lor(fx,6,.35);r2=lor(fx,6.65,.40)

insetcurve(fig,[.06,.70,.18,.18],fx,r1,BLUE,'Reference 1')

insetcurve(fig,[.29,.70,.18,.18],fx,r2,ORANGE,'Reference 2')

insetcurve(fig,[.06,.39,.18,.18],fx,.6*r1+.4*r2,INK,'One mixed spectrum')

canvas.add_patch(Rectangle((.29,.40),.18,.13,facecolor='#EDF3F6',edgecolor=GREY,lw=.9));canvas.text(.38,.465,'Fit known\nresponses',ha='center',va='center',fontsize=9)

arrow(canvas,(.245,.48),(.285,.48));arrow(canvas,(.38,.615),(.38,.54));arrow(canvas,(.38,.39),(.38,.255))

canvas.plot([.15,.15,.38,.38],[.635,.62,.62,.635],color=GREY,lw=1)

bar=fig.add_axes([.31,.065,.16,.14]);bar.bar([0,1],[.6,.4],color=[BLUE,ORANGE],width=.5);bar.set(ylim=(0,1),xticks=[0,1],xticklabels=['1','2'],ylabel='Contribution');bar.tick_params(labelsize=7);bar.set_ylabel('Contribution',fontsize=8,labelpad=2);bar.set_title('Recovered weights',fontsize=8)

for j,p in enumerate([.8,.5,.2]):insetcurve(fig,[.55+j*.145,.70,.125,.18],fx,p*r1+(1-p)*r2,INK,f'Mixture {j+1}')

canvas.add_patch(Rectangle((.62,.43),.29,.12,facecolor='#EDF3F6',edgecolor=GREY,lw=.9));canvas.text(.765,.49,'Estimate responses\nand their contributions',ha='center',va='center',fontsize=9)

canvas.plot([.605,.605,.91,.91],[.635,.62,.62,.635],color=GREY,lw=1)

arrow(canvas,(.765,.615),(.765,.56));arrow(canvas,(.765,.42),(.765,.32))

insetcurve(fig,[.56,.10,.16,.18],fx,r1,BLUE,'Recovered 1')

insetcurve(fig,[.78,.10,.16,.18],fx,r2,ORANGE,'Recovered 2')

canvas.plot([.51,.51],[.06,.93],color='#C7CED2',lw=.8,ls='--')

save(fig,8,'unmixing_workflow','Known responses allow mixture contributions to be fitted. When responses are unknown, multiple mixed spectra provide variation from which component responses and contributions may be estimated, subject to the method assumptions. Every small spectrum is one continuous Brillouin-band curve, with frequency in GHz; outputs are illustrative, not benchmark results.')



# 9: visual inputs and recovery targets for the four study configurations.

fig=plt.figure(figsize=(9.4,6.6));gs=fig.add_gridspec(2,4,left=.07,right=.98,bottom=.09,top=.87,wspace=.62,hspace=1.05)

fx=np.linspace(5.2,7.5,240);r1=lor(fx,6,.38);r2=lor(fx,6.65,.45)

headings=['1. Fixed mixture · known spectra','2. Fixed mixture · unknown spectra','3. Gradient · known spectra','4. Gradient · unknown spectra']

for i in range(4):

    row,col=divmod(i,2);left=fig.add_subplot(gs[row,col*2]);right=fig.add_subplot(gs[row,col*2+1])

    fig.text(.07+col*.495,.95-row*.47,headings[i],fontweight='bold',fontsize=10)

    if i==0:

        left.plot(fx,r1,color=BLUE,lw=1.6);left.plot(fx,r2,color=ORANGE,lw=1.6)

        title(left,'Known components');right.bar([0,1],[.4,.6],color=[BLUE,ORANGE],width=.6)

        right.set(xticks=[0,1],xticklabels=['Water','Solution'],ylim=(0,1),ylabel='Contribution');title(right,'Recover weights')

    elif i==1:

        for frac in [.2,.5,.8]:left.plot(fx,frac*r1+(1-frac)*r2,color=CMAP(frac),lw=1.4)

        title(left,'Mixed spectra');right.plot(fx,r1,color=BLUE,lw=1.6);right.plot(fx,r2,color=ORANGE,lw=1.6);plotaxes(right,xlim=(5.2,7.5));right.set_yticks([]);title(right,'Recover spectra')

        right.set_ylim(0,r1.max()*1.65)
        weights=right.inset_axes([.59,.70,.36,.24]);weights.bar([0,1],[.4,.6],color=[BLUE,ORANGE]);weights.set(xticks=[],yticks=[],ylim=(0,1));weights.set_title('+ weights',fontsize=7,pad=2)

    elif i==2:

        for wv in np.linspace(0,.1,7):left.plot(fx,response(fx,np.array(wv)),color=CMAP(wv/.1),lw=1.2)

        title(left,'Known local spectra');vv=np.linspace(0,10,21);weights=np.exp(-.5*((vv-5)/2.2)**2);weights/=weights.sum()

        right.bar(vv,weights,width=.45,color=CMAP(vv/10));right.set(xlabel='Glycerol (% by mass)',ylabel='Contribution',xlim=(-.5,10.5));title(right,'Recover distribution')

    else:

        for t in [.2,.5,.8]:left.plot(fx,t*response(fx,np.array(.02))+(1-t)*response(fx,np.array(.08)),color=CMAP(t),lw=1.4)

        title(left,'Mixed spectra');mass_percent=np.linspace(0,10,100);bf,ga,_=params(mass_percent/100)
        right.plot(mass_percent,bf,color=BLUE,lw=1.6,ls='--')
        right.set(xlabel='Glycerol (% by mass)',ylabel='BFS (GHz)',xlim=(0,10),ylim=(6,6.8),xticks=[0,5,10])
        right.tick_params(axis='y',colors=BLUE);right.yaxis.label.set_color(BLUE)
        title(right,'Recover relationship\n& contribution')
        width_axis=right.twinx();width_axis.plot(mass_percent,ga,color=ORANGE,lw=1.6)
        width_axis.set(ylabel='Linewidth (GHz)',ylim=(.18,.34));width_axis.yaxis.label.set_color(ORANGE)
        width_axis.tick_params(axis='y',labelsize=7,colors=ORANGE);width_axis.spines['right'].set_visible(True);width_axis.spines['right'].set_color(ORANGE)

    plotaxes(left,xlim=(5.2,7.5));left.set_yticks([]);left.set_xticks([5.5,6.5,7.5])

    left.set_xlabel('Shift (GHz)')

    if i==1:right.set_xlabel('Shift (GHz)')

    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();y_mid=(left.bbox.y0+left.bbox.y1)/2
    start_x=left.bbox.x1+10;end_x=right.get_tightbbox(renderer).x0-10
    assert end_x-start_x>15, 'Insufficient room between spectrum and target labels'
    start,end=fig.transFigure.inverted().transform([(start_x,y_mid),(end_x,y_mid)])
    fig.add_artist(ConnectionPatch(start,end,coordsA=fig.transFigure,coordsB=fig.transFigure,arrowstyle='-|>',color=GREY,lw=1.1,mutation_scale=10))

save(fig,9,'study_questions','Four simulation configurations and their recovery targets. (1) Known fixed spectra are used to estimate mixture weights. (2) Unknown fixed spectra require estimation of component spectra and weights. (3) Known concentration-dependent spectra are used to estimate a concentration distribution. (4) Unknown varying spectra require estimation of contributions and spectral relationships. The final panel illustrates the prescribed BFS (blue dashed line, left axis) and intrinsic linewidth (orange solid line, right axis) versus glycerol mass percentage. All example outputs are schematic, not recovery results. Unknown-spectrum weights require intensity calibration to represent collection fractions; the final configuration does not imply uniquely recoverable absolute concentration.')



# 10: matched 3D sample definitions, with red collection cylinders.

fig=plt.figure(figsize=(7.3,4.7),layout='constrained')

for j,kind in enumerate(['fixed','varying']):

    a=fig.add_subplot(1,2,j+1,projection='3d');geometry(a,kind,many=True)

    title(a,'(a) Uniform 10% solution' if j==0 else '(b) Radial concentration gradient')

sm=plt.cm.ScalarMappable(norm=colors.Normalize(0,10),cmap=CMAP)

cb=fig.colorbar(sm,ax=fig.axes,orientation='horizontal',shrink=.5,pad=.04,fraction=.035)

cb.set_label('Glycerol (% by mass)',fontsize=8);cb.ax.tick_params(labelsize=7)

save(fig,10,'sample_models_3d','Three-dimensional prescribed sample models in surrounding water. The uniform sphere has 10% glycerol by mass; the cutaway gradient sphere increases from 0% at the boundary to 10% at the centre. The red cylinder array illustrates collection regions at multiple lateral scan positions, each with the finite axial collection range (30 micrometres, from z=0 to 30). The cutaway reveals a radial model field, not a reconstructed volume. Concentration laws are test functions, not physical diffusion models; dimensions are in micrometres.')



# 11: fixed mixture, from 3D geometry to correctly weighted coloured contributions.

fig=plt.figure(figsize=(9.8,3.5));gs=fig.add_gridspec(1,3,width_ratios=[1.45,1,1],wspace=.4)

a=fig.add_subplot(gs[0],projection='3d');geometry(a,'fixed',labels=False);title(a,'(a) Collection region')

parts,centres,total=decomposition('fixed')

for j in [1,2]:

    a=fig.add_subplot(gs[j]);title(a,'(b) Weighted spectra' if j==1 else '(c) Their sum')

    for curve,wc in zip(parts,centres):a.plot(f,curve,color=CMAP(wc/.1),lw=1.6,label='Water: 33.3% collection' if wc==0 else 'Solution: 66.7% collection')

    if j==2:a.plot(f,total,color=INK,lw=2.1,label='Combined spectrum')

    plotaxes(a,xlim=(4.5,8.5),ylabel='Intensity');a.legend(frameon=False,fontsize=7)

    a.set_ylim(0,total.max()*1.4)

save(fig,11,'fixed_mixture_3d','A single axial collection region contains water and homogeneous 10% glycerol solution. The central ray has collection fractions 1/3 water and 2/3 solution (33.3% and 66.7%); their full-band signal fractions are 29.4% and 70.6%. The coloured curves already include those weights and the chosen concentration-dependent areas; their exact sum is the black curve. These coefficients are collection fractions, not glycerol mass fractions or necessarily equal signal fractions. Spectra include the assumed instrument response.')



# 12: many immediate depth points, each with a colour-coded local response.

fig=plt.figure(figsize=(9.8,3.8));gs=fig.add_gridspec(1,3,width_ratios=[1.45,1.1,1.05],wspace=.45)

a=fig.add_subplot(gs[0],projection='3d');geometry(a,'varying',points=True,labels=False);title(a,'(a) Sample along depth')

b=fig.add_subplot(gs[1]);pp=np.linspace(0,20,9);wc=.1*(1-pp/20)

for zp,wv in zip(pp,wc):

    yy=response(f,np.array(wv));b.plot(f,zp+6.5*yy/yy.max(),color=CMAP(wv/.1),lw=1.4)

b.set(xlim=(5.45,7.25),ylim=(-1,28),xlabel='Frequency shift (GHz)',ylabel=r'Depth, $z$ ($\mu$m)');title(b,'(b) Local spectra')

b.set_yticks([0,10,20]);b.grid(alpha=.15)

c=fig.add_subplot(gs[2]);parts,centres,total=decomposition('varying')

for curve,wv in zip(parts,centres):c.plot(f,curve,color=CMAP(wv/.1),lw=1,alpha=.8)

c.plot(f,total,color=INK,lw=2.2,label='Integrated spectrum');plotaxes(c,xlim=(5.45,7.25),ylabel='Intensity');title(c,'(c) Combined spectrum');c.legend(frameon=False,fontsize=7);c.set_ylim(0,total.max()*1.35)

key=c.inset_axes([.56,.77,.40,.055])
cb=fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0,10),cmap=CMAP),cax=key,orientation='horizontal')
cb.set_ticks([0,5,10]);cb.ax.tick_params(labelsize=6,pad=1,length=2);cb.set_label('Glycerol (% by mass)',fontsize=6.5,labelpad=2)

save(fig,12,'varying_mixture_depth','A concentration gradient produces different local spectra within one axial collection volume. Coloured sample points and curves represent depth positions, not separate lateral pixels. Local curves in panel (b) are individually scaled and vertically offset to show their shapes; these display offsets are not frequency shifts. Panel (c) groups the true depth-weighted signal into concentration bins and water; their exact sum is black. The colour bar identifies the concentration of the coloured contributions; it does not display additional spectra. The colours progress with glycerol mass fraction. The surrounding water contribution is retained.')



# 13: one stated, increasing intensity law rather than three competing display curves.

ww=np.linspace(0,.1,201);bfs,gamma,area=params(ww)

fig,axes=plt.subplots(1,3,figsize=(7.4,2.5),layout='constrained')

for a,y,col,ttl,yl in zip(axes,[bfs,gamma,area],[BLUE,ORANGE,BLUE],

    ['(a) Brillouin shift','(b) Intrinsic linewidth','(c) Integrated intensity'],

    ['BFS (GHz)','FWHM (GHz)',r'Relative area, $A/A_0$']):

    a.plot(ww*100,y,color=col,lw=2);a.set(xlabel='Glycerol (% by mass)',ylabel=yl,xlim=(0,10));title(a,ttl)

equations=[r'$f_{\mathrm{B}}=6.05(1+0.85w)$',r'$\Gamma=0.20[(1+0.85w)^2+2w]$',r'$A/A_0=1+2w$']

for a,eq in zip(axes,equations):

    lo,hi=a.get_ylim();a.set_ylim(lo,hi+(hi-lo)*.35);a.text(.03,.97,eq,transform=a.transAxes,va='top',fontsize=8)

save(fig,13,'concentration_parameters','Chosen local response laws over the initial 0–10% glycerol mass range. BFS and intrinsic linewidth use the provisional laws specified in the methodology. Integrated intensity is now prescribed to rise linearly by 20% between 0% and 10% glycerol. This increasing intensity is an explicit modelling choice, not a measured concentration calibration. Peak area and peak height are distinct.')

np.savetxt(DATA/'concentration_parameters.csv',np.column_stack([ww*100,bfs,gamma,area]),delimiter=',',header='mass_percent,bfs_GHz,intrinsic_FWHM_GHz,relative_area',comments='')



# 14: same forward model as figures 11–13, not unrelated illustrative spectra.

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

fig=plt.figure(figsize=(8.3,6.1),layout='constrained');gs=fig.add_gridspec(2,3,width_ratios=[1.25,1.1,1.15])

for row,kind in enumerate(['fixed','varying']):

    a=fig.add_subplot(gs[row,0],projection='3d');geometry(a,kind,labels=False);a.set_zlabel('');a.text2D(.79,.93,r'$z$ ($\mu$m)',transform=a.transAxes,fontsize=7)

    title(a,f'({chr(97+row*3)}) '+('Fixed sample' if row==0 else 'Gradient sample'))

    a=fig.add_subplot(gs[row,1]);im=a.imshow(cubes[kind][25],origin='lower',aspect='auto',extent=[4.495,8.505,-25.5,25.5],cmap='magma',vmin=0,vmax=vmax)

    a.set(xlim=(5.65,7.0),xlabel='Frequency shift (GHz)',ylabel=r'Scan position, $x$ ($\mu$m)')

    a.axhline(0,color='white',lw=.9,ls='--');title(a,f'({chr(98+row*3)}) Central line scan')

    cb=fig.colorbar(im,ax=a,fraction=.045,pad=.03);cb.set_label('Intensity',fontsize=7);cb.ax.tick_params(labelsize=7)

    a=fig.add_subplot(gs[row,2]);parts,centres,total=decomposition(kind)

    for i,(curve,wv) in enumerate(zip(parts,centres)):

        label='Water' if i==0 else ('10% solution' if kind=='fixed' else ('Concentration bins' if i==1 else None))

        a.plot(f,curve,color=CMAP(wv/.1),lw=1.1,label=label)

    a.plot(f,total,color=INK,lw=2,label='Sum');a.set_ylim(0,vmax*1.1)

    plotaxes(a,xlim=(5.65,7.0),ylabel='Intensity')

    title(a,f'({chr(99+row*3)}) One mixed pixel');a.legend(frameon=False,fontsize=6.8)

    a.text(.03,.56,r'$x=y=0$',transform=a.transAxes,fontsize=8)

save(fig,14,'noiseless_dataset','Example noiseless simulation datasets. Left: three-dimensional model fields with red axial collection indicators. Middle: spectra across the central lateral line, displayed over a cropped 5.65–7.00 GHz range; the complete simulated frequency axis remains 4.50–8.50 GHz. The white line selects x=y=0. Right: the coloured contributions for that one mixed pixel and their exact black sum. The fixed sample has two responses; the gradient sample is displayed as concentration-bin contributions plus water. These are forward-model results with instrument broadening and the chosen increasing intensity law, not measured data or reconstructed volumes. Both rows use common frequency and intensity scales.')

# 15: matching dataset view, with independent Poisson samples.

water_band=np.sum(response(f,np.array(0)))*(f[1]-f[0])

P['figure15_water_reference_photons']=10000

count_scale=P['figure15_water_reference_photons']/water_band

noisy={kind:rng.poisson(cube*count_scale*(f[1]-f[0])) for kind,cube in cubes.items()}

fig=plt.figure(figsize=(8.3,6.1),layout='constrained');gs=fig.add_gridspec(2,3,width_ratios=[1.25,1.1,1.15])

for row,kind in enumerate(['fixed','varying']):

    a=fig.add_subplot(gs[row,0],projection='3d');geometry(a,kind,labels=False);a.set_zlabel('');a.text2D(.79,.93,r'$z$ ($\mu$m)',transform=a.transAxes,fontsize=7)

    title(a,f'({chr(97+row*3)}) '+('Fixed sample' if row==0 else 'Gradient sample'))

    density=noisy[kind]/(count_scale*(f[1]-f[0]))

    a=fig.add_subplot(gs[row,1]);im=a.imshow(density[25],origin='lower',aspect='auto',extent=[4.495,8.505,-25.5,25.5],cmap='magma',vmin=0,vmax=vmax)

    a.set(xlim=(5.65,7.0),xlabel='Frequency shift (GHz)',ylabel=r'Scan position, $x$ ($\mu$m)');a.axhline(0,color='white',lw=.9,ls='--');title(a,f'({chr(98+row*3)}) Noisy line scan')

    cb=fig.colorbar(im,ax=a,fraction=.045,pad=.03);cb.set_label('Intensity',fontsize=7);cb.ax.tick_params(labelsize=7)

    a=fig.add_subplot(gs[row,2]);a.plot(f,cubes[kind][25,25],color=INK,alpha=.25,lw=2,label='Noiseless expectation');a.plot(f,density[25,25],color=ORANGE,lw=.8,label='Poisson sample')

    plotaxes(a,xlim=(5.65,7.0),ylabel='Intensity');a.set_ylim(0,max(vmax*1.1,density[25,25].max()*1.08));title(a,f'({chr(99+row*3)}) One noisy pixel');a.legend(frameon=False,fontsize=7)

save(fig,15,'noisy_dataset','Poisson-noise counterparts of the noiseless datasets, with the same geometry, frequency window and heatmap colour limits. The reference photon budget is 10000 expected water photons over the complete 4.50–8.50 GHz band per pixel. Integer counts are divided by the known count scale and bin width for display as spectral density; the noise is sampled before this conversion. Colour saturation uses the noiseless heatmap limit for direct comparison. No Gaussian noise is added; these are simulation examples, not measured data.')

np.savez_compressed(DATA/'noisy_dataset.npz',x_um=x,y_um=x,f_GHz=f,fixed_counts=noisy['fixed'],varying_counts=noisy['varying'],count_scale=count_scale)

assert all(np.issubdtype(v.dtype,np.integer) and np.all(v>=0) for v in noisy.values())

np.savez_compressed(DATA/'noiseless_dataset.npz',x_um=x,y_um=x,f_GHz=f,fixed=cubes['fixed'],varying=cubes['varying'])

(DATA/'model_parameters.json').write_text(json.dumps(P,indent=2)+'\n')

assert np.allclose(params(np.array([0,.1]))[0],[6.05,6.56425])

assert np.allclose(params(np.array([0,.1]))[1],[.2,.275445])

assert np.allclose(params(np.array([0,.1]))[2],[1,1.2])

assert len(CAPTIONS)==15

(OUT/'captions.json').write_text(json.dumps(CAPTIONS,indent=2),encoding='utf-8')

(OUT/'CAPTIONS.md').write_text('# Figure captions\n\n'+ '\n\n'.join(f'## Review figure {n}\n\n{CAPTIONS[n]["caption"]}' for n in sorted(CAPTIONS))+'\n',encoding='utf-8')

print('All 15 figures complete; forward-model consistency checks passed.',flush=True)

