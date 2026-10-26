"""Brillouin acquisition schematic, with serif type and blue/orange vector artwork."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Ellipse, Polygon, ConnectionPatch, Arc

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'drafts'
OUT.mkdir(exist_ok=True)
for name in ['arial.ttf','arialbd.ttf','ariali.ttf']:
    font_manager.fontManager.addfont('C:/Windows/Fonts/'+name)
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9,
    'mathtext.fontset':'cm',
    'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
BLUE,ORANGE,INK,GREY='#24649A','#BE5A26','#263238','#7A858B'
fig=plt.figure(figsize=(6.3,3.45),facecolor='white')
gs=fig.add_gridspec(1,3,width_ratios=[1.25,1,1.1],
    left=.025,right=.98,bottom=.18,top=.87,wspace=.46)
optic=fig.add_subplot(gs[0]);grid=fig.add_subplot(gs[1]);spec=fig.add_subplot(gs[2])

def arrow(ax,start,end,color=INK,lw=1.6,style='-|>',scale=11):
    ax.annotate('',xy=end,xytext=start,
        arrowprops=dict(arrowstyle=style,color=color,lw=lw,mutation_scale=scale))

optic.set(xlim=(-.1,11.2),ylim=(0,11.2))
optic.axis('off')
optic.text(0,1.08,'(a)  Illuminate and collect',transform=optic.transAxes,
    fontweight='bold',ha='left',va='bottom')
# Beam splitter and physically separate detection branch.
optic.add_patch(Rectangle((.0,7.40),2.6,1.50,facecolor='#EDF3F6',edgecolor=INK,lw=1.2))
optic.text(1.3,8.15,'Laser\n$f_{\\mathrm{inc}}$',ha='center',va='center')
optic.add_patch(Rectangle((2.0,9.3),6.0,1.55,facecolor='#F1F3F4',edgecolor=INK,lw=1.2))
optic.text(5,10.075,'Spectrometer\n+ detector',ha='center',va='center',fontsize=8)
optic.plot([2.2,4.87,4.87,5.0],[8.175,8.175,5.65,3.65],color=BLUE,lw=2)
arrow(optic,(4.87,7.0),(4.87,6.3),BLUE)
optic.plot([5.0,5.13,5.13],[3.65,5.65,9.5],color=ORANGE,lw=2,ls=(0,(5,2)))
arrow(optic,(5.13,8.72),(5.13,9.35),ORANGE)
optic.add_patch(Polygon([[4.5,8.175],[5,8.675],[5.5,8.175],[5,7.675]],
    facecolor='white',edgecolor=INK,lw=1.2,zorder=5))
optic.plot([4.68,5.32],[8.49,7.86],color=GREY,lw=1.1,zorder=6)
optic.annotate('Beam splitter',xy=(5.45,8.32),xytext=(6.5,8.85),ha='left',va='center',fontsize=7.5,arrowprops=dict(arrowstyle='-',color=GREY,lw=.9))
optic.add_patch(Ellipse((5,5.65),2.45,.43,facecolor='#D6DFE4',edgecolor=INK,lw=1.2,zorder=5))
optic.annotate('Objective',xy=(3.85,5.65),xytext=(.55,5.65),
    ha='left',va='center',arrowprops=dict(arrowstyle='-',color=GREY,lw=.9))
focus=(5,3.65)
optic.text(4.5,4.8,r'$f_{\mathrm{inc}}$',ha='right',color=BLUE,fontsize=9)
optic.text(6.0,6.3,'Scattered light\n$f_{\\mathrm{inc}}\\pm f_{\\mathrm{B}}$',ha='left',va='center',
    fontsize=8,color=ORANGE)

# Vertical wave motifs symbolise axial acoustic fluctuations, not particle tracks.
optic.add_patch(Rectangle((3.55,1.15),2.9,2.5,facecolor='#E7EFF2',edgecolor='#607C86',lw=1.3))
wave_z=np.linspace(1.4,3.35,250)
for centre in [3.95,4.98,6.0]:
    wave_x=centre+.19*np.sin(2*np.pi*(wave_z-1.4)/.6)
    optic.plot(wave_x,wave_z,color=BLUE,lw=1.6,alpha=.8)
optic.plot(*focus,marker='o',ms=4.5,color=INK)
optic.text(5,.68,'Sample',ha='center',va='center')
optic.annotate('Acoustic\nwaves',xy=(3.9,2.3),xytext=(.1,2.35),
    ha='left',va='center',fontsize=7.5,
    arrowprops=dict(arrowstyle='-',color=GREY,lw=.9))
# Integrate the angle into the optical drawing instead of adding disconnected arrows.
optic.add_patch(Arc(focus,1.45,1.45,theta1=-90,theta2=90,color=GREY,lw=1))
optic.text(6.5,3.55,'Backscattering\n'+r'$\theta=180^\circ$',
    ha='left',va='center',fontsize=7.5)

grid.set(xlim=(0,1),ylim=(0,1),aspect='equal')
grid.axis('off')
grid.text(0,1.15,'(b)  Scan the sample',transform=grid.transAxes,
    fontweight='bold',ha='left',va='bottom')
origin=.03; step=.185
selected=(4,2)
for row in range(5):
    for col in range(5):
        chosen=(col,row)==selected
        grid.add_patch(Rectangle((origin+col*step,.055+row*step),step,step,
            facecolor=ORANGE if chosen else '#F1F5F7',edgecolor=GREY,
            lw=1.0))
cx=origin+(selected[0]+.5)*step;cy=.055+(selected[1]+.5)*step
grid.plot(cx,cy,'o',color='white',ms=4)


grid.text(.49,-.065,'$x$ position',ha='center',va='center')
grid.text(-.08,.52,'$y$ position',ha='center',va='center',rotation=90)
grid.text(.40,-.18,'One spectrum at\neach scan position',ha='center',va='top',fontsize=8)

spec.set(xlim=(-1.8,1.8),ylim=(0,1.32))
spec.text(0,1.08,'(c)  Record a spectrum',transform=spec.transAxes,
    fontweight='bold',ha='left',va='bottom')
for side in ['top','right']:
    spec.spines[side].set_visible(False)
for side in ['left','bottom']:
    spec.spines[side].set_color(INK)
    spec.spines[side].set_linewidth(1.15)
spec.set_yticks([])
spec.set_xticks([-1.12,0,1.12],['$-f_{\\mathrm{B}}$','0','$+f_{\\mathrm{B}}$'])
spec.tick_params(axis='x',length=4,color=INK,pad=7)
spec.set_xlabel(r'Frequency shift, $f$ (GHz)',labelpad=8,fontsize=8)
spec.set_ylabel('Photon count',labelpad=5,fontsize=8)
f=np.linspace(-1.8,1.8,1200)
# Deliberately schematic bell shapes, not a calibration or fitted lineshape.
rayleigh=1.03/(1+(f/.125)**2)
stokes=.33/(1+((f+1.12)/.17)**2)
anti=.33/(1+((f-1.12)/.17)**2)
curve=.025+rayleigh+stokes+anti
assert np.isfinite(curve).all() and (curve>0).all()
assert np.isclose(stokes.max(),anti.max())
spec.plot(f,curve,color=GREY,lw=2)
for low,high,col in [(-1.65,-.57,BLUE),(.57,1.65,ORANGE)]:
    mask=(f>=low)&(f<=high)
    spec.plot(f[mask],curve[mask],color=col,lw=2.5)
spec.text(0,1.12,'Incident-light\npeak: $0$',ha='center',va='bottom',fontsize=8,color=INK)
spec.text(-1.12,.49,'Stokes\n$-f_{\\mathrm{B}}$',ha='center',va='bottom',fontsize=7.5,color=BLUE)
spec.text(1.12,.49,'Anti-Stokes\n$+f_{\\mathrm{B}}$',ha='center',va='bottom',fontsize=7.5,color=ORANGE)

# Workflow arrows sit between panels, avoiding the plotted data and labels.
fig.canvas.draw()
for a,title in [(optic,'(a)  Illuminate and collect'),(grid,'(b)  Scan the sample'),(spec,'(c)  Spectrum at selected pixel')]:
    a.texts[0].remove()
    fig.text(a.get_position().x0,.95,title,fontweight='bold',ha='left',va='bottom')
for a,b in []:
    left=a.get_position();right=b.get_position()
    start=(left.x1+.008,(left.y0+left.y1)/2)
    end=(right.x0-.025,(left.y0+left.y1)/2)
    fig.add_artist(ConnectionPatch(start,end,coordsA=fig.transFigure,
        coordsB=fig.transFigure,arrowstyle='-|>',mutation_scale=13,lw=1.5,color=GREY))

# Expanding dashed guides explicitly enlarge the selected pixel into its spectrum.
box=spec.get_position()
frame_left=box.x0-.060
frame_bottom=box.y0-.155
frame_top=box.y1+.025
fig.add_artist(Rectangle((frame_left,frame_bottom),box.x1+.012-frame_left,
    frame_top-frame_bottom,transform=fig.transFigure,facecolor='none',
    edgecolor=ORANGE,lw=1.0,zorder=0))
pixel_right=origin+(selected[0]+1)*step
for pixel_y,frame_y in [(.055+(selected[1]+1)*step,frame_top),
                       (.055+selected[1]*step,frame_bottom)]:
    fig.add_artist(ConnectionPatch((pixel_right,pixel_y),(frame_left,frame_y),
        coordsA=grid.transData,coordsB=fig.transFigure,color=ORANGE,
        ls=(0,(4,3)),lw=1.2,zorder=0))

stem=OUT/'figure_02_brillouin_scan_v2'
for ext in ['pdf','svg','png']:
    fig.savefig(stem.with_suffix('.'+ext),dpi=300,bbox_inches='tight',pad_inches=.15)
plt.close(fig)
(OUT/'figure_02_caption.txt').write_text(
    'Principle of Brillouin microscopy and spectral image formation. '
    '(a) Incident laser light at frequency f_inc is focused into the sample through an objective. '
    'The same objective collects backscattered light, which is directed through a beam splitter '
    'to a separate spectrometer and detector. Backscattering is labelled theta = 180 degrees. '
    'Vertical wave motifs symbolise longitudinal acoustic fluctuations along the optical axis; '
    'they are not literal trajectories of material particles. '
    '(b) Repeating the acquisition across x and y produces a spatial grid containing a spectrum '
    'at each position; the orange square identifies one position. '
    '(c) Expanding dashed guides link that pixel to its schematic spectrum, showing the central elastic Rayleigh peak and '
    'frequency-shifted Stokes and anti-Stokes peaks. Frequency shifts are relative to the laser. '
    'Optical paths, wave amplitudes and peak shapes are schematic and not to scale; the spectrum '
    'does not represent measured or simulated quantitative data.\n',encoding='utf-8')
print('Figure 2 candidate exported as PDF, SVG and PNG; schematic checks passed.')
