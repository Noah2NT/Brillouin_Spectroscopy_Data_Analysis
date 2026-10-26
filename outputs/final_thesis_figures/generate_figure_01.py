"""Generate Figure 1 as editable vector artwork and a print-resolution preview."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Ellipse, Polygon, Rectangle

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'drafts'
OUT.mkdir(exist_ok=True)
for name in ['arial.ttf', 'arialbd.ttf', 'ariali.ttf']:
    font = Path('C:/Windows/Fonts') / name
    if font.exists():
        font_manager.fontManager.addfont(str(font))
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 11,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'axes.linewidth': 1.15, 'savefig.facecolor': 'white',
})
BLUE, ORANGE = '#24649A', '#BE5A26'
INK, GREY = '#263238', '#7A858B'

fig = plt.figure(figsize=(6.8, 6.7), facecolor='white')
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=.36,
                      left=.15, right=.95, top=.96, bottom=.11)
ax = fig.add_subplot(gs[0])
ax.set(xlim=(0,10), ylim=(1.6,5.6), aspect='equal')
ax.axis('off')
ax.text(-.04, 1.02, '(a)  Cell indentation', transform=ax.transAxes,
        fontweight='bold', ha='left', va='bottom', color=INK)

# A neutral, original cell schematic: no copied microscopy image or scale bar.
vertices = [(3.0,2.05),(2.9,2.35),(3.35,2.8),(3.65,3.05),
            (4.0,3.6),(4.6,3.7),(5.0,3.42),
            (5.35,3.67),(5.95,3.55),(6.3,3.1),
            (6.6,2.8),(7.02,2.4),(7.0,2.05),
            (6.1,1.95),(3.85,1.95),(3.0,2.05)]
codes = [MplPath.MOVETO] + [MplPath.CURVE4]*15
ax.add_patch(PathPatch(MplPath(vertices,codes),facecolor='#E7EFF2',
                       edgecolor='#607C86',lw=1.6))
ax.add_patch(Ellipse((5.05,2.65),1.55,.88,angle=-6,
                     facecolor='#BCCDD5',edgecolor='#607C86',lw=1.1))
ax.plot([2.4,7.6],[1.99,1.99],color='#A1A9AE',lw=1.5,zorder=0)
ax.add_patch(Rectangle((4.45,4.35),3.3,.17,facecolor='#B4BDC2',
                        edgecolor=INK,lw=1.2))
ax.add_patch(Polygon([[4.66,4.35],[5.34,4.35],[5.0,3.42]],
                      facecolor='#D3DADF',edgecolor=INK,lw=1.2))
ax.annotate('',xy=(5,4.6),xytext=(5,5.35),
             arrowprops=dict(arrowstyle='-|>',lw=1.8,color=INK,mutation_scale=15))
ax.text(5.2,5.0,'$F$',fontsize=14,color=INK)
ax.annotate('AFM probe',xy=(5.3,4.08),xytext=(7.0,3.7),
             fontsize=11,color=INK,ha='left',va='center',
             arrowprops=dict(arrowstyle='-',lw=.9,color=GREY,
                             connectionstyle='angle3,angleA=0,angleB=80'))
ax.annotate('Cell',xy=(3.35,2.55),xytext=(1.7,2.7),
             fontsize=11,color=INK,ha='right',va='center',
             arrowprops=dict(arrowstyle='-',lw=.9,color=GREY))

ax = fig.add_subplot(gs[1])
ax.set(xlim=(0,1.13),ylim=(0,1.12))
ax.text(-.04,1.06,'(b)  Different mechanical responses',transform=ax.transAxes,
        fontweight='bold',ha='left',va='bottom',color=INK)
for side in ['top','right']:
    ax.spines[side].set_visible(False)
for side in ['bottom','left']:
    ax.spines[side].set_color(INK)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'Indentation, $\delta$',labelpad=24,color=INK)
ax.set_ylabel(r'Force, $F$',labelpad=13,color=INK)
ax.plot(1.13,0,marker='>',color=INK,clip_on=False,ms=5)
ax.plot(0,1.12,marker='^',color=INK,clip_on=False,ms=5)

# Same contact-law exponent for both illustrative responses; stiffness differs.
exponent=1.5
k_high,k_low=1.8,.72
xh=np.linspace(0,(1.04/k_high)**(1/exponent),240)
xl=np.linspace(0,1.07,240)
ax.plot(xh,k_high*xh**exponent,color=BLUE,lw=2.5)
ax.plot(xl,k_low*xl**exponent,color=ORANGE,lw=2.5,ls=(0,(5,2)))
force=.58
dh=(force/k_high)**(1/exponent)
dl=(force/k_low)**(1/exponent)
assert 0 < dh < dl < 1.07
assert np.isclose(k_high*dh**exponent,k_low*dl**exponent)
ax.plot([0,dl],[force,force],color=GREY,lw=1,ls=(0,(2,3)),zorder=0)
for x,col,suffix in [(dh,BLUE,'H'),(dl,ORANGE,'L')]:
    ax.plot([x,x],[0,force],color=GREY,lw=1,ls=(0,(2,3)),zorder=0)
    ax.plot(x,force,'o',color=col,ms=5)
    ax.annotate(r'$\delta_{\mathrm{'+suffix+r'}}$',xy=(x,0),xytext=(0,-8),
                 textcoords='offset points',va='top',ha='center',color=col,fontsize=12)
ax.text(.025,force+.035,'Same applied force',fontsize=9.5,color='#5F6B72',
        ha='left',va='bottom')
ax.text(.72,.99,'Higher apparent\nstiffness',color=BLUE,fontsize=11,
        ha='left',va='top',linespacing=1.2)
ax.text(.97,.47,'Lower apparent\nstiffness',color=ORANGE,fontsize=11,
        ha='center',va='top',linespacing=1.2)

stem=OUT/'figure_01_cell_mechanics_v1'
for ext in ['pdf','svg','png']:
    fig.savefig(stem.with_suffix('.'+ext),dpi=300,bbox_inches='tight',pad_inches=.12)
plt.close(fig)
(OUT/'figure_01_caption.txt').write_text(
    'Schematic illustration of how similar cell appearances can conceal different '
    'mechanical responses. (a) An AFM probe applies force to a cell. '
    '(b) Two illustrative force–indentation curves show that the same applied force '
    'produces less indentation for the response with higher apparent stiffness '
    '(delta_H < delta_L). Curves are schematic and do not represent measured data '
    'or particular healthy and cancerous cell populations.\n',encoding='utf-8')
print('Figure 1 v1 exported as PDF, SVG and 300 dpi PNG. Comparison checks passed.')

