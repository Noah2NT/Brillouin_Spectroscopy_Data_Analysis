"""Apply only the requested corrections to the three original Arial figures."""
from pathlib import Path
from io import BytesIO
import shutil
import json
import sys
import numpy as np
from pypdf import PdfReader
from matplotlib.backends.backend_pdf import PdfFile, RendererPdf

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'outputs/thesis_figure_batch/source/generate_all_figures.py'
REFERENCES=ROOT/'outputs/final_thesis_figures/references/requested_corrections'
OUT=ROOT/'outputs/Thesis_Figure_Corrections'
PROJECT=ROOT/'outputs/Thesis_Overleaf/figures'
OUT.mkdir(parents=True,exist_ok=True)

source=SOURCE.read_text(encoding='utf-8-sig')
namespace={'__file__':str(SOURCE)}
exec(compile(source.split('# 1: preserve the approved illustration',1)[0],str(SOURCE),'exec'),namespace)
figures={}
namespace['save']=lambda fig,n,stem,caption:figures.update({n:fig})
selected=set(sys.argv[1:])
manifest=json.loads((OUT/'checks.json').read_text(encoding='utf-8')) if selected and (OUT/'checks.json').exists() else []

for number,next_number,name in ((9,10,'09_study_design'),(11,12,'12_fixed_mixture_formation'),(12,13,'13_gradient_mixture_formation')):
    if selected and name not in selected:
        continue
    block=source[source.index(f'# {number}:'):source.index(f'# {next_number}:')]
    exec(compile(block,str(SOURCE),'exec'),namespace)
    fig=figures[number]
    fig.set_dpi(72)
    pdf=PdfFile(BytesIO())
    pdf.newPage(*fig.get_size_inches())
    renderer=RendererPdf(pdf,300,fig.get_figheight(),fig.get_figwidth())
    fig.draw(renderer)
    frame=fig.get_tightbbox(renderer).padded(.10)
    original=PdfReader(REFERENCES/(name+'.pdf')).pages[0]
    assert np.allclose([frame.width*72,frame.height*72],[float(original.mediabox.width),float(original.mediabox.height)],atol=1e-7)
    positions=[ax.get_position().bounds for ax in fig.axes]

    if number==9:
        names=['1. Fixed sample with known spectra','2. Fixed sample with unknown spectra','3. Varying sample with known spectra','4. Varying sample with unknown spectra']
        for text,label in zip(fig.texts,names):text.set_text(label)
        axes=fig.axes
        for bar,height in zip(axes[1].patches,[1/3,2/3]):bar.set_height(height)
        # Keep the small recovered-weight example consistent with the baseline.
        for child in axes[3].child_axes:
            for bar,height in zip(child.patches,[1/3,2/3]):bar.set_height(height)

        # Panel 3 stays exactly as drawn in the original schematic.
        assert len(axes[5].patches)==21

        axes[-1].remove()
        a=axes[7]
        a.clear()
        bfs,width,_=namespace['params'](np.linspace(0,.1,100))
        a.plot(bfs,width,color=namespace['ORANGE'],lw=1.6)
        a.set(xlabel='BFS (GHz)',ylabel='Linewidth (GHz)',xlim=(6,6.65),ylim=(.19,.29),xticks=[6,6.3,6.6])
        a.tick_params(axis='y',colors='black')
        namespace['title'](a,'Recover relationship\n& contribution')
    else:
        a=fig.axes[0]
        a.set_zlabel('')
        a.text2D(.82,1.015,r'$z$ ($\mu$m)',transform=a.transAxes,fontsize=7,ha='left',va='bottom')
        if number==11:
            fig.axes[2].set_title('(c) Combined spectrum',loc='left',fontweight='bold',pad=9)
        else:
            combined=fig.axes[2]
            for line in combined.lines:
                if line.get_label()=='Integrated spectrum':line.set_label('Combined spectrum')
            combined.legend(frameon=False,fontsize=7)

    for ax,before in zip(fig.axes,positions):
        assert np.allclose(ax.get_position().bounds,before,atol=1e-12),'An axes rectangle changed.'
    for extension in ('pdf','png','svg'):
        path=OUT/(name+'.'+extension)
        fig.savefig(path,dpi=300,bbox_inches=frame,pad_inches=0,transparent=(extension!='png'))
        shutil.copy2(path,PROJECT/path.name)
    result=PdfReader(OUT/(name+'.pdf')).pages[0]
    assert np.allclose(list(result.mediabox),list(original.mediabox),atol=1e-7)
    manifest=[entry for entry in manifest if entry['file']!=name+'.pdf']
    manifest.append({'file':name+'.pdf','width_pt':float(result.mediabox.width),'height_pt':float(result.mediabox.height),'font_family':'Arial','axes_positions_preserved':True})
    namespace['plt'].close(fig)
    print('Corrected',name,flush=True)

(OUT/'checks.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
