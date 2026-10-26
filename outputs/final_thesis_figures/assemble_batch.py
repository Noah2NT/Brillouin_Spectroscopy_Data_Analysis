"""Integrate the figure batch into a separate copy of the thesis and make review assets."""
from pathlib import Path
import json,re,shutil
from html import escape
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'thesis_figure_batch'
FIG=OUT/'figures'
OLD=HERE.parent/'thesis_revision'
caps={int(k):v for k,v in json.loads((OUT/'captions.json').read_text()).items()}
s=(OLD/'main.tex').read_text(encoding='utf-8-sig')
latex={
1:r'Similar appearances can conceal different mechanical responses. An AFM probe applies force to a cell; at the same force, the higher-stiffness illustrative response has less indentation. The curves are schematic.',
2:r'How Brillouin microscopy produces a spectral image. (a) Incident light is focused into a sample and backscattered light is directed to a separate spectrometer and detector. Acoustic-wave motifs are symbolic. (b) Each spatial scan position contains a spectrum. (c) Expanding guides connect one pixel to its cartoon photon-count spectrum. The peaks, geometry and counts are schematic; the angle is ideal backscattering.',
3:r'Reading a Brillouin spectrum. (a) BFS is the separation from the incident-light frequency, and FWHM is the width at half peak height. The lower panels distinguish (b) a centre shift from (c) increased linewidth. Curves are schematic; the simulation models only the positive-shift Brillouin band.',
4:r'Different materials can have separated Brillouin responses. These schematic water and Plexiglass peaks introduce material contrast; they are not the water--glycerol simulation calibration.',
5:r'A finite collection volume can include sample and surrounding medium. Their weighted local responses combine into one measured spectrum. The colours identify contributions, not separate reconstructed depth positions.',
6:r'Shorter exposure increases relative photon shot noise. Connected orange lines show independent Poisson-count realisations, with faint blue expected traces. The underlying mixed response and photon rate are fixed; only exposure changes. Raw-count axes change scale between panels. The illustrative scale is 1000 expected water-reference photons per second in the displayed 5.25--7.35 GHz band; these times are not an experimental calibration.',
7:r'A single peak misses the shape of this chosen two-component mixture. (a) The true components and their sum. (b) The same sum and its best single-Lorentzian least-squares fit. The panels have common frequency and intensity scales. Stronger overlap can make the mismatch less apparent.',
8:r'Known and unknown spectral unmixing. Known references permit fitting of contributions; unknown references must be estimated from variation across mixed measurements. Each small spectrum is one continuous curve in the Brillouin band. Recovered outputs are illustrative, not benchmark results.',
9:r'Four simulation configurations and their recovery targets. (1) Known fixed spectra are used to estimate mixture weights. (2) Unknown fixed spectra require estimation of component spectra and weights. (3) Known concentration-dependent spectra are used to estimate a concentration distribution. (4) Unknown varying spectra require estimation of contributions and spectral relationships. The final panel illustrates the prescribed BFS (blue dashed line, left axis) and intrinsic linewidth (orange solid line, right axis) versus glycerol mass percentage. All example outputs are schematic, not recovery results. Unknown-spectrum weights require intensity calibration to represent collection fractions; the final configuration does not imply uniquely recoverable absolute concentration.',
10:r'Three-dimensional sample definitions. A uniform 10\% solution and a radial gradient are surrounded by water. The gradient cutaway reveals prescribed internal concentration. The red array shows collection regions at representative lateral scan positions, each spanning the 30 $\mu\mathrm m$ axial collection range, from the sample centre at $z=0$ to $z=30\,\mu\mathrm m$. These are simulation test objects, not reconstructed volumes or diffusion predictions.',
11:r'Formation of a fixed-component spectrum along one collection ray. The central ray contains collection fractions 33.3\% water and 66.7\% homogeneous 10\% solution (full-band signal fractions 29.4\% and 70.6\%, respectively). Coloured curves include these weights and the concentration-dependent response areas; their exact sum is black. The spectra include the assumed instrument response. Intensity denotes spectral density in the common arbitrary scale.',
12:r'Formation of a varying-concentration spectrum. (a) Coloured points lie at different depths within one collection region. (b) Local spectra are individually scaled and offset vertically for visibility. (c) Water and concentration-bin contributions are depth-weighted and summed exactly. The colour bar identifies glycerol concentration for the coloured contributions. Colour represents glycerol mass fraction. Collection begins at the centre and traverses one side of the gradient. These are depth contributions to one pixel, not different lateral pixels.',
13:r'Here $w$ denotes glycerol mass fraction (percentage divided by 100). Prescribed response laws over the initial 0--10\% glycerol mass range. BFS and intrinsic linewidth follow the provisional model. Integrated intensity is chosen to increase linearly by 20\% across this range. The curves are simulation inputs, not measured or fitted calibrations; integrated intensity means peak area.',
14:r'Noiseless forward-model datasets. Left: three-dimensional sample models and red collection indicators. Middle: the central lateral spectral line, displayed over 5.65--7.00 GHz; the complete simulated band is 4.50--8.50 GHz. The white line selects $x=y=0$. Right: contributions to that single pixel and their exact black sum. Intensity denotes spectral density on the common arbitrary scale. Both rows use common scales. Instrument broadening and the chosen increasing-intensity law are included; random noise and additive background are absent.',
15:r'Poisson-noise examples corresponding to Figure~\ref{fig:noiseless_dataset}, with the same geometry and heatmap colour limits. The reference budget is 10000 expected water photons per pixel in the full band. Counts are converted to spectral density (labelled intensity) using the known count scale and bin width for display. Orange traces show noisy pixels and faint curves show their expectations. These are simulations, not experimental measurements.'}

def block(n,label,width='1.0'):
    return (r'\begin{figure}[htbp]\centering'+'\n'+
        rf'\includegraphics[width={width}\linewidth]{{figures/{caps[n]["stem"]}.pdf}}'+'\n'+
        r'\caption{'+latex[n]+'}\n'+rf'\label{{{label}}}\end{{figure}}')

mapping={'fig:healthy-cancer':1,'fig:brillouin_setup':2,'fig:brillouin_spectrum':3,
 'fig:water_plexiglas':4,'fig:collection_volume':5,'fig:noise':6,'fig:single_peak_fit':7,
 'fig:known_unknown_unmixing':8,'fig:study-design':9,'fig:concentration_parameters':13,
 'fig:noiseless_dataset':14}
for old in re.findall(r'\\begin\{figure\}.*?\\end\{figure\}',s,flags=re.S):
    label=re.search(r'\\label\{([^}]+)\}',old)[1]
    if label in mapping:s=s.replace(old,block(mapping[label],label,'.72' if mapping[label]==1 else '1.0'))

# Introduce material contrast before collection-volume mixing, matching the visual story.
b4=block(4,'fig:water_plexiglas');s=s.replace(b4,'')
anchor=r'\subsection{Mixed spectra and measurement limitations}'
s=s.replace(anchor,'Different materials can produce distinct spectral responses. Figure~\\ref{fig:water_plexiglas} introduces this contrast before the collection of mixed signals is considered.\n\n'+b4+'\n\n'+anchor)

anchor='These fields define mathematical test objects, not equilibrium droplets or physical diffusion models.'
s=s.replace(anchor,'Figure~\\ref{fig:sample_models_3d} shows both prescribed fields in three dimensions and identifies the collection range.\n\n'+block(10,'fig:sample_models_3d')+'\n\n'+anchor)
anchor='For the varying sample, it is useful to describe the collection-weighted distribution'
s=s.replace(anchor,'Figure~\\ref{fig:fixed_formation} follows the central collection ray from the three-dimensional model to its two weighted spectral contributions and their sum.\n\n'+block(11,'fig:fixed_formation')+'\n\n'+anchor)
anchor='The noiseless recorded spectrum includes instrument broadening:'
s=s.replace(anchor,'Figure~\\ref{fig:varying_formation} expands the corresponding process for the varying sample, showing local responses at immediate depth positions within one collection volume. These depths are distinct from the lateral scan positions in Figure~\\ref{fig:noiseless_dataset}.\n\n'+block(12,'fig:varying_formation')+'\n\n'+anchor)

s=s.replace('$a=0.85$, $b=2.0$, $\\delta_A=0$', '$a=0.85$, $b=2.0$, $\\delta_A=+0.20$')
old='Integrated intensity will remain constant in the baseline because no verified intensity calibration has been supplied. Sensitivity cases with $\\delta_A=-0.20$ and $+0.20$ produce a 20\\% decrease or increase at the upper concentration limit. These cases test the dependence of the recovery on the assumed response; they are not measured trends.'
new='Integrated intensity will increase linearly in the baseline, with $\\delta_A=+0.20$, so that the 10\\% solution has 1.20 times the full-band area of water. This is a prescribed simulation choice because no verified intensity calibration has been supplied. Constant-area ($\\delta_A=0$) and decreasing-area ($\\delta_A=-0.20$) cases can test sensitivity to this assumption; none is presented as a measured trend.'
assert old in s;s=s.replace(old,new)
s=s.replace('A baseline branch will assume constant $H$; a calibrated branch may supply $H$ from the simulation model to investigate the value of intensity information.', 'The primary distribution estimates will be signal-weighted, because the baseline response area varies with concentration. A reference-assisted branch may supply $H$ from the simulation model to investigate the value of intensity calibration; a constant-$H$ branch will be a deliberate model-mismatch comparison rather than an assumption that the baseline has constant area.')
s=s.replace('Figure~\\ref{fig:noise} illustrates the effect of random noise on an otherwise unchanged spectrum.', 'Figure~\\ref{fig:noise} illustrates how reducing exposure time at fixed photon rate increases relative shot noise in an otherwise unchanged mixed response. Its connected count traces use raw counts rather than area-normalised intensities.')
anchor='Counts will not be converted to seconds without a measured photon rate.'
s=s.replace(anchor,anchor+' The exposure-time illustration in Figure~\\ref{fig:noise} separately assumes 1000 expected water-reference photons per second within its displayed 5.25--7.35 GHz band. That assumed rate defines the illustrative seconds and is not a calibration of the reference microscope; the main benchmark remains specified by photon budgets.')
s=s.replace('The observed spectra remain dominated by water because the collection interval extends beyond the sample.', 'The displayed line-scan window is narrowed to 5.65--7.00 GHz to make the spectral variation visible, without changing the simulated frequency bins. The right-hand panels decompose the central mixed pixel into contributions whose sum equals the corresponding line-scan spectrum. The observed spectra remain dominated by water because the collection interval extends beyond the sample.')
# Match the shorter, one-sided collection interval throughout the methods.
s=s.replace(r'$-50\leq z\leq50\,\mu\mathrm m$',r'$0\leq z\leq30\,\mu\mathrm m$')
s=s.replace('400 midpoint samples','120 midpoint samples')
s=s.replace('Here the baseline collection length is 100', 'Here the illustrative baseline collection length is 30')
s=s.replace(r'\int_{-L_z/2}^{L_z/2}',r'\int_{0}^{L_z}')
s=s.replace('interval and zero outside it.', 'interval $0\\leq z\\leq L_z$ and zero outside it. This interval starts at the sample midplane and traverses one side of the sphere.')
s=s.replace(r'$L_z=100\,\mu\mathrm m$',r'$L_z=30\,\mu\mathrm m$').replace(r'$p_i=0.40$',r'$p_i=2/3$')
s=s.replace(r'$L_z=10,100\,\mu\mathrm',r'$L_z=10,30,100\,\mu\mathrm')
s=s.replace('The 100 $\\mu\\mathrm m$ baseline','The 30 $\\mu\\mathrm m$ baseline')
s=s.replace('The observed spectra remain dominated by water because the collection interval extends beyond the sample.', 'The observed spectra mix sample and water because the collection interval extends beyond the sample. The central fixed-sample ray now has two-thirds sample weighting and one-third water weighting.')
s=s.replace(block(14,'fig:noiseless_dataset'),block(14,'fig:noiseless_dataset')+'\n\n'+r'Figure~\ref{fig:noisy_dataset} shows matching Poisson-noise examples at a reference budget of 10000 water photons per pixel.'+'\n\n'+block(15,'fig:noisy_dataset'))
s=re.sub(r'% Existing artwork.*?\\newcommand\{\\draftimage\}[^\n]*\n','',s,flags=re.S)
assert 'Figure placeholder' not in s
assert len(re.findall(r'\\begin\{figure\}',s))==15
labels=re.findall(r'\\label\{([^}]+)\}',s);refs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',s)
assert len(labels)==len(set(labels));assert set(refs)<=set(labels)
for path in re.findall(r'\\includegraphics\[[^]]+\]\{([^}]+)\}',s):assert (OUT/path).exists(),path
(OUT/'main.tex').write_text(s,encoding='utf-8')
shutil.copy2(OLD/'BrillouinMicroscopy.bib',OUT/'BrillouinMicroscopy.bib')
for name in ['generate_all_figures.py','generate_figure_02.py','assemble_batch.py']:
    shutil.copy2(HERE/name,OUT/'source'/name)

order=[1,2,3,4,5,6,7,8,9,10,13,11,12,14,15]
manifest=[]
for paper_no,review_no in enumerate(order,1):
    manifest.append({'paper_figure':paper_no,'review_id':review_no,**caps[review_no]})
(OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',26)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
review=OUT/'review';review.mkdir(exist_ok=True)
for start in [0,4,8,12]:
    group=order[start:start+4]
    board=Image.new('RGB',(1840,1600 if len(group)>2 else 800),'#E8ECEF')
    for j,n in enumerate(group):
        tile=Image.new('RGB',(900,770),'white');d=ImageDraw.Draw(tile)
        short=caps[n]['stem'].split('_',2)[2].replace('_',' ')
        d.text((20,16),f'Figure {order.index(n)+1}: {short}',font=font,fill='#263238')
        paper_no=order.index(n)+1;d.text((20,55),f'Stable file ID {n}',font=small,fill='#68757D')
        im=Image.open(FIG/(caps[n]['stem']+'.png')).convert('RGB');im.thumbnail((860,660))
        tile.paste(im,((900-im.width)//2,100+(660-im.height)//2))
        board.paste(tile,(10+(j%2)*920,10+(j//2)*790))
    board.save(review/f'overview_{start//4+1}.png')
html=['<!doctype html><html><head><meta charset="utf-8"><title>Thesis figure review</title><style>body{font:16px Arial;color:#263238;max-width:1100px;margin:40px auto;padding:0 24px}img{width:100%;height:auto}article{margin:50px 0 80px}p{line-height:1.55}a{color:#24649A}</style></head><body><h1>Thesis figure review</h1><p>Figure numbers follow the paper and your latest feedback. Filenames retain stable IDs from the earlier round. Click a preview for the full PNG; PDF and SVG versions are beside it.</p>']
for entry in manifest:
    stem=entry['stem'];html.append(f'<article id="review-{entry["review_id"]}"><h2>Figure {entry["paper_figure"]} · File ID {entry["review_id"]}</h2><a href="figures/{stem}.png"><img src="figures/{stem}.png"></a><p>{escape(entry["caption"])}</p><a href="figures/{stem}.pdf">PDF</a> · <a href="figures/{stem}.svg">SVG</a></article>')
html.append('</body></html>');(OUT/'REVIEW.html').write_text('\n'.join(html),encoding='utf-8')
print('Paper integrated: 15 figures, all cross-references and image paths checked. Review gallery assembled.')
