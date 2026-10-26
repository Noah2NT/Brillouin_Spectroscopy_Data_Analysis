"""Four visual treatments of the same schematic; no scientific inputs change."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'style_options'
OUT.mkdir(exist_ok=True)
for name in ['times.ttf','timesbd.ttf','timesi.ttf','calibri.ttf','calibrib.ttf',
             'calibrii.ttf','consola.ttf','consolab.ttf','consolai.ttf']:
    font_manager.fontManager.addfont('C:/Windows/Fonts/'+name)

source=(ROOT/'generate_figure_01.py').read_text(encoding='utf-8-sig')
options=[
    ('A','Clean journal','Arial | restrained colour | flat vector shapes',
     'Arial','#24649A','#BE5A26','#263238','#7A858B','#E7EFF2','#607C86','#BCCDD5'),
    ('B','Classic academic','Times New Roman | monochrome | fine outlines',
     'Times New Roman','#202020','#737373','#222222','#999999','#F8F8F8','#555555','#DEDEDE'),
    ('C','Soft biological','Calibri | teal and violet | softer cell illustration',
     'Calibri','#147D7A','#8A55A1','#2F414B','#9AABB3','#E1F3EC','#5D9A88','#A8CFBE'),
    ('D','Technical schematic','Consolas | strong geometry | unfilled line drawing',
     'Consolas','#174A7B','#B85828','#263749','#8192A3','#FFFFFF','#263749','#FFFFFF'),
]
for key,title,description,font,blue,orange,ink,grey,cell,outline,nucleus in options:
    s=source.replace("OUT = ROOT / 'drafts'","OUT = ROOT / 'style_options'")
    s=s.replace("'Arial'",repr(font)).replace("'Arial:italic'",repr(font+':italic'))
    s=s.replace("'Arial:bold'",repr(font+':bold'))
    for old,new in [('#24649A',blue),('#BE5A26',orange),('#263238',ink),
                    ('#7A858B',grey),('#E7EFF2',cell),('#607C86',outline),('#BCCDD5',nucleus)]:
        s=s.replace(old,new)
    s=s.replace("for ext in ['pdf','svg','png']:","for ext in ['svg','png']:")
    s=s.replace("OUT/'figure_01_cell_mechanics_v1'",f"OUT/'style_{key}'")
    s=s.replace("OUT/'figure_01_caption.txt'",f"OUT/'style_{key}_caption.txt'")
    s=s.replace("print('Figure 1 v1 exported as PDF, SVG and 300 dpi PNG. Comparison checks passed.')",'')
    if key=='B':
        s=s.replace('lw=1.6','lw=1.1').replace('lw=2.5','lw=1.8')
        s=s.replace("facecolor='#D3DADF'","facecolor='#F6F6F6'")
    if key=='C':
        insertion="""
for cx,cy,angle in [(3.8,2.4,25),(6.3,2.5,-35),(4.05,3.0,-15),(5.95,3.0,30)]:
    ax.add_patch(Ellipse((cx,cy),.28,.13,angle=angle,facecolor='#B3DCCC',edgecolor='none'))
"""
        s=s.replace("ax.plot([2.4,7.6]",insertion+"\nax.plot([2.4,7.6]")
        s=s.replace('lw=2.5','lw=2.8')
    if key=='D':
        s=s.replace("facecolor='#B4BDC2'","facecolor='white'")
        s=s.replace("facecolor='#D3DADF'","facecolor='white'")
        s=s.replace('edgecolor=INK,lw=1.2','edgecolor=INK,lw=1.2,hatch=\'///\'')
        s=s.replace("edgecolor='#263749',lw=1.1", "edgecolor='#263749',lw=1.1,ls='--'")
        s=s.replace('fontweight=\'bold\'',"fontweight='normal'")
    exec(compile(s,str(ROOT/'generate_figure_01.py'),'exec'),{'__file__':str(ROOT/'generate_figure_01.py')})

# One readable board makes the same content easy to compare.
W,H=1000,1130
board=Image.new('RGB',(2*W+60,2*H+60),'#E9EDF0')
title_font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',32)
sub_font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23)
for i,(key,title,description,*_) in enumerate(options):
    tile=Image.new('RGB',(W,H),'white')
    draw=ImageDraw.Draw(tile)
    draw.text((30,22),key+'  '+title,font=title_font,fill='#202A33')
    draw.text((30,70),description,font=sub_font,fill='#59656D')
    im=Image.open(OUT/f'style_{key}.png').convert('RGB')
    im.thumbnail((W-60,H-140))
    tile.paste(im,((W-im.width)//2,125))
    board.paste(tile,(20+(i%2)*(W+20),20+(i//2)*(H+20)))
board.save(OUT/'four_style_comparison.png')
(OUT/'README.md').write_text('# Figure style options\n\nThese four options share identical schematic content, force curves and labels. Only visual styling changes.\n\n'+
    '\n'.join(f'- {key}: {title}. {description}.' for key,title,description,*_ in options)+
    '\n\nNone is approved yet. A PNG preview and editable SVG are supplied for each.\n',encoding='utf-8')
print('Four PNG/SVG options and comparison board generated; physics checks passed in all four.')
