"""Original feature-map drawing using the U-Net Fig. 1 visual convention."""
from pathlib import Path
from html import escape
import math,json
ROOT=Path(__file__).resolve().parents[1]
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="519pt" height="358.58pt" viewBox="0 0 1100 760">','<rect width="1100" height="760" fill="white"/>']
def text(x,y,t,size=17,anchor='middle'):
    parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial" font-size="{size}" fill="#182833">{escape(t)}</text>')
def arrow(id,pts,color='#314754',width=2):
    d='M'+' L'.join(f'{x},{y}' for x,y in pts)
    x,y=pts[-1];px,py=pts[-2];a=math.atan2(y-py,x-px)
    tri=[(x,y),(x-9*math.cos(a)+4*math.sin(a),y-9*math.sin(a)-4*math.cos(a)),(x-9*math.cos(a)-4*math.sin(a),y-9*math.sin(a)+4*math.cos(a))]
    parts.append(f'<g id="{id}"><path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"/><path d="M'+ ' L'.join(f'{u},{v}' for u,v in tri)+f' Z" fill="{color}"/></g>')
def fmap(id,x,y,w,h,ch,grid=None,fill='#aacbdc'):
    parts.append(f'<g id="{id}"><rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="#395569" stroke-width="1.2"/>')
    text(x+w/2,y-13,str(ch))
    if grid:text(x+w/2,y+h+25,grid,16)
    parts.append('</g>')
def pair(name,x,y,h,ch,grid):
    fmap(name+'-conv1',x,y,20,h,ch,grid);fmap(name+'-conv2',x+45,y,20,h,ch)
    arrow(name+'-conv',[(x+20,y+h/2),(x+45,y+h/2)])
    title=name.replace('E','Encoder Block ').replace('D','Decoder Block ')
    if name.startswith('D') and name!='D1':text(x+210,y+h/2,title,17)
    else:text(x+32.5,y-54,title,17)
def concat(name,x,y,h,left,right):
    fmap(name+'-skip',x,y,18,h,'',fill='#ffffff')
    fmap(name+'-incoming',x+18,y,16,h,'')
    text(x+17,y-13,f'{left+right}')
arrow('input',[(42,160),(90,160)])
arrow('residual',[(60,160),(60,8),(1000,8),(1000,147)],width=1.7)
text(520,30,'Identity residual',18)
arrow('pool-E2',[(145,220),(145,325),(190,325)],'#bd7237',2.6)
arrow('pool-E3',[(245,360),(245,451),(290,451)],'#bd7237',2.6)
arrow('to-bottleneck',[(345,472),(345,541),(440,541)])
arrow('to-D3',[(470,541),(520,541),(520,451),(550,451)],'#277e86',2.6)
arrow('to-D2',[(670,451),(710,451),(710,400),(687,400),(687,360)],'#277e86',2.6)
arrow('to-D1',[(790,325),(840,325),(840,265),(807,265),(807,220)],'#277e86',2.6)
for name,start,end,y in [('E1',155,790,160),('E2',255,670,325),('E3',355,550,451)]:
    arrow('skip-'+name,[(start,y),(end,y)],'#647f8d',1.8)
    text((start+end)/2,y-45,'Skip from',16)
    text((start+end)/2,y-25,'Encoder Block '+name[1:],16)
pair('E1',90,100,120,32,'20 × 20')
pair('E2',190,290,70,64,'10 × 10')
pair('E3',290,430,42,128,'5 × 5')
fmap('bottleneck',440,520,30,42,256,'5 × 5',fill='#68a9ad')
text(455,477,'Bottleneck',19)
for name,x,y,h,skip,inc,out,grid in [('D3',550,430,42,128,256,128,'5 × 5'),('D2',670,290,70,64,128,64,'10 × 10'),('D1',790,100,120,32,64,32,'20 × 20')]:
    concat(name,x,y,h,skip,inc)
    pair(name,x+55,y,h,out,grid)
    arrow(name+'-first-conv',[(x+34,y+h/2),(x+55,y+h/2)])
arrow('project',[(910,160),(935,160)])
fmap('projection',935,100,16,120,32,'20 × 20',fill='#68a9ad')
arrow('to-add',[(951,160),(987,160)])
parts.append('<g id="addition"><circle cx="1000" cy="160" r="13" fill="white" stroke="#314754" stroke-width="2"/></g>')
text(1000,166,'+',23);arrow('output',[(1013,160),(1060,160)])
text(45,90,'h',22);text(1045,115,'U(h)',22);text(1045,195,'32 × 20 × 20',16)
text(100,275,'2 × 2',16);text(200,405,'2 × 2',16)
text(935,270,'1 × 1 Convolution',16)
# Dedicated legend band below every architecture object and dimension label.
parts.append('<g id="legend-band">')
arrow('legend-conv',[(110,624),(145,624)])
text(165,630,'3 × 3 Convolution +',16,'start')
text(165,651,'Group Normalization +',16,'start')
text(165,672,'Gaussian Error Linear Unit',16,'start')
arrow('legend-pool',[(110,709),(145,709)],'#bd7237',2.6);text(165,715,'2 × 2 Average Pooling',16,'start')
arrow('legend-resize',[(620,624),(655,624)],'#277e86',2.6);text(675,630,'Bilinear Resize',16,'start')
parts.append('<g id="legend-copied-features"><rect x="627" y="656" width="22" height="18" fill="white" stroke="#395569" stroke-width="1.2"/></g>')
text(675,672,'Copied encoder features',16,'start')
text(620,715,'Above maps: number of channels',16,'start')
text(620,737,'Below maps: height × width (grid cells)',16,'start')
parts.append('</g>')
parts.append('</svg>')
(ROOT/'fig/adaptiveunet_structure.svg').write_text('\n'.join(parts),encoding='utf-8')
(ROOT/'revision_experiments/figure_references/provenance.json').write_text(json.dumps({'reuse':'visual conventions only; independently drawn from implementation','primary_reference':{'title':'U-Net: Convolutional Networks for Biomedical Image Segmentation','figure':1,'url':'https://arxiv.org/pdf/1505.04597'},'related_reference':{'title':'U-FNO: An enhanced Fourier neural operator-based deep-learning model for multiphase flow','figure':2,'doi':'10.1016/j.advwatres.2022.104180'},'implementation':'data_training.py: AdaptiveUNet; 3 encoder blocks, 2 average pools, 1 bottleneck convolution, 3 skip-connected decoder blocks, identity residual; one shared instance across 4 Fourier layers','feature_map_geometry':'schematic sizes encode resolution ordering; widths are schematic, channel counts explicit'},indent=2),encoding='utf-8')
print('Created literature-referenced feature-map schematic.')
