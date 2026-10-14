"""Blind adversarial review of a candidate round against its base by an independent headless Claude session.

Usage: review.py <workspace> <base_round> <cand_round> [--crops 4] [--seed N] [--model MODEL]

Renders source | X | Y panels with X/Y in random order: the whole scene for both layers (features = model
detections; objects = superpixels, the final polygons), then the largest changed areas of each layer that changed.
Asks the reviewer what each gets wrong and which is more accurate, then unblinds. Writes <cand_round>/review/verdict.json.
If neither layer changed, it writes a tie without calling the reviewer.
The reviewer never sees the hypothesis, the metrics, or which side is the candidate.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OVERLAY_ALPHA,ask_claude,blend,class_maps,colour,load_json,read_rgb,save_json,workspace

PROMPT='''You are reviewing two land-cover classifications, X and Y, of the same aerial image. Read these images:
{images}
Each image has three panels: left = source photo, middle = classification X, right = classification Y.
Colours are see-through tints over the photo and follow the legend at the bottom (swatches are shown as they look
over grey pavement); magenta = unclassified. Each image's title says which layer it shows:
- features: the model's detections, pixel by pixel (gaps are possible)
- objects: superpixels, each given one class; these are the final polygons
The first images show the whole scene; the others zoom in on the areas where X and Y differ most in that layer.
{changed}

Your job is to find mistakes. Compare each classification with the source photo and list concrete errors:
wrong class on a surface, real features left unclassified, one label swallowing distinct surfaces. Pay most
attention to the places where X and Y disagree. More coloured area is NOT better if the labels are wrong.

Classes: {classes}

Reply with only JSON:
{{"x_errors":["..."],"y_errors":["..."],"verdict":"X" | "Y" | "tie","confidence":"high" | "low","reason":"one sentence"}}'''

def overlay(rgb,names,alpha=OVERLAY_ALPHA):
    out=rgb.astype('float32')
    for n in np.unique(names):
        m=names==n;out[m]=(1-alpha)*out[m]+alpha*np.array(colour(n))
    return out.astype('uint8')

def panel(rgb,x,y,box,width,legend,title):
    y0,y1,x0,x1=box;f=width/(x1-x0);size=(width,max(1,round((y1-y0)*f)))
    tiles=[Image.fromarray(a[y0:y1,x0:x1]).resize(size,Image.Resampling.NEAREST) for a in (rgb,x,y)]
    rows=(len(legend)+5)//6;img=Image.new('RGB',(3*width+20,size[1]+28+18*rows),'white');d=ImageDraw.Draw(img)
    for i,(t,lab) in enumerate(zip(tiles,('source',f'X ({title})',f'Y ({title})'))):img.paste(t,(i*(width+10),22));d.text((i*(width+10)+4,4),lab,fill='black')
    for i,n in enumerate(legend):
        lx,ly=(i%6)*(img.width//6)+4,size[1]+30+18*(i//6);d.rectangle((lx,ly,lx+12,ly+12),fill=blend((128,128,128),colour(n)),outline='black');d.text((lx+16,ly),n,fill='black')
    return img

def main():
    ap=argparse.ArgumentParser();ap.add_argument('workspace');ap.add_argument('base');ap.add_argument('cand')
    ap.add_argument('--crops',type=int,default=4);ap.add_argument('--seed',type=int);ap.add_argument('--model');a=ap.parse_args()
    ws,cfg=workspace(a.workspace);out=Path(a.cand)/'review';out.mkdir(exist_ok=True)
    rgb,_=read_rgb(cfg['image']);base=class_maps(Path(a.base)/'run');cand=class_maps(Path(a.cand)/'run')
    layers=[l for l,b,c in zip(('features','objects'),base,cand) if b is not None and c is not None]
    changed=[l for l in layers if (base[layers.index(l)]!=cand[layers.index(l)]).any()]
    if not changed:
        verdict={'preferred':'tie','confidence':None,'reason':'identical in every layer; reviewer not called','layers_changed':[]}
        save_json(out/'verdict.json',verdict);print(json.dumps(verdict,indent=1));return
    rng=np.random.default_rng(a.seed);swap=bool(rng.integers(2));H,W=rgb.shape[:2];paths=[];maps={}
    for i,l in enumerate(layers):
        X,Y=(cand[i],base[i]) if swap else (base[i],cand[i]);maps[l]=(X,Y)
    legend=sorted({n for X,Y in maps.values() for n in (*np.unique(X),*np.unique(Y))})
    over={l:(overlay(rgb,X),overlay(rgb,Y)) for l,(X,Y) in maps.items()}
    for l in layers:
        p=out/f'{len(paths):02d}_scene_{l}.png';panel(rgb,*over[l],(0,H,0,W),620,legend,l).save(p);paths.append(p)
    for l in changed:
        X,Y=maps[l];diff=ndimage.binary_dilation(X!=Y,iterations=6);lab,n=ndimage.label(diff)
        sizes=ndimage.sum(diff,lab,range(1,n+1));k=a.crops if len(changed)==1 else max(2,a.crops//2)
        for i in np.argsort(sizes)[::-1][:k]:
            ys,xs=ndimage.find_objects(lab)[i];cy,cx=(ys.start+ys.stop)//2,(xs.start+xs.stop)//2
            half=max(80,(ys.stop-ys.start)//2+30,(xs.stop-xs.start)//2+30)
            box=(max(0,cy-half),min(H,cy+half),max(0,cx-half),min(W,cx+half))
            p=out/f'{len(paths):02d}_change_{l}.png';panel(rgb,*over[l],box,420,legend,l).save(p);paths.append(p)
    note=f"Only the {changed[0]} layer differs between X and Y; judge on that layer." if len(changed)==1 else 'Both layers differ between X and Y.'
    dict_classes=lambda r:[c['name'] for c in load_json(Path(r)/'run.work'/'dictionary.json')['classes']]
    classes=', '.join(dict.fromkeys(dict_classes(a.base)+dict_classes(a.cand)))
    reply=ask_claude(PROMPT.format(images='\n'.join(str(p) for p in paths),classes=classes,changed=note),a.model)
    v=reply.get('verdict','tie');cand_side='X' if swap else 'Y'
    verdict={'preferred':'tie' if v not in ('X','Y') else ('candidate' if v==cand_side else 'base'),
        'confidence':reply.get('confidence'),'reason':reply.get('reason'),
        'candidate_errors':reply.get(f'{cand_side.lower()}_errors',[]),
        'base_errors':reply.get(f"{'y' if cand_side=='X' else 'x'}_errors",[]),'candidate_was':cand_side,'layers_changed':changed,'images':[p.name for p in paths]}
    save_json(out/'verdict.json',verdict);print(json.dumps(verdict,indent=1))

if __name__=='__main__':main()
