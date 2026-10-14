"""Shared helpers for the autotune scripts."""
import json,re,subprocess,zlib
from pathlib import Path
import numpy as np
import rasterio
from automage.config import concepts

ROOT=Path(__file__).resolve().parent
NODATA=65535

def load_json(path):return json.loads(Path(path).read_text())
def save_json(path,value):Path(path).write_text(json.dumps(value,indent=2))

def workspace(path):
    """A workspace is <dir>/workspace.json (image path, taxonomy) + reference/ + rounds/."""
    path=Path(path).resolve();return path,load_json(path/'workspace.json')

def read_rgb(path):
    """Source bands 1-3 as uint8 (percentile stretch when not 8-bit) and a validity mask."""
    with rasterio.open(path) as src:a=src.read([1,2,3],masked=True)
    valid=~np.ma.getmaskarray(a).any(0);data=a.filled(0).astype('float32');valid&=data.sum(0)>0
    if data[:,valid].max()>255:
        lo,hi=np.percentile(data[:,valid],[2,98],axis=1)
        data=(data-lo[:,None,None])/np.maximum(hi-lo,1e-6)[:,None,None]*255
    return np.clip(data,0,255).astype('uint8').transpose(1,2,0),valid

def class_maps(run):
    """Per-pixel class names for a run's features (winning class) and objects."""
    run=Path(run);work=run.with_name(run.name+'.work')
    names={t['class_id']:t['class_name'] for t in concepts(load_json(work/'dictionary.json'))}
    names[0]='unclassified';names[NODATA]='unexecuted'
    with rasterio.open(work/'classes.tif') as s:cls=s.read(1)
    lut=np.array([names.get(i,'unclassified') for i in range(int(cls.max())+1)],dtype=object)
    features=lut[cls]
    objects=None
    if (run/'objects.tif').exists():
        import pyogrio.raw
        meta,_,_,data=pyogrio.raw.read(run/'classification.gpkg',layer='objects',read_geometry=False)
        d=dict(zip(meta['fields'],data))
        with rasterio.open(run/'objects.tif') as s:ids=s.read(1)
        olut=np.full(int(ids.max())+1,'unclassified',dtype=object);olut[d['object_id'].astype(int)]=d['class_final']
        objects=olut[ids]
    return features,objects

def presence(run):
    """Every class detected at each pixel, winning or not: {class name: bool mask} from the per-concept score bands."""
    run=Path(run);work=run.with_name(run.name+'.work')
    owner={t['id']:t['class_name'] for t in concepts(load_json(work/'dictionary.json'))};out={}
    with rasterio.open(work/'concept_scores.tif') as s:
        for band,cid in enumerate(s.descriptions,1):
            name=owner[cid];m=s.read(band)>0;out[name]=out[name]|m if name in out else m
    return out

def soft_points(ws):
    """Reference points the labellers didn't agree on (dropped from the strict reference), with lenient answers.

    Each gets "options": every class either labeller gave (label or also); empty when both said uncertain, meaning
    any class is acceptable. Only "unclassified" is always wrong. Built from the label files, so it works for any
    workspace with labels_A/B.json and reference.json.
    """
    ws=Path(ws);ref=ws/'reference';strict={p['id'] for p in load_json(ref/'reference.json')['points']}
    A,B=load_json(ref/'labels_A.json'),load_json(ref/'labels_B.json');out=[]
    for p in load_json(ref/'points.json'):
        if p['id'] in strict or p['id'] not in A or p['id'] not in B:continue
        a,b=A[p['id']],B[p['id']]
        out.append({**p,'options':sorted({a['label'],b['label'],*a['also'],*b['also']}-{'uncertain'})})
    return out

def soft_ok(p,pred):
    """A soft point is acceptable when classified, with one of its options (any class if it has none)."""
    return pred!='unclassified' and pred!='unexecuted' and (not p['options'] or pred in p['options'])

# Stable colours keyed by class name, so overlays are comparable when classes are dropped or reordered.
# Chosen so that, blended at OVERLAY_ALPHA over asphalt, grass, dry grass, concrete or shadow, every pair of classes
# stays clearly distinct (CIE76 dE >= 29) and each still reads as its own legend swatch drawn over mid grey.
OVERLAY_ALPHA=.65
PALETTE={'structures':(255,0,90),'roads':(255,90,0),'concrete':(170,255,255),'impervious other':(90,0,90),
    'artificial turf':(0,255,170),'pools':(0,0,255),'turf':(0,255,0),'canopy':(0,90,0),'ground cover':(255,170,0),
    'bare earth':(255,170,170),'vehicles':(255,255,90),'trampolines':(255,170,255),'undeveloped lands':(170,170,90),
    'horse corrals':(0,170,255),'open water':(0,90,255),'agricultural lands':(0,170,0),'shadow':(0,0,0),
    'unclassified':(255,0,255),'unexecuted':(255,255,255)}
def blend(rgb,c,alpha=OVERLAY_ALPHA):return tuple(int(round((1-alpha)*a+alpha*b)) for a,b in zip(rgb,c))
def colour(name):return PALETTE.get(name,(zlib.crc32(name.encode())%200+30,90,200))

def ask_claude(prompt,model=None,retries=2):
    """Run a fresh, independent headless Claude session (Read tool only) and return the JSON it replies with."""
    cmd=['claude','-p',prompt,'--allowedTools','Read']+(['--model',model] if model else [])
    last=''
    for _ in range(retries+1):
        out=subprocess.run(cmd,capture_output=True,text=True).stdout;last=out
        m=re.search(r'(\[.*\]|\{.*\})',out,re.S)
        if m:
            try:return json.loads(m.group(1))
            except json.JSONDecodeError:pass
    raise RuntimeError(f'No JSON in reply: {last[:500]}')
