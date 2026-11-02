"""Score one automage run for prompt tuning: metrics.json, prompts.csv and review images in <run>/eval/.

Usage: evaluate.py <run_dir>   (reads <run_dir> and its sibling <run_dir>.work)
"""
import csv,json,re,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyogrio
import pyogrio.raw
import rasterio
from PIL import Image,ImageDraw
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import shape
from automage.config import concepts

# Deliberately overlapping observations (see dictionary note); excluded from effective overlap.
OVERLAY_CLASSES={'shadow','vehicles'}
THUMB=1000

def slug(name):return re.sub(r'[^a-zA-Z0-9_-]+','_',name).strip('_')

def thumb(rgb,mask=None,color=(255,0,40),alpha=.55,dim=1.):
    out=rgb.astype('float32')*dim
    if mask is not None:out[mask]=(1-alpha)*rgb[mask]+alpha*np.array(color)
    return Image.fromarray(np.clip(out,0,255).astype('uint8'))

def resize_mask(mask,shape_):
    return np.asarray(Image.fromarray(mask.astype('uint8')*255).resize(shape_[::-1],Image.Resampling.NEAREST))>0

def main(run):
    run=Path(run).resolve();work=run.with_name(run.name+'.work');ev=run/'eval';ev.mkdir(exist_ok=True)
    summary=json.loads((run/'summary.json').read_text())
    types=concepts(json.loads((work/'dictionary.json').read_text()))
    class_names={t['class_id']:t['class_name'] for t in types}
    with rasterio.open(summary['source']) as src:
        rgb=np.clip(src.read([1,2,3]),0,255).astype('uint8').transpose(1,2,0);transform=src.transform;H,W=src.height,src.width
    with rasterio.open(work/'valid.tif') as v,rasterio.open(work/'executed.tif') as e:valid=(v.read(1)>0)&(e.read(1)>0)
    with rasterio.open(work/'concept_scores.tif') as s:scores=s.read()>0
    nvalid=int(valid.sum())

    # Class presence from per-concept score bands (same source as pipeline.export_outputs).
    cids=sorted(class_names);present={c:np.any(scores[[i for i,t in enumerate(types) if t['class_id']==c]],axis=0)&valid for c in cids}
    stack=np.stack([present[c] for c in cids]);anyclass=stack.any(0)
    core=[c for c in cids if class_names[c] not in OVERLAY_CLASSES];core_count=np.stack([present[c] for c in core]).sum(0)
    pairs={}
    for i,a in enumerate(cids):
        for b in cids[i+1:]:
            n=int((present[a]&present[b]).sum())
            if n:pairs[(a,b)]=n

    # Per-prompt masks from the raw observations.
    prompt_masks=defaultdict(lambda:np.zeros((H,W),bool));stats=defaultdict(lambda:{'detections':0,'scores':[]})
    with (work/'feature_observations.jsonl').open() as f:
        for line in f:
            r=json.loads(line);key=(r['class_id'],r['concept_id'],r['prompt'])
            prompt_masks[key]|=rasterize([(shape(r['geometry']),1)],out_shape=(H,W),transform=transform,dtype='uint8').astype(bool)
            stats[key]['detections']+=1;stats[key]['scores'].append(r['score'])
    rows=[]
    for t in types:
        for p in t['prompts']:
            key=(t['class_id'],t['id'],p);m=prompt_masks.get(key,np.zeros((H,W),bool))&valid
            same=np.zeros((H,W),bool)
            for k,o in prompt_masks.items():
                if k[0]==t['class_id'] and k!=key:same|=o
            conflicts={class_names[c]:int((m&present[c]).sum()) for c in cids if c!=t['class_id'] and (m&present[c]).any()}
            sc=stats[key]['scores'] if key in stats else []
            rows.append({'class':t['class_name'],'type':t['id'],'prompt':p,'detections':len(sc),
                'area_px':int(m.sum()),'area_pct':round(100*m.sum()/nvalid,2),'unique_px':int((m&~same).sum()),
                'mean_score':round(float(np.mean(sc)),3) if sc else '','min_score':round(float(min(sc)),3) if sc else '',
                'conflict_px':int(sum(conflicts.values())),
                'top_conflicts':'; '.join(f'{k}:{v}' for k,v in sorted(conflicts.items(),key=lambda x:-x[1])[:3])})
    with (ev/'prompts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    objects=None
    if 'objects' in dict(pyogrio.list_layers(run/'classification.gpkg')):
        meta,_,_,data=pyogrio.raw.read(run/'classification.gpkg',layer='objects',read_geometry=False)
        objects={name:np.asarray(col) for name,col in zip(meta['fields'],data)}
    metrics={
        'coverage_pct':round(100*(1-summary['coverage']['completed_unknown_pixels']/summary['coverage']['valid_pixels']),2),
        'effective_overlap_pct':round(100*int(((core_count>1)&valid).sum())/nvalid,2),
        'all_class_overlap_pct':round(100*int(((stack.sum(0)>1)&valid).sum())/nvalid,2),
        'all_class_overlap_px':int(((stack.sum(0)>1)&valid).sum()),
        'summary_cross_class_overlap_px':summary['coverage']['cross_class_overlap_pixels'],
        'feature_count':summary['feature_count'],'prompt_entries':summary['inference']['prompt_entries'],
        'class_coverage_pct':{class_names[c]:round(100*int(present[c].sum())/nvalid,2) for c in cids},
        'class_features':{c['class_name']:c['features'] for c in summary['classes']},
        'top_overlap_pairs':[{'pair':f'{class_names[a]} x {class_names[b]}','px':n,'pct':round(100*n/nvalid,2)} for (a,b),n in sorted(pairs.items(),key=lambda x:-x[1])[:10]],
        'redundant_prompts':[f"{r['type']}: {r['prompt']}" for r in rows if r['area_px'] and r['unique_px']<.01*r['area_px']],
        'silent_prompts':[f"{r['type']}: {r['prompt']}" for r in rows if not r['detections']],
        'seconds':round(summary['seconds'],1),'backend':summary['inference']['model'].get('backend'),'state':summary['state']}
    if objects is not None:
        metrics.update(objects=len(objects['purity']),objects_flagged_pct=round(100*objects['needs_review'].mean(),2),
            objects_unclassified_pct=round(100*(objects['class_final']=='unclassified').mean(),2),
            objects_mean_purity=round(float(objects['purity'].mean()),3))
    (ev/'metrics.json').write_text(json.dumps(metrics,indent=2))

    # Review images at thumbnail scale.
    scale=min(1,THUMB/max(H,W));tshape=(round(H*scale),round(W*scale))
    small=np.asarray(Image.fromarray(rgb).resize(tshape[::-1],Image.Resampling.BILINEAR))
    Image.fromarray(small).save(ev/'image.png')
    for c in cids:
        if present[c].any():thumb(small,resize_mask(present[c],tshape),dim=.8).save(ev/f'class_{slug(class_names[c])}.png')
    thumb(small,resize_mask(valid&~anyclass,tshape),color=(255,0,255),alpha=.7,dim=.8).save(ev/'unknown.png')
    for (a,b),_ in sorted(pairs.items(),key=lambda x:-x[1])[:5]:
        both=present[a]&present[b];lab,n=ndimage.label(both)
        if not n:continue
        sizes=ndimage.sum(both,lab,range(1,n+1));tiles=[]
        for idx in np.argsort(sizes)[::-1][:5]:
            ys,xs=ndimage.find_objects(lab)[idx];pad=40
            y0,y1,x0,x1=max(0,ys.start-pad),min(H,ys.stop+pad),max(0,xs.start-pad),min(W,xs.stop+pad)
            crop=rgb[y0:y1,x0:x1];f=min(1.,300/max(crop.shape[:2]))
            size=(max(1,round(crop.shape[1]*f)),max(1,round(crop.shape[0]*f)))
            orig=Image.fromarray(crop).resize(size);tint=thumb(crop,both[y0:y1,x0:x1],color=(255,200,0)).resize(size)
            tiles.append((orig,tint))
        tw=max(o.width for o,_ in tiles)*2+10;th=sum(o.height for o,_ in tiles)+10*len(tiles)+20
        sheet=Image.new('RGB',(tw,th),'white');d=ImageDraw.Draw(sheet);d.text((4,4),f'{class_names[a]} x {class_names[b]} (left: image, right: overlap)',fill='black');y=20
        for o,t in tiles:sheet.paste(o,(0,y));sheet.paste(t,(o.width+10,y));y+=o.height+10
        sheet.save(ev/f'conflict_{slug(class_names[a])}__{slug(class_names[b])}.png')
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('class_coverage_pct','class_features')},indent=2))

if __name__=='__main__':main(sys.argv[1])
