"""Bounded tiled GPU inference, immutable provenance and shared geospatial exports."""
from __future__ import annotations
import json,math,time,signal,os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
from .config import Config,load_dictionary,concepts,sha,model_sha,write_json,ROOT
from .raster import plan,rendering,read_window

NODATA=65535

def directory_bytes(path):return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())

def profile(src,count=1,dtype='uint16',nodata=None):
    return dict(driver='GTiff',height=src.height,width=src.width,count=count,dtype=dtype,crs=src.crs,transform=src.transform,
        tiled=True,blockxsize=256,blockysize=256,compress='deflate',predictor=2,nodata=nodata,BIGTIFF='IF_SAFER',SPARSE_OK=True)

def initialize(source,out,types):
    with rasterio.open(source) as src:
        with rasterio.open(out/'concept_scores.tif','w',**profile(src,len(types))) as dst:
            for i,t in enumerate(types,1):dst.set_band_description(i,t['id'])
            dst.update_tags(score_encoding='round(SAM3 detection score *10000); zero=no usable detection',accuracy='not measured')
        with rasterio.open(out/'executed.tif','w',**profile(src,dtype='uint8')) as dst:pass
        with rasterio.open(out/'valid.tif','w',**profile(src,dtype='uint8')) as dst:
            for _,w in dst.block_windows(1):dst.write((src.dataset_mask(window=w)>0).astype('uint8'),1,window=w)

def export_outputs(source,out,types,objects=False):
    class_counts={};concept_counts={};valid_n=unknown=unexecuted=conflicts=0
    with rasterio.open(source) as src,rasterio.open(out/'concept_scores.tif') as supports,rasterio.open(out/'valid.tif') as validity,rasterio.open(out/'executed.tif') as execution:
        with rasterio.open(out/'classes.tif','w',**profile(src,nodata=NODATA)) as cls,rasterio.open(out/'concepts.tif','w',**profile(src,nodata=NODATA)) as con,rasterio.open(out/'scores.tif','w',**profile(src)) as scr:
            cls.update_tags(unknown='0',nodata=str(NODATA),semantics='highest score surface projection; overlapping concept support retained separately')
            for _,w in cls.block_windows(1):
                s=supports.read(window=w);v=validity.read(1,window=w)>0;ex=execution.read(1,window=w)>0
                high=s.max(0);ci=s.argmax(0);win=np.where(high>0,ci+1,0).astype('uint16');labels=np.asarray([t['class_id'] for t in types],dtype='uint16')[ci];labels[high==0]=0
                win[~v]=NODATA;labels[~v]=NODATA;cls.write(labels,1,window=w);con.write(win,1,window=w);scr.write(high,1,window=w)
                valid_n+=int(v.sum());unknown+=int((v&ex&(high==0)).sum());unexecuted+=int((v&~ex).sum())
                present=np.stack([np.any(s[[i for i,t in enumerate(types) if t['class_id']==k]]>0,axis=0) for k in sorted({t['class_id'] for t in types})])
                conflicts+=int(((present.sum(0)>1)&v).sum())
                for k in np.unique(labels[v]):class_counts[str(k)]=class_counts.get(str(k),0)+int((labels[v]==k).sum())
                for i,t in enumerate(types):concept_counts[t['id']]=concept_counts.get(t['id'],0)+int(((s[i]>0)&v).sum())
        thumb_scale=min(1,1000/max(src.width,src.height));h,w=max(1,round(src.height*thumb_scale)),max(1,round(src.width*thumb_scale))
        rgb=src.read([1,2,3],out_shape=(3,h,w)).transpose(1,2,0);rgb=np.clip(rgb,0,255).astype('uint8')
    with rasterio.open(out/'classes.tif') as c:labels=c.read(1,out_shape=(h,w),resampling=rasterio.enums.Resampling.nearest)
    palette=np.array([[int((i*83)%225+30),int((i*137)%225+30),int((i*193)%225+30)] for i in range(18)],'uint8');ov=rgb.copy();sel=(labels>0)&(labels<NODATA);ov[sel]=(.45*rgb[sel]+.55*palette[labels[sel]%len(palette)]).astype('uint8');Image.fromarray(ov).save(out/'overlay.jpg')
    return {'valid_pixels':valid_n,'completed_unknown_pixels':unknown,'unexecuted_valid_pixels':unexecuted,'cross_class_overlap_pixels':conflicts,'winning_class_pixels':class_counts,'overlapping_concept_pixels':concept_counts,'semantic_accuracy':'not measured'}


def _classify_diagnostics(source,out,config=None,concept_ids=None,model=None,deadline=None):
    cfg=(config or Config()).validate();source=Path(source).resolve();out=Path(out).resolve();start=time.monotonic();deadline=min(deadline or float('inf'),start+cfg.wall_seconds)
    dictionary=load_dictionary(cfg.dictionary);types=concepts(dictionary);selected=set(concept_ids or [t['id'] for t in types]);entries=[{'concept_index':i,'concept_id':t['id'],'class_id':t['class_id'],'prompt':p} for i,t in enumerate(types) if t['id'] in selected for p in t['prompts']]
    if not entries or selected-set(t['id'] for t in types):raise ValueError('Unknown or empty concept selection')
    info=plan(source,cfg);identity={'source_sha256':sha(source),'dictionary_sha256':sha(cfg.dictionary),'config':{k:v for k,v in cfg.asdict().items() if k!='resume'},'selected_concepts':sorted(selected),'model_path':str(Path(cfg.model).resolve())}
    identity=json.loads(json.dumps(identity))
    if out.exists():
        if not cfg.resume:raise FileExistsError('Existing outputs immutable; pass --resume only for matching completed run')
        old=json.loads((out/'provenance.json').read_text());summary=json.loads((out/'summary.json').read_text())
        if old['identity']!=identity or summary['state']!='complete':raise ValueError('Resume identity mismatch or incomplete run')
        for name,digest in old['code_sha256'].items():
            if not (ROOT/name).exists() or sha(ROOT/name)!=digest:raise ValueError('Resume code differs: '+name)
        if old['model_sha256']!=model_sha(cfg.model):raise ValueError('Model changed')
        for name,digest in summary['output_sha256'].items():
            if sha(out/name)!=digest:raise ValueError('Output changed: '+name)
        return summary
    out.mkdir(parents=True);write_json(out/'input.json',info);write_json(out/'config.json',cfg.asdict());write_json(out/'dictionary.json',dictionary)
    write_json(out/'provenance.json',{'identity':identity,'model_sha256':model_sha(cfg.model),'code_sha256':{name:sha(ROOT/name) for name in ['config.py','raster.py','model.py','pipeline.py','features.py','objects.py','deliverables.py']},'originals_preserved':True})
    initialize(source,out,types)
    (out/'feature_observations.jsonl').touch()
    summary={'state':'started','planned_windows':len(info['windows']),'processed_windows':0,'prompt_entries':len(entries),'observations':0,'repairs':[]};own=model is None
    try:
        if own:
            from .model import Sam3
            model=Sam3(cfg)
        with rasterio.open(source) as src,rasterio.open(out/'concept_scores.tif','r+') as scores,rasterio.open(out/'executed.tif','r+') as executed,(out/'observations.jsonl').open('w') as observations:
            stretch=rendering(src,cfg.bands);summary['rendering_stretch']=stretch
            for wi,window in enumerate(info['windows']):
                if cfg.max_windows is not None and wi>=cfg.max_windows:break
                if time.monotonic()>=deadline-cfg.evaluation_reserve_seconds:summary['state']='inconclusive_inference_budget';break
                rgb,valid,native=read_window(src,window,info,cfg,stretch);local={}
                for entry,masks,confidence in model.infer(rgb,entries,deadline-cfg.evaluation_reserve_seconds):
                    target=local.setdefault(entry['concept_index'],np.zeros((cfg.tile,cfg.tile),np.uint16))
                    for mask,score in zip(masks,confidence):
                        m=np.asarray(mask,dtype=bool)&valid
                        if not m.any():continue
                        target[m]=np.maximum(target[m],round(float(score)*10000));ys,xs=np.nonzero(m)
                        from .features import observation_geometry
                        from shapely.geometry import mapping
                        geometry=observation_geometry(m,src.transform,window,info['factor'])
                        with (out/'feature_observations.jsonl').open('a') as of:
                            of.write(json.dumps({**entry,'window_index':wi,'score':float(score),'scale_factor':info['factor'],'geometry':mapping(geometry)})+'\n')
                        observations.write(json.dumps({**entry,'window_index':wi,'score':float(score),'mask_pixels':int(m.sum()),'analysis_bbox':[int(xs.min()+window[0]),int(ys.min()+window[1]),int(xs.max()+1+window[0]),int(ys.max()+1+window[1])],'scale_factor':info['factor']})+'\n');summary['observations']+=1
                for index,maskscore in local.items():
                    h,w=window[3],window[2];a=np.asarray(Image.fromarray(maskscore[:h,:w]).resize((int(native.width),int(native.height)),Image.Resampling.NEAREST),dtype='uint16')
                    old=scores.read(index+1,window=native);scores.write(np.maximum(old,a),index+1,window=native)
                executed.write(np.ones((int(native.height),int(native.width)),np.uint8),1,window=native);summary['processed_windows']+=1
                print(json.dumps({'window':wi+1,'total':len(info['windows']),'seconds':round(time.monotonic()-start,2),'observations':summary['observations']}),flush=True)
                if directory_bytes(out)>cfg.output_bytes:raise RuntimeError('Output byte budget exceeded')
        summary['coverage']=export_outputs(source,out,types,cfg.objects);summary['model']=model.metadata;summary['repairs']=model.repairs
        if summary['state']=='started':summary['state']='complete' if summary['processed_windows']==len(info['windows']) and not summary['coverage']['unexecuted_valid_pixels'] else 'incomplete_window_limit'
        if time.monotonic()>deadline:summary['state']='inconclusive_wall_budget'
    except Exception as exc:
        summary['state']='inconclusive_execution';summary['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        if own and model is not None:model.close()
        summary['seconds']=time.monotonic()-start;summary['retained_bytes']=directory_bytes(out)
        summary['output_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['summary.json']}
        write_json(out/'summary.json',summary)
    return summary


def classify(source,out,config=None,concept_ids=None,model=None,deadline=None):
    """Publish SAM3 features and optional superpixel objects using the shared engine."""
    from .deliverables import classify as publish
    return publish(source,out,config,concept_ids,model,deadline)
