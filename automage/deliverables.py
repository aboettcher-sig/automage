"""Publish linked SAM3 features and optional SLIC objects as one scene product."""
import dataclasses
import json
import os
from pathlib import Path
import time

import numpy as np
import rasterio
from rasterio.enums import Resampling

from .config import Config,load_dictionary,concepts,sha,write_json
from . import features


def work_directory(out):
    out=Path(out)
    return out.with_name(out.name+'.work')


def overview(source,out,work,cfg,types,feature_rows,object_rows,budget):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/sam3-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from skimage.segmentation import find_boundaries
    from .objects import rgb_window
    from .raster import rendering
    budget()
    with rasterio.open(source) as src:
        factor=min(1,900/max(src.width,src.height));h,w=max(1,round(src.height*factor)),max(1,round(src.width*factor))
        rgb,valid=rgb_window(src,None,cfg,rendering(src,cfg.bands),(h,w))
    colors=np.asarray(plt.get_cmap('tab20').colors)
    classes={t['class_id']:t['class_name'] for t in types};palette={c:colors[(c-1)%len(colors)] for c in classes}
    panels=3 if cfg.objects else 2
    fig,axes=plt.subplots(1,panels,figsize=(6*panels,7),squeeze=False)
    axes=axes[0]
    for ax in axes:ax.imshow(rgb);ax.axis('off')
    axes[0].set_title('Source imagery')
    for path in sorted((out/'features').glob('*.tif')):
        budget()
        with rasterio.open(path) as raster:
            cid=int(raster.tags()['class_id']);mask=raster.read(2,out_shape=(h,w),resampling=Resampling.nearest)>0
        rgba=np.zeros((h,w,4));rgba[:,:,:3]=palette[cid];rgba[:,:,3]=mask*.35
        axes[1].imshow(rgba)
        if mask.any():axes[1].contour(mask.astype(float),levels=[.5],colors=[palette[cid]],linewidths=.4)
    axes[1].set_title(f'SAM3 features ({len(feature_rows):,})')
    if cfg.objects:
        with rasterio.open(out/'objects.tif') as raster:
            ids=raster.read(1,out_shape=(h,w),resampling=Resampling.nearest)
        name_to_id={v:k for k,v in classes.items()}
        lookup=np.zeros((max((r['object_id'] for r in object_rows),default=0)+1,4))
        for row in object_rows:
            cid=name_to_id.get(row['class_proposed'],0)
            lookup[row['object_id']]=[*palette.get(cid,np.array([.5,.5,.5])),.48]
        rgba=lookup[ids];rgba[find_boundaries(ids,mode='inner')]=[.05,.05,.05,.85]
        axes[2].imshow(rgba);axes[2].set_title(f'Superpixel objects ({len(object_rows):,}) — proposed classes')
    present=sorted({r['class_id'] for r in feature_rows})
    handles=[Patch(facecolor=palette[c],label=classes[c]) for c in present]
    if cfg.objects:handles.append(Patch(facecolor='.5',label='unclassified object'))
    if handles:fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,fontsize=9,bbox_to_anchor=(.5,.035))
    fig.suptitle(Path(source).name,fontsize=11)
    fig.text(.5,.012,'Features may overlap. Object labels are proposals; review flags are not accuracy estimates.',ha='center',fontsize=9)
    fig.subplots_adjust(bottom=.22,top=.92,wspace=.025)
    fig.savefig(out/'overview.png',dpi=150,facecolor='white');plt.close(fig)


def classify(source,out,config=None,concept_ids=None,model=None,deadline=None):
    from .pipeline import _classify_diagnostics,directory_bytes
    cfg=(config or Config()).validate();source=Path(source).resolve();out=Path(out).resolve()
    if not cfg.model:
        raise ValueError('Model directory cannot be empty; use Config() for bundled SAM3')
    work=work_directory(out);start=time.monotonic()
    deadline=min(deadline or float('inf'),start+cfg.wall_seconds)
    if out.exists():
        if not cfg.resume:raise FileExistsError('Existing scene outputs are preserved; use a new directory')
        saved=json.loads((out/'summary.json').read_text())
        if saved['state']!='complete':raise ValueError('Cannot reuse an incomplete scene output')
        # Check input, configuration, code, model and working-result provenance.
        _classify_diagnostics(source,work,cfg,concept_ids,model,deadline)
        for name,digest in saved['output_sha256'].items():
            if sha(out/name)!=digest:raise ValueError('Output changed: '+name)
        return saved
    if work.exists():raise FileExistsError('Working directory exists; preserve it and select a new output name')
    out.mkdir(parents=True)
    summary={'schema':'features_objects_v1','state':'started','source':str(source),
             'working_directory':str(work),'configuration':json.loads(json.dumps(cfg.asdict()))}
    write_json(out/'summary.json',summary)
    def budget():
        if time.monotonic()>=deadline:raise TimeoutError('Scene inference/export allowance exhausted')
        if directory_bytes(out)+directory_bytes(work)>cfg.output_bytes:
            raise RuntimeError('Scene output allowance exhausted')
    try:
        raw=_classify_diagnostics(source,work,cfg,concept_ids,model,deadline)
        budget()
        dictionary=load_dictionary(cfg.dictionary);types=concepts(dictionary)
        info=json.loads((work/'input.json').read_text())
        with rasterio.open(source) as src:rows,project,area_crs=features.collect(work,src,types,budget)
        statistics=features.export(rows,source,out,work,budget)
        object_rows=[];object_statistics=None
        if cfg.objects:
            from .objects import export
            object_rows,object_statistics=export(source,out,work,cfg,types,rows,project,budget)
        overview(source,out,work,cfg,types,rows,object_rows,budget)
        budget()
        stats={r['class_id']:r for r in statistics}
        for t in types:
            stats.setdefault(t['class_id'],{'class_id':t['class_id'],'class_name':t['class_name'],
                             'features':0,'covered_pixels':0,'raster':None})
        summary.update(state=raw['state'],coverage=raw['coverage'],
            feature_count=len(rows),feature_observation_count=raw['observations'],
            classes=[stats[k] for k in sorted(stats)],objects=object_statistics,
            input={'width':info['width'],'height':info['height'],'crs':info['crs'],
                   'transform':info['transform'],'native_ground_gsd_m':info['native_ground_gsd_m']},
            inference={'presented_ground_gsd_m':info['presented_ground_gsd_m'],
                       'planned_windows':raw['planned_windows'],'completed_windows':raw['processed_windows'],
                       'prompt_entries':raw['prompt_entries'],'model':raw['model'],'repairs':raw['repairs']},
            output_grid={'description':'Source grid, CRS and footprint; invalid/unprocessed pixels masked',
                         'ground_gsd_m':info['native_ground_gsd_m']},
            area_crs=area_crs,dictionary=dictionary,
            provenance=json.loads((work/'provenance.json').read_text()),
            meanings={'features':'SAM3 detections accumulated within the same class using intersection/min-area >=0.5; overlapping features remain in the vector layer.',
                      'objects':'SLIC superpixels; disjoint review regions covering valid processed imagery.',
                      'feature_rasters':'Band 1: peak detection score. Band 2: integer-valued feature_id. Float64 preserves IDs exactly. Same-class overlap displays the highest-score feature; vector overlap remains intact.',
                      'object_features':'Positive-area intersections; fractions are intersection / object area and intersection / feature area, measured in the recorded metric CRS.'},
            semantic_accuracy='Not measured')
    except Exception as exc:
        summary.update(state='inconclusive_execution',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        summary['seconds']=time.monotonic()-start
        summary['retained_bytes']=directory_bytes(out)+directory_bytes(work)
        summary['output_sha256']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p!=out/'summary.json'}
        write_json(out/'summary.json',summary)
    return summary
