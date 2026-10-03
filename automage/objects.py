"""Non-overlapping SLIC superpixel objects and their links to SAM3 features."""
from collections import defaultdict
from contextlib import ExitStack
import json
import sqlite3

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.strtree import STRtree
from skimage.segmentation import slic

from .features import write_layer


def rgb_window(src,window,cfg,stretch,out_shape=None):
    options={'window':window,'masked':True}
    if out_shape is not None:options['out_shape']=(3,*out_shape)
    a=src.read(cfg.bands,**options)
    valid=(~np.ma.getmaskarray(a)).all(0)&np.isfinite(a.data).all(0)
    data=a.filled(0).astype('float32')
    if stretch:
        for b,(lo,hi) in enumerate(stretch):data[b]=(data[b]-lo)/max(hi-lo,1e-6)*255
    rgb=np.clip(data.transpose(1,2,0),0,255).astype('uint8')
    rgb[~valid]=0
    return rgb,valid


def export(source,out,work,cfg,types,features,project,budget):
    from .pipeline import profile
    from .raster import rendering
    classes={t['class_id']:t['class_name'] for t in types};classes[0]='unclassified'
    rows=[];next_id=1;max_class=max(classes)
    with rasterio.open(source) as src,ExitStack() as stack, \
         rasterio.open(work/'valid.tif') as vf,rasterio.open(work/'executed.tif') as ex, \
         rasterio.open(out/'objects.tif','w',**profile(src,dtype='uint32',nodata=0)) as dst:
        crs=src.crs.to_wkt();stretch=rendering(src,cfg.bands)
        feature_rasters=[stack.enter_context(rasterio.open(p)) for p in sorted((out/'features').glob('*.tif'))]
        dst.set_band_description(1,'object_id')
        dst.update_tags(object_layer='classification.gpkg:objects',algorithm='SLIC',
                        zero='source invalid or unprocessed; no object',
                        region_px=cfg.object_region_px,compactness=cfg.object_compactness,
                        block_px=cfg.object_block)
        for y in range(0,src.height,cfg.object_block):
            for x in range(0,src.width,cfg.object_block):
                budget()
                window=Window(x,y,min(cfg.object_block,src.width-x),min(cfg.object_block,src.height-y))
                rgb,render_valid=rgb_window(src,window,cfg,stretch)
                valid=render_valid&(vf.read(1,window=window)>0)&(ex.read(1,window=window)>0)
                if not valid.any():
                    dst.write(np.zeros(valid.shape,'uint32'),1,window=window);continue
                labels=slic(rgb,n_segments=max(1,round(valid.sum()/cfg.object_region_px**2)),
                    compactness=cfg.object_compactness,mask=valid,start_label=1,
                    enforce_connectivity=True,convert2lab=True,channel_axis=-1).astype('int32')
                labels[~valid]=0
                ids=np.unique(labels[valid])
                if ids.min()<=0:raise RuntimeError('SLIC failed to assign valid pixels to objects')
                if next_id+len(ids)>np.iinfo('uint32').max:raise RuntimeError('Object ID capacity exhausted')
                mapping=np.zeros(int(labels.max())+1,'uint32')
                mapping[ids]=np.arange(next_id,next_id+len(ids),dtype='uint32');next_id+=len(ids)
                global_labels=mapping[labels]
                dst.write(global_labels,1,window=window)
                proposed=np.zeros(valid.shape,'uint16');best=np.zeros(valid.shape,'float64')
                winning_id=np.zeros(valid.shape,'float64')
                for raster in feature_rasters:
                    score,fid=raster.read(window=window)
                    take=(fid>0)&((score>best)|((score==best)&((winning_id==0)|(fid<winning_id))))
                    best[take]=score[take];winning_id[take]=fid[take]
                    proposed[take]=int(raster.tags()['class_id'])
                histogram=np.bincount(labels[valid].astype('int64')*(max_class+1)+proposed[valid],
                    minlength=len(mapping)*(max_class+1)).reshape(len(mapping),max_class+1)
                pieces=defaultdict(list)
                # Polygonize local int32 labels; exported IDs remain uint32.
                for geo,value in shapes(labels,mask=valid,transform=src.window_transform(window)):
                    pieces[int(value)].append(shape(geo))
                for local_id in ids:
                    counts=histogram[local_id];pixels=int(counts.sum());cid=int(counts.argmax())
                    geometry=unary_union(pieces[int(local_id)])
                    purity=float(counts[cid]/pixels)
                    rows.append({'object_id':int(mapping[local_id]),'class_proposed':classes[cid],
                        'class_final':classes[cid],'purity':purity,
                        'needs_review':int(cid==0 or purity<cfg.object_purity),
                        'reviewed':0,'notes':'','pixel_count':pixels,
                        'area_m2':project(geometry).area,'feature_count':0,
                        'top_feature_id':0,'geometry':geometry})
    # Intersections preserve every feature assignment; they do not force exclusivity.
    tree=STRtree([f['geometry'] for f in features]);links=[]
    for i,obj in enumerate(rows):
        if i%100==0:budget()
        support=[]
        for index in sorted(tree.query(obj['geometry'])):
            feature=features[int(index)]
            intersection=obj['geometry'].intersection(feature['geometry'])
            if intersection.is_empty or intersection.area<=0:continue
            area=min(obj['area_m2'],feature['area_m2'],project(intersection).area)
            if area<=1e-9:continue
            links.append((obj['object_id'],feature['feature_id'],area,
                          area/obj['area_m2'],area/feature['area_m2']))
            support.append(feature)
        obj['feature_count']=len(support)
        if support:
            strongest=max(support,key=lambda f:(f['score'],-f['feature_id']))
            obj['top_feature_id']=strongest['feature_id']
            if len({f['class_id'] for f in support})>1 or strongest['class_name']!=obj['class_proposed']:
                obj['needs_review']=1
    fields={'object_id':'int64','class_proposed':object,'class_final':object,
            'purity':'float64','needs_review':'int32','reviewed':'int32','notes':object,
            'pixel_count':'int64','area_m2':'float64','feature_count':'int32','top_feature_id':'int64'}
    gpkg=out/'classification.gpkg'
    write_layer(gpkg,'objects',rows,fields,crs)
    with sqlite3.connect(gpkg) as db:
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('CREATE UNIQUE INDEX features_feature_id ON features(feature_id)')
        db.execute('CREATE UNIQUE INDEX objects_object_id ON objects(object_id)')
        db.execute('''CREATE TABLE object_features (
            object_id INTEGER NOT NULL REFERENCES objects(object_id),
            feature_id INTEGER NOT NULL REFERENCES features(feature_id),
            intersection_area_m2 REAL NOT NULL,
            object_fraction REAL NOT NULL CHECK(object_fraction>=0 AND object_fraction<=1),
            feature_fraction REAL NOT NULL CHECK(feature_fraction>=0 AND feature_fraction<=1),
            PRIMARY KEY(object_id,feature_id))''')
        db.executemany('INSERT INTO object_features VALUES (?,?,?,?,?)',links)
        db.execute("INSERT INTO gpkg_contents(table_name,data_type,identifier,description) VALUES ('object_features','attributes','object_features','Positive-area overlaps between SLIC objects and SAM3 features')")
        if db.execute('PRAGMA foreign_key_check').fetchall():raise RuntimeError('Broken object-feature references')
    return rows,{'count':len(rows),'links':len(links),'flagged_for_review':sum(r['needs_review'] for r in rows),
                 'unclassified':sum(r['class_proposed']=='unclassified' for r in rows),
                 'pixel_count':sum(r['pixel_count'] for r in rows),
                 'algorithm':'SLIC, non-overlapping blocks',
                 'region_px':cfg.object_region_px,'compactness':cfg.object_compactness,
                 'block_px':cfg.object_block,'purity_review_threshold':cfg.object_purity,
                 'proposal_rule':'Dominant published-feature class across object pixels (peak score wins; tied scores use lowest feature_id); includes unclassified pixels. Final class initially copies proposal; reviewed=0.'}
