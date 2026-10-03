"""SAM3 features: mask geometry and same-class accumulation."""
import json,math
import re
from collections import defaultdict
import numpy as np
from shapely.geometry import shape,mapping
from shapely.ops import unary_union
from rasterio.features import shapes
from affine import Affine
from pyproj import Transformer
from shapely.ops import transform as transform_geometry
from shapely.strtree import STRtree
import rasterio
from rasterio.features import rasterize
import pyogrio.raw


def metric_projection(src):
    lon,lat=Transformer.from_crs(src.crs,4326,always_xy=True).transform(
        *(src.transform*(src.width/2,src.height/2)))
    epsg=(32600 if lat>=0 else 32700)+min(60,int((lon+180)//6)+1)
    tr=Transformer.from_crs(src.crs,epsg,always_xy=True)
    return lambda geometry:transform_geometry(tr.transform,geometry),f'EPSG:{epsg}'


def write_layer(path,layer,records,fields,crs):
    """Explicit field types also produce usable empty GeoPackage layers."""
    geometries=np.asarray([r['geometry'].wkb for r in records],dtype=object)
    arrays=[np.asarray([r[name] for r in records],dtype=dtype) for name,dtype in fields.items()]
    pyogrio.raw.write(path,geometries,arrays,list(fields),driver='GPKG',layer=layer,
                     geometry_type='MultiPolygon',promote_to_multi=True,crs=crs)


def collect(work,src,types,budget):
    with (work/'feature_observations.jsonl').open() as f:
        def observations():
            for i,line in enumerate(f):
                if i%100==0:budget()
                yield json.loads(line)
        accumulated=stitch(observations(),max(abs(src.transform.a),abs(src.transform.e))*1008)
    lookup={t['id']:t for t in types}
    classes={t['class_id']:t['class_name'] for t in types}
    project,area_crs=metric_projection(src)
    result=[]
    for fid,(_,feature) in enumerate(sorted(accumulated.items()),1):
        if fid%100==0:budget()
        obs=feature['observations'];ids=sorted({r['concept_id'] for r in obs})
        result.append({'feature_id':fid,'class_id':feature['class_id'],
            'class_name':classes[feature['class_id']],
            'types':json.dumps(sorted({lookup[c]['name'] for c in ids})),
            'concept_ids':json.dumps(ids),'prompts':json.dumps(sorted({r['prompt'] for r in obs})),
            'score':max(r['score'] for r in obs),'observation_count':len(obs),
            'area_m2':project(feature['geometry']).area,'geometry':feature['geometry']})
    return result,project,area_crs


def export(features,source,out,work,budget):
    from .pipeline import profile
    fields={'feature_id':'int64','class_id':'int32','class_name':object,'types':object,
            'concept_ids':object,'prompts':object,'score':'float64',
            'observation_count':'int64','area_m2':'float64'}
    with rasterio.open(source) as src:
        write_layer(out/'classification.gpkg','features',features,fields,src.crs.to_wkt())
        (out/'features').mkdir()
        by_class=defaultdict(list)
        for feature in features:by_class[feature['class_id']].append(feature)
        stats=[];used=set()
        with rasterio.open(work/'valid.tif') as vf,rasterio.open(work/'executed.tif') as ex:
            for cid,rows in sorted(by_class.items()):
                budget();name=rows[0]['class_name']
                slug=re.sub(r'[^a-zA-Z0-9_-]+','_',name).strip('_') or str(cid)
                if slug in used:raise ValueError('Class filenames collide: '+name)
                used.add(slug)
                path=out/'features'/(slug+'.tif')
                tree=STRtree([r['geometry'] for r in rows]);covered=0
                # Both bands are float64 so all uint32 feature IDs remain exact.
                with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True),rasterio.open(path,'w',**profile(src,2,'float64')) as dst:
                    dst.set_band_description(1,'score');dst.set_band_description(2,'feature_id')
                    dst.update_tags(class_id=cid,class_name=name,feature_layer='classification.gpkg:features',
                        zero='no feature',overlap_rule='highest feature peak score; ties use lowest feature_id',
                        score_meaning='peak contributing SAM3 detection score; not calibrated accuracy')
                    for _,window in dst.block_windows(1):
                        budget();shape_=(int(window.height),int(window.width));tf=src.window_transform(window)
                        from shapely.geometry import Polygon
                        bounds=Polygon([tf*p for p in [(0,0),(shape_[1],0),(shape_[1],shape_[0]),(0,shape_[0])]])
                        ids=tree.query(bounds)
                        selected=sorted((rows[int(i)] for i in ids),key=lambda r:(r['score'],-r['feature_id']))
                        valid=(vf.read(1,window=window)>0)&(ex.read(1,window=window)>0)
                        for band,field in [(1,'score'),(2,'feature_id')]:
                            array=rasterize([(r['geometry'],r[field]) for r in selected],out_shape=shape_,
                                transform=tf,dtype='float64') if selected else np.zeros(shape_,'float64')
                            array[~valid]=0
                            dst.write(array,band,window=window)
                            if band==2:covered+=int(np.count_nonzero(array))
                        dst.write_mask(valid.astype('uint8')*255,window=window)
                stats.append({'class_id':cid,'class_name':name,'features':len(rows),
                              'covered_pixels':covered,'raster':str(path.relative_to(out))})
    return stats

def observation_geometry(mask,src_transform,window,factor):
    tf=src_transform*Affine.translation(window[0]*factor,window[1]*factor)*Affine.scale(factor,factor)
    pieces=[shape(g) for g,v in shapes(mask.astype('uint8'),mask=mask,transform=tf) if v]
    return unary_union(pieces)

def stitch(records,cell,containment=.5):
    features={};index=defaultdict(set);counter=0
    def cells(bounds):
        x0,y0,x1,y1=bounds
        for x in range(math.floor(x0/cell),math.floor(x1/cell)+1):
            for y in range(math.floor(y0/cell),math.floor(y1/cell)+1):yield (x,y)
    for row in records:
        g=shape(row['geometry']);keys=list(cells(g.bounds));candidates=set().union(*(index[k] for k in keys));hits=[]
        for i in candidates:
            obj=features.get(i)
            if obj is None or obj['class_id']!=row['class_id']:continue
            area=g.intersection(obj['geometry']).area
            if area and area/max(min(g.area,obj['geometry'].area),1e-12)>=containment:hits.append(i)
        if hits:
            chosen=min(hits);obj=features[chosen];geoms=[g]
            for i in hits:
                other=features.pop(i);geoms.append(other['geometry'])
                if i!=chosen:obj['observations'].extend(other['observations'])
            obj['geometry']=unary_union(geoms);obj['observations'].append({k:v for k,v in row.items() if k!='geometry'});features[chosen]=obj
        else:
            chosen=counter;counter+=1;features[chosen]={'class_id':row['class_id'],'geometry':g,'observations':[{k:v for k,v in row.items() if k!='geometry'}]}
        for key in cells(features[chosen]['geometry'].bounds):index[key].add(chosen)
    return features
