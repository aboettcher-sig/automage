import dataclasses
import json
import sqlite3

import numpy as np
import pytest
import pyogrio
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely import from_wkb
from shapely.ops import unary_union

from automage.config import Config
from automage.pipeline import classify


class Model:
    metadata={'test':'synthetic; not semantic validation'}
    repairs=[]
    def infer(self,rgb,entries,deadline):
        for entry in entries:
            mask=np.zeros((1,1008,1008),bool)
            if entry['class_id']==1:mask[0,10:42,10:50]=True
            else:mask[0,22:55,25:60]=True
            yield entry,mask,np.array([.8 if entry['class_id']==1 else .9])


class EmptyModel(Model):
    def infer(self,rgb,entries,deadline):
        for entry in entries:yield entry,np.zeros((0,1008,1008),bool),np.array([])


def fixture(tmp_path):
    source=tmp_path/'source.tif'
    a=np.full((3,64,80),120,'uint8');a[:,:5]=0;a[:,28:32,38:42]=0
    a[1,35:]+=50
    with rasterio.open(source,'w',driver='GTiff',width=80,height=64,count=3,dtype='uint8',
        crs='EPSG:32613',transform=from_origin(500000,4430000,.2,.2),nodata=0) as dst:dst.write(a)
    dictionary=tmp_path/'dictionary.json'
    dictionary.write_text(json.dumps({'classes':[
        {'name':'grass','types':[{'id':'grass','name':'grass','prompts':['grass']}]},
        {'name':'tree','types':[{'id':'tree','name':'tree','prompts':['tree']}]}]}))
    model=tmp_path/'weights';model.mkdir();(model/'model.safetensors').write_bytes(b'test stub')
    return source,Config(model=str(model),dictionary=str(dictionary),objects=True,
                         object_region_px=12,object_block=40,wall_seconds=600)


def read_layer(path,name):
    _,_,geometry,_=pyogrio.raw.read(path,layer=name)
    return [from_wkb(g) for g in geometry]


def test_feature_object_ids_geometry_links_and_unknowns(tmp_path):
    source,cfg=fixture(tmp_path);out=tmp_path/'scene'
    summary=classify(source,out,cfg,model=Model())
    assert summary['state']=='complete'
    assert {p.name for p in out.iterdir()}=={'classification.gpkg','features','objects.tif','overview.png','summary.json'}
    assert not (out/'concept_scores.tif').exists()
    gpkg=out/'classification.gpkg'
    assert set(pyogrio.list_layers(gpkg)[:,0])=={'features','objects','object_features'}
    with sqlite3.connect(gpkg) as db:
        frows=db.execute('SELECT feature_id,class_id FROM features ORDER BY feature_id').fetchall()
        objects=db.execute('SELECT object_id,pixel_count,class_proposed,class_final,reviewed,needs_review FROM objects ORDER BY object_id').fetchall()
        links=db.execute('SELECT object_id,feature_id,intersection_area_m2,object_fraction,feature_fraction FROM object_features').fetchall()
        assert db.execute('PRAGMA foreign_key_check').fetchall()==[]
        assert any(o[2]=='unclassified' and o[5]==1 for o in objects)
        assert all(o[2]==o[3] and o[4]==0 for o in objects)
    assert len(frows)==2
    fgeoms=read_layer(gpkg,'features');ogeoms=read_layer(gpkg,'objects')
    assert fgeoms[0].intersection(fgeoms[1]).area>0
    assert abs(sum(g.area for g in ogeoms)-unary_union(ogeoms).area)<1e-7
    with rasterio.open(source) as src,rasterio.open(out/'objects.tif') as dst:
        ids=dst.read(1);valid=src.dataset_mask()>0
        assert np.array_equal(ids>0,valid)
        assert dst.transform==src.transform and dst.crs==src.crs
        recovered=rasterize([(g,o[0]) for g,o in zip(ogeoms,objects)],out_shape=ids.shape,transform=dst.transform,dtype='uint32')
        assert np.array_equal(recovered,ids)
        for oid,n,*_ in objects:assert int((ids==oid).sum())==n
    for path in (out/'features').glob('*.tif'):
        with rasterio.open(path) as dst:
            assert dst.descriptions==('score','feature_id')
            scores,ids=dst.read();cid=int(dst.tags()['class_id'])
            assert set(np.unique(ids)) <= {0,*[fid for fid,c in frows if c==cid]}
            assert np.array_equal(scores>0,ids>0)
            assert not ids[~valid].any()
    fs={r[0]:g for r,g in zip(frows,fgeoms)};os_={r[0]:g for r,g in zip(objects,ogeoms)}
    assert links
    for oid,fid,area,of,ff in links:
        intersection=os_[oid].intersection(fs[fid]).area
        assert area==pytest.approx(intersection,abs=1e-7)
        assert of==pytest.approx(area/os_[oid].area)
        assert ff==pytest.approx(area/fs[fid].area)
    # Editing a primary product must not be silently replaced on resume.
    (out/'overview.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Output changed'):
        classify(source,out,dataclasses.replace(cfg,resume=True),model=Model())


def test_empty_features_still_have_reviewable_objects(tmp_path):
    source,cfg=fixture(tmp_path);out=tmp_path/'scene'
    summary=classify(source,out,cfg,model=EmptyModel())
    assert summary['feature_count']==0 and summary['objects']['count']>0
    assert list((out/'features').iterdir())==[]
    with sqlite3.connect(out/'classification.gpkg') as db:
        assert db.execute('SELECT COUNT(*) FROM features').fetchone()[0]==0
        assert db.execute('SELECT COUNT(*) FROM object_features').fetchone()[0]==0
        assert db.execute("SELECT COUNT(*) FROM objects WHERE class_proposed!='unclassified' OR needs_review!=1").fetchone()[0]==0


def test_optional_objects_are_omitted(tmp_path):
    source,cfg=fixture(tmp_path);out=tmp_path/'scene'
    classify(source,out,dataclasses.replace(cfg,objects=False),model=Model())
    assert not (out/'objects.tif').exists()
    assert set(pyogrio.list_layers(out/'classification.gpkg')[:,0])=={'features'}
