import dataclasses,json
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box,mapping
from automage.config import Config,load_dictionary,concepts
from automage.raster import plan,read_window,rendering
from automage.features import stitch
from automage.deliverables import work_directory
from automage.pipeline import classify


def fixture_raster(path):
    a=np.full((3,35,57),100,'uint8');a[:,:5,:]=0
    with rasterio.open(path,'w',driver='GTiff',width=57,height=35,count=3,dtype='uint8',crs='EPSG:3857',transform=from_origin(100,200,.2,.2),nodata=0) as dst:dst.write(a)
    return path


def test_default_dictionary_preserves_duplicate_prompt_provenance():
    types=concepts(load_dictionary());assert len(types)==44;assert sum(len(t['prompts']) for t in types)==87
    duplicate=[t['id'] for t in types if 'gravel road' in t['prompts']]
    assert len(duplicate)==2 and duplicate[0]!=duplicate[1]


def test_short_rectangular_scene_padded_without_distortion(tmp_path):
    p=fixture_raster(tmp_path/'a.tif');cfg=Config();info=plan(p,cfg)
    with rasterio.open(p) as src:a,v,native=read_window(src,info['windows'][0],info,cfg,rendering(src,cfg.bands))
    assert a.shape==(1008,1008,3);assert native.width==57 and native.height==35
    assert v.sum()==30*57;assert not v[35:].any();assert not v[:,:][... ,57:].any();assert not a[:5].any()






def test_feature_merge_uses_actual_mask_overlap_and_class():
    def row(i,cls,g):return {'concept_id':str(i),'prompt':str(i),'class_id':cls,'score':.8,'geometry':mapping(g)}
    rows=[row(1,1,box(0,0,4,4)),row(2,1,box(1,0,5,4)),row(3,2,box(0,0,4,4)),row(4,1,box(5,0,8,4))]
    features=stitch(rows,10)
    assert len(features)==3;assert sorted(len(v['observations']) for v in features.values())==[1,1,2]


class FakeModel:
    metadata={'test':'stub; no GPU or semantic validation'};repairs=[]
    def infer(self,rgb,entries,deadline):
        mask=np.zeros((1,1008,1008),bool);mask[0,8:16,8:16]=1
        yield entries[0],mask,np.array([.8])


def test_explicit_empty_model_path_fails_before_creating_outputs(tmp_path):
    out = tmp_path / 'result'
    with pytest.raises(ValueError, match='Model directory cannot be empty'):
        classify(tmp_path / 'input.tif', out, Config(model=''))
    assert not out.exists()
    assert not work_directory(out).exists()


@pytest.mark.parametrize('sharded',[False,True])
def test_real_raster_invariants_and_resume_provenance(tmp_path,sharded):
    source=fixture_raster(tmp_path/'input.tif');weights=tmp_path/'weights';weights.mkdir();(weights/'model.safetensors').write_bytes(b'unit test stub')
    weight_file=weights/'model.safetensors'
    if sharded:
        weight_file=weight_file.rename(weights/'model-00001-of-00002.safetensors')
        (weights/'model-00002-of-00002.safetensors').write_bytes(b'second shard')
        (weights/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'first':weight_file.name,'second':'model-00002-of-00002.safetensors'}}))
    cfg=Config(model=str(weights),wall_seconds=600)
    out=tmp_path/'result';result=classify(source,out,cfg,model=FakeModel())
    assert result['state']=='complete'
    with rasterio.open(work_directory(out)/'classes.tif') as f:a=f.read(1)
    assert (a[:5]==65535).all();assert (a[8:16,8:16]==1).all();assert a[20,20]==0
    assert result['coverage']['unexecuted_valid_pixels']==0
    again=classify(source,out,dataclasses.replace(cfg,resume=True),model=FakeModel());assert again==result
    with pytest.raises(ValueError,match='identity'):classify(source,out,dataclasses.replace(cfg,resume=True,scale=2),model=FakeModel())
    original=weight_file.read_bytes();weight_file.write_bytes(b'changed model')
    with pytest.raises(ValueError,match='Model changed'):classify(source,out,dataclasses.replace(cfg,resume=True),model=FakeModel())
    weight_file.write_bytes(original)
    (work_directory(out)/'observations.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='Output changed'):classify(source,out,dataclasses.replace(cfg,resume=True),model=FakeModel())
