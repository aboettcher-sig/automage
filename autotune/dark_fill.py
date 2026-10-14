"""Dark-gap fill: label still-unclassified dark pixels as shadow, as a new round built from an existing one.

Usage: dark_fill.py <workspace> <base_round_dir> <new_round_dir> [--threshold auto|N] [--smooth-m 0.4]

The model often detects nothing on dark ground in shadow (forest floor, shaded yards), and no prompt reaches it.
Shadow is mostly just dark, so this fills pixels the model left unclassified where the lightly smoothed image
brightness (mean of R, G, B, 0-255) is below a threshold. Detections are never changed; only unclassified pixels are.

- `--threshold auto` (default) picks the threshold that maximises features accuracy on the tune points (searching
  20-100 in steps of 5; the lowest value within 1 point of the best, because higher thresholds
  start filling sunlit dry ground that has few reference points), so it adapts to each image's exposure.
- Objects: an unclassified object becomes shadow when most of its pixels were filled.

It copies the base round to <new_round_dir>, rewrites run.work/classes.tif and the objects' class_final in
run/classification.gpkg, refreshes the diagnostics (run/eval/), saves the fill mask as run/dark_fill.tif and the settings in dark_fill.json, then scores
the new round against the base (the usual gate). Review it with review.py like any other round.
The base round must keep the shadow class in its dictionary. Because later run.sh rounds re-run the model, apply
this as a final step on the best round, not in the middle of tuning.
"""
import argparse,json,shutil,sqlite3,struct,subprocess,sys
from pathlib import Path
import numpy as np
import rasterio
from scipy import ndimage
from shapely import wkb
sys.path.insert(0,str(Path(__file__).resolve().parent))
from automage.config import concepts
from common import NODATA,class_maps,load_json,read_rgb,save_json,workspace

def gpkg_geometry(blob):
    """Shapely geometry from a GeoPackage geometry blob (header + WKB), or None when empty."""
    if blob is None:return None
    flags=blob[3];env=(0,32,48,48,64)[(flags>>1)&7]
    if (flags>>4)&1:return None
    return wkb.loads(bytes(blob[8+env:]))

def open_gpkg(path):
    """SQLite connection with the spatial functions GeoPackage rtree triggers call, so attribute updates work."""
    db=sqlite3.connect(path);bounds=lambda b,i:(None if gpkg_geometry(b) is None else gpkg_geometry(b).bounds[i])
    db.create_function('ST_IsEmpty',1,lambda b:int(gpkg_geometry(b) is None or gpkg_geometry(b).is_empty),deterministic=True)
    for name,i in (('ST_MinX',0),('ST_MinY',1),('ST_MaxX',2),('ST_MaxY',3)):db.create_function(name,1,lambda b,i=i:bounds(b,i),deterministic=True)
    return db

def main():
    ap=argparse.ArgumentParser();ap.add_argument('workspace');ap.add_argument('base');ap.add_argument('new')
    ap.add_argument('--threshold',default='auto');ap.add_argument('--smooth-m',type=float,default=.4);a=ap.parse_args()
    ws,cfg=workspace(a.workspace);base,new=Path(a.base).resolve(),Path(a.new).resolve()
    if new.exists():sys.exit(f'{new} already exists')
    work=base/'run.work';ids={t['class_name']:t['class_id'] for t in concepts(load_json(work/'dictionary.json'))}
    if 'shadow' not in ids:sys.exit('the base round has no shadow class in its dictionary; dark-gap fill needs it')
    rgb,valid=read_rgb(cfg['image']);gsd=cfg.get('native_ground_gsd_m') or .1
    size=max(3,round(a.smooth_m/gsd))|1;L=ndimage.uniform_filter(rgb.mean(2).astype('float32'),size)
    with rasterio.open(work/'classes.tif') as s:cls=s.read(1);prof=s.profile
    with rasterio.open(work/'executed.tif') as s:ex=s.read(1)>0
    gap=(cls==0)&valid&ex
    ref=[p for p in load_json(ws/'reference'/'reference.json')['points'] if p['split']=='tune']
    feats,_=class_maps(base/'run')
    def tune_acc(t):return sum(('shadow' if gap[p['y'],p['x']] and L[p['y'],p['x']]<t else feats[p['y'],p['x']]) in p['accept'] for p in ref)
    if a.threshold=='auto':
        scores={t:tune_acc(t) for t in range(20,101,5)};best=max(scores.values())
        t=min(k for k,v in scores.items() if v>=best-1)  # conservative: higher thresholds start filling sunlit dry ground
    else:t=float(a.threshold);scores={}
    fill=gap&(L<t)
    shutil.copytree(base,new,symlinks=False,ignore=shutil.ignore_patterns('review','compare.json','score.json','final_score.json'))
    out=cls.copy();out[fill]=ids['shadow']
    with rasterio.open(new/'run.work'/'classes.tif','w',**prof) as d:d.write(out,1)
    mp=dict(prof,dtype='uint8',nodata=None)
    with rasterio.open(new/'run'/'dark_fill.tif','w',**mp) as d:d.write(fill.astype('uint8'),1)
    with rasterio.open(new/'run'/'objects.tif') as s:obj=s.read(1)
    n=np.bincount(obj.ravel());f=np.bincount(obj.ravel(),weights=fill.ravel().astype(float),minlength=len(n))
    candidates=[int(i) for i in np.nonzero((n>0)&(f>.5*n))[0]]
    db=open_gpkg(new/'run'/'classification.gpkg')
    before=db.execute("select count(*) from objects where class_final='unclassified'").fetchone()[0]
    db.executemany("update objects set class_final='shadow',notes=trim(coalesce(notes,'')||' dark-gap fill') where object_id=? and class_final='unclassified'",[(i,) for i in candidates])
    db.commit();changed=before-db.execute("select count(*) from objects where class_final='unclassified'").fetchone()[0];db.close()
    v=valid&ex;info={'base_round':str(base),'threshold':t,'threshold_mode':a.threshold,'smooth_px':size,'tune_correct_by_threshold':scores,
        'filled_pct_of_image':round(100*fill.sum()/v.sum(),2),'unclassified_pct_before':round(100*gap.sum()/v.sum(),2),
        'unclassified_pct_after':round(100*(gap&~fill).sum()/v.sum(),2),'objects_set_to_shadow':int(changed)}
    save_json(new/'dark_fill.json',info);(new/'flags.txt').write_text((base/'flags.txt').read_text().strip()+f' + dark-gap fill (threshold {t})\n')
    print(json.dumps({k:v for k,v in info.items() if k!='tune_correct_by_threshold'},indent=1),flush=True)
    # Refresh the diagnostics so coverage and unclassified objects describe the filled result, not the copied base.
    summ=load_json(new/'run'/'summary.json');summ['coverage']['completed_unknown_pixels']=int((gap&~fill).sum())
    save_json(new/'run'/'summary.json',summ)
    here=Path(__file__).resolve().parent
    subprocess.run([sys.executable,str(here/'evaluate.py'),str(new/'run')],check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'score.py'),str(ws),str(new),'--base',str(base)],check=True)

if __name__=='__main__':main()
