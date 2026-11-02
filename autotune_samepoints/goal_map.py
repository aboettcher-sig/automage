"""Translate a round's classification into the goal's categories (e.g. irrigation status, fire fuel type).

Usage: goal_map.py <workspace> <round_dir>

Reads the workspace's goal.json and the round's run.work/classes.tif, and writes run/goal_map.tif (one band, one
code per goal category), run/goal_map.json (code -> category, share of the image) and run/goal_map.png (preview).
Classes whose category is "depends" (the class alone doesn't decide it, e.g. shadow over unknown ground) and
classes without a category get the code for "undetermined"; 0 stays unclassified. Report the undetermined share:
it is the part of the answer the classification can't give on its own.
"""
import json,sys
from pathlib import Path
import numpy as np
import rasterio
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
from automage.config import concepts
from common import load_goal,load_json,save_json,workspace

PALETTE=[(230,230,230),(31,119,180),(255,127,14),(44,160,44),(214,39,40),(148,103,189),(140,86,75),(227,119,194),(188,189,34),(23,190,207)]

def main(ws,rd):
    ws,_=workspace(ws);rd=Path(rd);goal=load_goal(ws)
    if not goal:sys.exit('no goal.json in the workspace')
    names={t['class_id']:t['class_name'] for t in concepts(load_json(rd/'run.work'/'dictionary.json'))}
    cats=sorted({v['category'] for v in goal['classes'].values() if v.get('category') and v['category']!='depends'})
    codes={c:i+2 for i,c in enumerate(cats)}  # 0 unclassified, 1 undetermined
    with rasterio.open(rd/'run.work'/'classes.tif') as s:cls=s.read(1);prof=s.profile
    out=np.zeros(cls.shape,'uint8')
    for k,n in names.items():
        cat=goal['classes'].get(n,{}).get('category')
        out[cls==k]=codes.get(cat,1)
    valid=cls!=65535
    with rasterio.open(rd/'run'/'goal_map.tif','w',**dict(prof,dtype='uint8',nodata=None)) as d:d.write(out,1)
    legend={0:'unclassified',1:'undetermined (class alone does not decide)',**{v:k for k,v in codes.items()}}
    share={legend[c]:round(100*float(((out==c)&valid).sum())/valid.sum(),2) for c in legend}
    save_json(rd/'run'/'goal_map.json',{'goal':goal.get('goal'),'purpose':goal.get('purpose'),'codes':legend,'share_pct':share})
    rgb=np.array([PALETTE[c%len(PALETTE)] for c in range(256)],'uint8')[out];rgb[out==0]=(0,0,0)
    Image.fromarray(rgb).save(rd/'run'/'goal_map.png')
    print(json.dumps({'codes':legend,'share_pct':share},indent=1))

if __name__=='__main__':
    if len(sys.argv)!=3:sys.exit(__doc__)
    main(*sys.argv[1:])
