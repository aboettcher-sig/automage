"""Show everything that fires at given points of a round: per class best score, and every prompt with its scores.

Usage: point_prompts.py <workspace> <round_dir> <selector> [...]

Selectors (any mix):
  p017            a point id (tune or unlabelled points only; held-out points are refused)
  1200,340        a pixel as col,row
  truth=<class>   every tune point whose reference label is <class>
  wins=<class>    every tune point where <class> wins in this round's features
  errors          every tune point this round gets wrong in features
Quote class names with spaces: 'truth=undeveloped lands'. In zsh a variable holding several ids isn't split into
separate arguments; pass ids literally, use ${=ids}, or use a selector.

For each point: the reference label (if it is a reference point), the winning class in features and objects, the
best score of every class detected there, and every prompt whose detections cover the point (best score, number of
detections, and the windows they came from). Use it to find which prompt wins or competes at an error, e.g. when
two classes both cover a point and the winner is decided by a small score difference.
"""
import collections,json,sys
from pathlib import Path
import numpy as np
import rasterio
from shapely.geometry import Point,shape
from shapely.strtree import STRtree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from automage.config import concepts
from common import class_maps,load_json,workspace

def main(ws,rd,targets):
    ws,_=workspace(ws);rd=Path(rd);work=rd/'run.work'
    ref={p['id']:p for p in load_json(ws/'reference'/'reference.json')['points']} if (ws/'reference'/'reference.json').exists() else {}
    allpts={p['id']:p for p in load_json(ws/'reference'/'points.json')}
    feats,objs=class_maps(rd/'run');tune=[p for p in ref.values() if p['split']=='tune'];pts=[];seen=set()
    def add(p):
        if p['id'] not in seen:seen.add(p['id']);pts.append(p)
    for t in targets:
        if t.startswith('truth='):[add(p) for p in tune if p['label']==t[6:]]
        elif t.startswith('wins='):[add(p) for p in tune if feats[p['y'],p['x']]==t[5:]]
        elif t=='errors':[add(p) for p in tune if feats[p['y'],p['x']] not in p['accept']]
        elif t in allpts:
            if ref.get(t,{}).get('split')=='holdout':sys.exit(f'{t} is a held-out point; holdout is only for the final report')
            add({**allpts[t],**ref.get(t,{})})
        else:x,y=map(int,t.split(','));add({'id':t,'x':x,'y':y})
    if not pts:sys.exit('no points selected')
    types=concepts(load_json(work/'dictionary.json'));cls=np.array([t['class_name'] for t in types])
    with rasterio.open(work/'concept_scores.tif') as s:S=np.stack([s.read(i+1)[[p['y'] for p in pts],[p['x'] for p in pts]] for i in range(s.count)]).astype(float)/10000
    with rasterio.open(load_json(work/'input.json')['source']) as s:tr=s.transform
    tree=STRtree([Point(*(tr*(p['x']+.5,p['y']+.5))) for p in pts]);hits=[collections.defaultdict(list) for _ in pts]
    with (work/'feature_observations.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            for i in tree.query(shape(r['geometry']),predicate='contains'):hits[int(i)][(cls[r['concept_index']],r['prompt'])].append((r['score'],r['window_index']))
    for i,p in enumerate(pts):
        best={}
        for k,n in enumerate(cls):
            if S[k,i]>0:best[n]=max(best.get(n,0),S[k,i])
        print(f"\n{p['id']}  col {p['x']} row {p['y']}"+(f"  reference: {p['label']} (accept {', '.join(p['accept'])}, {p['split']})" if 'label' in p else ''))
        print(f"  winner: features {feats[p['y'],p['x']]}, objects {objs[p['y'],p['x']] if objs is not None else '-'}")
        print('  classes detected (best score): '+(', '.join(f'{n} {v:.3f}' for n,v in sorted(best.items(),key=lambda x:-x[1])) or 'none'))
        for (n,prompt),obs in sorted(hits[i].items(),key=lambda x:-max(s for s,_ in x[1])):
            print(f"    {n:18s} '{prompt}': best {max(s for s,_ in obs):.3f}, {len(obs)} detection(s), windows {sorted({w for _,w in obs})}")

if __name__=='__main__':
    if len(sys.argv)<4:sys.exit(__doc__)
    main(sys.argv[1],sys.argv[2],sys.argv[3:])
