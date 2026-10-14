"""Build the reference from two labellers: keep points where both agree, split into tune/holdout.

Usage: merge_labels.py <workspace> [--holdout 0.3]

Writes reference/reference.json. A prediction is correct if it is in the point's "accept" set
(the agreed label plus anything either labeller listed under "also").
"""
import argparse,collections,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load_json,save_json,workspace

def main():
    ap=argparse.ArgumentParser();ap.add_argument('workspace');ap.add_argument('--holdout',type=float,default=.3);a=ap.parse_args()
    ws,cfg=workspace(a.workspace);ref=ws/'reference'
    pts=load_json(ref/'points.json');A=load_json(ref/'labels_A.json');B=load_json(ref/'labels_B.json')
    keep=[]
    for p in pts:
        la,lb=A.get(p['id']),B.get(p['id'])
        if la and lb and la['label']==lb['label']!='uncertain':
            keep.append({**p,'label':la['label'],'accept':sorted({la['label'],*la['also'],*lb['also']})})
    rng=np.random.default_rng(cfg['seed']);by=collections.defaultdict(list)
    for p in keep:by[p['label']].append(p)
    for group in by.values():
        idx=rng.permutation(len(group));n=round(a.holdout*len(group))
        for k,i in enumerate(idx):group[i]['split']='holdout' if k<n else 'tune'
    both=[p for p in pts if p['id'] in A and p['id'] in B]
    stats={'points':len(pts),'labelled_by_both':len(both),'agreed':len(keep),
        'uncertain_either':sum(1 for p in both if 'uncertain' in (A[p['id']]['label'],B[p['id']]['label'])),
        'tune':sum(p['split']=='tune' for p in keep),'holdout':sum(p['split']=='holdout' for p in keep),
        'per_class':dict(collections.Counter(p['label'] for p in keep).most_common())}
    save_json(ref/'reference.json',{'stats':stats,'points':keep});print(stats)

if __name__=='__main__':main()
