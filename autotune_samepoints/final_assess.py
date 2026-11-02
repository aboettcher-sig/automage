"""Final accuracy assessment of the finished map, reusing the tuning reference's points and topping up thin classes.

Usage:
  final_assess.py <workspace> <round_dir> sample [--layer features|objects] [--min-per-class 10] [--seed 101]
  label_points.py <workspace> A --dir final_assessment     (and B, in parallel; only if sample drew top-up points)
  final_assess.py <workspace> <round_dir> estimate

This is the low-cost variant of autotune/final_assess.py, which samples and labels a whole fresh set of points.
Here the initial points (reference/points.json) and both labellers' labels are reused. `sample` counts the reused
points with a reference in each class of the final map; any map class with fewer than --min-per-class gets new
random points inside it up to that number (a class with none gets the full number). Only those top-up points need
labelling. If no class needs topping up, no sheets are made and `estimate` can run straight away.

`estimate` combines both sets by map class, as in the original: each map class's share of the map is known exactly,
and its accuracy and reference mix are estimated from all points that fall in it. Reused points keep their sampling
weight within a class (the edge stratum was over-sampled), top-up points are a simple random sample of their class.

Reference uncertainty is reported as in the original:
- lenient: the map is right if it matches a class either labeller gave (points where both said uncertain/other
  have no reference and are left out; their count is reported);
- strict: the map is right only if both labellers gave exactly the map's class. The truth lies between the two.
With a goal, it also reports goal agreement: the map is right when its error costs nothing for the goal.

Limits, stated in the output: the tune points (and, through the gate's soft score, the points the labellers
disagreed on) were used to choose the map, so this estimate is optimistic. The holdout and top-up points were not;
they are reported alongside.
"""
import argparse,collections,json,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import class_maps,load_goal,load_json,point_cost,read_rgb,save_json,workspace
from sample_points import CLOSE_M,CLOSE_MIN_PX,PER_SHEET,WIDE_M,WIDE_MIN_PX,sheet

DIR='final_assessment'

def wilson(k,n,z=1.96):
    """95% Wilson interval (low, high) in %, sound for small samples and for 0% or 100%."""
    if not n:return None
    p=k/n;c=(p+z*z/(2*n))/(1+z*z/n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [round(100*max(0,c-h),1),round(100*min(1,c+h),1)]

def stratum_pixels(ws,cfg):
    d=ws/'reference'/'design.json'
    if d.exists():return load_json(d)['strata_pixels']
    from sample_points import strata  # older workspace: recompute the same strata (deterministic for the seed)
    rgb,valid=read_rgb(cfg['image']);grid,_=strata(rgb,valid,10,cfg['seed']);k=grid.max()
    return {('edge' if c==k else f'cluster_{c}'):int((grid==c).sum()) for c in range(k+1)}

def map_layer(rd,layer):
    f,o=class_maps(Path(rd)/'run')
    return f if layer=='features' else o

def ref_sets(la,lb):
    if not la or not lb:return None
    given=[l for l in (la,lb) if l['label'] not in ('uncertain','other')]
    if not given:return None
    lenient={x for l in given for x in (l['label'],*l['also'])}
    strict={la['label']} if la['label']==lb['label'] and la['label'] not in ('uncertain','other') else set()
    return {'lenient':lenient,'strict':strict|({x for x in la['also']} & {x for x in lb['also']} if strict else set()),'first':given[0]['label']}

def reused(ws,m):
    """The initial points with a reference, each with its map class at this round."""
    ref=ws/'reference';A,B=load_json(ref/'labels_A.json'),load_json(ref/'labels_B.json')
    split={p['id']:p['split'] for p in load_json(ref/'reference.json')['points']} if (ref/'reference.json').exists() else {}
    rows,no_ref=[],collections.Counter()
    for p in load_json(ref/'points.json'):
        mc=str(m[p['y'],p['x']])
        if mc=='unexecuted':continue
        r=ref_sets(A.get(p['id']),B.get(p['id']))
        if r is None:no_ref[mc]+=1;continue
        rows.append({**p,**r,'map':mc,'split':split.get(p['id'],'disagreed')})
    return rows,no_ref

def sample(ws,cfg,rd,a):
    out=ws/DIR;out.mkdir(exist_ok=False)
    m=map_layer(rd,a.layer);rgb,valid=read_rgb(cfg['image']);valid&=m!='unexecuted'
    rows,_=reused(ws,m);have=collections.Counter(r['map'] for r in rows)
    classes,counts=np.unique(m[valid],return_counts=True);rng=np.random.default_rng(a.seed);pts=[];plan={}
    for c,n in zip(classes,counts):
        c=str(c);need=min(int(n),max(0,a.min_per_class-have[c]));plan[c]={'pixels':int(n),'reused':have[c],'top_up':need}
        if not need:continue
        ys,xs=np.nonzero(valid&(m==c));pick=rng.choice(len(ys),need,replace=False)
        pts+=[{'x':int(xs[i]),'y':int(ys[i]),'map':c} for i in pick]
    order=rng.permutation(len(pts));pts=[{'id':f'f{i:03d}',**pts[j]} for i,j in enumerate(order)]
    save_json(out/'points.json',pts)
    save_json(out/'design.json',{'round':str(Path(rd).resolve()),'layer':a.layer,'seed':a.seed,'min_per_class':a.min_per_class,
        'valid_pixels':int(valid.sum()),'classes':plan})
    print(json.dumps({'top_up_points':len(pts),'per_class':{c:f"{v['reused']} reused + {v['top_up']} new" for c,v in plan.items()}},indent=1))
    if not pts:print(f'No class needs topping up; run: final_assess.py {ws} {rd} estimate');return
    (out/'sheets').mkdir();g=cfg.get('native_ground_gsd_m') or .1;px=lambda v:max(8,round(v/g))
    cp,wp=max(CLOSE_MIN_PX,px(CLOSE_M)),max(WIDE_MIN_PX,px(WIDE_M))
    labelled=[{k:v for k,v in p.items() if k!='map'} for p in pts]  # labellers never see the map's class
    for name,seq in (('A',labelled),('B',[labelled[i] for i in rng.permutation(len(labelled))])):
        for j in range(0,len(seq),PER_SHEET):sheet(rgb,seq[j:j+PER_SHEET],out/'sheets'/f'{name}_{j//PER_SHEET:02d}.png',cp,wp)
    print(f'{math.ceil(len(pts)/PER_SHEET)} sheets per labeller. Next: label_points.py {ws} A --dir {DIR}  and  ... B --dir {DIR}, '
          f'then: final_assess.py {ws} {rd} estimate')

def estimate(ws,cfg,rd,a):
    out=ws/DIR;design=load_json(out/'design.json');goal=load_goal(ws)
    if Path(design['round'])!=Path(rd).resolve():sys.exit(f"the sample was drawn from {design['round']}; assess that round")
    m=map_layer(rd,design['layer']);rows,no_ref=reused(ws,m)
    sizes=stratum_pixels(ws,cfg)
    for r in rows:r['w']=sizes[r['stratum']]/sum(1 for x in rows if x['stratum']==r['stratum'])  # pixels each reused point stands for
    new=load_json(out/'points.json')
    if new:
        if not((out/'labels_A.json').exists() and (out/'labels_B.json').exists()):sys.exit('label the top-up points first (label_points.py --dir final_assessment)')
        A,B=load_json(out/'labels_A.json'),load_json(out/'labels_B.json')
        for p in new:
            r=ref_sets(A.get(p['id']),B.get(p['id']))
            if r is None:no_ref[p['map']]+=1;continue
            rows.append({**p,**r,'split':'top-up','w':None})
    for r in rows:
        r['ref']=r['map'] if r['map'] in r['lenient'] else r['first']
        r['ok_lenient']=r['map'] in r['lenient'];r['ok_strict']=r['map'] in r['strict']
        r['ok_goal']=goal is not None and point_cost(goal,{'label':r['first'],'accept':sorted(r['lenient'])},r['map'])==0
    N=design['valid_pixels'];W={c:v['pixels']/N for c,v in design['classes'].items()}
    by=collections.defaultdict(list)
    for r in rows:by[r['map']].append(r)
    for h,rs in by.items():  # within a map class: reused points keep relative weights (scaled to count as their number), top-ups weigh 1
        ru=[r for r in rs if r['w'] is not None];tot=sum(r['w'] for r in ru)
        for r in ru:r['v']=r['w']*len(ru)/tot
        for r in rs:
            if r['w'] is None:r['v']=1.
    def share(rs,ind):
        n=sum(r['v'] for r in rs);return sum(r['v'] for r in rs if ind(r))/n
    def overall(ind):
        est=var=0.
        for h,w in W.items():
            rs=by.get(h,[])
            if not rs:continue
            p=share(rs,ind);est+=w*p;var+=w**2*p*(1-p)/max(len(rs)-1,1)
        return round(100*est,1),round(196*math.sqrt(var),1)
    classes=sorted(set(W)|{r['ref'] for r in rows})
    P={h:{k:W[h]*share(by[h],lambda r,k=k:r['ref']==k) if by.get(h) else 0. for k in classes} for h in W}
    total_m2=N*(cfg.get('native_ground_gsd_m') or 0)**2;per_class={}
    for k in classes:
        ref_share=sum(P[h][k] for h in W)
        se=math.sqrt(sum(W[h]**2*(q:=share(by[h],lambda r:r['ref']==k))*(1-q)/max(len(by[h])-1,1) for h in W if by.get(h)))
        rs=by.get(k,[])
        per_class[k]={'map_share_pct':round(100*W.get(k,0),2),'sample_n':len(rs),
            'sample_reused':sum(r['split']!='top-up' for r in rs),'sample_top_up':sum(r['split']=='top-up' for r in rs),
            'users_accuracy_pct':round(100*share(rs,lambda r:r['ok_lenient']),1) if rs else None,
            'users_ci95_range_pct':wilson(sum(r['ok_lenient'] for r in rs),len(rs)),
            'producers_accuracy_pct':round(100*P[k][k]/ref_share,1) if k in P and ref_share else None,
            'estimated_share_pct':round(100*ref_share,2),'estimated_share_ci95_pct':round(196*se,2),
            **({'estimated_area_m2':round(ref_share*total_m2),'area_ci95_m2':round(1.96*se*total_m2)} if total_m2 else {})}
    def unweighted(s,key):
        rs=[r for r in rows if r['split']==s];return round(100*sum(r[key] for r in rs)/len(rs),1) if rs else None
    splits={s:{'n':sum(r['split']==s for r in rows),'lenient':unweighted(s,'ok_lenient'),'strict':unweighted(s,'ok_strict'),
        **({'goal':unweighted(s,'ok_goal')} if goal else {})} for s in ('tune','holdout','disagreed','top-up')}
    res={'round':design['round'],'layer':design['layer'],
        'design':f"tuning reference points reused, map classes with fewer than {design['min_per_class']} topped up with new points; estimated by map class",
        'with_reference':len(rows),'reused':sum(r['split']!='top-up' for r in rows),'top_up':sum(r['split']=='top-up' for r in rows),
        'no_reference':{'count':sum(no_ref.values()),'by_map_class':dict(no_ref),'note':'both labellers uncertain or other; excluded'},
        'overall_accuracy_pct':{'lenient':overall(lambda r:r['ok_lenient']),'strict':overall(lambda r:r['ok_strict']),
            'note':'value, +/- 95% CI, area-weighted by map class; lenient = either labeller agrees, strict = both agree exactly'},
        **({'goal_agreement_pct':overall(lambda r:r['ok_goal'])} if goal else {}),
        'by_split_unweighted_pct':splits,
        'caveat':'tune points, and through the soft score the disagreed points, were used to choose the map, so these figures are optimistic; holdout and top-up points were not',
        'per_class':per_class}
    save_json(out/'assessment.json',res);oa=res['overall_accuracy_pct']
    lines=[f"# Final accuracy assessment ({design['layer']}, {Path(design['round']).name})","",
        f"{res['reused']} reused tuning points + {res['top_up']} new top-up points (classes under {design['min_per_class']}), all with a reference.","",
        f"Overall accuracy (area-weighted by map class): lenient {oa['lenient'][0]}% ± {oa['lenient'][1]}, strict {oa['strict'][0]}% ± {oa['strict'][1]}"
        +(f"; goal agreement {res['goal_agreement_pct'][0]}% ± {res['goal_agreement_pct'][1]}" if goal else ''),"",
        "**Optimistic:** the map was tuned on the tune points (and, via the soft score, the disagreed points).",
        "The holdout and top-up points were not used, so compare with them:","",
        "| points | n | lenient | strict |"+(" goal |" if goal else ""),"|---|---|---|---|"+("---|" if goal else "")]
    for s,v in splits.items():lines.append(f"| {s} | {v['n']} | {v['lenient']}% | {v['strict']}% |"+(f" {v['goal']}% |" if goal else ""))
    lines+=["","User's accuracy has a 95% Wilson interval (on unweighted counts) in brackets; with few points these are wide.","",
        "| class | map share | sample (reused + new) | user's acc. | producer's acc. | estimated share |","|---|---|---|---|---|---|"]
    for k,v in per_class.items():
        ua='—' if v['users_accuracy_pct'] is None else f"{v['users_accuracy_pct']}% ({v['users_ci95_range_pct'][0]}–{v['users_ci95_range_pct'][1]})"
        pa='—' if v['producers_accuracy_pct'] is None else f"{v['producers_accuracy_pct']}%"
        lines.append(f"| {k+' (map gap)' if k=='unclassified' else k} | {v['map_share_pct']}% | {v['sample_reused']} + {v['sample_top_up']} | {ua} | {pa} | {v['estimated_share_pct']}% ± {v['estimated_share_ci95_pct']} |")
    (out/'assessment.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('workspace');ap.add_argument('round');ap.add_argument('step',choices=['sample','estimate'])
    ap.add_argument('--layer',choices=['features','objects'],default='features')
    ap.add_argument('--min-per-class',type=int,default=10);ap.add_argument('--seed',type=int,default=101);a=ap.parse_args()
    ws,cfg=workspace(a.workspace);(sample if a.step=='sample' else estimate)(ws,cfg,Path(a.round),a)

if __name__=='__main__':main()
