"""Final accuracy assessment of the finished map, separate from tuning (good practice: Olofsson et al. 2014).

Usage:
  final_assess.py <workspace> <round_dir> sample [--layer features|objects] [--total 300] [--min-per-class 10] [--seed 101]
  label_points.py <workspace> A --dir final_assessment     (and B, in parallel: blind labellers, photo only)
  final_assess.py <workspace> <round_dir> estimate

The tuning reference was sampled before the map existed (colour/texture strata, edges over-sampled) and keeps only
points both labellers agreed on, so its accuracy is a tuning signal, not an estimate of map accuracy. This step
samples fresh points stratified by the final map's classes (unclassified included), with a known inclusion
probability for each, has them labelled blind, and weights each class by its share of the map area. It reports
overall accuracy with a 95% confidence interval, user's accuracy (when the map says X, how often it is X),
producer's accuracy (how much of the real X the map finds) and error-adjusted area estimates.

Reference uncertainty is reported, not hidden:
- lenient: the map is right if it matches a class either labeller gave (points where both said uncertain/other
  have no reference and are left out; their share is reported);
- strict: the map is right only if both labellers gave exactly the map's class. The truth lies between the two.
With a goal, it also reports goal agreement: the map is right when its error costs nothing for the goal.
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

def map_layer(rd,layer):
    f,o=class_maps(Path(rd)/'run')
    return f if layer=='features' else o

def sample(ws,cfg,rd,a):
    out=ws/DIR;(out/'sheets').mkdir(parents=True,exist_ok=False)
    m=map_layer(rd,a.layer);rgb,valid=read_rgb(cfg['image']);valid&=m!='unexecuted'
    classes,counts=np.unique(m[valid],return_counts=True);N=int(valid.sum());rng=np.random.default_rng(a.seed)
    alloc={c:min(int(n),max(a.min_per_class,round(a.total*n/N))) for c,n in zip(classes,counts)}
    pts=[];strata={}
    for c,n in zip(classes,counts):
        ys,xs=np.nonzero(valid&(m==c));pick=rng.choice(len(ys),alloc[c],replace=False)
        strata[str(c)]={'pixels':int(n),'weight':n/N,'n':int(alloc[c])}
        pts+=[{'x':int(xs[i]),'y':int(ys[i]),'map':str(c)} for i in pick]
    order=rng.permutation(len(pts));pts=[{'id':f'f{i:03d}',**pts[j]} for i,j in enumerate(order)]
    save_json(out/'points.json',pts);save_json(out/'design.json',{'round':str(Path(rd).resolve()),'layer':a.layer,'seed':a.seed,'valid_pixels':N,'strata':strata})
    g=cfg.get('native_ground_gsd_m') or .1;px=lambda v:max(8,round(v/g));cp,wp=max(CLOSE_MIN_PX,px(CLOSE_M)),max(WIDE_MIN_PX,px(WIDE_M))
    labelled=[{k:v for k,v in p.items() if k!='map'} for p in pts]  # labellers never see the map's class
    for name,seq in (('A',labelled),('B',[labelled[i] for i in rng.permutation(len(labelled))])):
        for j in range(0,len(seq),PER_SHEET):sheet(rgb,seq[j:j+PER_SHEET],out/'sheets'/f'{name}_{j//PER_SHEET:02d}.png',cp,wp)
    print(json.dumps({'points':len(pts),'sheets_per_labeller':math.ceil(len(pts)/PER_SHEET),'per_class':{c:s['n'] for c,s in strata.items()}},indent=1))
    print(f'Next: label_points.py {ws} A --dir {DIR}  and  ... B --dir {DIR}, then: final_assess.py {ws} {rd} estimate')

def estimate(ws,cfg,rd,a):
    out=ws/DIR;design=load_json(out/'design.json');pts=load_json(out/'points.json');goal=load_goal(ws)
    if Path(design['round'])!=Path(rd).resolve():sys.exit(f"the sample was drawn from {design['round']}; assess that round")
    A,B=load_json(out/'labels_A.json'),load_json(out/'labels_B.json');strata=design['strata']
    def ref_sets(p):
        la,lb=A.get(p['id']),B.get(p['id'])
        if not la or not lb:return None
        given=[l for l in (la,lb) if l['label'] not in ('uncertain','other')]
        if not given:return None
        lenient={x for l in given for x in (l['label'],*l['also'])}
        strict={la['label']} if la['label']==lb['label'] and la['label'] not in ('uncertain','other') else set()
        return {'lenient':lenient,'strict':strict|({x for x in la['also']} & {x for x in lb['also']} if strict else set()),'first':given[0]['label']}
    rows=[];no_ref=collections.Counter()
    for p in pts:
        r=ref_sets(p)
        if r is None:no_ref[p['map']]+=1;continue
        ref_class=p['map'] if p['map'] in r['lenient'] else r['first']
        rows.append({**p,**r,'ref':ref_class,'ok_lenient':p['map'] in r['lenient'],'ok_strict':p['map'] in r['strict'],
            'ok_goal':goal is not None and point_cost(goal,{'label':r['first'],'accept':sorted(r['lenient'])},p['map'])==0})
    by=collections.defaultdict(list)
    for r in rows:by[r['map']].append(r)
    def overall(key):
        acc=var=0.
        for h,s in strata.items():
            rs=by.get(h,[]);n=len(rs)
            if not n:continue
            ph=sum(r[key] for r in rs)/n;acc+=s['weight']*ph;var+=s['weight']**2*ph*(1-ph)/max(n-1,1)
        return round(100*acc,1),round(196*math.sqrt(var),1)
    classes=sorted(set(strata)|{r['ref'] for r in rows})
    P={h:{k:0. for k in classes} for h in strata}  # estimated area proportions: map class h x reference class k
    for h,s in strata.items():
        rs=by.get(h,[]);n=len(rs)
        for r in rs:P[h][r['ref']]+=s['weight']/n
    total_m2=design['valid_pixels']*(cfg.get('native_ground_gsd_m') or 0)**2
    per_class={}
    for k in classes:
        ref_share=sum(P[h][k] for h in strata)
        se=math.sqrt(sum(strata[h]['weight']**2*((nk:=sum(r['ref']==k for r in by.get(h,[])))/len(by[h]))*(1-nk/len(by[h]))/max(len(by[h])-1,1) for h in strata if by.get(h)))
        rs=by.get(k,[]);ua=sum(r['ok_lenient'] for r in rs)/len(rs) if rs else None
        per_class[k]={'map_share_pct':round(100*strata[k]['weight'],2) if k in strata else 0.,'sample_n':len(rs),
            'users_accuracy_pct':None if ua is None else round(100*ua,1),
            'users_ci95_range_pct':wilson(sum(r['ok_lenient'] for r in rs),len(rs)),
            'producers_accuracy_pct':round(100*P[k][k]/ref_share,1) if k in P and ref_share else None,
            'estimated_share_pct':round(100*ref_share,2),'estimated_share_ci95_pct':round(196*se,2),
            **({'estimated_area_m2':round(ref_share*total_m2),'area_ci95_m2':round(1.96*se*total_m2)} if total_m2 else {})}
    res={'round':design['round'],'layer':design['layer'],'points':len(pts),'with_reference':len(rows),
        'no_reference':{'count':sum(no_ref.values()),'by_map_class':dict(no_ref),
            'note':'both labellers uncertain or other; excluded, so accuracy for these map classes rests on fewer points'},
        'overall_accuracy_pct':{'lenient':overall('ok_lenient'),'strict':overall('ok_strict'),
            'note':'value, +/- 95% CI; lenient = either labeller agrees, strict = both agree exactly'},
        **({'goal_agreement_pct':overall('ok_goal')} if goal else {}),'per_class':per_class}
    save_json(out/'assessment.json',res)
    lines=[f"# Final accuracy assessment ({design['layer']}, {Path(design['round']).name})","",
        f"{len(pts)} points stratified by map class, labelled blind; {len(rows)} with a reference.","",
        f"Overall accuracy (area-weighted): lenient {res['overall_accuracy_pct']['lenient'][0]}% ± {res['overall_accuracy_pct']['lenient'][1]}, strict {res['overall_accuracy_pct']['strict'][0]}% ± {res['overall_accuracy_pct']['strict'][1]}"
        +(f"; goal agreement {res['goal_agreement_pct'][0]}% ± {res['goal_agreement_pct'][1]}" if goal else ''),"",
        "User's accuracy has a 95% Wilson interval in brackets; with few points per class these are wide.","",
        "| class | map share | sample | user's acc. | producer's acc. | estimated share |","|---|---|---|---|---|---|"]
    for k,v in per_class.items():
        ua='—' if v['users_accuracy_pct'] is None else f"{v['users_accuracy_pct']}% ({v['users_ci95_range_pct'][0]}–{v['users_ci95_range_pct'][1]})"
        pa='—' if v['producers_accuracy_pct'] is None else f"{v['producers_accuracy_pct']}%"
        lines.append(f"| {k+' (map gap)' if k=='unclassified' else k} | {v['map_share_pct']}% | {v['sample_n']} | {ua} | {pa} | {v['estimated_share_pct']}% ± {v['estimated_share_ci95_pct']} |")
    (out/'assessment.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('workspace');ap.add_argument('round');ap.add_argument('step',choices=['sample','estimate'])
    ap.add_argument('--layer',choices=['features','objects'],default='features');ap.add_argument('--total',type=int,default=300)
    ap.add_argument('--min-per-class',type=int,default=10);ap.add_argument('--seed',type=int,default=101);a=ap.parse_args()
    ws,cfg=workspace(a.workspace);(sample if a.step=='sample' else estimate)(ws,cfg,Path(a.round),a)

if __name__=='__main__':main()
