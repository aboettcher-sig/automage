"""Score a run against the reference points, optionally against a base run (the numeric acceptance gate).

Usage: score.py <workspace> <round_dir> [--base <round_dir>] [--final]

Uses tune points only and writes <round_dir>/score.json (and compare.json with --base). --final also scores the
held-out points and writes everything to <round_dir>/final_score.json instead, leaving score.json (which is read
during tuning) free of holdout results. Accuracy is on features (winning class per
pixel) and on objects (class_final). False detections are wrong classes detected at a point whose winning class
is right (the hidden overlap that can start winning; at already-wrong points they aren't counted twice); "prompts" attributes every point hit to the prompt that made it.

Soft points are the points the labellers didn't agree on (dropped from the strict reference): a soft point is
acceptable when classified with a class either labeller gave (any class if both were uncertain). They measure gap
filling where the strict points are thin; they never count in accuracy.

Gate vs. base: features net >= 0, objects net >= -1, no reference class losing more than 1 point net in features
or (unless features net >= LARGE_FEATURE_GAIN) in objects,
and gain > 0, where gain = features net + FALSE_WEIGHT * false-detection net (+ SOFT_WEIGHT * soft net in features
when no strict feature point is broken). When no feature point and no false detection changes (an objects-only
round), gain = objects net (+ SOFT_WEIGHT * soft net in objects when no strict object point is broken).
"""
import argparse,collections,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import rasterio
from shapely.geometry import Point,shape
from shapely.strtree import STRtree
from automage.config import concepts
from common import class_maps,load_json,presence,save_json,soft_ok,soft_points,workspace

FALSE_WEIGHT=.5  # a hidden wrong detection counts half as much as a wrong winning class
LARGE_FEATURE_GAIN=5  # features net from which a per-class loss in objects is tolerated
SOFT_WEIGHT=.5   # a lenient soft point counts half as much as a strict one, and only when no strict point is lost

def predict(round_dir,pts):
    f,o=class_maps(Path(round_dir)/'run');pr=presence(Path(round_dir)/'run')
    return {p['id']:{'features':f[p['y'],p['x']],'objects':None if o is None else o[p['y'],p['x']],
        'present':sorted(c for c,m in pr.items() if m[p['y'],p['x']])} for p in pts}

def false_detections(pts,pred):
    """Wrong classes detected at points whose winning class is right: the hidden overlap that can start winning.

    Points whose winner is already wrong are counted once, as errors; a wrong detection there changes nothing, so it
    is not counted again (otherwise filling a gap with a second plausible class is penalised twice).
    """
    pairs={(p['id'],c) for p in pts if pred[p['id']]['features'] in p['accept'] for c in pred[p['id']]['present'] if c not in p['accept']}
    every={(p['id'],c) for p in pts for c in pred[p['id']]['present'] if c not in p['accept']}
    by=collections.defaultdict(list);truth={p['id']:p['label'] for p in pts}
    for i,c in sorted(pairs):by[c].append(i)
    return pairs,{'count':len(pairs),'points':len({i for i,_ in pairs}),'count_including_wrong_points':len(every),
        'by_class':{c:{'count':len(ids),'truth':dict(collections.Counter(truth[i] for i in ids).most_common()),'points':ids}
            for c,ids in sorted(by.items(),key=lambda x:-len(x[1]))}}

def prompt_hits(round_dir,pts):
    """For each prompt: reference points it fires on, split into right (class accepted there) and wrong."""
    work=Path(round_dir)/'run.work';names={t['class_id']:t['class_name'] for t in concepts(load_json(work/'dictionary.json'))}
    with rasterio.open(load_json(work/'input.json')['source']) as s:t=s.transform
    tree=STRtree([Point(*(t*(p['x']+.5,p['y']+.5))) for p in pts]);hits=collections.defaultdict(set)
    with (work/'feature_observations.jsonl').open() as f:
        for line in f:
            r=json.loads(line);hits[(names[r['class_id']],r['prompt'])].update(int(i) for i in tree.query(shape(r['geometry']),predicate='contains'))
    rows=[]
    for (cls,prompt),idx in hits.items():
        wrong=[pts[i] for i in sorted(idx) if cls not in pts[i]['accept']]
        rows.append({'class':cls,'prompt':prompt,'right':len(idx)-len(wrong),'wrong':len(wrong),
            'wrong_truth':dict(collections.Counter(p['label'] for p in wrong).most_common()),'wrong_points':[p['id'] for p in wrong]})
    return sorted(rows,key=lambda r:(-r['wrong'],r['right']))

def accuracy(pts,pred,layer):
    per=collections.defaultdict(lambda:[0,0]);conf=collections.defaultdict(list)
    for p in pts:
        ok=pred[p['id']][layer] in p['accept'];per[p['label']][0]+=ok;per[p['label']][1]+=1
        if not ok:conf[(p['label'],pred[p['id']][layer])].append(p['id'])
    n=sum(v[1] for v in per.values());c=sum(v[0] for v in per.values())
    return {'accuracy':round(100*c/n,2),'correct':c,'n':n,
        'per_class':{k:{'correct':v[0],'n':v[1],'accuracy':round(100*v[0]/v[1],1)} for k,v in sorted(per.items())},
        'errors':[{'truth':t,'predicted':pr,'count':len(ids),'points':ids} for (t,pr),ids in sorted(conf.items(),key=lambda x:-len(x[1]))]}

def compare(pts,base,cand,layer):
    fixed=[p for p in pts if base[p['id']][layer] not in p['accept'] and cand[p['id']][layer] in p['accept']]
    broken=[p for p in pts if base[p['id']][layer] in p['accept'] and cand[p['id']][layer] not in p['accept']]
    per=collections.Counter()
    for p in fixed:per[p['label']]+=1
    for p in broken:per[p['label']]-=1
    return {'net':len(fixed)-len(broken),'per_class_net':{k:v for k,v in per.items() if v},
        'fixed':[{'id':p['id'],'x':p['x'],'y':p['y'],'truth':p['label'],'was':base[p['id']][layer],'now':cand[p['id']][layer]} for p in fixed],
        'broken':[{'id':p['id'],'x':p['x'],'y':p['y'],'truth':p['label'],'was':base[p['id']][layer],'now':cand[p['id']][layer]} for p in broken]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('workspace');ap.add_argument('round');ap.add_argument('--base');ap.add_argument('--final',action='store_true')
    a=ap.parse_args();ws,_=workspace(a.workspace);rd=Path(a.round)
    ref=load_json(ws/'reference'/'reference.json')['points']
    splits={'tune':[p for p in ref if p['split']=='tune']}
    if a.final:splits['holdout']=[p for p in ref if p['split']=='holdout']
    pred=predict(rd,ref);has_obj=next(iter(pred.values()))['objects'] is not None
    score={s:{'features':accuracy(pts,pred,'features'),**({'objects':accuracy(pts,pred,'objects')} if has_obj else {}),
        'false_detections':false_detections(pts,pred)[1]} for s,pts in splits.items()}
    score['tune']['prompts']=prompt_hits(rd,splits['tune'])
    soft=soft_points(ws);spred=predict(rd,soft) if soft else {}
    soft_score=lambda pr:{l:sum(soft_ok(p,pr[p['id']][l]) for p in soft) for l in (['features','objects'] if has_obj else ['features'])}
    if soft:score['soft']={'n':len(soft),'acceptable':soft_score(spred),
        'unclassified':{l:sum(spred[p['id']][l]=='unclassified' for p in soft) for l in soft_score(spred)}}
    m=rd/'run'/'eval'/'metrics.json'
    if m.exists():score['diagnostics']={k:v for k,v in load_json(m).items() if k in ('coverage_pct','effective_overlap_pct','prompt_entries','objects_mean_purity','seconds')}
    summ=load_json(rd/'run'/'summary.json');g=lambda v:round(sum(v)/len(v),4)
    score.setdefault('diagnostics',{}).update(native_ground_gsd_m=g(summ['input']['native_ground_gsd_m']),
        presented_ground_gsd_m=g(summ['inference']['presented_ground_gsd_m']),state=summ['state'])
    save_json(rd/('final_score.json' if a.final else 'score.json'),score)
    out={s:{**{l:{'accuracy':v[l]['accuracy'],'correct':f"{v[l]['correct']}/{v[l]['n']}"} for l in ('features','objects') if l in v},
        'false_detections':v['false_detections']['count']} for s,v in score.items() if s in ('tune','holdout')}
    out['diagnostics']=score.get('diagnostics');out['soft']=score.get('soft');out['top_errors']=[f"{e['truth']} -> {e['predicted']}: {e['count']}" for e in score['tune']['features']['errors'][:8]]
    out['top_false_detections']=[f"{c} at {v['count']} points (truth: {v['truth']})" for c,v in list(score['tune']['false_detections']['by_class'].items())[:6]]
    out['worst_prompts']=[f"{r['class']}: '{r['prompt']}' wrong {r['wrong']} / right {r['right']}" for r in score['tune']['prompts'] if r['wrong']][:8]
    if a.base:
        tune=splits['tune'];base=predict(a.base,tune)
        cmp={l:compare(tune,base,pred,l) for l in (['features','objects'] if has_obj else ['features'])}
        # Compare false detections only at points whose winner is right in both rounds: a newly fixed point would
        # otherwise expose its pre-existing wrong detections as "added" (penalising the fix itself), and a newly
        # broken point would hide them as "removed". Points that change correctness are counted by the features net.
        both=[p for p in tune if base[p['id']]['features'] in p['accept'] and pred[p['id']]['features'] in p['accept']]
        fb,fc=false_detections(both,base)[0],false_detections(both,pred)[0]
        cmp['false_detections']={'net':len(fb-fc),'removed':len(fb-fc),'added':len(fc-fb),'added_pairs':sorted(fc-fb),'removed_pairs':sorted(fb-fc)}
        cmp['false_detections']['net']=len(fb-fc)-len(fc-fb)
        sb=predict(a.base,soft) if soft else {}
        cmp['soft']={l:sum(soft_ok(p,spred[p['id']][l]) for p in soft)-sum(soft_ok(p,sb[p['id']][l]) for p in soft) for l in soft_score(spred)} if soft else {}
        gain=cmp['features']['net']+FALSE_WEIGHT*cmp['false_detections']['net']
        formula=f"features net {cmp['features']['net']} + {FALSE_WEIGHT} x false-detection net {cmp['false_detections']['net']}"
        if soft and not cmp['features']['broken'] and cmp['soft']['features']:
            gain+=SOFT_WEIGHT*cmp['soft']['features'];formula+=f" + {SOFT_WEIGHT} x soft net {cmp['soft']['features']}"
        # Objects-only levers (--object-region-px/--object-compactness) leave features untouched: let the objects net decide then.
        if has_obj and not cmp['features']['fixed'] and not cmp['features']['broken'] and not cmp['false_detections']['added'] and not cmp['false_detections']['removed']:
            gain=cmp['objects']['net'];formula=f"objects net {gain} (objects-only round: no feature point or false detection changed)"
            if soft and not cmp['objects']['broken'] and cmp['soft'].get('objects'):
                gain+=SOFT_WEIGHT*cmp['soft']['objects'];formula+=f" + {SOFT_WEIGHT} x soft net {cmp['soft']['objects']}"
        cmp['gain']=gain;cmp['gain_formula']=formula
        reasons=[]
        if cmp['features']['net']<0:reasons.append(f"features net {cmp['features']['net']} (needs >= 0)")
        if gain<=0:reasons.append(f"gain {gain} = {formula} (needs > 0)")
        if has_obj and cmp['objects']['net']<-1:reasons.append(f"objects net {cmp['objects']['net']} (needs >= -1)")
        reasons+=[f'class {k} lost {-v} points net' for k,v in cmp['features']['per_class_net'].items() if v<-1]
        # A class losing points in objects fails the gate unless the features gain is large (>= LARGE_FEATURE_GAIN):
        # broad area phrases can swallow narrow surfaces in objects while features get them right.
        obj_losses={k:v for k,v in cmp['objects']['per_class_net'].items() if v<-1} if has_obj else {}
        if obj_losses and cmp['features']['net']<LARGE_FEATURE_GAIN:
            reasons+=[f'class {k} lost {-v} points net in objects (features net {cmp["features"]["net"]} < {LARGE_FEATURE_GAIN})' for k,v in obj_losses.items()]
        cmp['objects_class_losses_allowed']=obj_losses if obj_losses and cmp['features']['net']>=LARGE_FEATURE_GAIN else {}
        cmp['gate']='pass' if not reasons else 'fail';cmp['gate_reasons']=reasons;save_json(rd/'compare.json',cmp)
        out['vs_base']={'gate':cmp['gate'],'gain':gain,'gain_formula':formula,'reasons':reasons,
            **{l:{'net':cmp[l]['net'],'fixed':len(cmp[l]['fixed']),'broken':len(cmp[l]['broken']),'per_class_net':cmp[l]['per_class_net']} for l in ('features','objects') if l in cmp},
            'false_detections':{k:cmp['false_detections'][k] for k in ('net','removed','added')},
            'soft_net':cmp['soft'],'soft_says_worse':any(v<0 for v in cmp['soft'].values()),
            'objects_class_losses_allowed':cmp['objects_class_losses_allowed']}
    print(json.dumps(out,indent=1))

if __name__=='__main__':main()
