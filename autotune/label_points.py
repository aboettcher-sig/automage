"""Label reference points with independent headless Claude sessions, each labelling a few sheets.

Usage: label_points.py <workspace> <A|B> [--jobs 4] [--per-session 3] [--model MODEL] [--dir reference]

--model overrides the model chosen for this labeller in workspace.json (sample_points.py --labellers).
--dir picks the folder inside the workspace holding sheets/ (default: reference; final_assess.py uses final_assessment).
--per-session sets how many sheets one session labels (default 3). Each sheet stays a separate full-resolution image;
grouping only pays each session's fixed start-up cost once per group. A sheet counts as done only if the reply has
labels for it, so a partial reply keeps the sheets it did label.

Labellers see only the source image chips and the class definitions, never a classification.
Writes reference/labels_<A|B>.json: {point_id: {"label": ..., "also": [...]}}, saved after every sheet. A labeller can
answer "other" with a "describe" text for a surface no class fits; those points are left out of accuracy and
listed by merge_labels.py as surfaces the class list doesn't cover.
Finished sheets are listed in reference/labels_<A|B>.sheets.json, so a rerun (e.g. after a usage limit)
labels only the sheets still missing. Exits with status 1 if any sheet failed.
"""
import argparse,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ask_claude,load_json,save_json,workspace

PROMPT='''You are labelling land cover in aerial imagery to build an accuracy reference. Read each of these images:
{sheets}
Each image shows numbered points. Each point has two views: left = close-up with the point at the centre of the red circle;
right = wider context with the close-up area boxed in red. Black areas are outside the image.

Label the surface exactly at the centre of the circle, as seen from above. Classes (use these exact names):
{classes}

Rules:
- "label": the one class that best describes the centre pixel. Whatever is on top counts: a car on a road is
  "vehicles", a tree crown over a lawn is "canopy".
- If the centre is in shadow but the surface beneath is recognisable, label that surface and put "shadow" in "also".
  If nothing beneath is recognisable, label "shadow".
- "also": other classes that would be equally correct for this exact pixel. Usually empty.
- Label "uncertain" if you cannot tell with reasonable confidence or the centre sits right on a boundary.
- Label "other" only if you can clearly see what the surface is but none of the classes fits it (e.g. a golf sand
  bunker, a solar carport); then add "describe" with a short plain name for it, e.g. "sand bunker". Don't use
  "other" for something that fits a class, even loosely.
- Judge only what is visible. Do not guess from what is typical for the neighbourhood.

Reply with only a JSON object mapping each image's file name to an array with one object per point on it, e.g.
{{"A_00.png": [{{"id":"p017","label":"turf","also":[]}}, {{"id":"p018","label":"other","also":[],"describe":"sand bunker"}}],
"A_01.png": [...]}}'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('workspace');ap.add_argument('labeller',choices=['A','B'])
    ap.add_argument('--jobs',type=int,default=4);ap.add_argument('--per-session',type=int,default=3);ap.add_argument('--model');ap.add_argument('--dir',default='reference');a=ap.parse_args()
    ws,cfg=workspace(a.workspace);ref=ws/a.dir
    model=a.model or cfg.get('labeller_models',{}).get(a.labeller)  # the run's choice (sample_points.py --labellers) unless overridden
    print(f'labeller {a.labeller}: model {model or "default"}',flush=True)
    names={t['name'] for t in cfg['taxonomy']}|{'uncertain','other'}
    classes='\n'.join(f"- {t['name']}: {t['definition']}" for t in cfg['taxonomy'])
    out=ref/f'labels_{a.labeller}.json';done_file=ref/f'labels_{a.labeller}.sheets.json'
    labels=load_json(out) if out.exists() else {};done=set(load_json(done_file)) if done_file.exists() else set()
    def parse(rows):
        out={}
        for r in rows if isinstance(rows,list) else []:
            if not(isinstance(r,dict) and 'id' in r and 'label' in r):continue
            lab=r['label'] if r['label'] in names else 'uncertain';desc=str(r.get('describe','')).strip().lower()
            if lab=='other' and not desc:lab='uncertain'  # "other" needs a description to be useful
            out[r['id']]={'label':lab,'also':[x for x in r.get('also',[]) if x in names-{'uncertain','other'}],**({'describe':desc} if lab=='other' else {})}
        return out
    def run(group):
        reply=ask_claude(PROMPT.format(sheets='\n'.join(str(s) for s in group),classes=classes),model)
        if not isinstance(reply,dict):raise RuntimeError('reply is not an object keyed by image name')
        return {s.name:parse(reply.get(s.name)) for s in group}
    sheets=[p for p in sorted((ref/'sheets').glob(f'{a.labeller}_*.png')) if p.name not in done]
    groups=[sheets[i:i+a.per_session] for i in range(0,len(sheets),a.per_session)]
    print(f'{len(done)} sheets already labelled, {len(sheets)} to go in {len(groups)} sessions',flush=True);failed=[]
    with ThreadPoolExecutor(a.jobs) as ex:
        futures={ex.submit(run,group):group for group in groups}
        for f in as_completed(futures):
            group=futures[f]
            try:got=f.result()
            except Exception as e:
                failed+=[s.name for s in group];print(f'{", ".join(s.name for s in group)}: FAILED ({str(e)[:200]})',flush=True);continue
            for name,sheet_labels in got.items():
                if not sheet_labels:failed.append(name);print(f'{name}: FAILED (no labels in reply)',flush=True);continue
                labels.update(sheet_labels);done.add(name);print(f'{name}: {len(sheet_labels)} labels',flush=True)
            save_json(out,labels);save_json(done_file,sorted(done))
    print(f'{len(labels)} labels -> {out}')
    if failed:
        print(f'{len(failed)} sheets failed; rerun the same command to label only those.',flush=True);sys.exit(1)

if __name__=='__main__':main()
