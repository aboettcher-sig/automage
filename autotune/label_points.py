"""Label reference points with an independent headless Claude session per sheet.

Usage: label_points.py <workspace> <A|B> [--jobs 4] [--model MODEL]

Labellers see only the source image chips and the class definitions, never a classification.
Writes reference/labels_<A|B>.json: {point_id: {"label": ..., "also": [...]}}, saved after every sheet.
Finished sheets are listed in reference/labels_<A|B>.sheets.json, so a rerun (e.g. after a usage limit)
labels only the sheets still missing. Exits with status 1 if any sheet failed.
"""
import argparse,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ask_claude,load_json,save_json,workspace

PROMPT='''You are labelling land cover in aerial imagery to build an accuracy reference. Read the image {sheet}.
It shows numbered points. Each point has two views: left = close-up with the point at the centre of the red circle;
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
- Judge only what is visible. Do not guess from what is typical for the neighbourhood.

Reply with only a JSON array, one object per point, e.g. [{{"id":"p017","label":"turf","also":[]}}]'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('workspace');ap.add_argument('labeller',choices=['A','B'])
    ap.add_argument('--jobs',type=int,default=4);ap.add_argument('--model');a=ap.parse_args()
    ws,cfg=workspace(a.workspace);ref=ws/'reference'
    names={t['name'] for t in cfg['taxonomy']}|{'uncertain'}
    classes='\n'.join(f"- {t['name']}: {t['definition']}" for t in cfg['taxonomy'])
    out=ref/f'labels_{a.labeller}.json';done_file=ref/f'labels_{a.labeller}.sheets.json'
    labels=load_json(out) if out.exists() else {};done=set(load_json(done_file)) if done_file.exists() else set()
    def run(sheet):
        reply=ask_claude(PROMPT.format(sheet=sheet,classes=classes),a.model)
        return {r['id']:{'label':r['label'] if r['label'] in names else 'uncertain','also':[x for x in r.get('also',[]) if x in names]}
                for r in reply if isinstance(r,dict) and 'id' in r and 'label' in r}
    sheets=[p for p in sorted((ref/'sheets').glob(f'{a.labeller}_*.png')) if p.name not in done]
    print(f'{len(done)} sheets already labelled, {len(sheets)} to go',flush=True);failed=[]
    with ThreadPoolExecutor(a.jobs) as ex:
        futures={ex.submit(run,sheet):sheet for sheet in sheets}
        for f in as_completed(futures):
            sheet=futures[f]
            try:got=f.result()
            except Exception as e:
                failed.append(sheet.name);print(f'{sheet.name}: FAILED ({str(e)[:200]})',flush=True);continue
            labels.update(got);done.add(sheet.name);save_json(out,labels);save_json(done_file,sorted(done))
            print(f'{sheet.name}: {len(got)} labels',flush=True)
    print(f'{len(labels)} labels -> {out}')
    if failed:
        print(f'{len(failed)} sheets failed; rerun the same command to label only those.',flush=True);sys.exit(1)

if __name__=='__main__':main()
