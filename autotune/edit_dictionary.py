"""Apply prompt edits to a dictionary: edit_dictionary.py <base.json> <edits.json> <out.json>

edits.json: [{"type": "<type id>", "remove": [...], "add": [...], "why": "..."},
             {"type": "<type id>", "drop_type": true, "why": "..."},
             {"class": "<class name>", "drop_class": true, "why": "..."},
             {"class": "<class name>", "new_type": "<type id>", "add": [...], "why": "..."}, ...]
"""
import json,sys
from automage.config import load_dictionary

def main(base,edits,out):
    d=json.load(open(base));classes={c['name']:c for c in d['classes']}
    for e in json.load(open(edits)):
        if e.get('drop_class'):d['classes']=[c for c in d['classes'] if c['name']!=e['class']];continue
        if 'new_type' in e:classes[e['class']]['types'].append({'id':e['new_type'],'name':e['new_type'].split('.')[-1],'prompts':e['add']});continue
        owner=next(c for c in d['classes'] if any(t['id']==e['type'] for t in c['types']))
        if e.get('drop_type'):owner['types']=[t for t in owner['types'] if t['id']!=e['type']];continue
        t=next(t for t in owner['types'] if t['id']==e['type'])
        missing=[p for p in e.get('remove',[]) if p not in t['prompts']]
        if missing:raise ValueError(f"{e['type']}: not present {missing}")
        t['prompts']=[p for p in t['prompts'] if p not in e.get('remove',[])]+[p for p in e.get('add',[]) if p not in t['prompts']]
    d['classes']=[c for c in d['classes'] if c['types']]
    json.dump(d,open(out,'w'),indent=2);load_dictionary(out)  # re-validate (no empty prompt lists, no duplicate ids)

if __name__=='__main__':main(*sys.argv[1:])
