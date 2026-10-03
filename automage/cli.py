"""One configuration/API for command line and notebook."""
import argparse,json
from pathlib import Path
from .config import Config,load_dictionary,concepts
from .raster import plan

def main(argv=None):
    parser=argparse.ArgumentParser(prog="automage", description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ['classify','plan']:
        p=sub.add_parser(command);p.add_argument('--input',required=True);p.add_argument('--out',required=command=='classify');p.add_argument('--model',default=Config.model,help='Optional model directory override; defaults to bundled SAM3');p.add_argument('--dictionary',default=Config.dictionary)
        p.add_argument('--scale',type=float,default=1);p.add_argument('--target-gsd',type=float);p.add_argument('--overlap',type=float,default=.2);p.add_argument('--prompt-batch',type=int,default=8);p.add_argument('--max-windows',type=int);p.add_argument('--wall-seconds',type=int,default=5400);p.add_argument('--objects',action='store_true');p.add_argument('--resume',action='store_true')
        p.add_argument('--object-region-px',type=int,default=28,help='Target superpixel width in source pixels')
        p.add_argument('--object-compactness',type=float,default=12.)
        p.add_argument('--autocast-dtype',choices=['float16','bfloat16'],default=None)
    a=parser.parse_args(argv)
    cfg=Config(model=a.model,dictionary=a.dictionary,scale=a.scale,target_gsd=a.target_gsd,overlap=a.overlap,prompt_batch=a.prompt_batch,max_windows=a.max_windows,wall_seconds=a.wall_seconds,objects=a.objects,resume=a.resume,object_region_px=a.object_region_px,object_compactness=a.object_compactness,autocast_dtype=a.autocast_dtype)
    if a.command=='plan':
        info=plan(a.input,cfg);types=concepts(load_dictionary(cfg.dictionary));info.update(concepts=len(types),prompt_entries=sum(len(t['prompts']) for t in types),configuration=cfg.asdict());print(json.dumps(info,indent=2));return info
    from .pipeline import classify
    source=Path(a.input)
    if source.is_dir():
        paths=sorted(p for p in source.rglob('*') if p.suffix.lower() in ('.tif','.tiff'))
        if not paths:raise ValueError('No source GeoTIFFs')
        if Path(a.out).resolve().is_relative_to(source.resolve()):raise ValueError('Output root must not be inside input discovery root')
        result={str(p.relative_to(source)):classify(p,Path(a.out)/p.relative_to(source).with_suffix(''),cfg) for p in paths}
    else:result=classify(a.input,a.out,cfg)
    print(json.dumps(result,indent=2));return result


def entrypoint():
    """Console entry point; command results are printed by main."""
    main()
