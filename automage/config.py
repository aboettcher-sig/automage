"""Classification settings and dictionary loading."""
from __future__ import annotations
import dataclasses,hashlib,json,re,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def load_dictionary(path=None):
    value=json.loads(Path(path or ROOT/'dictionary.json').read_text())
    classes=value['classes']; seen=set()
    for c in classes:
        for t in c['types']:
            if t['id'] in seen or not t['prompts']: raise ValueError('Duplicate concept or empty prompt list')
            seen.add(t['id'])
    return value

def concepts(dictionary):
    return [{**t,'class_id':i+1,'class_name':c['name']} for i,c in enumerate(dictionary['classes']) for t in c['types']]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def model_sha(path):
    """Fingerprint either a single weight file or every indexed model shard."""
    root=Path(path)
    if (root/'model.safetensors').is_file():return sha(root/'model.safetensors')
    index=root/'model.safetensors.index.json'
    names=set(json.loads(index.read_text())['weight_map'].values())|{index.name}
    return {name:sha(root/name) for name in sorted(names)}

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))

@dataclasses.dataclass
class Config:
    model: str = str(ROOT/'models'/'sam3')
    dictionary: str = str(ROOT/'dictionary.json')
    tile: int = 1008
    overlap: float = .20
    prompt_batch: int = 8
    threshold: float = .5
    mask_threshold: float = .5
    scale: float = 1.
    target_gsd: float|None = None
    bands: tuple = (1,2,3)
    device: str = 'cuda'
    dtype: str = 'float32'
    autocast_dtype: str|None = None
    gpu_bytes: int = 32*2**30
    wall_seconds: int = 5400
    evaluation_reserve_seconds: int = 300
    output_bytes: int = 10*2**30
    max_windows: int|None = None
    objects: bool = False
    object_region_px: int = 28
    object_compactness: float = 12.
    object_block: int = 2048
    object_purity: float = .80
    resume: bool = False

    def validate(self):
        if self.tile != 1008: raise ValueError('SAM3 model window is fixed at1008; alter scale/GSD, not model tile')
        if not 0<=self.overlap<.75 or self.prompt_batch<1 or not math.isfinite(self.scale) or self.scale<=0: raise ValueError('Invalid window configuration')
        if self.target_gsd is not None and (not math.isfinite(self.target_gsd) or self.target_gsd<=0):raise ValueError('target GSD must be finite and positive')
        if not 0<self.threshold<1 or not 0<self.mask_threshold<1:raise ValueError('thresholds must be between0and1')
        if self.device!='cuda':raise ValueError('GPU inference required; CUDA API includes ROCm')
        if self.dtype not in ('float32','bfloat16'):raise ValueError('Unsupported dtype')
        if self.autocast_dtype not in (None,'float16','bfloat16'):raise ValueError('Unsupported autocast dtype')
        if self.object_region_px<2 or self.object_block<self.object_region_px or not math.isfinite(self.object_compactness) or self.object_compactness<=0 or not 0<self.object_purity<=1:raise ValueError('Invalid superpixel parameters')
        if self.wall_seconds<=self.evaluation_reserve_seconds:raise ValueError('No inference allowance after evaluation reserve')
        if len(self.bands)!=3 or min(self.bands)<1:raise ValueError('Exactly three RGB bands required')
        return self
    def asdict(self):return dataclasses.asdict(self)
