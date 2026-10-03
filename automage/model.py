"""One GPU image encoding per window; bounded prompt batching, no CPU fallback."""
import gc,time
from contextlib import nullcontext
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model,Sam3Processor

class Sam3:
    def __init__(self,cfg):
        if not torch.cuda.is_available():raise RuntimeError('GPU unavailable; CPU inference is disabled')
        self.cfg=cfg;self.batch=cfg.prompt_batch;self.repairs=[];torch.set_num_threads(8)
        if torch.version.hip:torch.backends.cudnn.enabled=False  # Native GPU conv; avoids gfx1151 MIOpen batch failure
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.95,cfg.gpu_bytes/total))
        dtype=getattr(torch,cfg.dtype)
        self.model=Sam3Model.from_pretrained(cfg.model,local_files_only=True,dtype=dtype)
        # Owned host storage avoids the measured mmap-to-HIP transfer penalty.
        self.model._apply(lambda t:(t.clone() if t.device.type=='cpu' else t).to('cuda'))
        self.model.eval();self.processor=Sam3Processor.from_pretrained(cfg.model,local_files_only=True)
        self.metadata={'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'hip':torch.version.hip,'dtype':cfg.dtype,'autocast_dtype':cfg.autocast_dtype,'cudnn_miopen_enabled':torch.backends.cudnn.enabled}
    def infer(self,rgb,entries,deadline):
        if time.monotonic()>=deadline:raise TimeoutError('Inference budget exhausted')
        precision = torch.autocast('cuda',dtype=getattr(torch,self.cfg.autocast_dtype)) if self.cfg.autocast_dtype else nullcontext()
        with torch.inference_mode(), precision:
            image=self.processor(images=Image.fromarray(rgb),return_tensors='pt').to('cuda')
            image['pixel_values']=image['pixel_values'].to(getattr(torch,self.cfg.dtype))
            vision=self.model.get_vision_features(pixel_values=image['pixel_values']);del image
            cursor=0
            while cursor<len(entries):
                if time.monotonic()>=deadline:raise TimeoutError('Inference budget exhausted')
                selected=entries[cursor:cursor+self.batch];n=len(selected)
                expanded=type(vision)(fpn_hidden_states=tuple(t.expand(n,*t.shape[1:]).contiguous() for t in vision.fpn_hidden_states),fpn_position_encoding=tuple(t.expand(n,*t.shape[1:]).contiguous() for t in vision.fpn_position_encoding))
                text=self.processor(text=[r['prompt'] for r in selected],return_tensors='pt').to('cuda')
                try:
                    output=self.model(vision_embeds=expanded,input_ids=text['input_ids'],attention_mask=text['attention_mask'])
                    results=self.processor.post_process_instance_segmentation(output,threshold=self.cfg.threshold,mask_threshold=self.cfg.mask_threshold,target_sizes=[(self.cfg.tile,self.cfg.tile)]*n)
                    packed=[(r['masks'].cpu().numpy(),r['scores'].float().cpu().numpy()) for r in results]
                    del output,results,expanded,text
                except (torch.cuda.OutOfMemoryError,RuntimeError) as exc:
                    msg=str(exc)
                    known=isinstance(exc,torch.cuda.OutOfMemoryError) or any(s in msg.lower() for s in ['miopen','out of memory','cudnn'])
                    del expanded,text;gc.collect();torch.cuda.empty_cache()
                    if not known or self.batch==1:raise
                    new=max(1,self.batch//2);self.repairs.append({'from':self.batch,'to':new,'reason':msg[:200]});self.batch=new
                    continue
                for entry,(masks,scores) in zip(selected,packed):yield entry,masks,scores
                cursor+=n
            del vision
        torch.cuda.synchronize()
    def close(self):
        del self.model;gc.collect();torch.cuda.empty_cache()
