"""Mask-aware, georeferenced RGB windows with explicit native information support."""
import math
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from pyproj import CRS,Transformer,Geod

def native_gsd(src):
    """Ground distances near image centre, including geographic/WebMercator distortion."""
    transformer=Transformer.from_crs(src.crs,4326,always_xy=True);geod=Geod(ellps='WGS84')
    x,y=src.width/2,src.height/2
    p=[transformer.transform(*(src.transform*v)) for v in [(x,y),(x+1,y),(x,y+1)]]
    return [abs(geod.inv(*p[0],*q)[2]) for q in p[1:]]

def rendering(src,bands):
    if max(bands)>src.count:raise ValueError('Requested RGB bands not present')
    if all(src.dtypes[b-1]=='uint8' for b in bands):return None
    scale=min(1,1024/max(src.width,src.height));a=src.read(bands,out_shape=(3,max(1,round(src.height*scale)),max(1,round(src.width*scale))),masked=True)
    ranges=[]
    for band in a:
        v=band.compressed();v=v[np.isfinite(v)]
        if not len(v):raise ValueError('No finite valid RGB samples')
        ranges.append(np.percentile(v,[2,98]).tolist())
    return ranges

def plan(source,cfg):
    cfg.validate()
    with rasterio.open(source) as src:
        if src.crs is None:raise ValueError('GeoTIFF CRS required')
        gsd=native_gsd(src);factor=cfg.target_gsd/(sum(gsd)/2) if cfg.target_gsd else cfg.scale
        W,H=math.ceil(src.width/factor),math.ceil(src.height/factor)
        stride=max(1,round(cfg.tile*(1-cfg.overlap)))
        origins=lambda n:list(range(0,max(1,n-cfg.tile+1),stride))+([] if n<=cfg.tile or (n-cfg.tile)%stride==0 else [n-cfg.tile])
        windows=[(x,y,min(cfg.tile,W-x),min(cfg.tile,H-y)) for y in origins(H) for x in origins(W)]
        return {'source':str(source),'width':src.width,'height':src.height,'crs':src.crs.to_wkt(),'transform':list(src.transform),
            'native_ground_gsd_m':gsd,'presented_ground_gsd_m':[v*factor for v in gsd], 'factor':factor,'analysis_size':[W,H],
            'native_window_footprint_pixels':cfg.tile*factor,'windows':windows,'window_count':len(windows),
            'information_note':'Resampling changes presented scale, not acquired spatial information; ground GSD estimated at scene centre.'}

def read_window(src,window,info,cfg,stretch):
    x,y,w,h=window;f=info['factor'];native=Window(x*f,y*f,w*f,h*f)
    a=src.read(cfg.bands,window=native,out_shape=(3,h,w),boundless=True,fill_value=0,resampling=Resampling.bilinear,masked=True)
    v=(~np.ma.getmaskarray(a)).all(0)&np.isfinite(a.data).all(0)
    data=a.filled(0).astype(np.float32)
    if stretch:
        for b,(lo,hi) in enumerate(stretch):data[b]=(data[b]-lo)/max(hi-lo,1e-6)*255
    rgb=np.zeros((cfg.tile,cfg.tile,3),np.uint8);valid=np.zeros((cfg.tile,cfg.tile),bool)
    rgb[:h,:w]=np.clip(data.transpose(1,2,0),0,255).astype(np.uint8);valid[:h,:w]=v
    rgb[~valid]=0
    # Georeferenced native footprint for output, clipped at source edges.
    x0,y0=max(0,round(native.col_off)),max(0,round(native.row_off));x1,y1=min(src.width,round(native.col_off+native.width)),min(src.height,round(native.row_off+native.height))
    return rgb,valid,Window(x0,y0,x1-x0,y1-y0)
