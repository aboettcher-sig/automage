"""Create a workspace: sample stratified reference points and render labelling sheets.

Usage: sample_points.py <image.tif> <workspace_dir> [--n 300] [--seed 0] [--dictionary bundled]

Strata: k-means clusters of colour+texture, plus high-gradient edge pixels (thin features, boundaries).
Chip extents and point spacing are set in ground metres, so labellers see the same ground on any GSD, with a minimum
size in pixels so coarse imagery (e.g. 0.5 m) isn't shown as a few enlarged pixels.
Writes workspace.json, reference/points.json and reference/sheets/{A,B}_NN.png: the same points in two
independent orders/groupings, one set per labeller.
"""
import argparse,math,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage
from scipy.cluster.vq import kmeans2,whiten
from skimage.color import rgb2lab
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import read_rgb,save_json
import rasterio
from automage.config import load_dictionary
from automage.raster import native_gsd

PER_SHEET=10;TILE=240
CLOSE_M=4.6;WIDE_M=13.7;MIN_DIST_M=1.4  # half-widths of the close-up and context views, and point spacing, in metres
CLOSE_MIN_PX=40;WIDE_MIN_PX=120  # but never fewer source pixels than this, so coarse imagery still shows recognisable shapes

def strata(rgb,valid,k,seed):
    s=max(1,round(max(rgb.shape[:2])/800));small=rgb[::s,::s];v=valid[::s,::s]
    lab=rgb2lab(small/255.);L=lab[...,0]
    tex=np.sqrt(np.maximum(ndimage.uniform_filter(L**2,5)-ndimage.uniform_filter(L,5)**2,0))
    feat=np.stack([L,lab[...,1],lab[...,2],tex],-1)[v];w=whiten(feat)
    _,lbl=kmeans2(w,k,seed=seed,minit='++');grid=np.full(v.shape,-1);grid[v]=lbl
    edge=np.hypot(ndimage.sobel(L,0),ndimage.sobel(L,1));grid[v&(edge>np.percentile(edge[v],85))]=k
    return grid,s

def sample(grid,s,valid,n,seed,min_dist,edge_share=.2,floor=8):
    rng=np.random.default_rng(seed);k=grid.max();sizes={c:int((grid==c).sum()) for c in range(k+1)}
    n_edge=round(n*edge_share);rest=n-n_edge;total=sum(v for c,v in sizes.items() if c<k)
    quota={c:max(floor,round(rest*sizes[c]/total)) for c in range(k) if sizes[c]}
    f=rest/sum(quota.values());quota={c:max(1,math.floor(q*f)) for c,q in quota.items()};quota[k]=max(0,n-sum(quota.values()))
    pts=[]
    for c,q in quota.items():
        if q<=0:continue
        ys,xs=np.nonzero(grid==c);got=0
        for i in rng.permutation(len(ys)):
            x=min(valid.shape[1]-1,xs[i]*s+int(rng.integers(s)));y=min(valid.shape[0]-1,ys[i]*s+int(rng.integers(s)))
            if not valid[y,x] or any((x-p['x'])**2+(y-p['y'])**2<min_dist**2 for p in pts):continue
            pts.append({'x':int(x),'y':int(y),'stratum':'edge' if c==k else f'cluster_{c}'});got+=1
            if got==q:break
    order=rng.permutation(len(pts))
    return [{'id':f'p{i:03d}',**pts[j]} for i,j in enumerate(order)]

def crop(rgb,x,y,half):
    H,W=rgb.shape[:2];out=np.zeros((2*half,2*half,3),'uint8')
    y0,y1,x0,x1=y-half,y+half,x-half,x+half
    out[max(0,-y0):2*half-max(0,y1-H),max(0,-x0):2*half-max(0,x1-W)]=rgb[max(0,y0):min(H,y1),max(0,x0):min(W,x1)]
    return Image.fromarray(out).resize((TILE,TILE),Image.Resampling.BICUBIC)

def marker(d,c,r):
    for w,col in ((4,'white'),(2,'red')):
        d.ellipse((c-r,c-r,c+r,c+r),outline=col,width=w)
        for a,b in (((c-r-12,c),(c-r-3,c)),((c+r+3,c),(c+r+12,c)),((c,c-r-12),(c,c-r-3)),((c,c+r+3),(c,c+r+12))):d.line((a,b),fill=col,width=w)

def sheet(rgb,pts,path,close_px,wide_px):
    cw,ch=2*TILE+20,TILE+30;img=Image.new('RGB',(2*cw+10,5*ch),'white');d=ImageDraw.Draw(img)
    for i,p in enumerate(pts):
        ox,oy=(i%2)*(cw+10),(i//2)*ch
        close=crop(rgb,p['x'],p['y'],close_px);marker(ImageDraw.Draw(close),TILE//2,14)
        wide=crop(rgb,p['x'],p['y'],wide_px);r=TILE*close_px/(2*wide_px);c=TILE/2
        ImageDraw.Draw(wide).rectangle((c-r,c-r,c+r,c+r),outline='red',width=2)
        img.paste(close,(ox,oy+26));img.paste(wide,(ox+TILE+10,oy+26))
        d.text((ox+4,oy+6),f"{p['id']}   (left: close-up, point at circle centre / right: context, close-up area boxed)",fill='black')
    img.save(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('image');ap.add_argument('workspace')
    ap.add_argument('--n',type=int,default=300);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--clusters',type=int,default=10)
    ap.add_argument('--dictionary');a=ap.parse_args()
    ws=Path(a.workspace).resolve();ref=ws/'reference';(ref/'sheets').mkdir(parents=True,exist_ok=False);(ws/'rounds').mkdir()
    image=str(Path(a.image).resolve());rgb,valid=read_rgb(image)
    taxonomy=[{'name':c['name'],'definition':c.get('definition','')} for c in load_dictionary(a.dictionary)['classes']]
    with rasterio.open(image) as src:gsd=native_gsd(src)
    save_json(ws/'workspace.json',{'image':image,'native_ground_gsd_m':round(sum(gsd)/2,4),'taxonomy':taxonomy,'seed':a.seed})
    g=sum(gsd)/2;px=lambda m:max(8,round(m/g))
    grid,s=strata(rgb,valid,a.clusters,a.seed);pts=sample(grid,s,valid,a.n,a.seed,px(MIN_DIST_M))
    save_json(ref/'points.json',pts)
    rng=np.random.default_rng(a.seed+1)
    for name,order in (('A',pts),('B',[pts[i] for i in rng.permutation(len(pts))])):
        for j in range(0,len(order),PER_SHEET):sheet(rgb,order[j:j+PER_SHEET],ref/'sheets'/f'{name}_{j//PER_SHEET:02d}.png',max(CLOSE_MIN_PX,px(CLOSE_M)),max(WIDE_MIN_PX,px(WIDE_M)))
    print(f'native ground GSD {sum(gsd)/2:.4f} m; {len(pts)} points, {math.ceil(len(pts)/PER_SHEET)} sheets per labeller -> {ref}')

if __name__=='__main__':main()
