import argparse, json, time, hashlib, os, csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
torch.set_num_threads(int(os.environ.get("PINN_THREADS", "4")))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def raw(path, crop=None):
    with open(path,'rb') as f:
        re,_=map(float,f.readline().split()); f.readline(); nt,n=map(int,f.readline().split()); f.readline(); frames=[]; times=[]; xy=None
        for i in range(nt):
            times.append(float(f.readline())); block=f.read(70*n)
            if len(block)!=70*n: raise ValueError('Truncated field file')
            a=np.fromstring(block.decode('ascii'),sep=' ').reshape(n,5)
            if xy is None:
                xy=a[:,:2].copy(); mask=np.ones(n,bool) if crop is None else ((xy[:,0]>=crop[0])&(xy[:,0]<=crop[1])&(xy[:,1]>=crop[2])&(xy[:,1]<=crop[3])); coords=xy[mask]
            elif not np.array_equal(a[:,:2],xy): raise ValueError('Coordinates changed')
            frames.append(a[mask,2:].astype('float32'))
        if f.read(1): raise ValueError('Trailing data')
    return dict(xy=coords.astype('float32'),times=np.array(times),fields=np.array(frames),re=np.array(re))

def load(path):
    if str(path).endswith('.npz'):
        with np.load(path,allow_pickle=False) as z: d={k:z[k] for k in z.files}
    else: d=raw(path)
    if d['fields'].shape!=(len(d['times']),len(d['xy']),3) or not np.isfinite(d['fields']).all(): raise ValueError('Invalid fields')
    return d

def split(d,path):
    xy=d['xy']; sig=hashlib.sha256(xy.tobytes()).hexdigest()
    if len(xy)<65: raise ValueError('At least 65 nodes required')
    if path.exists():
        with np.load(path) as z:
            if str(z['signature'])!=sig: raise ValueError('Sensor split mismatch')
            return z['sensors']
    z=(xy-xy.min(0))/np.maximum(np.ptp(xy,axis=0),1e-9); ids=[int(np.argmin(((z-.5)**2).sum(1)))]; dist=((z-z[ids[0]])**2).sum(1)
    for _ in range(63):
        ids.append(int(dist.argmax())); dist=np.minimum(dist,((z-z[ids[-1]])**2).sum(1))
    np.savez(path,sensors=ids,signature=sig); return np.array(ids)

def interpolate(d,sensors,out):
    xy=d['xy']; held=np.ones(len(xy),bool); held[sensors]=False; target=xy[held]; tri=mtri.Triangulation(*xy[sensors].T); tid=tri.get_trifinder()(*target.T); inside=tid>=0
    vertices=np.zeros((len(target),3),int); w=np.zeros((len(target),3)); vertices[inside]=tri.triangles[tid[inside]]
    q=xy[sensors][vertices[inside]]; p=target[inside]; a=q[:,0]-q[:,2]; b=q[:,1]-q[:,2]; c=p-q[:,2]; det=a[:,0]*b[:,1]-a[:,1]*b[:,0]
    w[inside,0]=(c[:,0]*b[:,1]-c[:,1]*b[:,0])/det; w[inside,1]=(a[:,0]*c[:,1]-a[:,1]*c[:,0])/det; w[inside,2]=1-w[inside,:2].sum(1)
    missing=np.flatnonzero(~inside)
    for start in range(0,len(missing),2048):
        ix=missing[start:start+2048]; vertices[ix,0]=np.argmin(((target[ix,None]-xy[sensors][None])**2).sum(2),axis=1); w[ix,0]=1
    sums=np.zeros(2); refs=np.zeros(2); count=0
    for f in d['fields']:
        pred=(f[sensors][vertices,:2]*w[:,:,None]).sum(1); truth=f[held,:2]; sums+=((pred-truth)**2).sum(0); refs+=(truth**2).sum(0); count+=len(truth)
    result=dict(method='linear Delaunay; nearest outside convex hull',scope='all snapshots and non-sensor nodes',held_out_nodes=int(held.sum()),snapshots=len(d['times']),inside_hull_fraction=float(inside.mean()),components=['u','v'],rmse=np.sqrt(sums/count).tolist(),relative_l2_percent=(100*np.sqrt(sums/refs)).tolist())
    out.write_text(json.dumps(result,indent=2)); print(result,flush=True)

class Net(nn.Module):
    def __init__(self,width,attention=False,center=None,scale=None):
        super().__init__(); self.attention=attention; self.register_buffer('center',torch.tensor(center if center is not None else [0.,0.,0.],dtype=torch.float32)); self.register_buffer('scale',torch.tensor(scale if scale is not None else [1.,1.,1.],dtype=torch.float32)); self.embed=nn.Linear(3,width)
        if attention: self.q=nn.Linear(width,width); self.k=nn.Linear(width,width); self.v=nn.Linear(width,width)
        self.body=nn.Sequential(nn.Linear(width,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,3))
    def forward(self,x):
        if not self.attention: h=torch.tanh(self.embed((x-self.center)/self.scale))
        else:
            offsets=x.new_tensor([-.1,0.,.1]); tokens=x[:,None,:]+torch.stack([offsets*0,offsets*0,offsets],-1)[None]; h=torch.tanh(self.embed((tokens-self.center)/self.scale)); q=self.q(h[:,1:2]); k=self.k(h); v=self.v(h); h=(torch.softmax((q*k).sum(-1,keepdim=True).transpose(1,2)/(h.shape[-1]**.5),-1)@v).squeeze(1)
        return self.body(h)

def residual(model,x,re):
    x=x.requires_grad_(True); y=model(x)
    def grad(z): return torch.autograd.grad(z,x,torch.ones_like(z),create_graph=True,retain_graph=True)[0]
    u,v,p=y[:,0:1],y[:,1:2],y[:,2:3]; du,dv,dp=grad(u),grad(v),grad(p)
    uxx=grad(du[:,0:1])[:,0:1]; uyy=grad(du[:,1:2])[:,1:2]; vxx=grad(dv[:,0:1])[:,0:1]; vyy=grad(dv[:,1:2])[:,1:2]
    return torch.cat((du[:,0:1]+dv[:,1:2],du[:,2:3]+u*du[:,0:1]+v*du[:,1:2]+dp[:,0:1]-(uxx+uyy)/re,dv[:,2:3]+u*dv[:,0:1]+v*dv[:,1:2]+dp[:,1:2]-(vxx+vyy)/re),1)

def route_segment(edges,t):
    return min(max(int(np.searchsorted(edges,t,side="right")-1),0),len(edges)-2)

def train(args,d,sensors,c,kind):
    out=Path(args.run_dir)/kind; out.mkdir(parents=True,exist_ok=True); device=torch.device(args.device if args.device!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu')); torch.manual_seed(c['seed']); rng=np.random.default_rng(c['seed']); times=d['times']; xy=d['xy']; fields=d['fields']; nseg=c['segments'] if kind=='segmented' else 1; edges=np.linspace(times[0],times[-1],nseg+1)
    lo=np.r_[xy.min(0),times[0]]; hi=np.r_[xy.max(0),times[-1]]; model=Net(c['width'],kind=='attention',(lo+hi)/2,np.maximum((hi-lo)/2,1e-6)).to(device); opt=torch.optim.Adam(model.parameters(),lr=c['lr']); ck=out/'checkpoint.pt'; segstart=0; step=0; completed=[]; elapsed=0.; sig=hashlib.sha256(xy.tobytes()+times.tobytes()+fields.tobytes()+sensors.tobytes()).hexdigest()
    if args.resume and ck.exists():
        state=torch.load(ck,map_location=device,weights_only=False)
        if state['config']!=c or state['signature']!=sig: raise ValueError('Checkpoint mismatch')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); segstart=state['segment']; step=state['step']; completed=state['completed']; rng.bit_generator.state=state['rng']; torch.set_rng_state(state['torch_rng'].cpu()); elapsed=state['elapsed']
    tic=time.perf_counter()
    def checkpoint(seg,st):
        tmp=ck.with_suffix('.tmp'); torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),segment=seg,step=st,completed=completed,config=c,signature=sig,rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),elapsed=elapsed+time.perf_counter()-tic),tmp); tmp.replace(ck)
    for seg in range(segstart,nseg):
        eligible=np.flatnonzero((times>=edges[seg]) & ((times<edges[seg+1]) if seg<nseg-1 else (times<=edges[seg+1])))
        if not len(eligible): raise ValueError('Empty segment')
        for i in range(step,c['steps']):
            ti=rng.choice(eligible,c['batch']); si=rng.choice(sensors,c['batch']); x=np.column_stack((xy[si],times[ti])).astype('float32'); y=fields[ti,si,:2]; ci=rng.integers(0,len(xy),c['collocation']); ct=rng.uniform(edges[seg],edges[seg+1],c['collocation']); xc=np.column_stack((xy[ci],ct)).astype('float32')
            opt.zero_grad(); pred=model(torch.tensor(x,device=device)); loss=((pred[:,:2]-torch.tensor(y,device=device))**2).mean()+c['physics_weight']*(residual(model,torch.tensor(xc,device=device),float(d['re']))**2).mean(); loss.backward(); opt.step()
            if (i+1)%c['checkpoint_every']==0: checkpoint(seg,i+1)
        completed.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()}); step=0; opt=torch.optim.Adam(model.parameters(),lr=c['lr']); checkpoint(seg+1,0); print(kind,'segment',seg+1,'loss',float(loss.detach()),flush=True)
        if args.stop_after_segment and seg+1 == args.stop_after_segment: return
    if len(completed)!=nseg: raise RuntimeError('Incomplete checkpoint')
    model.eval(); held=np.ones(len(xy),bool); held[sensors]=False; ti=len(times)//2; coords=xy[held]; preds=[]; seg=route_segment(edges,times[ti]); model.load_state_dict(completed[seg])
    with torch.no_grad():
        for chunk in np.array_split(coords,max(1,int(np.ceil(len(coords)/4096)))):
            xx=torch.tensor(np.column_stack((chunk,np.full(len(chunk),times[ti]))),device=device,dtype=torch.float32); preds.append(model(xx).cpu().numpy())
    pred=np.concatenate(preds); truth=fields[ti,held].copy(); pred[:,2]+=np.mean(truth[:,2]-pred[:,2]); error=pred-truth; rmse=np.sqrt((error**2).mean(0)); ref=np.sqrt((truth**2).mean(0)); result=dict(components=['u','v','pressure_gauge_aligned'],rmse=rmse.tolist(),relative_l2_percent=(100*rmse/np.maximum(ref,1e-12)).tolist(),snapshot_time=float(times[ti]),held_out_nodes=len(coords),steps_per_segment=c['steps'],segments=nseg,total_optimizer_steps=nseg*c['steps'],elapsed_training_seconds=elapsed+time.perf_counter()-tic,label=c['label'],device=str(device),pressure_alignment='held-out snapshot mean offset, evaluation only')
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    table=Path(args.run_dir)/'results.csv'; rows=[]
    if table.exists():
        with table.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    row={'experiment':kind,'label':c['label'],'snapshot_time':float(times[ti]),'optimizer_steps':nseg*c['steps'],'u_rmse':float(rmse[0]),'v_rmse':float(rmse[1]),'pressure_gauge_aligned_rmse':float(rmse[2]),'u_relative_l2_percent':float(result['relative_l2_percent'][0]),'v_relative_l2_percent':float(result['relative_l2_percent'][1]),'pressure_relative_l2_percent':float(result['relative_l2_percent'][2])}
    rows=[r for r in rows if r['experiment']!=kind]+[row]
    with table.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader(); writer.writerows(rows)
    fig,axs=plt.subplots(3,3,figsize=(13,10),constrained_layout=True)
    for j,name in enumerate(['u','v','pressure (gauge aligned)']):
        lim=(min(truth[:,j].min(),pred[:,j].min()),max(truth[:,j].max(),pred[:,j].max()))
        for k,(val,title) in enumerate([(truth[:,j],'truth'),(pred[:,j],'prediction'),(abs(error[:,j]),'absolute error')]):
            im=axs[j,k].scatter(coords[:,0],coords[:,1],c=val,s=3,cmap='viridis',vmin=lim[0] if k<2 else None,vmax=lim[1] if k<2 else None); axs[j,k].set_title(f'{name}: {title}'); axs[j,k].set_aspect('equal'); fig.colorbar(im,ax=axs[j,k])
    fig.suptitle(f'{kind} - {c["label"]}, t={times[ti]:g}'); fig.savefig(out/'snapshot.png',dpi=120); plt.close(fig); print(kind,result,flush=True)

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True); q=sub.add_parser('prepare'); q.add_argument('--data-path',required=True); q.add_argument('--output',required=True); q.add_argument('--crop',nargs=4,type=float,default=[-1.5,8,-2,2])
    for cmd in ['verify','interpolate','train']:
        q=sub.add_parser(cmd); q.add_argument('--data-path',required=True); q.add_argument('--run-dir',required=True)
        if cmd=='interpolate': q.add_argument('--sensor-data-path',help='Dataset whose frozen sensor coordinates are mapped onto this evaluation dataset')
        if cmd=='train':
            q.add_argument('--config',required=True); q.add_argument('--experiment',choices=['all','plain','segmented','attention'],default='all'); q.add_argument('--device',choices=['auto','cpu','cuda'],default='auto'); q.add_argument('--resume',action='store_true'); q.add_argument('--stop-after-segment',type=int,default=0,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.command=='prepare':
        d=raw(args.data_path,args.crop); Path(args.output).parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(args.output,**d); print('Prepared',args.output,d['fields'].shape,flush=True); return
    d=load(args.data_path); out=Path(args.run_dir); out.mkdir(parents=True,exist_ok=True)
    if args.command=='interpolate' and args.sensor_data_path:
        sensor_data=load(args.sensor_data_path); source=split(sensor_data,out/'sensors.npz'); lookup={tuple(v):i for i,v in enumerate(d['xy'])}
        try: sensors=np.array([lookup[tuple(v)] for v in sensor_data['xy'][source]])
        except KeyError as exc: raise ValueError('Sensor coordinates absent in evaluation dataset') from exc
    else: sensors=split(d,out/'sensors.npz')
    print('Dataset',d['fields'].shape,'Re',float(d['re']),'time',d['times'][[0,-1]],'sensors',len(sensors),flush=True)
    if args.command=='interpolate': interpolate(d,sensors,out/'interpolation.json')
    if args.command=='train':
        c=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        for kind in (['plain','segmented','attention'] if args.experiment=='all' else [args.experiment]): train(args,d,sensors,c,kind)
if __name__=='__main__': main()
