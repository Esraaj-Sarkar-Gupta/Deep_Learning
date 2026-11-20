import sys, subprocess, json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train import Net, residual, route_segment
root=Path(__file__).resolve().parents[1]; out=root/'runs'/'synthetic'; out.mkdir(parents=True,exist_ok=True)
xy=np.stack(np.meshgrid(np.linspace(-1,1,12),np.linspace(-1,1,10)),axis=-1).reshape(-1,2).astype('float32'); t=np.linspace(0,1,7); x=xy[:,0]; y=xy[:,1]
fields=np.array([np.stack([-np.cos(x)*np.sin(y)*np.exp(-2*z/100),np.sin(x)*np.cos(y)*np.exp(-2*z/100),-.25*(np.cos(2*x)+np.cos(2*y))*np.exp(-4*z/100)],axis=1) for z in t],dtype='float32'); path=out/'synthetic_fixture.npz'; np.savez_compressed(path,xy=xy,times=t,fields=fields,re=np.array(100.))
class TaylorGreen(torch.nn.Module):
    def forward(self,z):
        x,y,t=z[:,0:1],z[:,1:2],z[:,2:3]; a=torch.exp(-2*t/100); return torch.cat((-torch.cos(x)*torch.sin(y)*a,torch.sin(x)*torch.cos(y)*a,-.25*(torch.cos(2*x)+torch.cos(2*y))*a*a),1)
r=residual(TaylorGreen(),torch.rand(20,3,dtype=torch.float64),100); assert r.abs().max().item()<1e-12, r.abs().max()
m=Net(16,True); a=torch.randn(1,3); b=torch.randn(9,3); assert torch.allclose(m(a),m(torch.cat((a,b)))[:1],atol=1e-7); opt=torch.optim.Adam(m.parameters()); opt.zero_grad(); (residual(m,b,100)**2).mean().backward(); assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()); opt.step()
def run(*args): subprocess.run([sys.executable,str(root/'train.py'),*map(str,args)],cwd=root,check=True)
run('interpolate','--data-path',path,'--run-dir',out/'smoke')
run('train','--data-path',path,'--run-dir',out/'smoke','--config',root/'configs/smoke.json','--experiment','all','--device','cpu')
run('train','--data-path',path,'--run-dir',out/'boundary','--config',root/'configs/smoke.json','--experiment','segmented','--stop-after-segment','1')
state=torch.load(out/'boundary/segmented/checkpoint.pt',weights_only=False); assert state['segment']==1 and state['step']==0 and len(state['completed'])==1
for k,v in state['model'].items(): assert torch.equal(v,state['completed'][0][k])
run('train','--data-path',path,'--run-dir',out/'boundary','--config',root/'configs/smoke.json','--experiment','segmented','--resume')
a=torch.load(out/'smoke/segmented/checkpoint.pt',weights_only=False); b=torch.load(out/'boundary/segmented/checkpoint.pt',weights_only=False)
assert b['segment']==2 and len(b['completed'])==2
for i in range(2):
    for k in a['completed'][i]: assert torch.equal(a['completed'][i][k],b['completed'][i][k]), (i,k)
edges=np.linspace(t[0],t[-1],3); route=lambda z:route_segment(edges,z)
assert [route(v) for v in [0,.5,1]]==[0,1,1]
run('train','--data-path',path,'--run-dir',out/'smoke','--config',root/'configs/smoke.json','--experiment','plain','--resume')
print('PASS manufactured Navier-Stokes, attention optimizer and batch invariance, transfer, endpoint routing, segment-boundary resume, all model smoke tests',flush=True)
