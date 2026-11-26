"""Paired channelized benchmark. Run with the project Python environment."""
from pathlib import Path
import contextlib
import csv
import io
import json
import sys
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "low_resolution_baseline"))
from controlled_training import HERE, MODELS, SEEDS, build_model, file_hash, normalize
from evaluate_controlled import physical
from data_generation import ReservoirGrid, ReservoirProperties, ReservoirSimulator
from scipy.ndimage import label
import torch

CASES = range(8)
STEPS = (25, 50, 75, 100)
OUT = BASE / "channelized_paired"
OUT.mkdir(exist_ok=True)


def simulate(job):
    n,case = job
    rng = np.random.default_rng(2026112300 + case)
    amplitude, phase, halfwidth = rng.uniform(7,14), rng.uniform(-np.pi,np.pi), rng.uniform(8,11)
    coord = (np.arange(n)+.5)*100/n
    x,y = np.meshgrid(coord,coord)
    center = x + amplitude*np.sin(np.pi*x/100)*np.sin(2*np.pi*x/100+phase)
    mask = abs(y-center)<=halfwidth
    components,_ = label(mask)
    assert components[0,0] and components[0,0]==components[-1,-1]
    k = np.where(mask, 1e-13, 1e-15)
    # Same internal time step and same saved time spacing at both resolutions.
    # Save only the initial and four scoring states.
    props=ReservoirProperties()
    with contextlib.redirect_stdout(io.StringIO()):
        grid=ReservoirGrid(nx=n,ny=n,total_length=(100,100,1));grid.build_grid()
    sim=ReservoirSimulator(grid,props)
    sim.set_properties(k.ravel());sim.set_well_conditions();sim.compute_geometric_transmissibility()
    # Vectorized form of the archived stencil; preserve direction-specific mobility on pressure ties.
    oldp=np.full(n*n,props.p_init);olds=np.array([c.Sw for c in grid.celllist])
    ids=np.arange(n*n).reshape(n,n)
    left=np.r_[ids[:,:-1].ravel(),ids[:-1,:].ravel()]
    right=np.r_[ids[:,1:].ravel(),ids[1:,:].ravel()]
    i=np.r_[left,right];j=np.r_[right,left];cells=ids.ravel()
    geom=2*k.ravel()[i]*k.ravel()[j]/(k.ravel()[i]+k.ravel()[j])
    V=(100/n)**2;dt=10800/8
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    rows=np.r_[cells,i];cols=np.r_[cells,j]
    q=np.array(sim.qtvec,dtype=float)
    states=[];max_cfl=0
    for sub in range(1,809):
        u=np.clip((olds-.2)/.6,0,1)
        mw=u**4/props.mu_w/props.Bw
        mt=mw+(1-u)**2/props.mu_o/props.Bo
        fw=mw/mt
        trans=geom*np.where(oldp[j]>oldp[i],mt[j],mt[i])
        accum=.2*V/dt*((1-olds)*(props.Cr+props.Co)/props.Bo+olds*(props.Cr+props.Cw)/props.Bw)
        mat=coo_matrix((np.r_[accum+np.bincount(i,trans),-trans],(rows,cols)),shape=(n*n,n*n)).tocsr()
        p=spsolve(mat,((accum*oldp).astype(np.float32)+q.astype(np.float32)).astype(float))
        inflow=trans*np.maximum(p[j]-p[i],0)
        mix=np.bincount(i,inflow*(fw[j]-fw[i]))
        mix[0]+=props.q_in;mix[-1]-=props.q_in*fw[-1]
        s=np.clip(olds+dt*props.Bw/(.2*V)*mix-olds*(props.Cr+props.Cw)*(p-oldp)/V,.2,1)
        max_cfl=max(max_cfl,np.bincount(i,inflow).max()*dt*4.882266/(.2*V))
        if sub==1:
            sim.compute_mobility();sim.compute_total_transmissibility();ref=sim.solve_pressure_implicit(dt)
            for c,v in zip(grid.celllist,ref):c.press=float(v)
            sim.update_saturation(dt)
            np.testing.assert_allclose(p,ref,rtol=1e-9,atol=.002)
            np.testing.assert_allclose(s,[c.Sw for c in grid.celllist],atol=1e-8,rtol=0)
        if sub in [8*(1+t) for t in (0,*STEPS)]:
            states.append(np.stack((p,s)).reshape(2,n,n))
        oldp,olds=p,s
    assert max_cfl<1 and np.isfinite(states).all()
    np.savez_compressed(OUT/f"ref_{n}_{case}.npz",k=k,states=states,max_cfl=max_cfl,
                        amplitude=amplitude,phase=phase,halfwidth=halfwidth)
    return n,case


@torch.inference_mode()
def evaluate():
    norm=json.loads((HERE/"training_normalization.json").read_text())
    fits=json.loads((HERE/"training_summary.json").read_text())
    rows=[]
    for n in (20,64):
        refs=[np.load(OUT/f"ref_{n}_{c}.npz") for c in CASES]
        k=np.stack([r["k"] for r in refs])
        states=np.stack([r["states"] for r in refs])
        inputs=np.stack([normalize(a,key,norm) for a,key in zip([k,np.full_like(k,.2),states[:,0,0],states[:,0,1]],["permeability","porosity","pressure","saturation"])],axis=1)
        for model in MODELS:
            for seed in SEEDS:
                path=HERE/"checkpoints"/(model.replace("-","_")+f"_seed{seed}_best.pt")
                assert file_hash(path)==next(f["sha256"] for f in fits if f["model"]==model and f["seed"]==seed)
                net=build_model(model,seed).cuda().eval()
                net.load_state_dict(torch.load(path,map_location="cpu",weights_only=False)["model_state_dict"])
                x=torch.tensor(inputs,device="cuda",dtype=torch.float32)
                for t in range(1,101):
                    out=net(x);x=torch.cat((x[:,:2],out),dim=1)
                    if t not in STEPS:continue
                    pred=physical(out,norm).cpu().numpy()
                    truth=states[:,1+STEPS.index(t)]
                    for c in CASES:
                        band=(truth[c,1]>.23)&(truth[c,1]<.77)
                        assert band.any()
                        err=abs(pred[c]-truth[c])
                        rows.append(dict(grid=n,case=c,model=model,seed=seed,step=t,cells=int(band.sum()),band_mae=float(err[1][band].mean()),sw_mae=float(err[1].mean()),pressure_mae_pa=float(err[0].mean())))
    with (OUT/"metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for n in (20,64):
        for model in MODELS:
            print(n,model,[np.mean([r["band_mae"] for r in rows if (r["grid"],r["model"],r["step"])==(n,model,t)]) for t in STEPS],flush=True)


if __name__=="__main__":
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(4) as pool:
        for job in pool.map(simulate,[(n,c) for n in (20,64) for c in CASES]):print("reference",job,flush=True)
    evaluate()
