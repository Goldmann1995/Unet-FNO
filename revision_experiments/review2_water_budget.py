"""Water-volume accounting over the first 100 transitions of the independent test set."""
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.optimize import minimize_scalar

BASE = Path(__file__).resolve().parent
DT, PV, VOLUME = 10800., 5., 25.
INJECTED = 5. * DT / 86400.
CW_CR = 14e-6 / 6894.76


def mobility(s):
    x = np.clip((s - .2) / .6, 0., 1.)
    water = x**4 / .001
    total = water + (1-x)**2 / (.0018 * 1.2)
    return total, water / total


def slope(x):
    den = 2.16*x**4 + (1-x)**2
    return 2.16*(4*x**3*(1-x)**2 + 2*x**4*(1-x)) / (.6*den**2)


max_slope = -minimize_scalar(lambda x: -slope(x), bounds=(0, 1), method='bounded').fun
shock_x = minimize_scalar(lambda x: -mobility(.2+.6*x)[1]/x,
                          bounds=(1e-8, 1.), method='bounded').x


def global_budget(p, s):
    old = s[:, :-1]
    _, fw = mobility(old)
    net_well = INJECTED * (1-fw[..., -1, -1])
    storage = PV * (np.diff(s, axis=1) + CW_CR*old*np.diff(p, axis=1)).sum(axis=(-2,-1))
    return storage-net_well


def trajectory_metrics(p, s):
    residual = global_budget(p,s)
    swept = s >= .5
    retreat = swept[:,:-1] & ~swept[:,1:]
    return dict(mean_abs_step_residual_pct=100*np.abs(residual).mean()/INJECTED,
                signed_cumulative_residual_m3=residual.sum(1).mean(),
                retreat_case_percent=100*retreat.any((1,2,3)).mean(),
                retreat_events=int(retreat.sum()), lower_pct=100*(s[:,1:]<.2).mean(),
                upper_pct=100*(s[:,1:]>1).mean())


with h5py.File(BASE/'independent_20x20_100.h5') as f:
    groups=[f['data'][k] for k in sorted(f['data'])]
    def read(key):
        return np.stack([g[key][:101].reshape(101,20,20) for g in groups]).astype(float)
    pressure, saturation = read('pressure'), read('saturation')
    perm=np.stack([g['permeability'][:].reshape(20,20) for g in groups])
records=[]
for p,s,k in zip(pressure,saturation,perm):
    old=s[:-1]
    total,fw=mobility(old)
    incoming=np.zeros_like(old)
    mixing=np.zeros_like(old)
    for axis in (1,2):
        for shift in (-1,1):
            kn=np.roll(k,shift,axis-1)
            geom=2*k*kn/(k+kn)
            neighbor_p=np.roll(p,shift,axis)
            up=np.where(neighbor_p[:-1]>p[:-1],np.roll(total,shift,axis),total)
            flow=DT*geom*up*np.maximum(neighbor_p[1:]-p[1:],0)
            edge=[slice(None)]*3
            edge[axis]=0 if shift==1 else -1
            flow[tuple(edge)]=0
            incoming+=flow
            mixing+=flow*(np.roll(fw,shift,axis)-fw)
    well=np.zeros_like(old)
    well[:,0,0]=INJECTED
    well[:,-1,-1]=-INJECTED*fw[:,-1,-1]
    compression=PV*CW_CR*old*np.diff(p,axis=0)
    raw=old+(mixing+well-compression/VOLUME)/PV
    clipped=np.clip(raw,.2,1)
    discrepancy=np.max(abs(clipped-s[1:]))
    assert discrepancy<2e-6,discrepancy
    records.append(dict(replay_max=float(discrepancy),mixing_m3=mixing.sum(),
                        clipping_m3=PV*(clipped-raw).sum(),
                        physical_compression_m3=compression.sum(),
                        net_well_m3=well.sum(), max_advective_CFL=(incoming*max_slope/PV).max()))
result=dict(reservoirs=100,steps=100,injected_m3_per_reservoir=100*INJECTED,
            shock_sw=.2+.6*shock_x,max_fractional_flow_slope=max_slope,
            replay_max=max(r['replay_max'] for r in records),
            mean_budget={k:np.mean([r[k] for r in records]) for k in records[0]},
            max_inflow_CFL=max(r['max_advective_CFL'] for r in records),
            reference=trajectory_metrics(pressure,saturation),runs={})
for path in sorted((BASE/'low_resolution_baseline').glob('*_seed*_rollout_predictions.npz')):
    with np.load(path) as f:
        pred=f['predictions']
    p=np.concatenate((pressure[:,:1],pred[:,:,0]),axis=1)
    s=np.concatenate((saturation[:,:1],pred[:,:,1]),axis=1)
    result['runs'][path.stem]=trajectory_metrics(p,s)
assert len(result['runs'])==12
(BASE/'review2_water_budget.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
for model in ('FNO','U_FNO','U_FNO_LR','UNet_FNO'):
    runs=[v for k,v in result['runs'].items() if k.split('_seed')[0]==model]
    print(model,{k:np.mean([v[k] for v in runs]) for k in runs[0]})
