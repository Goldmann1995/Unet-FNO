from pathlib import Path
import numpy as np,h5py,json
from scipy.optimize import minimize_scalar
base=Path(__file__).resolve().parent
with h5py.File(base/'independent_20x20_100.h5') as f:
    groups=list(f['data'].values())
    def field(key,t):
        return np.stack([g[key][t].reshape(20,20) for g in groups])
    s=field('saturation',100);p=field('pressure',100);p0=field('pressure',0)
mask=(s>.23)&(s<.77)
assert mask.any(axis=(1,2)).all()
out={}
for model in ['FNO','U_FNO','U_FNO_LR','UNet_FNO']:
    runs=[]; violations=[]; fractions=[]
    for seed in (42,43,44):
        with np.load(base/'low_resolution_baseline'/f'{model}_seed{seed}_rollout_predictions.npz') as z:a=z['predictions']
        assert a.shape==(100,100,2,20,20) and np.isfinite(a).all()
        e=abs(a[:,-1,1]-s);dp=abs(a[:,-1,0]-p)
        runs.append(np.column_stack([e.mean((1,2)),(e*mask).sum((1,2))/mask.sum((1,2)),100*dp.mean((1,2))/abs(p-p0).mean((1,2))]))
        bad=(a[:,:,1]<.2)|(a[:,:,1]>1)
        violations.append(int(bad.any((1,2,3)).sum()));fractions.append(float(bad.mean()*100))
    vals=np.mean(runs,axis=0)
    rng=np.random.default_rng(20261117);boot=vals[rng.integers(100,size=(10000,100))].mean(axis=1)
    out[model]=dict(mean=vals.mean(axis=0).tolist(),ci=np.quantile(boot,[.025,.975],axis=0).tolist(),violating_cases=violations,violating_percent=fractions)
def shock(r):
    def slope(x):return -(r*x**4/(r*x**4+(1-x)**2))/x
    return .2+.6*minimize_scalar(slope,bounds=(.00001,1),method='bounded').x
result=dict(metrics=['step100 whole-grid saturation MAE','step100 reference 0.23<Sw<0.77 saturation MAE','step100 pressure MAE divided by mean absolute reference change since stored state 0, percent'],reservoirs=len(p),mask_fraction=float(mask.mean()),shock_viscosity_only=shock(1.8),shock_simulator_mobility=shock(2.16),models=out)
(base/'review2_diagnostics.json').write_text(json.dumps(result,indent=2))
print(result)
