"""Independent evaluation of six archived ablation checkpoints; no training."""
from pathlib import Path
import sys, json, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_experiments as r
import numpy as np
import torch
import h5py
from ablation_study import build_ablation_models

OUT = Path(__file__).resolve().parent
FILES = ['best_v1-full.pth', 'best_v2-w_o_unet.pth', 'best_v3-w_o_residual.pth',
         'best_v4-w_o_bias.pth', 'best_v5-depth1.pth', 'best_v6-independent_unet.pth']

def stats(a):
    a = np.asarray(a, dtype=float)
    if not np.isfinite(a).all():
        return {'mean': None, 'ci_low': None, 'ci_high': None,
                'nonfinite_count': int((~np.isfinite(a)).sum())}
    mean, low, high = r.bootstrap_mean(a)
    return dict(mean=float(mean), ci_low=float(low), ci_high=float(high))

@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    refs = r.load_reference(r.NEW)
    assert len(refs) == 100 and all(len(s['pressure']) == 500 for s in refs)
    new_hashes = [r.field_hash(s['permeability']) for s in refs]
    with h5py.File(r.SOURCE) as f:
        old_hashes = {r.field_hash(g['permeability'][:]) for g in f['data'].values()}
    assert len(set(new_hashes)) == 100 and not set(new_hashes) & old_hashes
    norm = json.loads((r.HERE/'normalization.json').read_text())
    manifest = {'test_file': str(r.NEW), 'test_sha256': r.file_hash(r.NEW),
                'normalization': norm, 'normalization_basis': 'Reconstructed from all original 500 cases, population std + 1e-10; independent test excluded. Historical exact preprocessing was not saved.',
                'n_independent_reservoirs': 100, 'one_step_transitions_per_case': 499,
                'rollout_steps': 100, 'bootstrap_resamples': 10000,
                'failure_definition': 'Any nonfinite predicted state, or pressure relative error >=100% at any rollout step.',
                'limitation': 'One archived checkpoint per variant; historical training split and seeds unavailable. CI measures test-reservoir variability only.',
                'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
                'source_hashes': {n:r.file_hash(r.CODE/n) for n in ['ablation_study.py','data_training.py']},
                'checkpoints': {}}
    models = build_ablation_models(modes=8,width=32,n_layers=4,base_ch=32)
    summary = {}; coarse_rows = []; rollout_rows = []; case_arrays = {}
    k = np.stack([s['permeability'] for s in refs]); phi = np.stack([s['porosity'] for s in refs])
    truth = np.stack([np.stack([s['pressure'][1:101],s['saturation'][1:101]],axis=1) for s in refs])
    for (name,model), filename in zip(models.items(), FILES):
        start = time.perf_counter()
        path = r.CODE/'ablation/checkpoints'/filename
        before = r.file_hash(path)
        checkpoint = torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        manifest['checkpoints'][name] = {'path':str(path),'sha256':before,
            'saved_epochs':len(checkpoint['train_losses']), 'parameters':sum(p.numel() for p in model.parameters())}
        model.cuda().eval(); metrics = []
        for index,s in enumerate(refs):
            pieces=[]
            for lo in range(0,499,128):
                hi=min(lo+128,499); b=hi-lo
                x=r.input_tensor(np.broadcast_to(s['permeability'],(b,20,20)),np.broadcast_to(s['porosity'],(b,20,20)),s['pressure'][lo:hi],s['saturation'][lo:hi],norm)
                pred=r.to_physical(model(x),norm).cpu().numpy()
                target=np.stack([s['pressure'][lo+1:hi+1],s['saturation'][lo+1:hi+1]],axis=1)
                pieces.append(np.concatenate([np.abs(pred-target).mean((-2,-1)),r.relative_error(pred,target)],axis=1))
            values=np.concatenate(pieces).mean(axis=0); metrics.append(values)
            coarse_rows.append(dict(model=name,id=s['id'],pressure_mae_pa=values[0],saturation_mae=values[1],pressure_rel_percent=values[2],saturation_rel_percent=values[3],combined_rel_percent=values[2:].mean()))
            if (index+1)%25==0: print(name,'single step',index+1,'/100',round(time.perf_counter()-start,1),'s',flush=True)
        metrics=np.asarray(metrics); case_arrays[name]=metrics
        x=r.input_tensor(k,phi,np.stack([s['pressure'][0] for s in refs]),np.stack([s['saturation'][0] for s in refs]),norm)
        predictions=[]
        for step in range(100):
            out=model(x); predictions.append(r.to_physical(out,norm).cpu().numpy()); x=torch.cat([x[:,:2],out],dim=1)
        predictions=np.stack(predictions,axis=1); errors=r.relative_error(predictions,truth)
        bad_states=~np.isfinite(predictions).all(axis=(2,3,4))
        failures=(bad_states | (errors[:,:,0]>=100)).any(axis=1)
        slug=name[:2]
        np.savez_compressed(OUT/(slug+'_results.npz'),one_step_metrics=metrics,rollout_predictions=predictions,rollout_errors=errors,failures=failures,ids=np.array([s['id'] for s in refs]))
        for i,s in enumerate(refs):
            for t in range(100):
                rollout_rows.append(dict(model=name,id=s['id'],step=t+1,pressure_rel_percent=errors[i,t,0],saturation_rel_percent=errors[i,t,1],nonfinite_state=bool(bad_states[i,t])))
        summary[name]={'single_step':{key:stats(metrics[:,j]) for j,key in enumerate(['pressure_mae_pa','saturation_mae','pressure_rel_percent','saturation_rel_percent'])},
            'combined_rel_percent':stats(metrics[:,2:].mean(axis=1)),
            'rollout_step100_pressure_rel_percent':stats(errors[:,-1,0]),
            'rollout_step100_saturation_rel_percent':stats(errors[:,-1,1]),
            'rollout_failures':int(failures.sum()),'nonfinite_trajectories':int(bad_states.any(axis=1).sum()),
            'elapsed_seconds':time.perf_counter()-start}
        assert before == r.file_hash(path)
        print(name,json.dumps(summary[name]),flush=True)
        model.cpu(); torch.cuda.empty_cache()
        (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    summary['paired_combined_difference_from_V1_pp']={name:stats(a[:,2:].mean(axis=1)-case_arrays['V1-Full'][:,2:].mean(axis=1)) for name,a in case_arrays.items() if name!='V1-Full'}
    r.write_csv(OUT/'single_step_per_reservoir.csv',coarse_rows)
    r.write_csv(OUT/'rollout_per_reservoir_step.csv',rollout_rows)
    assert len(coarse_rows)==600 and len(rollout_rows)==60000
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print('COMPLETE: 600 reservoir summaries and 60000 rollout-step rows.',flush=True)

if __name__=='__main__': main()
