"""Full independent evaluation only after all12 validation-selected fits finish."""
from pathlib import Path
import sys,json,time,csv
import numpy as np
import h5py
import torch
from controlled_training import HERE,EXPERIMENTS,TEST,FINE,MODELS,SEEDS,build_model,normalize,file_hash
sys.path.insert(0,str(EXPERIMENTS))
from run_experiments import load_reference,relative_error,bootstrap_mean,timing

def write_csv(path,rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def normalized_input(s,lo,hi,norm):
    b=hi-lo;n=s['permeability'].shape[0]
    return torch.as_tensor(np.stack([
        np.broadcast_to(normalize(s['permeability'],'permeability',norm),(b,n,n)),
        np.broadcast_to(normalize(s['porosity'],'porosity',norm),(b,n,n)),
        normalize(s['pressure'][lo:hi],'pressure',norm),normalize(s['saturation'][lo:hi],'saturation',norm)],axis=1),dtype=torch.float32,device='cuda')

def physical(out,norm):
    return torch.stack([out[...,i,:,:].double()*norm[k]['scale']+norm[k]['mean'] for i,k in enumerate(['pressure','saturation'])],dim=-3)

def failure_mask(errors):
    return (~np.isfinite(errors).all(axis=2)|(errors[:,:,0]>=100)).any(axis=1)

@torch.inference_mode()
def single_step(model,refs,norm):
    metrics=[]
    for s in refs:
        n=len(s['pressure'])-1;steps=[]
        for lo in range(0,n,128):
            hi=min(lo+128,n);x=normalized_input(s,lo,hi,norm)
            pred=physical(model(x),norm).cpu().numpy()
            truth=np.stack([s['pressure'][lo+1:hi+1],s['saturation'][lo+1:hi+1]],axis=1)
            mae=np.abs(pred-truth).mean(axis=(-2,-1));err=relative_error(pred,truth)
            steps.append(np.concatenate([mae,err],axis=1))
        allsteps=np.concatenate(steps)
        assert allsteps.shape==(499,4)
        metrics.append(allsteps.mean(axis=0))
    result=np.asarray(metrics)
    if not np.isfinite(result).all():raise FloatingPointError('Single-step predictions nonfinite; inspect before reporting.')
    return result

@torch.inference_mode()
def rollout(model,refs,norm):
    x=torch.cat([normalized_input(s,0,1,norm) for s in refs],dim=0)
    truth=np.stack([np.stack([s['pressure'][1:101],s['saturation'][1:101]],axis=1) for s in refs])
    preds=[]
    for i in range(100):
        out=model(x);preds.append(physical(out,norm).cpu().numpy());x=torch.cat([x[:,:2],out],dim=1)
    pred=np.stack(preds,axis=1)
    errors=relative_error(pred,truth)
    failures=failure_mask(errors)
    return pred,errors,failures

def conditional_summary(a):
    """Average the three fixed training runs before resampling reservoir units."""
    avg=a.mean(axis=0)
    flat=avg.reshape(100,-1)
    valid=np.isfinite(flat).all(axis=0)
    mean=np.full(flat.shape[1],np.nan);lo=mean.copy();hi=mean.copy()
    if valid.any():mean[valid],lo[valid],hi[valid]=bootstrap_mean(flat[:,valid])
    shape=avg.shape[1:]
    def values(v):
        o=v.reshape(shape).astype(object);o[~np.isfinite(v.reshape(shape))]=None
        return o.tolist()
    return {'mean':values(mean),'ci_low':values(lo),'ci_high':values(hi),'nonfinite':bool((~valid).any()),
        'definition':'Only complete100-realisation cells have finite mean/CI; no individual nonfinite cases are excluded.'}

def evaluate():
    from provenance import verify,identity,digest
    verify()
    provenance=identity()
    evaluation_implementation=digest(Path(__file__))
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=True
    complete=json.loads((HERE/'training_summary.json').read_text())
    assert len(complete)==12 and {(x['model'],x['seed']) for x in complete}=={(m,s) for m in MODELS for s in SEEDS}
    norm=json.loads((HERE/'training_normalization.json').read_text())
    refs_coarse=load_reference(TEST);refs_fine=load_reference(FINE)
    allresults={};rows_coarse=[];rows_fine=[];rows_roll=[]
    summaries={}
    for name in MODELS:
        coarse=[];fine=[];recursive=[];fails=[]
        for seed in SEEDS:
            stem=name.replace('-','_')+f'_seed{seed}'
            result_path=HERE/(stem+'_evaluated.npz')
            cache_meta=HERE/(stem+'_evaluation_provenance.json')
            ckpt_path=HERE/'checkpoints'/(stem+'_best.pt')
            expected=next(x for x in complete if x['model']==name and x['seed']==seed)
            assert file_hash(ckpt_path)==expected['sha256']
            ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
            assert ckpt['normalization']==norm
            expected_cache={'checkpoint_sha256':expected['sha256'],'normalization_sha256':digest(HERE/'training_normalization.json'),
                'evaluator_sha256':evaluation_implementation,'data_sha256':provenance['data_sha256']}
            if result_path.exists():
                assert cache_meta.exists(),'Missing evaluation-cache provenance; refuse stale results.'
                saved=json.loads(cache_meta.read_text())
                assert saved['inputs']==expected_cache,'Changed checkpoint, preprocessing, evaluator or test data.'
                assert saved['result_sha256']==digest(result_path),'Changed evaluation result cache.'
                assert saved['predictions_sha256']==digest(HERE/(stem+'_rollout_predictions.npz')),'Changed raw prediction cache.'
                z=np.load(result_path);c=z['coarse'];f=z['fine'];r=z['rollout'];fail=z['failures']
            else:
                model=build_model(name,seed).cuda().eval();model.load_state_dict(ckpt['model_state_dict'],strict=True)
                began=time.perf_counter()
                c=single_step(model,refs_coarse,norm);print(stem,'coarse full evaluation complete',flush=True)
                f=single_step(model,refs_fine,norm);print(stem,'fine full evaluation complete',flush=True)
                pred,r,fail=rollout(model,refs_coarse,norm)
                np.savez_compressed(HERE/(stem+'_rollout_predictions.npz'),predictions=pred)
                np.savez_compressed(result_path,coarse=c,fine=f,rollout=r,failures=fail)
                cache_meta.write_text(json.dumps({'inputs':expected_cache,'result_sha256':digest(result_path),
                    'predictions_sha256':digest(HERE/(stem+'_rollout_predictions.npz'))},indent=2))
                print(stem,'EVALUATED',f'elapsed={time.perf_counter()-began:.1f}s','coarse mean',c.mean(axis=0),'rollout failures',int(fail.sum()),flush=True)
                del model;torch.cuda.empty_cache()
            coarse.append(c);fine.append(f);recursive.append(r);fails.append(fail)
            for dataset,metrics,rows in [('coarse',c,rows_coarse),('fine',f,rows_fine)]:
                refs=refs_coarse if dataset=='coarse' else refs_fine
                for i,s in enumerate(refs):
                    v=metrics[i]
                    rows.append({'model':name,'seed':seed,'id':s['id'],'transitions':499,'pressure_mae_pa':v[0],
                        'saturation_mae':v[1],'pressure_rel_percent':v[2],'saturation_rel_percent':v[3],'combined_rel_percent':(v[2]+v[3])/2})
            for i,s in enumerate(refs_coarse):
                for step in range(100):
                    rows_roll.append({'model':name,'seed':seed,'id':s['id'],'step':step+1,'pressure_rel_percent':r[i,step,0],
                        'saturation_rel_percent':r[i,step,1],'failed_by100':bool(fail[i])})
        coarse=np.stack(coarse);fine=np.stack(fine);recursive=np.stack(recursive);fails=np.stack(fails)
        allresults[name]={'coarse':coarse,'fine':fine,'rollout':recursive,'failures':fails}
        summary={}
        for dataset,a in [('coarse',coarse),('fine',fine)]:
            v=conditional_summary(a)
            seedmeans=a.mean(axis=1)
            v['seed_mean_metrics']=seedmeans.tolist();v['training_seed_sd']=seedmeans.std(axis=0,ddof=1).tolist()
            combined=a[:,:,2:].mean(axis=2)
            v['combined']=conditional_summary(combined)
            v['combined_seed_means']=combined.mean(axis=1).tolist();v['combined_training_seed_sd']=float(combined.mean(axis=1).std(ddof=1))
            summary[dataset]=v
        summary['rollout']=conditional_summary(recursive)
        summary['rollout']['failure_counts_per_seed']=fails.sum(axis=1).tolist()
        failure_events=~np.isfinite(recursive).all(axis=3)|(recursive[:,:,:,0]>=100)
        failed_by_step=np.maximum.accumulate(failure_events,axis=2)
        summary['rollout']['failure_rate_percent']=conditional_summary(failed_by_step.astype(float)*100)
        summary['rollout']['nonfinite_counts_final_per_seed']=(~np.isfinite(recursive[:,:,-1,:]).all(axis=2)).sum(axis=1).tolist()
        summary['rollout']['final_medians_per_seed']=np.median(recursive[:,:,-1,:],axis=1).tolist()
        summary['rollout']['final_seed_mean_errors']=recursive[:,:,-1,:].mean(axis=1).tolist()
        summary['rollout']['final_training_seed_sd']=recursive[:,:,-1,:].mean(axis=1).std(axis=0,ddof=1).tolist()
        summary['n_params']=next(x['n_params'] for x in complete if x['model']==name)
        summaries[name]=summary
        print('MODEL SUMMARY',name,json.dumps(summary['coarse']),flush=True)
    write_csv(HERE/'coarse_per_seed_realisation.csv',rows_coarse)
    write_csv(HERE/'fine_per_seed_realisation.csv',rows_fine)
    write_csv(HERE/'rollout_per_seed_realisation.csv',rows_roll)
    differences={}
    for dataset in ['coarse','fine']:
        a=allresults['U-FNO-LR'][dataset][:,:,2:].mean(axis=2)
        b=allresults['U-FNO'][dataset][:,:,2:].mean(axis=2)
        delta=(a-b).mean(axis=0)
        m,lo,hi=bootstrap_mean(delta)
        differences[dataset]={'combined_difference_pp':float(m),'ci_low':float(lo),'ci_high':float(hi),
            'seed_differences_pp':(a-b).mean(axis=1).tolist(),'definition':'U-FNO-LR minus U-FNO; equal seed mean within realisation, paired reservoir bootstrap'}
    def finite_json(obj):
        if isinstance(obj,dict):return {k:finite_json(v) for k,v in obj.items()}
        if isinstance(obj,list):return [finite_json(v) for v in obj]
        if isinstance(obj,float) and not np.isfinite(obj):return None
        return obj
    (HERE/'evaluation_summary.json').write_text(json.dumps(finite_json({'models':summaries,'paired_differences':differences}),indent=2,allow_nan=False))
    # Prespecified representative seed42, not the run with best test performance.
    models={}
    for name in MODELS:
        ckpt=torch.load(HERE/'checkpoints'/(name.replace('-','_')+'_seed42_best.pt'),map_location='cpu',weights_only=False)
        model=build_model(name,42).cuda().eval();model.load_state_dict(ckpt['model_state_dict']);models[name]=model
    compatibility_norm={key:{'mean':v['mean'],'std':v['scale']} for key,v in norm.items()}
    timing(models,compatibility_norm,output_dir=HERE)
    print('ALL controlled evaluations and timing complete.',flush=True)

if __name__=='__main__':evaluate()
