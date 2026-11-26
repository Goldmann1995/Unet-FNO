"""Frozen-checkpoint, independent-data reviewer experiments. No training."""
import os
for _k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[_k]='1'
os.environ['PYTHONIOENCODING']='utf-8'
from pathlib import Path
import sys,time,json,hashlib,platform,subprocess,contextlib,io,argparse,csv
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import h5py
import torch
import scipy

HERE=Path(__file__).resolve().parent
CODE=Path(os.environ.get('UNET_FNO_ROOT', Path(__file__).resolve().parents[1])).expanduser().resolve()
sys.path.insert(0,str(CODE))
SOURCE=CODE/'reservoir_fno_dataset_20x20/reservoir_fno_20260319_193856_samples_500.h5'
FINE=CODE/'reservoir_fno_dataset_40x40/reservoir_fno_20260322_175636_samples_100.h5'
NEW=HERE/'independent_20x20_100.h5'
SEED=20261117

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def field_hash(a):
    a=np.asarray(a,dtype='<f8')
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()

def relative_error(pred,truth):
    return np.abs(pred-truth).mean(axis=(-2,-1))/(np.abs(truth).mean(axis=(-2,-1))+1e-8)*100

def bootstrap_mean(data,reps=10000):
    data=np.asarray(data,dtype=float)
    if not np.isfinite(data).all():
        raise ValueError('Nonfinite data require explicit failure reporting, not omission.')
    rng=np.random.default_rng(SEED)
    counts=rng.multinomial(len(data),np.full(len(data),1/len(data)),size=reps)
    boot=counts@data/len(data)
    lo,hi=np.quantile(boot,[.025,.975],axis=0)
    return data.mean(axis=0),lo,hi

def write_csv(path,rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def generate_one(i):
    from data_generation import ReservoirDatasetCreator
    torch.set_num_threads(1)
    path=HERE/'simulation_cache'/f'sample_{i:03d}.npz'
    if path.exists():return i,str(path)
    seed=2026111700+i
    pattern=['gaussian','uniform','lognormal'][i%3]
    with contextlib.redirect_stdout(io.StringIO()):
        creator=ReservoirDatasetCreator(output_dir=str(HERE/'generator_work'),nx=20,ny=20)
        np.random.seed(seed)
        s=creator.generate_single_sample(i,pattern,nt=500,dt=10800.)
    np.savez_compressed(path,permeability=s['permeability'].reshape(20,20),
        porosity=s['porosity'].reshape(20,20),pressure=s['pressure'],saturation=s['saturation'],
        time_steps=s['time_steps'],seed=seed,pattern=pattern,simulation_seconds=s['sim_time'])
    return i,str(path)

def generate():
    (HERE/'simulation_cache').mkdir(exist_ok=True)
    if not NEW.exists():
        started=time.perf_counter()
        with ProcessPoolExecutor(max_workers=8) as pool:
            futures=[pool.submit(generate_one,i) for i in range(100)]
            for completed,fut in enumerate(as_completed(futures),1):
                i,p=fut.result()
                if completed%10==0:print(f'Generated {completed}/100; elapsed {time.perf_counter()-started:.1f}s',flush=True)
        with h5py.File(NEW,'w') as f:
            f.attrs['protocol_seed']=SEED;f.attrs['dt_seconds']=10800.;f.attrs['domain_m']=[100,100,1]
            dg=f.create_group('data')
            for i in range(100):
                z=np.load(HERE/'simulation_cache'/f'sample_{i:03d}.npz')
                g=dg.create_group(f'sample_{i:03d}')
                for key in ['permeability','porosity','pressure','saturation','time_steps']:
                    g.create_dataset(key,data=z[key],compression='gzip',compression_opts=1)
                g.attrs.update(sample_id=i,seed=int(z['seed']),pattern=str(z['pattern']),simulation_seconds=float(z['simulation_seconds']))
    with h5py.File(SOURCE) as f:
        known={field_hash(g['permeability'][:]) for g in f['data'].values()}
    manifest=[]
    for path in [NEW,FINE]:
        seen=set()
        with h5py.File(path) as f:
            assert len(f['data'])==100
            for key,g in f['data'].items():
                h=field_hash(g['permeability'][:]);assert h not in known;assert h not in seen;seen.add(h)
                manifest.append({'dataset':path.name,'id':key,'pattern':str(g.attrs['pattern']),
                    'seed':int(g.attrs.get('seed',-1)),'permeability_sha256':h,'source500_hash_match':False})
    write_csv(HERE/'test_manifest.csv',manifest)
    print('Verified 100 new coarse and all 100 fine fields against all 500 original fields.',flush=True)

def normalization():
    target=HERE/'normalization.json'
    if target.exists():return json.loads(target.read_text())
    result={}
    with h5py.File(SOURCE) as f:
        for key in ['permeability','porosity','pressure','saturation']:
            count=0;mean=0.;m2=0.
            for g in f['data'].values():
                a=g[key][:].astype(np.float64).ravel();n=a.size;mu=float(a.mean());v=float(a.var());delta=mu-mean
                m2+=n*v+delta*delta*count*n/(count+n)
                mean+=delta*n/(count+n);count+=n
            result[key]={'mean':mean,'std':float(np.sqrt(m2/count))+1e-10,'count':count}
    target.write_text(json.dumps(result,indent=2));return result

def load_models():
    from fno import FNO2d
    from ufno import Net2d
    from ablation_study import build_ablation_models
    models={'FNO':FNO2d(8,8,32),'U-FNO':Net2d(8,8,32),'UNet-FNO':build_ablation_models(modes=8,width=32,n_layers=4,base_ch=32)['V1-Full']}
    checkpoints={}
    for name,model in models.items():
        p=CODE/'checkpoints'/('best_'+name.lower().replace('-','_')+'.pth')
        checkpoint=torch.load(p,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        model.to('cuda').eval()
        checkpoints[name]={'file':str(p),'sha256':file_hash(p),'n_params':sum(p.numel() for p in model.parameters()),'saved_training_epochs':len(checkpoint.get('train_losses',[]))}
    return models,checkpoints

def tensor(a):return torch.as_tensor(np.asarray(a),dtype=torch.float32,device='cuda')

def input_tensor(k,phi,p,s,norm):
    arrays=[(a-norm[key]['mean'])/norm[key]['std'] for a,key in zip([k,phi,p,s],['permeability','porosity','pressure','saturation'])]
    return tensor(np.stack(arrays,axis=-3))

def to_physical(out,norm):
    out=out.to(torch.float64)
    return torch.stack([out[...,i,:,:]*norm[key]['std']+norm[key]['mean'] for i,key in enumerate(['pressure','saturation'])],dim=-3)

def load_reference(path,steps=None):
    data=[]
    with h5py.File(path) as f:
        for key,g in f['data'].items():
            n=g['permeability'].shape[0]
            count=g['pressure'].shape[0] if steps is None else steps+1
            data.append({'id':key,'permeability':g['permeability'][:],'porosity':g['porosity'][:],
                'pressure':g['pressure'][:count].reshape(-1,n,n),'saturation':g['saturation'][:count].reshape(-1,n,n),
                'time_steps':g['time_steps'][:count]})
    return data

@torch.inference_mode()
def evaluate_rollout(models,norm):
    references=load_reference(NEW,100)
    k=np.stack([s['permeability'] for s in references]);phi=np.stack([s['porosity'] for s in references])
    initp=np.stack([s['pressure'][0] for s in references]);inits=np.stack([s['saturation'][0] for s in references])
    truth=np.stack([np.stack([s['pressure'],s['saturation']],axis=1) for s in references])
    summary={};allrows=[];cirows=[]
    for name,model in models.items():
        start=time.perf_counter();x=input_tensor(k,phi,initp,inits,norm);preds=[]
        for step in range(100):
            out=model(x);preds.append(to_physical(out,norm).cpu().numpy())
            x=torch.cat([x[:,:2],out],dim=1)
        predictions=np.stack(preds,axis=1)
        errors=relative_error(predictions,truth[:,1:])
        assert errors.shape==(100,100,2)
        np.savez_compressed(HERE/(name.replace('-','_')+'_rollout.npz'),predictions=predictions,reference=truth[:,1:],relative_errors=errors)
        finite=np.isfinite(errors).all()
        if not finite:raise RuntimeError(f'{name} nonfinite predictions saved; must report failures explicitly.')
        m,lo,hi=bootstrap_mean(errors.reshape(100,-1));m=m.reshape(100,2);lo=lo.reshape(100,2);hi=hi.reshape(100,2)
        combined=bootstrap_mean(errors.mean(axis=2))
        for i in range(100):
            for j,s in enumerate(references):
                allrows.append({'model':name,'sample_id':s['id'],'step':i+1,'pressure_rel_percent':errors[j,i,0],'saturation_rel_percent':errors[j,i,1]})
            cirows.append({'model':name,'step':i+1,'n':100,'pressure_mean':m[i,0],'pressure_ci_low':lo[i,0],'pressure_ci_high':hi[i,0],
                'saturation_mean':m[i,1],'saturation_ci_low':lo[i,1],'saturation_ci_high':hi[i,1],
                'combined_mean':combined[0][i],'combined_ci_low':combined[1][i],'combined_ci_high':combined[2][i]})
        crossings=np.argmax(errors[:,:,0]>=100,axis=1)+1
        crossings[~(errors[:,:,0]>=100).any(axis=1)]=0
        summary[name]={'mean':m.tolist(),'ci_low':lo.tolist(),'ci_high':hi.tolist(),'combined_mean':combined[0].tolist(),
            'combined_ci_low':combined[1].tolist(),'combined_ci_high':combined[2].tolist(),
            'n_pressure_failures_by100':int((crossings>0).sum()),'first_pressure_failure_steps':crossings.tolist(),
            'mean_pressure_first_ge100':next((i+1 for i,x in enumerate(m[:,0]) if x>=100),None)}
        print(name,'rollout final',m[-1], '95% CI',lo[-1],hi[-1],f'elapsed {time.perf_counter()-start:.1f}s',flush=True)
    write_csv(HERE/'rollout_per_realisation.csv',allrows);write_csv(HERE/'rollout_mean_ci.csv',cirows)
    (HERE/'rollout_summary.json').write_text(json.dumps(summary,indent=2))
    return summary

@torch.inference_mode()
def evaluate_fine(models,norm):
    refs=load_reference(FINE)
    summary={};rows=[]
    for name,model in models.items():
        started=time.perf_counter();sample_metrics=[]
        for index,s in enumerate(refs):
            p=s['pressure'];sw=s['saturation'];n=len(p)-1
            metrics=[]
            for lo in range(0,n,128):
                hi=min(lo+128,n);b=hi-lo
                x=input_tensor(np.broadcast_to(s['permeability'],(b,40,40)),np.broadcast_to(s['porosity'],(b,40,40)),p[lo:hi],sw[lo:hi],norm)
                pred=to_physical(model(x),norm).cpu().numpy()
                truth=np.stack([p[lo+1:hi+1],sw[lo+1:hi+1]],axis=1)
                mae=np.abs(pred-truth).mean(axis=(-2,-1));err=relative_error(pred,truth)
                metrics.append(np.concatenate([mae,err],axis=1))
            means=np.concatenate(metrics).mean(axis=0)
            sample_metrics.append(means)
            rows.append({'model':name,'id':s['id'],'transitions':n,'pressure_mae_pa':means[0],'saturation_mae':means[1],
                'pressure_rel_percent':means[2],'saturation_rel_percent':means[3],'combined_rel_percent':(means[2]+means[3])/2})
            if (index+1)%25==0:print(f'{name} fine-grid {index+1}/100; {time.perf_counter()-started:.1f}s',flush=True)
        metrics=np.asarray(sample_metrics);m,lo,hi=bootstrap_mean(metrics)
        c=bootstrap_mean(metrics[:,2:].mean(axis=1))
        summary[name]={'mean':m.tolist(),'ci_low':lo.tolist(),'ci_high':hi.tolist(),'combined_mean':float(c[0]),'combined_ci_low':float(c[1]),'combined_ci_high':float(c[2]),'n_realisations':100,'transitions_per_realisation':499}
        print(name,'fine-grid metrics',summary[name],flush=True)
    write_csv(HERE/'fine_grid_per_realisation.csv',rows)
    (HERE/'fine_grid_summary.json').write_text(json.dumps(summary,indent=2));return summary

def distribution(values):
    a=np.asarray(values,dtype=float)
    return {'median':float(np.median(a)),'q1':float(np.quantile(a,.25)),'q3':float(np.quantile(a,.75)),'mean':float(a.mean()),'n':len(a),'values':a.tolist()}

@torch.inference_mode()
def timing(models,norm,output_dir=None):
    refs=load_reference(NEW,100)[:10]
    result={};steps=100
    # CPU counterpart advances from precisely the first stored simulator state.
    from data_generation import ReservoirGrid,ReservoirProperties,ReservoirSimulator
    cpu_times=[]
    for s in refs:
        with contextlib.redirect_stdout(io.StringIO()):
            grid=ReservoirGrid(nx=20,ny=20,total_length=(100,100,1));grid.build_grid()
            sim=ReservoirSimulator(grid,ReservoirProperties());sim.set_properties(s['permeability'].ravel());sim.set_well_conditions();sim.compute_geometric_transmissibility()
        for i,c in enumerate(grid.celllist):
            c.press=float(s['pressure'][0].ravel()[i]);c.Sw=float(s['saturation'][0].ravel()[i]);sim.presslast[i]=c.press
        pstore=[];sstore=[];t=time.perf_counter()
        for j in range(steps):
            sim.compute_mobility();sim.compute_total_transmissibility();p=sim.solve_pressure_implicit(10800.)
            for i,c in enumerate(grid.celllist):c.press=p[i].item()
            sim.update_saturation(10800.);sim.presslast=p.copy()
            pstore.append(np.array([c.press for c in grid.celllist]));sstore.append(np.array([c.Sw for c in grid.celllist]))
        cpu_times.append((time.perf_counter()-t)*1000)
        assert np.allclose(np.asarray(pstore).reshape(100,20,20),s['pressure'][1:101],rtol=1e-5,atol=2.)
        assert np.allclose(np.asarray(sstore).reshape(100,20,20),s['saturation'][1:101],rtol=1e-5,atol=1e-6)
    result['IMPES_100steps_ms']=distribution(cpu_times)
    for name,model in models.items():
        local={}
        for batch in (1,128):
            x=torch.zeros(batch,4,20,20,device='cuda')
            for _ in range(20):model(x)
            torch.cuda.synchronize();values=[]
            for _ in range(100):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();model(x);end.record();end.synchronize();values.append(start.elapsed_time(end))
            local['step_batch'+str(batch)+'_ms']=distribution(values)
        # Warm up the sequence path separately; only ten measured trajectories.
        s=refs[0];x=input_tensor(s['permeability'],s['porosity'],s['pressure'][0],s['saturation'][0],norm).unsqueeze(0)
        for _ in range(20):out=model(x);x=torch.cat([x[:,:2],out],dim=1)
        torch.cuda.synchronize();times=[]
        for s in refs:
            torch.cuda.synchronize();t=time.perf_counter()
            x=input_tensor(s['permeability'],s['porosity'],s['pressure'][0],s['saturation'][0],norm).unsqueeze(0)
            outputs=[]
            for _ in range(steps):
                out=model(x);outputs.append(out);x=torch.cat([x[:,:2],out],dim=1)
            physical=to_physical(torch.stack(outputs,dim=1),norm).cpu().numpy()
            torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
            assert physical.shape==(1,100,2,20,20)
        local['trajectory_100steps_ms']=distribution(times)
        local['paired_speedup']=distribution(np.asarray(cpu_times)/times)
        result[name]=local
        print(name,'timing trajectory',local['trajectory_100steps_ms']['median'],'speedup',local['paired_speedup']['median'],flush=True)
    destination=HERE if output_dir is None else Path(output_dir)
    (destination/'timing.json').write_text(json.dumps(result,indent=2));return result

def environment(checkpoints):
    cpu=subprocess.check_output(['powershell','-NoProfile','-Command','Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json'],text=True).strip()
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
    manifest={'cpu':json.loads(cpu),'gpu':gpu,'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
        'cudnn':torch.backends.cudnn.version(),'numpy':np.__version__,'scipy':scipy.__version__,'h5py':h5py.__version__,
        'cpu_threads':torch.get_num_threads(),'cudnn_benchmark':torch.backends.cudnn.benchmark,'platform':platform.platform(),
        'checkpoints':checkpoints,'original_source_file':str(SOURCE),'fine_grid_file':str(FINE),'seed':SEED,
        'bootstrap_resamples':10000,'new_data_file':str(NEW),'normalization_source':'entire original source file, reproducing archived checkpoint preprocessing; independent test data excluded'}
    for filename in ['data_generation.py','data_training.py','fno.py','ufno.py','ablation_study.py']:
        manifest.setdefault('source_sha256',{})[filename]=file_hash(CODE/filename)
    (HERE/'environment.json').write_text(json.dumps(manifest,indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','generate','rollout','fine','timing'],default='all');args=parser.parse_args()
    torch.set_num_threads(1)
    if args.stage in ['all','generate']:generate()
    if args.stage=='generate':return
    assert torch.cuda.is_available(),'CUDA required for the registered timing protocol'
    torch.backends.cudnn.benchmark=True
    norm=normalization();models,checkpoints=load_models();environment(checkpoints)
    if args.stage in ['all','rollout']:evaluate_rollout(models,norm)
    if args.stage in ['all','fine']:evaluate_fine(models,norm)
    if args.stage in ['all','timing']:timing(models,norm)
    print('Requested experiment stages complete.',flush=True)

if __name__=='__main__':main()
