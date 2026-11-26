"""Train four architectures on a shared leakage-free matched-budget protocol."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
os.environ['PYTHONIOENCODING']='utf-8'
from pathlib import Path
import sys,json,time,random,argparse,hashlib
import numpy as np
import h5py
import torch
from torch import nn
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
EXPERIMENTS=HERE.parent
CODE=Path(os.environ.get('UNET_FNO_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve()
sys.path.insert(0,str(CODE))
SOURCE=CODE/'reservoir_fno_dataset_20x20/reservoir_fno_20260319_193856_samples_500.h5'
TEST=EXPERIMENTS/'independent_20x20_100.h5'
FINE=CODE/'reservoir_fno_dataset_40x40/reservoir_fno_20260322_175636_samples_100.h5'
MODELS=['FNO','U-FNO','U-FNO-LR','UNet-FNO']
SEEDS=[42,43,44]

class LowResolutionUNet(nn.Module):
    """Two-level BN encoder and interpolation decoder for the U-FNO branch."""
    def __init__(self,width=32):
        super().__init__()
        def conv(cin,cout,stride):
            return nn.Sequential(nn.Conv2d(cin,cout,3,stride=stride,padding=1,bias=False),
                nn.BatchNorm2d(cout),nn.LeakyReLU(.1,inplace=True),nn.Dropout(0.))
        self.enc1=conv(width,width,2)
        self.enc2=nn.Sequential(conv(width,width,2),conv(width,width,1))
        self.dec1=nn.Sequential(nn.Conv2d(width,width,3,padding=1),nn.LeakyReLU(.1,inplace=True))
        self.dec0=nn.Sequential(nn.Conv2d(width*2,width,3,padding=1),nn.LeakyReLU(.1,inplace=True))
        self.output=nn.Conv2d(width*2,width,3,padding=1)
    def forward(self,x):
        e1=self.enc1(x);e2=self.enc2(e1)
        d1=self.dec1(F.interpolate(e2,size=e1.shape[-2:],mode='bilinear',align_corners=False))
        d0=self.dec0(F.interpolate(torch.cat([e1,d1],dim=1),size=x.shape[-2:],mode='bilinear',align_corners=False))
        return self.output(torch.cat([x,d0],dim=1))

def build_model(name,seed):
    from fno import FNO2d
    from ufno import Net2d
    from ablation_study import build_ablation_models
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    if name=='FNO':return FNO2d(8,8,32)
    if name=='UNet-FNO':return build_ablation_models(modes=8,width=32,n_layers=4,base_ch=32)['V1-Full']
    if name in ['U-FNO','U-FNO-LR']:
        model=Net2d(8,8,32)
        if name=='U-FNO-LR':
            for key in ['unet3','unet4','unet5']:setattr(model.conv1,key,LowResolutionUNet(32))
        return model
    raise ValueError(name)

def split_ids(ids):
    ids=sorted(ids);rng=np.random.default_rng(20261117)
    order=rng.permutation(len(ids));cut=int(.8*len(ids))
    return [ids[i] for i in order[:cut]],[ids[i] for i in order[cut:]]

def scale_for(mean,std):
    constant=bool(std<=max(abs(mean)*1e-12,np.finfo(float).tiny))
    return (1. if constant else float(std)),constant

def channel_stats(a):
    a=np.asarray(a,dtype=np.float64);mean=float(a.mean());std=float(a.std())
    scale,constant=scale_for(mean,std)
    return mean,scale,constant

def fit_normalization(path,train_ids):
    result={}
    with h5py.File(path) as f:
        for key in ['permeability','porosity','pressure','saturation']:
            count=0;mean=0.;m2=0.
            for name in train_ids:
                a=f['data'][name][key][:].astype(np.float64).ravel()
                n=a.size;mu=float(a.mean());var=float(a.var());delta=mu-mean
                m2+=n*var+delta*delta*count*n/(count+n)
                mean+=delta*n/(count+n);count+=n
            std=float(np.sqrt(m2/count));scale,constant=scale_for(mean,std)
            result[key]={'mean':mean,'std':std,'scale':scale,'constant':constant,'count':count}
    return result

def temporal_indices(n_samples,n_transitions,per_sample,seed,epoch):
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,20261117]))
    pairs=np.concatenate([np.column_stack([np.full(per_sample,i),rng.choice(n_transitions,per_sample,replace=False)]) for i in range(n_samples)])
    return pairs[rng.permutation(len(pairs))]

def relative_l2(pred,target):
    return ((pred-target).flatten(1).norm(dim=1)/target.flatten(1).norm(dim=1).clamp_min(1e-8)).mean()

def training_step(model,optimizer,x,y):
    optimizer.zero_grad(set_to_none=True)
    pred=model(x);loss=relative_l2(pred,y)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss; no silent skip.')
    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    return float(loss.detach())

def normalize(a,key,norm):
    return (a-norm[key]['mean'])/norm[key]['scale']

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def prepare_data():
    manifest_path=HERE/'split.json'
    if manifest_path.exists():manifest=json.loads(manifest_path.read_text())
    else:
        with h5py.File(SOURCE) as f:ids=sorted(f['data'])
        train,val=split_ids(ids)
        assert len(train)==400 and len(val)==100 and not(set(train)&set(val))
        with h5py.File(TEST) as f:test_ids=sorted(f['data'])
        manifest={'split_seed':20261117,'source':str(SOURCE),'train_ids':train,'validation_ids':val,
            'test_file':str(TEST),'test_ids':test_ids,'test_used_for_selection':False}
        manifest_path.write_text(json.dumps(manifest,indent=2))
    norm_path=HERE/'training_normalization.json'
    if norm_path.exists():norm=json.loads(norm_path.read_text())
    else:
        norm=fit_normalization(SOURCE,manifest['train_ids'])
        norm_path.write_text(json.dumps(norm,indent=2))
    cache=HERE/'source_normalized.pt'
    if cache.exists():data=torch.load(cache,map_location='cpu',weights_only=True)
    else:
        with h5py.File(SOURCE) as f:
            ids=sorted(f['data']);statics=[];dynamics=[]
            for key in ids:
                g=f['data'][key]
                statics.append(np.stack([normalize(g[k][:],k,norm) for k in ['permeability','porosity']]).astype(np.float32))
                dynamics.append(np.stack([normalize(g[k][:].reshape(-1,20,20),k,norm) for k in ['pressure','saturation']],axis=1).astype(np.float32))
        data={'ids':ids,'static':torch.from_numpy(np.stack(statics)),'dynamic':torch.from_numpy(np.stack(dynamics))}
        torch.save(data,cache)
    idmap={key:i for i,key in enumerate(data['ids'])}
    train=np.array([idmap[x] for x in manifest['train_ids']]);val=np.array([idmap[x] for x in manifest['validation_ids']])
    print('Fixed disjoint split and training-only normalization ready.',norm,flush=True)
    return data,train,val,norm

def gather(static,dynamic,ids,times):
    x=torch.cat([static[ids],dynamic[ids,times]],dim=1)
    y=dynamic[ids,times+1]
    return x,y

@torch.inference_mode()
def validation_loss(model,static,dynamic,pairs):
    model.eval();total=0.;count=0
    for lo in range(0,len(pairs),128):
        batch=pairs[lo:lo+128];x,y=gather(static,dynamic,batch[:,0],batch[:,1])
        loss=relative_l2(model(x),y)
        if not torch.isfinite(loss):return float('inf')
        total+=float(loss)*len(batch);count+=len(batch)
    return total/count

def train_all():
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=True
    assert torch.cuda.is_available()
    (HERE/'checkpoints').mkdir(exist_ok=True)
    data,train_ids,val_ids,norm=prepare_data()
    static=data['static'].cuda();dynamic=data['dynamic'].cuda()
    del data
    vt=np.unique(np.linspace(0,498,32,dtype=int))
    val_pairs=torch.tensor(np.array([(i,t) for i in val_ids for t in vt]),dtype=torch.long,device='cuda')
    config={'seeds':SEEDS,'models':MODELS,'max_epochs':100,'batch_size':128,'train_pairs_per_realisation_epoch':16,
        'validation_pairs_per_realisation':32,'validation_transition_indices':vt.tolist(),'optimizer':'AdamW',
        'learning_rate':.001,'weight_decay':.0001,'cosine_min_lr':1e-6,'patience':20,'gradient_clip_norm':1.,
        'loss':'mean per-transition relative L2 on both standardized output channels','precision':'float32',
        'normalization_file':str(HERE/'training_normalization.json'),'normalization_sha256':file_hash(HERE/'training_normalization.json'),
        'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,
        'design_sha256':file_hash(HERE/'design.txt'),'source_hashes':{k:file_hash(CODE/k) for k in ['fno.py','ufno.py','ablation_study.py','data_training.py']}}
    (HERE/'config.json').write_text(json.dumps(config,indent=2))
    summary=[]
    for seed in SEEDS:
        for name in MODELS:
            stem=name.replace('-','_')+f'_seed{seed}'
            best_path=HERE/'checkpoints'/(stem+'_best.pt');last_path=HERE/'checkpoints'/(stem+'_last.pt')
            log_path=HERE/(stem+'_history.json')
            done_path=HERE/(stem+'_completed.json')
            if done_path.exists():
                summary.append(json.loads(done_path.read_text()));print('Already complete',stem,flush=True);continue
            model=build_model(name,seed).cuda()
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=1e-6)
            start_epoch=1;best=float('inf');patience=0;history=[];elapsed_before=0.
            if last_path.exists():
                state=torch.load(last_path,map_location='cpu',weights_only=False)
                model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);schedule.load_state_dict(state['scheduler'])
                for v in opt.state.values():
                    for k,x in v.items():
                        if isinstance(x,torch.Tensor):v[k]=x.cuda()
                start_epoch=state['epoch']+1;best=state['best'];patience=state['patience'];history=state['history'];elapsed_before=state['elapsed_seconds']
                torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
                if patience>=20:start_epoch=101
            began=time.perf_counter()
            for epoch in range(start_epoch,101):
                tick=time.perf_counter();model.train()
                indices=temporal_indices(400,499,16,seed,epoch)
                indices[:,0]=train_ids[indices[:,0]]
                pairs=torch.tensor(indices,dtype=torch.long,device='cuda');total=0.
                for lo in range(0,len(pairs),128):
                    b=pairs[lo:lo+128];x,y=gather(static,dynamic,b[:,0],b[:,1]);total+=training_step(model,opt,x,y)*len(b)
                val=validation_loss(model,static,dynamic,val_pairs)
                if not np.isfinite(val):raise FloatingPointError(f'{stem} validation is nonfinite')
                improved=val<best
                if improved:
                    best=val;patience=0
                    torch.save({'model_state_dict':model.state_dict(),'seed':seed,'model_name':name,'epoch':epoch,'validation_loss':val,
                        'normalization':norm,'config':config},best_path)
                else:patience+=1
                schedule.step();torch.cuda.synchronize()
                record={'epoch':epoch,'train_relative_l2':total/len(pairs),'validation_relative_l2':val,
                    'best_validation_loss':best,'learning_rate_next':schedule.get_last_lr()[0],
                    'epoch_seconds':time.perf_counter()-tick,'training_pairs':len(pairs),'validation_pairs':len(val_pairs)}
                history.append(record);elapsed=elapsed_before+time.perf_counter()-began
                log_path.write_text(json.dumps(history,indent=2))
                torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':schedule.state_dict(),
                    'epoch':epoch,'best':best,'patience':patience,'history':history,'elapsed_seconds':elapsed,
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},last_path)
                if epoch==1 or epoch%10==0 or patience>=20:
                    print(f'{stem} epoch {epoch}/100 train={record["train_relative_l2"]:.6g} val={val:.6g} best={best:.6g} elapsed={elapsed:.1f}s',flush=True)
                if patience>=20:break
            chosen=torch.load(best_path,map_location='cpu',weights_only=False)
            done={'model':name,'seed':seed,'n_params':sum(p.numel() for p in model.parameters()),'epochs_completed':len(history),
                'best_epoch':chosen['epoch'],'best_validation_loss':chosen['validation_loss'],
                'elapsed_seconds':elapsed_before+time.perf_counter()-began,'checkpoint':str(best_path),'sha256':file_hash(best_path)}
            done_path.write_text(json.dumps(done,indent=2));summary.append(done)
            print('COMPLETE',json.dumps(done),flush=True)
            del model,opt,schedule;torch.cuda.empty_cache()
    assert len(summary)==12
    (HERE/'training_summary.json').write_text(json.dumps(summary,indent=2))
    print('ALL TWELVE TRAINING RUNS COMPLETE. No test evaluation used for selection.',flush=True)

if __name__=='__main__':
    import sys
    from provenance import guarded_train
    guarded_train(train_all)
