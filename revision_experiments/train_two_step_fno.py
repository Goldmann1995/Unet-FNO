"""Two-step FNO comparison, using the archived one-step validation criterion."""
from pathlib import Path
import sys
import json
import time
import csv
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'low_resolution_baseline'))
from controlled_training import (HERE, SEEDS, prepare_data, build_model, temporal_indices,
    gather, relative_l2, validation_loss, file_hash, TEST)
from evaluate_controlled import rollout
from run_experiments import load_reference

OUT=ROOT/'fno_two_step'
OUT.mkdir(exist_ok=True)


def train():
    data,train_ids,val_ids,norm=prepare_data()
    static=data['static'].cuda()
    dynamic=data['dynamic'].cuda()
    del data
    validation=torch.tensor([(i,t) for i in val_ids for t in np.unique(np.linspace(0,498,32,dtype=int))],device='cuda')
    for seed in SEEDS:
        net=build_model('FNO',seed).cuda()
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=1e-6)
        best=float('inf');history=[];best_epoch=0
        start=time.perf_counter()
        for epoch in range(1,101):
            net.train()
            indices=temporal_indices(400,498,16,seed,epoch)
            indices[:,0]=train_ids[indices[:,0]]
            pairs=torch.as_tensor(indices,device='cuda')
            losses=[]
            for ids,t in (b.T for b in pairs.split(128)):
                x,y=gather(static,dynamic,ids,t)
                optimizer.zero_grad(set_to_none=True)
                first=net(x)
                second=net(torch.cat((x[:,:2],first),1))
                loss=(relative_l2(first,y)+relative_l2(second,dynamic[ids,t+2]))/2
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                optimizer.step()
                losses.append(float(loss.detach()))
            val=validation_loss(net,static,dynamic,validation)
            assert np.isfinite(val)
            if val<best:
                best=val;best_epoch=epoch
                torch.save(net.state_dict(),OUT/f'seed{seed}.pt')
            schedule.step()
            history.append([epoch,np.mean(losses),val])
            if epoch%20==0:print(seed,epoch,flush=True)
            if epoch-best_epoch>=20:break
        torch.cuda.synchronize()
        info=dict(seed=seed,best_epoch=best_epoch,elapsed_seconds=time.perf_counter()-start,history=history,
                  checkpoint_sha256=file_hash(OUT/f'seed{seed}.pt'))
        (OUT/f'seed{seed}.json').write_text(json.dumps(info,indent=2))
    return norm


def evaluate(norm):
    refs=load_reference(TEST,100)
    rows=[]
    for seed in SEEDS:
        net=build_model('FNO',seed).cuda().eval()
        net.load_state_dict(torch.load(OUT/f'seed{seed}.pt',weights_only=True))
        _,err,fail=rollout(net,refs,norm)
        assert err.shape==(100,100,2)
        for case in range(100):
            for t in range(100):
                rows.append([seed,case,t+1,*err[case,t],int(fail[case])])
        print(seed,err.mean(0)[-1],fail.sum(),flush=True)
    with (OUT/'rollout.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','case','step','pressure_percent','saturation_percent','trajectory_failed']);w.writerows(rows)


if __name__=='__main__':
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=True
    norm=train()
    torch.cuda.empty_cache()
    evaluate(norm)
