"""Rebuild the archived loss figure and verify the plotted histories."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'low_resolution_baseline'))
from controlled_training import CODE
from compare_models import plot_loss_curves

def audit():
    trainers={}
    records={}
    for name in ('FNO','U-FNO','UNet-FNO'):
        filename='best_'+name.lower().replace('-','_')+'.pth'
        state=torch.load(CODE/'experiments/checkpoints'/filename,map_location='cpu',weights_only=False)
        losses=np.asarray(state['test_losses'])
        trainers[name]=SimpleNamespace(train_losses=state['train_losses'],test_losses=losses)
        records[name]=dict(entries=len(losses),min_epoch=int(losses.argmin()+1),
            test_loss_at_epochs_71_81_91=losses[[70,80,90]].tolist(),
            logged_optimizer_lr=state['optimizer_state_dict']['param_groups'][0]['lr'])
    with TemporaryDirectory() as tmp:
        plot_loss_curves(trainers,tmp)
        def rgb(path):
            with Image.open(path) as im:return np.array(im.convert('RGB'))
        records['identical_image_pixels']=bool(np.array_equal(rgb(Path(tmp)/'loss_curves.png'),
            rgb(ROOT.parent/'fig/figures/loss_curves.png')))
    assert records['identical_image_pixels']
    (ROOT/'v5_paired/archived_curve_audit.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':
    audit()
