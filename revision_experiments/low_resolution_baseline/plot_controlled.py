"""Controlled four-model evidence, all3 seeds and100 test reservoir units.

Accuracy figure: new baseline's mean accuracy and paired reservoir heterogeneity.
Rollout figure: pressure error magnitude and operational failure incidence.
Training figure: every validation trajectory, no averaging of survivors.
Python quantitative grids, 178mm width, editable >=5pt PDF text, no excluded runs.
"""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    import warnings

    def require_matplotlib_panel_alignment(*args, **kwargs):
        warnings.warn("Optional audit_panel_alignment tool unavailable; figure alignment was not audited.", RuntimeWarning)

HERE=Path(__file__).resolve().parent
OUT=HERE.parent.parent/'fig/controlled'
OUT.mkdir(parents=True,exist_ok=True)
MODELS=['FNO','U-FNO','U-FNO-LR','UNet-FNO'];SEEDS=[42,43,44]
COLORS=['#4477AA','#CC6677','#AA7733','#228833']
summary=json.loads((HERE/'evaluation_summary.json').read_text())['models']
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'xtick.labelsize':8,
 'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})

def save(fig,name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=HERE/(name+'_alignment.json'),strict=True)
    fig.savefig(OUT/(name+'.pdf'));fig.savefig(OUT/(name+'.svg'));fig.savefig(OUT/(name+'.png'),dpi=600)
    plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(178/25.4,85/25.4))
fig.subplots_adjust(left=.10,right=.98,bottom=.23,top=.86,wspace=.40)
for i,name in enumerate(MODELS):
    means=np.array(summary[name]['coarse']['combined_seed_means'])
    axes[0].bar(i,means.mean(),width=.6,color=COLORS[i],alpha=.75)
    axes[0].errorbar(i,means.mean(),yerr=means.std(ddof=1),fmt='none',ecolor='black',capsize=3,lw=.8)
    axes[0].scatter(i+np.array([-.12,0,.12]),means,s=12,c='black',zorder=3)
axes[0].set_xticks(range(4),['FNO','U-FNO','U-FNO-\nLR','UNet-\nFNO'])
axes[0].set_ylabel('Combined relative error (%)')
axes[0].set_title('(a) Independent single-step accuracy',loc='left',fontsize=8.5,pad=10)
axes[0].grid(axis='y',alpha=.15)
def field_combined(name):
    a=[np.load(HERE/(name.replace('-','_')+f'_seed{s}_evaluated.npz'))['coarse'][:,2:].mean(axis=1) for s in SEEDS]
    return np.mean(a,axis=0)
x=field_combined('U-FNO');y=field_combined('U-FNO-LR')
assert len(x)==len(y)==100 and (x>0).all() and (y>0).all()
lo=min(x.min(),y.min())*.8;hi=max(x.max(),y.max())*1.2
axes[1].scatter(x,y,s=10,color=COLORS[2],alpha=.6,edgecolors='none')
axes[1].plot([lo,hi],[lo,hi],ls='--',color='gray',lw=.8)
axes[1].set_xscale('log');axes[1].set_yscale('log');axes[1].set_xlim(lo,hi);axes[1].set_ylim(lo,hi)
axes[1].set_xlabel('U-FNO error (%)');axes[1].set_ylabel('U-FNO-LR error (%)')
axes[1].set_title('(b) Paired reservoir comparison',loc='left',fontsize=8.5,pad=10)
save(fig,'accuracy')

fig,axes=plt.subplots(1,2,figsize=(178/25.4,80/25.4))
fig.subplots_adjust(left=.105,right=.98,bottom=.22,top=.78,wspace=.37)
steps=np.arange(1,101)
for name,color in zip(MODELS,COLORS):
    r=summary[name]['rollout'];m=np.asarray(r['mean'],dtype=float)[:,0]
    lo=np.asarray(r['ci_low'],dtype=float)[:,0];hi=np.asarray(r['ci_high'],dtype=float)[:,0]
    # A nonfinite cell has no mean/CI; never estimate it after excluding runs.
    assert np.all(m[np.isfinite(m)]>0) and np.all(lo[np.isfinite(lo)]>0)
    axes[0].plot(steps,m,label=name,color=color,lw=1.1)
    axes[0].fill_between(steps,lo,hi,color=color,alpha=.15,linewidth=0)
    rate=r['failure_rate_percent'];p=np.asarray(rate['mean'],dtype=float)
    axes[1].plot(steps,p,label=name,color=color,lw=1.1)
    axes[1].fill_between(steps,rate['ci_low'],rate['ci_high'],color=color,alpha=.15,linewidth=0)
axes[0].set_yscale('log');axes[0].set_ylabel('Mean pressure error (%)')
axes[1].set_ylabel('Failed by step (%)');axes[1].set_ylim(-2,102)
for i,ax in enumerate(axes):
    ax.set_xlim(1,100);ax.set_xticks([1,25,50,75,100]);ax.set_xlabel('Autoregressive step')
    ax.set_title(['(a) Error magnitude','(b) Failure incidence'][i],loc='left',fontsize=9,pad=10)
    ax.grid(axis='y',alpha=.15)
fig.legend(*axes[0].get_legend_handles_labels(),ncol=4,loc='upper center',bbox_to_anchor=(.54,.99),frameon=False,columnspacing=1.)
save(fig,'rollout')

fig,ax=plt.subplots(figsize=(178/25.4,70/25.4))
fig.subplots_adjust(left=.10,right=.98,bottom=.22,top=.80)
for name,color in zip(MODELS,COLORS):
    for seed in SEEDS:
        history=json.loads((HERE/(name.replace('-','_')+f'_seed{seed}_history.json')).read_text())
        values=[r['validation_relative_l2'] for r in history]
        assert np.isfinite(values).all() and (np.array(values)>0).all()
        ax.plot([r['epoch'] for r in history],values,color=color,alpha=.55,lw=.8,label=name if seed==42 else None)
ax.set_yscale('log');ax.set_xlabel('Epoch');ax.set_ylabel('Validation relative L2 loss')
ax.legend(ncol=4,loc='lower center',bbox_to_anchor=(.5,1.02),frameon=False)
ax.set_xlim(1,100);ax.grid(axis='y',alpha=.15)
save(fig,'training')
print('Saved controlled accuracy, rollout and validation-history figures.')
