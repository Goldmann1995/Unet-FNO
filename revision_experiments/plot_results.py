"""Independent rollout mean and realisation-level pointwise uncertainty.

Figure contract: compare pressure and transport-error accumulation of frozen
models on the same 100 independent simulations. Two aligned panels distinguish
the elliptic pressure response from saturation transport. All runs are used.
Quantitative grid, 178 x 78 mm, editable PDF text >= 5 pt, Python backend.
"""
from pathlib import Path
import json,sys
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
OUTPUT=HERE.parent/'fig/revision'
OUTPUT.mkdir(parents=True,exist_ok=True)
d=json.loads((HERE/'rollout_summary.json').read_text())
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,
 'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
fig,axes=plt.subplots(1,2,figsize=(178/25.4,78/25.4))
fig.subplots_adjust(left=.09,right=.985,bottom=.20,top=.80,wspace=.29)
colors=['#4477AA','#CC6677','#228833']
x=np.arange(1,101)
for name,color in zip(['FNO','U-FNO','UNet-FNO'],colors):
    res=d[name];mean=np.asarray(res['mean']);lo=np.asarray(res['ci_low']);hi=np.asarray(res['ci_high'])
    assert np.isfinite(mean).all() and (lo>0).all(),'Log axis requires positive complete data.'
    for i,ax in enumerate(axes):
        ax.plot(x,mean[:,i],color=color,lw=1.25,label=name)
        ax.fill_between(x,lo[:,i],hi[:,i],color=color,alpha=.18,linewidth=0)
for i,ax in enumerate(axes):
    ax.set_yscale('log');ax.set_xlim(1,100);ax.set_xticks([1,25,50,75,100])
    ax.set_xlabel('Autoregressive step');ax.set_ylabel('Mean relative error (%)')
    ax.set_title(['(a) Pressure','(b) Water saturation'][i],loc='left',fontsize=9,pad=8)
    ax.grid(which='major',axis='y',alpha=.18,linewidth=.5)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,bbox_to_anchor=(.53,1.01),frameon=False)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,json_out=HERE/'rollout_figure_alignment.json',strict=True)
fig.savefig(OUTPUT/'rollout100_ci.pdf')
fig.savefig(OUTPUT/'rollout100_ci.svg')
fig.savefig(OUTPUT/'rollout100_ci.png',dpi=600)
plt.close(fig)
print('Saved complete-data mean/CI rollout figure.')
