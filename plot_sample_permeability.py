"""
plot_sample_permeability.py
===========================
生成论文 §3.1 中的渗透率场示意图（Fig. sample_permeability）。

用法：
    python plot_sample_permeability.py --h5 path/to/dataset.h5
    python plot_sample_permeability.py --h5 dataset.h5 --sample 0 --out fig/sample_0_visualization.png
"""

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── 论文字体设置（与 CMAME 双栏格式匹配）────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':        9,
    'axes.titlesize':   10,
    'axes.labelsize':   9,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'legend.fontsize':  8,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'savefig.pad_inches': 0.05,
})


def load_sample(h5_path: str, sample_idx: int):
    """从 HDF5 加载指定样本的渗透率、压力、饱和度数据。"""
    with h5py.File(h5_path, 'r') as f:
        keys = sorted(f['data'].keys())
        if sample_idx >= len(keys):
            raise IndexError(
                f"样本索引 {sample_idx} 超出范围（共 {len(keys)} 个样本）")
        key = keys[sample_idx]
        sg  = f['data'][key]

        perm = sg['permeability'][:]        # (ny, nx)  单位 m²
        pres = sg['pressure'][:]            # (nt, ncell)
        sat  = sg['saturation'][:]          # (nt, ncell)
        ts   = sg['time_steps'][:]          # (nt,)  单位 day
        pat  = sg.attrs.get('pattern', 'unknown')
        sid  = sg.attrs.get('sample_id', sample_idx)

        ny, nx = perm.shape
        pres_grid = pres.reshape(-1, ny, nx)   # (nt, ny, nx)
        sat_grid  = sat.reshape(-1, ny, nx)

    return perm, pres_grid, sat_grid, ts, pat, sid, nx, ny


def mD_label(val):
    """将 m² 转换为 mD 并格式化为标签（1 mD ≈ 9.869e-16 m²）。"""
    return val / 9.869e-16


def plot_paper_figure(h5_path: str, sample_idx: int, out_path: str):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.ticker as ticker

    # ── 数据加载 ─────────────────────────────────────────────
    perm, pres, sat, ts, pattern, sid, nx, ny = load_sample(h5_path, sample_idx)

    nt = pres.shape[0]
    t_ids = [0, nt // 2, nt - 1]
    t_labels = [f'$t={ts[i]:.0f}$ d' for i in t_ids]

    # ── helper：子图编号（放在下方）──────────────────────────
    def add_panel_label(ax, label):
        ax.text(0.5, -0.22, label,
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=10,
                fontweight='bold')

    # ── 画布 ───────────────────────────────────────────────
    fig = plt.figure(figsize=(10.5, 6))

    gs = GridSpec(2, 4, figure=fig,
                  hspace=0.7,
                  wspace=0.50,
                  left=0.06, right=0.98,
                  top=0.93, bottom=0.12)  # ⚠️ bottom调大给(a)-(h)

    # ── 子图 ───────────────────────────────────────────────
    ax_perm = fig.add_subplot(gs[0, 0])
    ax_p = [fig.add_subplot(gs[0, i]) for i in range(1, 4)]
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_s = [fig.add_subplot(gs[1, i]) for i in range(1, 4)]

    # ── 空间尺度 ───────────────────────────────────────────
    dx = 100.0 / nx
    dy = 100.0 / ny
    extent = [0, nx * dx, 0, ny * dy]

    # ════════════════════════════════════════════════════════
    # (a) Permeability
    # ════════════════════════════════════════════════════════
    perm_mD = perm / 9.869e-16

    im_k = ax_perm.imshow(
        perm_mD, cmap='plasma', origin='lower',
        extent=extent, aspect='equal'
    )

    # colorbar
    div = make_axes_locatable(ax_perm)
    cax = div.append_axes('right', size='4%', pad=0.04)
    cb = fig.colorbar(im_k, cax=cax)
    cb.set_label('$K$ (mD)', fontsize=8)

    # ── 井位标注（恢复）────────────────────────────
    dy_cell = 100.0 / ny

    ax_perm.plot(dx * 0.5, dy_cell * 0.5,
                 marker='^', ms=7, color='#D62728',
                 markeredgecolor='white', markeredgewidth=0.6,
                 zorder=5, label='Injector')

    ax_perm.plot(dx * (nx - 0.5), dy_cell * (ny - 0.5),
                 marker='v', ms=7, color='#2CA02C',
                 markeredgecolor='white', markeredgewidth=0.6,
                 zorder=5, label='Producer')

    ax_perm.legend(loc='upper right',
                   fontsize=10,
                   framealpha=0.85,
                   edgecolor='none')

    ax_perm.set_xlabel('$x$ (m)')
    ax_perm.set_ylabel('$y$ (m)')
    ax_perm.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax_perm.yaxis.set_major_locator(ticker.MultipleLocator(25))

    add_panel_label(ax_perm, '(a)')

    # ════════════════════════════════════════════════════════
    # (b–d) Pressure
    # ════════════════════════════════════════════════════════
    p_all = pres[t_ids] / 1e6
    vmin, vmax = p_all.min(), p_all.max()

    for i, (ax, ti, tlab) in enumerate(zip(ax_p, t_ids, t_labels)):
        im = ax.imshow(
            pres[ti] / 1e6,
            cmap='inferno',
            origin='lower',
            extent=extent,
            aspect='equal',
            vmin=vmin, vmax=vmax
        )

        if i == 0:
            ax.set_ylabel('$y$ (m)')
        else:
            ax.set_yticklabels([])

        ax.set_xlabel('$x$ (m)')
        ax.tick_params(labelsize=6)

        ax.set_title(f'Pressure {tlab}', fontsize=10)

        if i == 2:
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', size='5%', pad=0.04)
            cb = fig.colorbar(im, cax=cax)
            cb.set_label('$P$ (MPa)', fontsize=7)

        add_panel_label(ax, f'({chr(ord("b")+i)})')

    # ════════════════════════════════════════════════════════
    # (e) Histogram
    # ════════════════════════════════════════════════════════
    ax_hist.hist(perm_mD.ravel(), bins=20,
                 color='#9467BD', edgecolor='white', linewidth=0.4)

    ax_hist.set_xlabel('$K$ (mD)')
    ax_hist.set_ylabel('Count')

    add_panel_label(ax_hist, '(e)')

    # ════════════════════════════════════════════════════════
    # (f–h) Saturation
    # ════════════════════════════════════════════════════════
    for i, (ax, ti, tlab) in enumerate(zip(ax_s, t_ids, t_labels)):
        im = ax.imshow(
            sat[ti],
            cmap='coolwarm',
            origin='lower',
            extent=extent,
            aspect='equal',
            vmin=0.2, vmax=1.0
        )

        if i == 0:
            ax.set_ylabel('$y$ (m)')
        else:
            ax.set_yticklabels([])

        ax.set_xlabel('$x$ (m)')
        ax.tick_params(labelsize=6)

        ax.set_title(f'$S_w$ {tlab}', fontsize=10)

        if i == 2:
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', size='5%', pad=0.04)
            cb = fig.colorbar(im, cax=cax)
            cb.set_label('$S_w$', fontsize=7)

        add_panel_label(ax, f'({chr(ord("f")+i)})')

    # ── 保存 ───────────────────────────────────────────────
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved → {out_path}")







# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--h5',     required=True,
                   help='HDF5 数据集路径')
    p.add_argument('--sample', type=int, default=0,
                   help='样本索引（默认 0）')
    p.add_argument('--out',    default='fig/sample_0_visualization.png',
                   help='输出图片路径')
    return p.parse_args()


if __name__ == '__main__':
    import os
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plot_paper_figure(args.h5, args.sample, args.out)
