# =============================================================================
# data_visualization.py  —  模块三：可视化
#
# 包含：
#   ReservoirDataVisualizer    数据集可视化（单样本 / 时间演化 / 多样本对比 / 统计报告）
#   PredictionVisualizer       模型预测结果可视化（误差图 / 对比图）
#   plot_kr_curves             Brooks-Corey 相渗曲线 + 分流曲线 + 导数曲线
#   visualize_dataset          一键全量可视化入口
#
# 用法：
#   # 可视化数据集
#   from data_visualization import visualize_dataset
#   visualize_dataset('dataset.h5', output_dir='./figs')
#
#   # 可视化预测结果
#   from data_visualization import PredictionVisualizer
#   vis = PredictionVisualizer(model, test_loader, dataset, device='cuda')
#   vis.visualize_predictions(num_samples=3, save_fig=True, output_dir='./figs')
# =============================================================================

import os, json
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch


# ─────────────────────────── 数据集可视化 ────────────────────────────────────

class ReservoirDataVisualizer:
    """
    油藏数据集可视化器。
    读取 HDF5 文件，提供四种视图：
      1. 单样本总览（渗透率 / 孔隙度 / 压力 / 饱和度 + 时均曲线）
      2. 压力场时间演化（9 时间步网格）
      3. 饱和度场时间演化（9 时间步网格）
      4. 多样本对比仪表板
    """

    def __init__(self, h5_file_path: str, output_dir: str = './visualizations'):
        self.h5_file_path = h5_file_path
        self.output_dir   = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 色彩映射
        self.cmap_perm = 'viridis'
        self.cmap_poro = 'Blues'
        self.cmap_pres = 'hot'
        self.cmap_sat  = 'coolwarm'

        self._load_meta()

    # ── 元数据加载 ────────────────────────────────────────────────────────────

    def _load_meta(self):
        with h5py.File(self.h5_file_path, 'r') as f:
            self.sample_names = list(f['data'].keys())
            self.n_samples    = len(self.sample_names)
            sg = f['data'][self.sample_names[0]]
            self.ny, self.nx  = sg['permeability'].shape
            if 'time_steps' in sg:
                self.time_steps = sg['time_steps'][:]
                self.nt = len(self.time_steps) - 1
            else:
                self.nt = sg['pressure'].shape[0] - 1
                self.time_steps = np.arange(self.nt + 1, dtype=float)
        print(f"Dataset: {self.n_samples} samples  "
              f"grid {self.nx}×{self.ny}  nt={self.nt}")

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    def _read_sample(self, sample_name: str):
        with h5py.File(self.h5_file_path, 'r') as f:
            sg = f['data'][sample_name]
            perm = sg['permeability'][:]
            poro = sg['porosity'][:]
            pres = sg['pressure'][:].reshape(-1, self.ny, self.nx)
            sat  = sg['saturation'][:].reshape(-1, self.ny, self.nx)
            ts   = sg['time_steps'][:] if 'time_steps' in sg else self.time_steps
            pattern   = sg.attrs.get('pattern',   'unknown')
            sample_id = sg.attrs.get('sample_id', 0)
        return perm, poro, pres, sat, ts, pattern, sample_id

    @staticmethod
    def _add_colorbar(ax, im, label='', fontsize=8):
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax).set_label(label, fontsize=fontsize)

    def _display_times(self, n_panels=9):
        if self.nt + 1 <= n_panels:
            return np.arange(self.nt + 1)
        return np.linspace(0, self.nt, n_panels, dtype=int)

    # ── 1. 单样本总览 ─────────────────────────────────────────────────────────

    def visualize_single_sample(self, sample_idx: int = 0,
                                 save_fig: bool = False):
        """
        3×4 布局：
        行1 — 渗透率 / 孔隙度 / 渗透率直方图 / 孔隙度直方图
        行2 — 压力场（初始 / 中间 / 最终）+ 时均曲线（压力+饱和度）
        行3 — 饱和度场（初始 / 中间 / 最终）+ (空)
        """
        perm, poro, pres, sat, ts, pat, sid = self._read_sample(
            self.sample_names[sample_idx])

        fig = plt.figure(figsize=(20, 14))
        fig.suptitle(f'Sample {sid}  (pattern: {pat})',
                     fontsize=15, fontweight='bold', y=0.98)

        # ── 行1：岩石属性 ────────────────────────────────────────────────────
        ax = plt.subplot(3, 4, 1)
        im = ax.imshow(perm, cmap=self.cmap_perm, origin='lower')
        ax.set_title('Permeability (m²)', fontsize=11)
        self._add_colorbar(ax, im)

        ax = plt.subplot(3, 4, 2)
        im = ax.imshow(poro, cmap=self.cmap_poro, origin='lower')
        ax.set_title('Porosity', fontsize=11)
        self._add_colorbar(ax, im)

        ax = plt.subplot(3, 4, 3)
        ax.hist(perm.flatten(), bins=30, edgecolor='k', alpha=0.75)
        ax.set_title('Permeability distribution', fontsize=11)
        ax.set_xlabel('K (m²)'); ax.grid(True, alpha=0.3)

        ax = plt.subplot(3, 4, 4)
        ax.hist(poro.flatten(), bins=30, edgecolor='k', alpha=0.75,
                color='steelblue')
        ax.set_title('Porosity distribution', fontsize=11)
        ax.set_xlabel('φ'); ax.grid(True, alpha=0.3)

        # ── 行2：压力场 ──────────────────────────────────────────────────────
        t_ids = [0, self.nt // 2, self.nt]
        labels = ['Initial', f'Mid (t={ts[self.nt//2]:.0f}d)', 'Final']
        for col, (ti, lab) in enumerate(zip(t_ids, labels)):
            ax = plt.subplot(3, 4, 5 + col)
            im = ax.imshow(pres[ti], cmap=self.cmap_pres, origin='lower')
            ax.set_title(f'Pressure — {lab}', fontsize=10)
            if col == 0: ax.set_ylabel('Y')
            self._add_colorbar(ax, im, 'Pa')

        # 时均曲线（双 y 轴）
        ax_p = plt.subplot(3, 4, 8)
        color_p, color_s = 'tab:red', 'tab:blue'
        ax_p.plot(ts, pres.mean(axis=(1,2)), color=color_p, lw=2)
        ax_p.set_ylabel('Mean pressure (Pa)', color=color_p, fontsize=9)
        ax_p.tick_params(axis='y', labelcolor=color_p)
        ax_p.set_title('Temporal mean evolution', fontsize=10)
        ax_p.grid(True, alpha=0.3)
        ax_s = ax_p.twinx()
        ax_s.plot(ts, sat.mean(axis=(1,2)),  color=color_s, lw=2, ls='--')
        ax_s.set_ylabel('Mean Sw', color=color_s, fontsize=9)
        ax_s.tick_params(axis='y', labelcolor=color_s)
        ax_p.set_xlabel('Time (days)', fontsize=9)

        # ── 行3：饱和度场 ────────────────────────────────────────────────────
        for col, (ti, lab) in enumerate(zip(t_ids, labels)):
            ax = plt.subplot(3, 4, 9 + col)
            im = ax.imshow(sat[ti], cmap=self.cmap_sat,
                           origin='lower', vmin=0, vmax=1)
            ax.set_title(f'Water sat. — {lab}', fontsize=10)
            ax.set_xlabel('X')
            if col == 0: ax.set_ylabel('Y')
            self._add_colorbar(ax, im, 'Sw')

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        self._save(fig, f'sample_{sid}_overview.png', save_fig)
        return fig

    # ── 2. 压力场时间演化 ─────────────────────────────────────────────────────

    def visualize_pressure_evolution(self, sample_idx: int = 0,
                                      save_fig: bool = False):
        """9 个时间步的压力场网格图"""
        perm, poro, pres, sat, ts, pat, sid = self._read_sample(
            self.sample_names[sample_idx])
        dt = self._display_times(9)

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'Pressure evolution — Sample {sid}  (pattern: {pat})',
                     fontsize=14, fontweight='bold', y=0.97)
        for ax, ti in zip(axes.flatten(), dt):
            im = ax.imshow(pres[ti], cmap=self.cmap_pres, origin='lower')
            ax.set_title(f't={ts[ti]:.1f} d', fontsize=10)
            ax.set_xlabel('X', fontsize=8); ax.set_ylabel('Y', fontsize=8)
            ax.tick_params(labelsize=7)
            self._add_colorbar(ax, im, 'Pa', fontsize=7)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self._save(fig, f'sample_{sid}_pressure_evolution.png', save_fig)
        return fig

    # ── 3. 饱和度场时间演化 ───────────────────────────────────────────────────

    def visualize_saturation_evolution(self, sample_idx: int = 0,
                                        save_fig: bool = False):
        """9 个时间步的饱和度场网格图"""
        perm, poro, pres, sat, ts, pat, sid = self._read_sample(
            self.sample_names[sample_idx])
        dt = self._display_times(9)

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'Water saturation evolution — Sample {sid}  (pattern: {pat})',
                     fontsize=14, fontweight='bold', y=0.97)
        for ax, ti in zip(axes.flatten(), dt):
            im = ax.imshow(sat[ti], cmap=self.cmap_sat,
                           origin='lower', vmin=0, vmax=1)
            ax.set_title(f't={ts[ti]:.1f} d', fontsize=10)
            ax.set_xlabel('X', fontsize=8); ax.set_ylabel('Y', fontsize=8)
            ax.tick_params(labelsize=7)
            self._add_colorbar(ax, im, 'Sw', fontsize=7)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self._save(fig, f'sample_{sid}_saturation_evolution.png', save_fig)
        return fig

    # ── 4. 多样本对比仪表板 ───────────────────────────────────────────────────

    def create_comparison_dashboard(self, sample_indices=None,
                                     save_fig: bool = False):
        """
        每行一个样本，四列：渗透率 / 孔隙度 / 最终压力 / 最终饱和度
        """
        if sample_indices is None:
            sample_indices = list(range(min(4, self.n_samples)))
        n = len(sample_indices)

        fig = plt.figure(figsize=(20, 5 * n))
        fig.suptitle('Multi-sample comparison dashboard',
                     fontsize=16, fontweight='bold', y=0.99)

        for row, idx in enumerate(sample_indices):
            perm, poro, pres, sat, ts, pat, sid = self._read_sample(
                self.sample_names[idx])

            titles = [f'Sample {sid}\nPermeability ({pat})',
                      'Porosity', 'Final pressure', 'Final water sat.']
            fields = [perm, poro, pres[-1], sat[-1]]
            cmaps  = [self.cmap_perm, self.cmap_poro,
                      self.cmap_pres, self.cmap_sat]
            vmins  = [None, None, None, 0]
            vmaxs  = [None, None, None, 1]

            for col, (title, field, cmap, vmin, vmax) in enumerate(
                    zip(titles, fields, cmaps, vmins, vmaxs)):
                ax = plt.subplot(n, 4, row * 4 + col + 1)
                im = ax.imshow(field, cmap=cmap, origin='lower',
                               vmin=vmin, vmax=vmax)
                ax.set_title(title, fontsize=11)
                ax.set_xlabel('X', fontsize=9)
                if col == 0: ax.set_ylabel('Y', fontsize=9)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        self._save(fig, 'comparison_dashboard.png', save_fig)
        return fig

    # ── 5. 统计报告 ───────────────────────────────────────────────────────────

    def create_statistics_report(self, save_report: bool = False):
        """打印每个样本的渗透率/孔隙度统计，可选保存 JSON"""
        stats_all = []
        with h5py.File(self.h5_file_path, 'r') as f:
            for name in self.sample_names:
                sg   = f['data'][name]
                perm = sg['permeability'][:]
                poro = sg['porosity'][:]
                pat  = sg.attrs.get('pattern', 'unknown')
                sid  = sg.attrs.get('sample_id', 0)
                stats_all.append({
                    'sample_id': int(sid), 'pattern': str(pat),
                    'permeability': {
                        'mean':   float(perm.mean()),
                        'std':    float(perm.std()),
                        'min':    float(perm.min()),
                        'max':    float(perm.max()),
                        'median': float(np.median(perm)),
                    },
                    'porosity': {
                        'mean':   float(poro.mean()),
                        'std':    float(poro.std()),
                        'min':    float(poro.min()),
                        'max':    float(poro.max()),
                        'median': float(np.median(poro)),
                    },
                })

        print('=' * 80)
        print('RESERVOIR DATASET STATISTICAL REPORT')
        print('=' * 80)
        for s in stats_all:
            k = s['permeability']
            print(f"\nSample {s['sample_id']}  pattern={s['pattern']}")
            print(f"  K  mean={k['mean']:.2e}  std={k['std']:.2e}  "
                  f"[{k['min']:.2e}, {k['max']:.2e}]")
            p = s['porosity']
            print(f"  φ  mean={p['mean']:.3f}  std={p['std']:.4f}  "
                  f"[{p['min']:.3f}, {p['max']:.3f}]")

        if save_report:
            path = os.path.join(self.output_dir, 'statistics_report.json')
            with open(path, 'w') as fp:
                json.dump(stats_all, fp, indent=2)
            print(f"\nSaved → {path}")
        return stats_all

    # ── 保存辅助 ──────────────────────────────────────────────────────────────

    def _save(self, fig, filename: str, do_save: bool):
        if do_save:
            path = os.path.join(self.output_dir, filename)
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved → {path}")
        plt.show()
        plt.close(fig)


# ─────────────────────── 预测结果可视化 ──────────────────────────────────────

class PredictionVisualizer:
    """
    可视化训练好的 UNet-FNO 的单步预测结果。

    布局（3行 × 4列）：
      行1 — 输入 4 通道（K / φ / p^t / Sw^t）
      行2 — 压力：真值 / 预测 / 绝对误差 / 相对误差
      行3 — 饱和度：真值 / 预测 / 绝对误差 / 相对误差
    """

    def __init__(self, model, test_loader, dataset, device: str = 'cpu'):
        self.model       = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.dataset     = dataset
        self.device      = device

    # ── 主接口 ────────────────────────────────────────────────────────────────

    def visualize_predictions(self, num_samples: int = 3,
                               save_fig: bool = False,
                               output_dir: str = './visualizations'):
        os.makedirs(output_dir, exist_ok=True)
        ds = self.dataset

        with torch.no_grad():
            for bi, batch in enumerate(self.test_loader):
                if bi >= num_samples:
                    break

                x = batch['input'].to(self.device)
                y = batch['output'].to(self.device)
                pred = self.model(x)

                # 取 batch 中第一个样本
                x_np    = x[0].cpu().numpy()       # (4, ny, nx)
                y_np    = y[0].cpu().numpy()        # (2, ny, nx)
                pred_np = pred[0].cpu().numpy()     # (2, ny, nx)

                # 反归一化（如果 dataset 有 normalize）
                if ds.normalize:
                    y_p    = [ds._denorm(y_np[0],    'pressure'),
                               ds._denorm(y_np[1],    'saturation')]
                    p_p    = [ds._denorm(pred_np[0],  'pressure'),
                               ds._denorm(pred_np[1],  'saturation')]
                    x_show = [
                        ds._denorm(x_np[0], 'permeability'),
                        ds._denorm(x_np[1], 'porosity'),
                        ds._denorm(x_np[2], 'pressure'),
                        ds._denorm(x_np[3], 'saturation'),
                    ]
                else:
                    y_p    = [y_np[0],    y_np[1]]
                    p_p    = [pred_np[0], pred_np[1]]
                    x_show = [x_np[i] for i in range(4)]

                fig = self._make_figure(x_show, y_p, p_p, sample_no=bi+1)

                if save_fig:
                    path = os.path.join(output_dir,
                                        f'prediction_sample_{bi+1}.png')
                    fig.savefig(path, dpi=150, bbox_inches='tight')
                    print(f"Saved → {path}")
                plt.show()
                plt.close(fig)

    # ── 图像构建 ──────────────────────────────────────────────────────────────

    def _make_figure(self, x_show, y_p, p_p, sample_no: int):
        in_titles  = ['Permeability (K)', 'Porosity (φ)',
                       'Pressure  p^t',   'Saturation  Sw^t']
        out_labels = ['Pressure', 'Saturation']
        cmaps_in   = ['viridis', 'Blues', 'hot', 'coolwarm']
        cmaps_out  = ['hot', 'coolwarm']

        fig = plt.figure(figsize=(20, 12))

        # 行1：输入通道
        for col in range(4):
            ax = plt.subplot(3, 4, col + 1)
            im = ax.imshow(x_show[col], cmap=cmaps_in[col])
            ax.set_title(f'Input: {in_titles[col]}', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 行2–3：真值 / 预测 / 绝对误差 / 相对误差
        stats_list = []
        for row, (name, cmap, truth, pred) in enumerate(zip(
                out_labels, cmaps_out, y_p, p_p)):

            abs_err = np.abs(truth - pred)
            denom   = np.abs(truth)
            mask    = denom > denom.max() * 1e-3
            rel_err = np.zeros_like(abs_err)
            rel_err[mask] = abs_err[mask] / denom[mask]

            # vmax for relative error: 95th percentile to clip outliers
            v_rel = np.percentile(rel_err[mask], 95) if mask.any() else 1.0
            v_rel = max(v_rel, 1e-8)

            stats_list.append({
                'name':     name,
                'abs_mean': abs_err.mean(),
                'abs_max':  abs_err.max(),
                'rel_mean': rel_err[mask].mean() if mask.any() else 0.0,
                'rel_max':  rel_err[mask].max()  if mask.any() else 0.0,
            })

            base = 4 + row * 4

            ax = plt.subplot(3, 4, base + 1)
            im = ax.imshow(truth, cmap=cmap, origin='lower')
            ax.set_title(f'{name} (true)', fontsize=10); ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax = plt.subplot(3, 4, base + 2)
            im = ax.imshow(pred,  cmap=cmap, origin='lower')
            ax.set_title(f'{name} (pred)', fontsize=10); ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax = plt.subplot(3, 4, base + 3)
            im = ax.imshow(abs_err, cmap='hot', vmin=0,
                           vmax=abs_err.max() or 1.0)
            ax.set_title(f'{name} abs error', fontsize=10); ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            s = stats_list[-1]
            ax.text(0.05, 0.95,
                    f"mean={s['abs_mean']:.2e}\nmax={s['abs_max']:.2e}",
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.8))

            ax = plt.subplot(3, 4, base + 4)
            im = ax.imshow(rel_err, cmap='plasma', vmin=0, vmax=v_rel)
            ax.set_title(f'{name} rel error', fontsize=10); ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.text(0.05, 0.95,
                    f"mean={s['rel_mean']:.2%}\nmax={s['rel_max']:.2%}",
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        s0, s1 = stats_list
        fig.suptitle(
            f"Sample {sample_no} — "
            f"Pressure: MAE={s0['abs_mean']:.2e} Pa  RelErr={s0['rel_mean']:.2%} | "
            f"Saturation: MAE={s1['abs_mean']:.4f}  RelErr={s1['rel_mean']:.2%}",
            fontsize=12, y=1.01)
        plt.tight_layout()

        # 控制台输出
        print(f"\n{'='*60}\nSample {sample_no}")
        for s in stats_list:
            print(f"  {s['name']:12s}  "
                  f"AbsMean={s['abs_mean']:.2e}  AbsMax={s['abs_max']:.2e}  "
                  f"RelMean={s['rel_mean']:.2%}  RelMax={s['rel_max']:.2%}")

        return fig


# ──────────────────────── Brooks-Corey 相渗曲线 ───────────────────────────────

def plot_kr_curves(props, output_dir: str = '.', prefix: str = 'brooks_corey',
                   save: bool = True):
    """
    绘制并保存 Brooks-Corey 相渗曲线、分流曲线、分流导数。

    参数：
        props      ReservoirProperties 实例
        output_dir 输出目录
        prefix     文件名前缀
        save       是否保存图片和 CSV
    返回：
        dict — 关键物理量（交叉点、BL 前缘等）
    """
    os.makedirs(output_dir, exist_ok=True)
    Swr, Sor = props.Swr, props.Sor
    nw,  no  = props.nw,  props.no
    mu_w, mu_o = props.mu_w, props.mu_o
    Bw,   Bo   = props.Bw,   props.Bo
    denom = 1.0 - Swr - Sor

    Sw = np.linspace(0.0, 1.0, 1000)
    krw = np.zeros_like(Sw)
    kro = np.zeros_like(Sw)
    mob = (Sw > Swr) & (Sw < 1.0 - Sor)
    Swn = np.clip((Sw[mob] - Swr) / denom, 0.0, 1.0)
    krw[mob] = Swn ** nw
    kro[mob] = (1.0 - Swn) ** no
    krw[Sw >= 1.0 - Sor] = 1.0
    kro[Sw <= Swr]        = 1.0

    lw = krw / mu_w / Bw
    lo = kro / mu_o / Bo
    fw = np.where((lw + lo) > 0, lw / (lw + lo), 0.0)
    dfw = np.gradient(fw, Sw)

    # Buckley-Leverett 切线（切点法）
    err = np.abs(dfw * (Sw - Swr) - fw)
    err[Sw < Swr + 0.01] = np.inf
    i_front  = np.argmin(err)
    Sw_front = Sw[i_front]
    fw_front = fw[i_front]

    # 交叉点
    diff = np.abs(krw - kro); diff[~mob] = np.inf
    i_cross  = np.argmin(diff)
    Sw_cross = Sw[i_cross]
    kr_cross = (krw[i_cross] + kro[i_cross]) / 2

    # ── 图 ────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Brooks-Corey  Swr={Swr}  Sor={Sor}  nw={nw}  no={no}  "
        f"μw={mu_w:.3e}  μo={mu_o:.3e}",
        fontsize=12, fontweight='bold')

    # krw / kro
    ax = axes[0]
    ax.plot(Sw, krw, color='#1D9E75', lw=2.5, label=r'$k_{rw}$')
    ax.plot(Sw, kro, color='#E24B4A', lw=2.5, label=r'$k_{ro}$')
    ax.axvspan(0.0, Swr,       alpha=0.07, color='#1D9E75')
    ax.axvspan(1-Sor, 1.0,     alpha=0.07, color='#E24B4A')
    ax.axvline(Swr,     color='#1D9E75', lw=1, ls='--', alpha=0.7)
    ax.axvline(1-Sor,   color='#E24B4A', lw=1, ls='--', alpha=0.7)
    ax.plot(Sw_cross, kr_cross, 'ko', ms=7,
            label=f'crossover Sw={Sw_cross:.3f}')
    ax.set_xlabel('$S_w$'); ax.set_ylabel('$k_r$')
    ax.set_title('Relative permeability'); ax.set_xlim(0,1); ax.set_ylim(0,1.05)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    # fw
    ax = axes[1]
    ax.plot(Sw, fw, color='#534AB7', lw=2.5, label=r'$f_w$')
    ax.axvline(Swr,   color='#1D9E75', lw=1, ls='--', alpha=0.6)
    ax.axvline(1-Sor, color='#E24B4A', lw=1, ls='--', alpha=0.6)
    ax.plot([Swr, Sw_front, 1.0], [0, fw_front, 1.0],
            color='orange', lw=1.5, ls='--',
            label=f'BL tangent  Sw={Sw_front:.3f}')
    ax.plot(Sw_front, fw_front, 'o', color='orange', ms=7)
    ax.set_xlabel('$S_w$'); ax.set_ylabel('$f_w$')
    ax.set_title('Fractional flow'); ax.set_xlim(0,1); ax.set_ylim(0,1.05)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    # dfw/dSw
    ax = axes[2]
    m = (Sw >= Swr) & (Sw <= 1 - Sor)
    ax.plot(Sw[m], dfw[m], color='#BA7517', lw=2.5)
    ax.axvline(Sw_front, color='orange', lw=1.5, ls='--',
               label=f'BL front Sw={Sw_front:.3f}')
    ax.axvline(Swr,   color='#1D9E75', lw=1, ls='--', alpha=0.5)
    ax.axvline(1-Sor, color='#E24B4A', lw=1, ls='--', alpha=0.5)
    ax.set_xlabel('$S_w$'); ax.set_ylabel(r'$\partial f_w / \partial S_w$')
    ax.set_title('Fractional flow derivative\n(shock front velocity)')
    ax.set_xlim(Swr - 0.02, 1 - Sor + 0.02)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    plt.tight_layout()

    result = {
        'Sw_crossover': Sw_cross, 'kr_crossover': kr_cross,
        'Sw_BL_front':  Sw_front, 'fw_BL_front':  fw_front,
    }

    if save:
        png_path = os.path.join(output_dir, f'{prefix}_kr_curves.png')
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {png_path}")

        csv_path = os.path.join(output_dir, f'{prefix}_kr_data.csv')
        header = 'Sw,krw,kro,fw,dfw_dSw,lambda_w,lambda_o'
        np.savetxt(csv_path,
                   np.column_stack([Sw, krw, kro, fw, dfw, lw, lo]),
                   delimiter=',', header=header, comments='', fmt='%.8f')
        print(f"Saved → {csv_path}")

    plt.show()
    plt.close(fig)

    print(f"\nBrooks-Corey summary:")
    print(f"  Swr={Swr}  Sor={Sor}  nw={nw}  no={no}")
    print(f"  Mobile range: [{Swr:.3f}, {1-Sor:.3f}]  denom={denom:.3f}")
    print(f"  krw(1-Sor) = {krw[Sw >= 1-Sor][0]:.4f}  (should be 1.0)")
    print(f"  kro(Swr)   = {kro[Sw <= Swr][-1]:.4f}  (should be 1.0)")
    print(f"  Crossover  : Sw={Sw_cross:.4f}  kr={kr_cross:.4f}")
    print(f"  BL front   : Sw={Sw_front:.4f}  fw={fw_front:.4f}")
    return result


# ─────────────────────── 一键全量可视化 ──────────────────────────────────────

def visualize_dataset(h5_file_path: str, output_dir: str = './visualizations'):
    """
    对 HDF5 数据集执行全套可视化并保存。
    """
    vis = ReservoirDataVisualizer(h5_file_path, output_dir)
    print(f"\n{'='*60}\nDataset Visualization\n{'='*60}")

    # print("\n[1/4] Statistical report...")
    # vis.create_statistics_report(save_report=True)

    print("\n[2/4] Single sample overview...")
    vis.visualize_single_sample(sample_idx=0, save_fig=True)

    print("\n[3/4] Pressure evolution...")
    vis.visualize_pressure_evolution(sample_idx=0, save_fig=True)

    print("\n[4/4] Saturation evolution...")
    vis.visualize_saturation_evolution(sample_idx=0, save_fig=True)

    if vis.n_samples > 1:
        print("\n[5/5] Comparison dashboard...")
        vis.create_comparison_dashboard(
            sample_indices=list(range(min(4, vis.n_samples))),
            save_fig=True)

    print(f"\nDone. All outputs → {output_dir}")


# ──────────────────────────── 入口 ───────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visualization tools')
    parser.add_argument('--h5',     default=None, help='HDF5 数据集路径')
    parser.add_argument('--outdir', default='./visualizations')
    parser.add_argument('--kr',     action='store_true',
                        help='仅绘制 Brooks-Corey 相渗曲线（不需要 --h5）')
    args = parser.parse_args()

    if args.kr:
        # 独立绘制相渗曲线（使用默认参数，无需完整数据集）
        class _Props:
            Swr=0.2; Sor=0.2; nw=4.0; no=2.0
            mu_w=1e-3; mu_o=1.8e-3; Bw=1.0; Bo=1.2
        plot_kr_curves(_Props(), output_dir=args.outdir)
    elif args.h5:
        visualize_dataset(args.h5, output_dir=args.outdir)
    else:
        print("用法：")
        print("  # 可视化数据集")
        print("  python data_visualization.py --h5 dataset.h5 --outdir ./figs")
        print("  # 仅绘制相渗曲线")
        print("  python data_visualization.py --kr --outdir ./figs")
