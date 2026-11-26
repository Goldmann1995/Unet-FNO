# =============================================================================
# compare_models.py  —  三模型统一训练与对比
#
# 包含模型：
#   FNO2d        标准 Fourier Neural Operator（原始论文，4 层）
#   UFNO2d       U-FNO（Wen et al. 2022，后 3 层加 U-Net 分支）
#   UNetFNO2d    本文改进版（每层前置 AdaptiveUNet + GroupNorm + 残差）
#
# 评估指标（逐通道 + 综合）：
#   MAE  / RelErr(%)  for Pressure and Saturation
#   推断时间（ms/step）
#   参数量
#
# 用法：
#   python compare_models.py --h5 dataset.h5 --outdir ./compare --epochs 100
#
#   # 跳过训练，仅从已保存 checkpoint 加载并评估：
#   python compare_models.py --h5 dataset.h5 --outdir ./compare --eval_only
# =============================================================================

import os, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Tuple
import operator
from functools import reduce
# 从已有模块导入公共部分
from data_training import (
    LpLoss,
    ReservoirOperatorDataset,
    create_data_loaders,
    SpectralConv2d,          # 共用 SpectralConv2d
    AdaptiveUNet,
    UNetFNO2d,
    Trainer,
)
torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
np.random.seed(42)


# ═════════════════════════════════════════════════════════════════════════════
# 模型一：标准 FNO2d（原始论文，4 FNO 层，无 U-Net）
# ═════════════════════════════════════════════════════════════════════════════

class FNO2d(nn.Module):
    """
    标准 Fourier Neural Operator（Li et al., 2020）。
    4 层 SpectralConv2d + Conv1d bypass，ReLU 激活。
    输入 4 通道 [K, φ, p^t, Sw^t]，输出 2 通道 [p^{t+1}, Sw^{t+1}]。
    """

    def __init__(self, modes1: int = 8, modes2: int = 8, width: int = 32,
                 in_channels: int = 4, out_channels: int = 2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.width = width

        self.fc0 = nn.Linear(in_channels, width)

        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)

        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x: (B, 4, ny, nx)
        B, _, H, W = x.shape
        x = x.permute(0, 2, 3, 1)          # (B, H, W, 4)
        x = self.fc0(x)                      # (B, H, W, width)
        x = x.permute(0, 3, 1, 2)           # (B, width, H, W)

        def _layer(conv, w_conv, h, activate=True):
            x1 = conv(h)
            x2 = w_conv(h.view(B, self.width, -1)).view(B, self.width, H, W)
            out = x1 + x2
            return F.relu(out) if activate else out

        x = _layer(self.conv0, self.w0, x)
        x = _layer(self.conv1, self.w1, x)
        x = _layer(self.conv2, self.w2, x)
        x = _layer(self.conv3, self.w3, x, activate=False)

        x = x.permute(0, 2, 3, 1)           # (B, H, W, width)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                      # (B, H, W, 2)
        return x.permute(0, 3, 1, 2)         # (B, 2, H, W)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())






# ═════════════════════════════════════════════════════════════════════════════
# 统一评估工具
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, test_loader, dataset, device: str) -> Dict:
    """
    在测试集上对模型做完整评估，返回逐通道和综合指标。

    指标：
        pressure_mae     / saturation_mae     — 物理空间 MAE（反归一化后）
        pressure_rel_err / saturation_rel_err — 相对误差 (%)
        combined_rel_err                       — 两者均值 (%)
        inference_ms                          — 单 batch 平均推断时间 (ms)
    """
    model.eval()
    ds = dataset

    p_mae_all,  s_mae_all  = [], []
    p_rel_all,  s_rel_all  = [], []
    t_infer = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch['input'].to(device)
            y = batch['output'].to(device)

            t0   = time.perf_counter()
            pred = model(x)
            t_infer.append((time.perf_counter() - t0) * 1000)   # ms

            # 反归一化到物理空间
            if ds.normalize:
                p_true = ds._denorm(y[:, 0].cpu().numpy(),    'pressure')
                s_true = ds._denorm(y[:, 1].cpu().numpy(),    'saturation')
                p_pred = ds._denorm(pred[:, 0].cpu().numpy(), 'pressure')
                s_pred = ds._denorm(pred[:, 1].cpu().numpy(), 'saturation')
            else:
                p_true = y[:, 0].cpu().numpy()
                s_true = y[:, 1].cpu().numpy()
                p_pred = pred[:, 0].cpu().numpy()
                s_pred = pred[:, 1].cpu().numpy()

            # MAE
            p_mae_all.append(np.abs(p_true - p_pred).mean())
            s_mae_all.append(np.abs(s_true - s_pred).mean())

            # 相对误差（避免除零）
            p_denom = np.abs(p_true).mean(axis=(-1, -2), keepdims=True)
            s_denom = np.abs(s_true).mean(axis=(-1, -2), keepdims=True)
            p_rel_all.append(
                (np.abs(p_true - p_pred) / (p_denom + 1e-8)).mean() * 100)
            s_rel_all.append(
                (np.abs(s_true - s_pred) / (s_denom + 1e-8)).mean() * 100)

    return {
        'pressure_mae':     float(np.mean(p_mae_all)),
        'saturation_mae':   float(np.mean(s_mae_all)),
        'pressure_rel_err': float(np.mean(p_rel_all)),
        'saturation_rel_err': float(np.mean(s_rel_all)),
        'combined_rel_err': float((np.mean(p_rel_all) + np.mean(s_rel_all)) / 2),
        'inference_ms':     float(np.mean(t_infer)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 可视化：对比结果
# ═════════════════════════════════════════════════════════════════════════════

def plot_comparison_bar(results: Dict[str, Dict], output_dir: str):
    """
    绘制 4 张条形图：压力 MAE / 压力相对误差 / 饱和度 MAE / 饱和度相对误差。
    模型用不同颜色区分，最优值标星。
    """
    os.makedirs(output_dir, exist_ok=True)
    names  = list(results.keys())
    colors = {'FNO': '#85B7EB', 'U-FNO': '#E24B4A', 'UNet-FNO': '#1D9E75'}
    bar_colors = [colors.get(n, '#888780') for n in names]

    metrics = [
        ('pressure_mae',      'Pressure MAE (Pa)',       'Pressure MAE'),
        ('pressure_rel_err',  'Pressure Rel. Error (%)', 'Pressure Rel. Error'),
        ('saturation_mae',    'Saturation MAE',           'Saturation MAE'),
        ('saturation_rel_err','Saturation Rel. Error (%)','Saturation Rel. Error'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')

    for ax, (key, ylabel, title) in zip(axes.flatten(), metrics):
        vals = [results[n][key] for n in names]
        best = min(range(len(vals)), key=lambda i: vals[i])

        bars = ax.bar(names, vals, color=bar_colors, edgecolor='white',
                      linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # 数值标注
        for i, (bar, v) in enumerate(zip(bars, vals)):
            label = f'{v:.3e}' if v > 1e3 or v < 0.01 else f'{v:.4f}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    label, ha='center', va='bottom', fontsize=9,
                    fontweight='bold' if i == best else 'normal')

        # 最优值加金色边框
        bars[best].set_edgecolor('#F5A623')
        bars[best].set_linewidth(2.5)
        ax.set_ylim(0, max(vals) * 1.20)
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    for ax, lbl in zip(axes.flatten(), panel_labels):
        ax.text(0.5, -0.18, lbl, transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='center', va='top')

    plt.tight_layout()
    path = os.path.join(output_dir, 'comparison_bar.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")


def plot_loss_curves(trainers: Dict[str, 'Trainer'], output_dir: str):
    """三模型的训练/测试损失曲线对比"""
    colors = {'FNO': '#85B7EB', 'U-FNO': '#E24B4A', 'UNet-FNO': '#1D9E75'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Training & Test Loss Curves', fontsize=13, fontweight='bold')

    for name, trainer in trainers.items():
        c = colors.get(name, 'gray')
        axes[0].plot(trainer.train_losses, label=name, color=c, lw=2)
        axes[1].plot(trainer.test_losses,  label=name, color=c, lw=2)

    for ax, title in zip(axes, ['Training Loss', 'Test Loss']):
        ax.set_xlabel('Epoch'); ax.set_ylabel('Relative L2 Loss')
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)

    for ax, lbl in zip(axes, ['(a)', '(b)']):
        ax.text(0.5, -0.18, lbl, transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='center', va='top')

    plt.tight_layout()
    path = os.path.join(output_dir, 'loss_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")


# def plot_prediction_comparison(models: Dict[str, nn.Module],
#                                 test_loader, dataset,
#                                 device: str, output_dir: str,
#                                 n_samples: int = 2):
#     """
#     并排展示 n_samples 个测试样本的预测结果。
#     每个样本 2 行（压力 / 饱和度），每行 = 真值 + 各模型预测 + 各模型误差图。
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     ds = dataset
#     names = list(models.keys())

#     # 预计算颜色映射
#     cmaps = {'pressure': 'hot', 'saturation': 'coolwarm'}

#     for m in models.values():
#         m.eval()

#     with torch.no_grad():
#         for bi, batch in enumerate(test_loader):
#             if bi >= n_samples:
#                 break

#             x = batch['input'].to(device)
#             y = batch['output'].to(device)

#             preds = {}
#             for name, model in models.items():
#                 preds[name] = model(x)[0].cpu().numpy()  # (2, H, W)

#             y_np = y[0].cpu().numpy()

#             # 反归一化
#             def dn(arr, key):
#                 return ds._denorm(arr, key) if ds.normalize else arr

#             truth = {
#                 'pressure':   dn(y_np[0],  'pressure'),
#                 'saturation': dn(y_np[1],  'saturation'),
#             }
#             pred_phys = {
#                 name: {
#                     'pressure':   dn(preds[name][0], 'pressure'),
#                     'saturation': dn(preds[name][1], 'saturation'),
#                 }
#                 for name in names
#             }

#             # 布局：n_var 行 × (1 真值 + n_models 预测 + n_models 误差) 列
#             n_var    = 2
#             n_models = len(names)
#             n_cols   = 1 + n_models * 2   # 真值 + (pred+err) × n_models
#             fig, axes = plt.subplots(n_var, n_cols,
#                                       figsize=(4.5 * n_cols, 4.5 * n_var))
#             # fig.suptitle(f'Prediction comparison — test sample {bi+1}',
#             #              fontsize=13, fontweight='bold', y=1.01)

#             var_keys  = ['pressure', 'saturation']
#             var_units = ['Pa', '']

#             panel_idx = [0]   # mutable counter for label across rows/cols

#             for ri, (vk, unit) in enumerate(zip(var_keys, var_units)):
#                 row = axes[ri] if n_var > 1 else axes

#                 # 真值
#                 ax = row[0]
#                 im = ax.imshow(truth[vk], cmap=cmaps[vk], origin='lower',
#                                **({'vmin': 0, 'vmax': 1} if vk == 'saturation' else {}))
#                 lbl = f'({chr(ord("a") + panel_idx[0])})'
#                 ax.set_title(f'{vk.capitalize()}\n(Ground truth)', fontsize=9)
#                 ax.text(0.5, -0.04, lbl, transform=ax.transAxes,
#                         fontsize=10, fontweight='bold', ha='center', va='top')
#                 ax.axis('off')
#                 plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#                 panel_idx[0] += 1

#                 col = 1
#                 for name in names:
#                     pv    = pred_phys[name][vk]
#                     err   = np.abs(truth[vk] - pv)
#                     mae   = err.mean()
#                     denom = np.abs(truth[vk]).mean()
#                     rel   = mae / (denom + 1e-8) * 100

#                     # 预测
#                     ax = row[col]
#                     im = ax.imshow(pv, cmap=cmaps[vk], origin='lower',
#                                    **({'vmin': 0, 'vmax': 1} if vk == 'saturation' else {}))
#                     lbl = f'({chr(ord("a") + panel_idx[0])})'
#                     ax.set_title(f'{name}\nMAE={mae:.2e}{unit}  Rel={rel:.2f}%', fontsize=8)
#                     ax.text(0.5, -0.04, lbl, transform=ax.transAxes,
#                             fontsize=10, fontweight='bold', ha='center', va='top')
#                     ax.axis('off')
#                     plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#                     col += 1
#                     panel_idx[0] += 1

#                     # 误差图
#                     ax = row[col]
#                     im = ax.imshow(err, cmap='plasma', origin='lower',
#                                    vmin=0, vmax=np.percentile(err, 95) or err.max())
#                     lbl = f'({chr(ord("a") + panel_idx[0])})'
#                     ax.set_title(f'{name} |error|', fontsize=8)
#                     ax.text(0.5, -0.04, lbl, transform=ax.transAxes,
#                             fontsize=10, fontweight='bold', ha='center', va='top')
#                     ax.axis('off')
#                     plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#                     col += 1
#                     panel_idx[0] += 1

#             plt.tight_layout()
#             path = os.path.join(output_dir, f'pred_compare_sample{bi+1}.png')
#             fig.savefig(path, dpi=150, bbox_inches='tight')
#             plt.close()
#             print(f"  Saved → {path}")



def plot_prediction_comparison(models: Dict[str, nn.Module],
                                test_loader, dataset,
                                device: str, output_dir: str,
                                n_samples: int = 2):
    """
    纵向排列展示 n_samples 个测试样本的预测结果。
    每一行 = 某个变量在某个模型上的对比（真值 / 预测 / 误差），
    行数 = 变量数 × 模型数，列数固定为3，宽度恒定，更适合论文排版。
    """
    os.makedirs(output_dir, exist_ok=True)
    ds = dataset
    names = list(models.keys())

    cmaps = {'pressure': 'hot', 'saturation': 'coolwarm'}

    for m in models.values():
        m.eval()

    var_keys = ['pressure', 'saturation']
    var_titles = ['Pressure', 'Saturation']
    var_units = ['Pa', '']

    n_var = len(var_keys)
    n_models = len(names)
    n_rows = n_var * n_models   # 纵向堆叠：变量 × 模型
    n_cols = 3                   # 真值 / 预测 / 误差

    # 每个子图固定尺寸，保证清晰可读（可按需微调）
    panel_w, panel_h = 4.2, 3.8

    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            if bi >= n_samples:
                break

            x = batch['input'].to(device)
            y = batch['output'].to(device)

            preds = {}
            for name, model in models.items():
                preds[name] = model(x)[0].cpu().numpy()  # (2, H, W)

            y_np = y[0].cpu().numpy()

            def dn(arr, key):
                return ds._denorm(arr, key) if ds.normalize else arr

            truth = {
                'pressure':   dn(y_np[0],  'pressure'),
                'saturation': dn(y_np[1],  'saturation'),
            }
            pred_phys = {
                name: {
                    'pressure':   dn(preds[name][0], 'pressure'),
                    'saturation': dn(preds[name][1], 'saturation'),
                }
                for name in names
            }

            fig, axes = plt.subplots(n_rows, n_cols,
                                      figsize=(panel_w * n_cols, panel_h * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, n_cols)


            panel_idx = 0
            row_i = 0

            for vk, vtitle, unit in zip(var_keys, var_titles, var_units):
                for name in names:
                    pv = pred_phys[name][vk]
                    err = np.abs(truth[vk] - pv)
                    mae = err.mean()
                    denom = np.abs(truth[vk]).mean()
                    rel = mae / (denom + 1e-8) * 100

                    vmin_kw = {'vmin': 0, 'vmax': 1} if vk == 'saturation' else {}

                    # --- 真值 ---
                    ax = axes[row_i, 0]
                    im = ax.imshow(truth[vk], cmap=cmaps[vk], origin='lower', **vmin_kw)
                    ax.set_title(f'Ground Truth — {vtitle}', fontsize=11, fontweight='bold')
                    ax.text(0.5, -0.1, f'({chr(ord("a") + panel_idx)})',
                            transform=ax.transAxes, fontsize=11, fontweight='bold',
                            ha='center', va='top')
                    ax.axis('off')
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=8)
                    panel_idx += 1

                    # --- 预测 ---
                    ax = axes[row_i, 1]
                    im = ax.imshow(pv, cmap=cmaps[vk], origin='lower', **vmin_kw)
                    ax.set_title(f'{name} — {vtitle}\nMAE={mae:.2e}{unit}  Rel={rel:.2f}%',
                                 fontsize=10, fontweight='bold')
                    ax.text(0.5, -0.1, f'({chr(ord("a") + panel_idx)})',
                            transform=ax.transAxes, fontsize=11, fontweight='bold',
                            ha='center', va='top')
                    ax.axis('off')
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=8)
                    panel_idx += 1

                    # --- 误差 ---
                    ax = axes[row_i, 2]
                    p95 = np.percentile(err, 95)
                    vmax = p95 if p95 > 0 else (err.max() if err.max() > 0 else 1e-8)
                    im = ax.imshow(err, cmap='plasma', origin='lower', vmin=0, vmax=vmax)
                    ax.set_title(f'{name} — |Error|', fontsize=10, fontweight='bold')
                    ax.text(0.5, -0.1, f'({chr(ord("a") + panel_idx)})',
                            transform=ax.transAxes, fontsize=11, fontweight='bold',
                            ha='center', va='top')
                    ax.axis('off')
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=8)
                    panel_idx += 1

                    row_i += 1

            plt.tight_layout(rect=[0, 0, 1, 0.97])



            path = os.path.join(output_dir, f'pred_compare_sample{bi + 1}.png')
            fig.savefig(path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"  Saved → {path}")

def plot_metrics_radar(results: Dict[str, Dict], output_dir: str):
    """
    雷达图：5 个维度对比（压力相对误差 / 饱和度相对误差 /
    综合误差 / 推断时间 / 参数量）。
    每项归一化为 [0,1]，值越小越好。
    """
    import matplotlib.patches as mpatches
    os.makedirs(output_dir, exist_ok=True)
    names  = list(results.keys())
    colors = ['#85B7EB', '#E24B4A', '#1D9E75']

    keys = ['pressure_rel_err', 'saturation_rel_err',
            'combined_rel_err', 'inference_ms', 'n_params']
    labels = ['Pressure\nRel Err', 'Saturation\nRel Err',
              'Combined\nRel Err', 'Inference\nTime', 'Param\nCount']

    # 归一化（值越大越差 → 归一化后越接近1越差）
    raw = np.array([[results[n].get(k, 0) for k in keys] for n in names],
                   dtype=float)
    col_max = raw.max(axis=0) + 1e-12
    normed  = raw / col_max      # [0,1]，越小越好

    N = len(keys)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7),
                            subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7)
    ax.set_title('Model comparison radar\n(smaller area = better)',
                 fontsize=12, pad=18)

    patches = []
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = normed[i].tolist() + normed[i][:1].tolist()
        ax.plot(angles, vals, color=color, lw=2)
        ax.fill(angles, vals, color=color, alpha=0.15)
        patches.append(mpatches.Patch(color=color, label=name))

    ax.legend(handles=patches, loc='upper right',
              bbox_to_anchor=(1.35, 1.15), fontsize=10)
    ax.text(0.5, -0.08, '(a)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', ha='center', va='top')
    plt.tight_layout()
    path = os.path.join(output_dir, 'radar_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 汇总打印 + JSON 保存
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(results: Dict[str, Dict]):
    w = 14
    header = f"{'Metric':<26}" + "".join(f"{n:>{w}}" for n in results)
    sep    = "─" * len(header)
    print(f"\n{'='*len(header)}")
    print("MODEL COMPARISON SUMMARY")
    print(sep)
    print(header)
    print(sep)

    rows = [
        ('n_params',           'Parameters',      '{:>12,.0f}'),
        ('pressure_mae',       'Pressure MAE(Pa)', '{:>12.3e}'),
        ('pressure_rel_err',   'Pressure Rel(%)',  '{:>12.3f}'),
        ('saturation_mae',     'Saturation MAE',   '{:>12.5f}'),
        ('saturation_rel_err', 'Saturation Rel(%)','{:>12.3f}'),
        ('combined_rel_err',   'Combined Rel(%)',  '{:>12.3f}'),
        ('inference_ms',       'Infer.(ms/batch)', '{:>12.3f}'),
    ]

    for key, label, fmt in rows:
        vals = [results[n].get(key, 0) for n in results]
        best = min(range(len(vals)), key=lambda i: vals[i])
        row  = f"{label:<26}"
        for i, v in enumerate(vals):
            cell = fmt.format(v)
            row += f"{'★ '+cell if i == best else cell:>{w}}"
        print(row)

    print(sep)
    winner = min(results, key=lambda n: results[n]['combined_rel_err'])
    print(f"Best combined accuracy: {winner}")
    print('='*len(header))


# ═════════════════════════════════════════════════════════════════════════════
# 主函数
# ═════════════════════════════════════════════════════════════════════════════

def run(args):
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Device: {device}")

    # ── 数据 ─────────────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_loader, test_loader = create_data_loaders(
        args.h5,
        batch_size=args.batch,
        train_test_split=0.8,
        num_workers=args.workers,
    )
    # 取 dataset 用于反归一化
    train_ds = train_loader.dataset

    # ── 模型定义 ──────────────────────────────────────────────────────────────
    models_def = {
        'FNO':      FNO2d(modes1=args.modes, modes2=args.modes, width=args.width),
        'U-FNO':    UFNO2d(modes1=args.modes, modes2=args.modes, width=args.width),
        'UNet-FNO': UNetFNO2d(modes1=args.modes, modes2=args.modes, width=args.width,
                               n_fno_layers=args.depth),
    }

    for name, m in models_def.items():
        n = m.count_params()
        print(f"  {name:<12}  {n:>10,} params")

    # ── 训练 / 加载 ──────────────────────────────────────────────────────────
    trainers: Dict[str, Trainer] = {}
    results:  Dict[str, Dict]    = {}

    for name, model in models_def.items():
        ckpt = os.path.join(args.outdir, f'best_{name.lower().replace("-","_")}.pth')

        if args.eval_only and os.path.exists(ckpt):
            print(f"\n[{name}] Loading checkpoint: {ckpt}")
            trainer = Trainer(model, train_loader, test_loader,
                              device=device, model_name=name,
                              output_dir=args.outdir)
            trainer.load_model(ckpt)
        else:
            print(f"\n{'='*60}\nTraining {name}\n{'='*60}")
            trainer = Trainer(
                model, train_loader, test_loader,
                device=device,
                learning_rate=args.lr,
                weight_decay=args.wd,
                model_name=name,
                output_dir=args.outdir,
            )
            trainer.train(num_epochs=args.epochs,
                          early_stopping_patience=args.patience)
            trainer.save_model(
                os.path.join(args.outdir,
                             f'final_{name.lower().replace("-","_")}.pth'))

        trainers[name] = trainer

        # 评估
        print(f"\n[{name}] Evaluating...")
        metrics = evaluate_model(model, test_loader, train_ds, device)
        metrics['n_params'] = model.count_params()
        results[name] = metrics
        print(f"  P-MAE={metrics['pressure_mae']:.3e}  "
              f"P-Rel={metrics['pressure_rel_err']:.2f}%  "
              f"S-MAE={metrics['saturation_mae']:.5f}  "
              f"S-Rel={metrics['saturation_rel_err']:.2f}%  "
              f"Infer={metrics['inference_ms']:.2f}ms")

    # ── 汇总输出 ─────────────────────────────────────────────────────────────
    print_summary(results)

    json_path = os.path.join(args.outdir, 'comparison_results.json')
    with open(json_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f"\nResults saved → {json_path}")

    # ── 可视化 ────────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_comparison_bar(results,   output_dir=args.outdir)
    plot_loss_curves(trainers,     output_dir=args.outdir)
    plot_prediction_comparison(
        models_def, test_loader, train_ds, device,
        output_dir=args.outdir, n_samples=args.vis_samples)
    plot_metrics_radar(results,    output_dir=args.outdir)

    print(f"\nAll done. Outputs → {args.outdir}/")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and compare FNO / U-FNO / UNet-FNO')

    # 数据
    parser.add_argument('--h5',      required=True, help='HDF5 数据集路径')
    parser.add_argument('--outdir',  default='./compare', help='输出目录')
    parser.add_argument('--workers', type=int, default=4)

    # 模型超参
    parser.add_argument('--modes',   type=int,   default=8,    help='Fourier 模式数')
    parser.add_argument('--width',   type=int,   default=32,   help='隐藏通道数')
    parser.add_argument('--depth',   type=int,   default=4,    help='UNet-FNO 的 FNO 层数')

    # 训练超参
    parser.add_argument('--epochs',  type=int,   default=100)
    parser.add_argument('--batch',   type=int,   default=128)
    parser.add_argument('--lr',      type=float, default=1e-3)
    parser.add_argument('--wd',      type=float, default=1e-4)
    parser.add_argument('--patience',type=int,   default=20,   help='早停耐心轮数')

    # 其他
    parser.add_argument('--eval_only',   action='store_true',
                        help='跳过训练，直接从 checkpoint 加载评估')
    parser.add_argument('--vis_samples', type=int, default=2,
                        help='可视化对比的测试样本数')

    args = parser.parse_args()
    run(args)
