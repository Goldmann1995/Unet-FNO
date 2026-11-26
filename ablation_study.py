#!/usr/bin/env python3
# =============================================================================
# ablation_study.py  —  消融实验
#
# 基准模型（V1-Full）与原始 UNetFNO2d 完全一致：
#   - 单个 AdaptiveUNet 实例，所有 FNO 层共享（self.unet = AdaptiveUNet(...)）
#   - GroupNorm + 残差连接 + depth=3
#
# 六个变体（从完整模型逐步去除或替换一个组件）：
#   V1  Full              原始完整模型（共享 UNet，基准）
#   V2  w/o UNet          去掉 UNet，退化为纯 FNO                 → 验证 A：UNet 整体贡献
#   V3  w/o Residual      去掉 UNet 内残差连接（仍共享）            → 验证 C：残差连接贡献
#   V4  w/o Bias          去掉逐层可学习偏置 b_i（仍有 UNet）       → 验证 F：bias 项贡献
#   V5  depth=1           UNet 深度 3→1（仍共享）                  → 验证 D：多尺度深度
#   V6  Independent UNet  每层独立 UNet（不共享参数）               → 验证 E：共享的参数效率

#
# 输出：
#   checkpoints/   各变体的 best/final checkpoint
#   figures/       条形图、损失曲线、贡献热力图、雷达图、预测对比网格
#   report/        ablation_summary.txt + ablation_results.json
#
# 用法：
#   python ablation_study.py --h5 dataset.h5
#   python ablation_study.py --h5 dataset.h5 --epochs 50 --batch 16
#   python ablation_study.py --h5 dataset.h5 --eval_only
#   python ablation_study.py --h5 dataset.h5 --variants V1-Full V2-w/o_UNet
# =============================================================================

import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_training import (
    LpLoss, create_data_loaders, Trainer,
    SpectralConv2d,
)
from compare_models import evaluate_model
torch.manual_seed(42)
np.random.seed(42)

# ═════════════════════════════════════════════════════════════════════════════
#  消融变体模型定义
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 公共构件（与 data_training.py 保持完全一致，在此独立定义以便消融替换）
# ─────────────────────────────────────────────────────────────────────────────

class _EncGN(nn.Module):
    """编码块：GroupNorm（原版）"""
    def __init__(self, ic, oc, downsample=True):
        super().__init__()
        ng = min(32, oc)
        self.encode = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1),
            nn.GroupNorm(ng, oc), nn.GELU(),
            nn.Conv2d(oc, oc, 3, padding=1),
            nn.GroupNorm(ng, oc), nn.GELU(),
        )
        self.pool = nn.AvgPool2d(2, 2) if downsample else None

    def forward(self, x):
        if self.pool is not None: x = self.pool(x)
        return self.encode(x)


class _EncBN(nn.Module):
    """编码块：BatchNorm（消融 B）"""
    def __init__(self, ic, oc, downsample=True):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1),
            nn.BatchNorm2d(oc), nn.GELU(),
            nn.Conv2d(oc, oc, 3, padding=1),
            nn.BatchNorm2d(oc), nn.GELU(),
        )
        self.pool = nn.AvgPool2d(2, 2) if downsample else None

    def forward(self, x):
        if self.pool is not None: x = self.pool(x)
        return self.encode(x)


class _Dec(nn.Module):
    """解码块：双线性上采样 + skip concat（GroupNorm）"""
    def __init__(self, ic, sc, oc):
        super().__init__()
        ng = min(32, oc)
        self.decode = nn.Sequential(
            nn.Conv2d(ic + sc, oc, 3, padding=1),
            nn.GroupNorm(ng, oc), nn.GELU(),
            nn.Conv2d(oc, oc, 3, padding=1),
            nn.GroupNorm(ng, oc), nn.GELU(),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                          align_corners=False)
        return self.decode(torch.cat([x, skip], dim=1))


class _DecBN(nn.Module):
    """解码块：BatchNorm（消融 B）"""
    def __init__(self, ic, sc, oc):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(ic + sc, oc, 3, padding=1),
            nn.BatchNorm2d(oc), nn.GELU(),
            nn.Conv2d(oc, oc, 3, padding=1),
            nn.BatchNorm2d(oc), nn.GELU(),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                          align_corners=False)
        return self.decode(torch.cat([x, skip], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# UNet 变体工厂（enc_cls / dec_cls 可注入）
# ─────────────────────────────────────────────────────────────────────────────

def _build_unet(in_ch, out_ch, base_ch, depth,
                use_residual: bool, enc_cls, dec_cls):
    """
    返回一个按指定配置构建的 UNet nn.Module。
    use_residual=False 时去掉残差连接（消融 C）。
    enc_cls / dec_cls 控制归一化方式（消融 B）。
    """

    class _UNetVariant(nn.Module):
        def __init__(self):
            super().__init__()
            enc_channels = [in_ch]
            ch = base_ch
            self.encoders = nn.ModuleList()
            for d in range(depth):
                self.encoders.append(enc_cls(enc_channels[-1], ch,
                                             downsample=(d > 0)))
                enc_channels.append(ch)
                ch = min(ch * 2, 256)

            bt_ng = min(32, ch)
            self.bottleneck = nn.Sequential(
                nn.Conv2d(enc_channels[-1], ch, 3, padding=1),
                nn.GroupNorm(bt_ng, ch), nn.GELU())

            self.decoders = nn.ModuleList()
            dec_in = ch
            for d in reversed(range(depth)):
                sc = enc_channels[d + 1]; oc = enc_channels[d + 1]
                self.decoders.append(dec_cls(dec_in, sc, oc))
                dec_in = oc

            self.output_conv = nn.Conv2d(enc_channels[1], out_ch, 1)
            if use_residual:
                self.res_proj = (nn.Conv2d(in_ch, out_ch, 1)
                                 if in_ch != out_ch else nn.Identity())
            self._use_res = use_residual

        def forward(self, x):
            res = self.res_proj(x) if self._use_res else None
            feats, h = [], x
            for enc in self.encoders:
                h = enc(h); feats.append(h)
            h = self.bottleneck(h)
            for dec, skip in zip(self.decoders, reversed(feats)):
                h = dec(h, skip)
            out = self.output_conv(h)
            return out + res if self._use_res else out

    return _UNetVariant()


# ─────────────────────────────────────────────────────────────────────────────
# 统一的 UNetFNO 骨架（接受 unet_factory 函数或 None）
# ─────────────────────────────────────────────────────────────────────────────

class _UNetFNOVariant(nn.Module):
    """
    可配置的 UNet-FNO 变体骨架。
    unet_factory : None  → 不使用 UNet（纯 FNO，消融 A）
                  callable → 每层独立调用一次创建 UNet
    shared_unet  : True → 所有层共享同一 UNet 实例（消融 E：共享参数）
    """

    def __init__(self, modes1, modes2, width, n_layers,
                 unet_factory=None, shared_unet=False,use_bias=True,
                 in_channels=4, out_channels=2):
        super().__init__()
        self.width   = width
        self.n_layers = n_layers
        self.use_bias = use_bias
        self.fc0 = nn.Linear(in_channels, width)
        self.convs = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)])
        self.ws    = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        # 可学习偏置：shape (n_layers, 1, width, 1, 1)，广播到全空间
        if use_bias:
            self.biases = nn.Parameter(torch.zeros(n_layers, 1, width, 1, 1))
        else:
            self.biases = None
        # UNet 初始化
        self._has_unet = unet_factory is not None
        if self._has_unet:
            if shared_unet:
                unet = unet_factory()
                self.unets = nn.ModuleList([unet] * n_layers)   # 共享同一实例
            else:
                # 每层独立 UNet（但权重不共享 — 需要创建独立实例）
                self.unets = nn.ModuleList([unet_factory() for _ in range(n_layers)])

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.fc0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        for i, (conv, w) in enumerate(zip(self.convs, self.ws)):
            if self._has_unet:
                x = self.unets[i](x)
            xw = w(x.view(B, self.width, -1)).view(B, self.width, H, W)
            if self.use_bias:
                x = conv(x) + xw + self.biases[i]
            else:
                x = conv(x) + xw
            if i < self.n_layers - 1:
                x = F.gelu(x)

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).permute(0, 3, 1, 2)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# 六个消融变体工厂函数
# ─────────────────────────────────────────────────────────────────────────────

def build_ablation_models(modes=8, width=32, n_layers=4,
                           base_ch=32) -> Dict[str, _UNetFNOVariant]:
    """
    返回消融实验的六个模型（key = 变体名称）。

    基准模型（V1-Full）与原始 UNetFNO2d 代码完全一致：
      self.unet = AdaptiveUNet(...)   # 单个实例
      for i, (conv, w) in ...:
          x = self.unet(x)            # 每层调用同一 UNet（shared）
    即 shared_unet=True，所有 FNO 层共享同一 AdaptiveUNet。

    消融方向：去掉或替换单个组件，量化各组件贡献。
    """

    # ── UNet 配置工厂 ──────────────────────────────────────────────────────────
    def _full_unet():           # GroupNorm + 残差 + depth=3（标准配置）
        return _build_unet(width, width, base_ch, depth=3,
                           use_residual=True, enc_cls=_EncGN, dec_cls=_Dec)

    def _shallow_unet():        # GroupNorm + 残差 + depth=1（浅层）
        return _build_unet(width, width, base_ch, depth=1,
                           use_residual=True, enc_cls=_EncGN, dec_cls=_Dec)

    def _no_res_unet():         # GroupNorm + 无残差
        return _build_unet(width, width, base_ch, depth=3,
                           use_residual=False, enc_cls=_EncGN, dec_cls=_Dec)

    def _bn_unet():             # BatchNorm + 残差
        return _build_unet(width, width, base_ch, depth=3,
                           use_residual=True, enc_cls=_EncBN, dec_cls=_DecBN)

    kw = dict(modes1=modes, modes2=modes, width=width, n_layers=n_layers)

    models = {
        # ── V1：完整模型（基准）─────────────────────────────────────────────
        # 与原始 UNetFNO2d 完全一致：单个 AdaptiveUNet，所有层共享（shared=True）
        'V1-Full':
            _UNetFNOVariant(**kw, unet_factory=_full_unet,   shared_unet=True ,use_bias=True),

        # ── V2：去掉 UNet，退化为纯 FNO ──────────────────────────────────────
        # 验证 A：AdaptiveUNet 整体多尺度特征提取的贡献
        'V2-w/o UNet':
            _UNetFNOVariant(**kw, unet_factory=None,          shared_unet=False, use_bias=False),

        # ── V3：UNet 内去掉残差连接 ───────────────────────────────────────────
        # 验证 C：残差连接防止信息丢失的贡献（其余与 V1 相同，含 shared）
        'V3-w/o Residual':
            _UNetFNOVariant(**kw, unet_factory=_no_res_unet,  shared_unet=True, use_bias=True),

        # ── V4：去掉逐层可学习偏置 b_i ───────────────────────────────────────
        # 验证 F：bias 项对整体精度的独立贡献
        # V4 与 V1 的唯一区别是去掉 self.biases（128 个参数），其余完全相同
        # 原始 FNO 无此 bias；V4 = "标准 FNO 层 + UNet"，是最公平的 UNet 贡献基准
        'V4-w/o Bias':
            _UNetFNOVariant(**kw, unet_factory=_full_unet,    shared_unet=True,
                            use_bias=False),
        # ── V5：UNet depth=1（浅层） ─────────────────────────────────────────
        # 验证 D：多尺度深度（depth=3 vs depth=1）的影响（含 shared）
        'V5-depth=1':
            _UNetFNOVariant(**kw, unet_factory=_shallow_unet, shared_unet=True, use_bias=True),

        # ── V6：每层独立 UNet（不共享参数） ──────────────────────────────────
        # 验证 E：共享单个 UNet（V1 原始设计）vs 各层独立 UNet 的差异
        # 独立版参数量约为 V1 的 n_layers 倍，验证共享是否已足够
        'V6-Independent UNet':
            _UNetFNOVariant(**kw, unet_factory=_full_unet,    shared_unet=False, use_bias=True),


    }
    return models


# ═════════════════════════════════════════════════════════════════════════════
#  可视化
# ═════════════════════════════════════════════════════════════════════════════

def plot_ablation_bar(results: Dict[str, Dict], output_dir: str):
    """四指标条形图，V1（完整模型）用金色标注基准。"""
    os.makedirs(output_dir, exist_ok=True)
    names = list(results.keys())

    # 调色：V1 金色基准，其余蓝色系渐变
    palette = ['#F5A623'] + [plt.cm.Blues(0.4 + 0.12 * i)
                              for i in range(len(names) - 1)]

    metrics = [
        ('pressure_mae',       'Pressure MAE (Pa)',        'Pressure MAE'),
        ('pressure_rel_err',   'Pressure Rel. Error (%)',  'Pressure Rel. Error'),
        ('saturation_mae',     'Saturation MAE',           'Saturation MAE'),
        ('saturation_rel_err', 'Saturation Rel. Error (%)', 'Saturation Rel. Error'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Ablation Study — Component Contribution',
                 fontsize=14, fontweight='bold')

    for ax, (key, ylabel, title) in zip(axes.flatten(), metrics):
        vals  = [results[n][key] for n in names]
        best  = min(range(len(vals)), key=lambda i: vals[i])
        bars  = ax.bar(names, vals, color=palette,
                       edgecolor='white', linewidth=0.8)

        # 标注数值
        for i, (bar, v) in enumerate(zip(bars, vals)):
            lbl = f'{v:.3e}' if (v > 1e3 or v < 0.01) else f'{v:.4f}'
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    lbl, ha='center', va='bottom', fontsize=8,
                    fontweight='bold' if i == best else 'normal')

        # V1 基准水平线
        v1_val = results[names[0]][key]
        ax.axhline(v1_val, color='#F5A623', lw=1.5, ls='--', alpha=0.8,
                   label='V1 baseline')

        # 最优值标星
        bars[best].set_edgecolor('#E24B4A')
        bars[best].set_linewidth(2.5)

        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(fontsize=8)

    plt.tight_layout()
    p = os.path.join(output_dir, 'ablation_bar.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {p}')


def plot_ablation_loss_curves(trainers: Dict[str, Trainer],
                               output_dir: str):
    """所有变体的训练 / 测试损失曲线在同一图中。"""
    os.makedirs(output_dir, exist_ok=True)
    names   = list(trainers.keys())
    palette = ['#F5A623', '#1D9E75', '#E24B4A', '#534AB7',
               '#BA7517', '#0F6E56', '#712B13']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Ablation Loss Curves', fontsize=13, fontweight='bold')

    for i, (name, trainer) in enumerate(trainers.items()):
        c  = palette[i % len(palette)]
        ls = '-' if i == 0 else '--'
        axes[0].plot(trainer.train_losses, label=name, color=c, lw=2, ls=ls)
        axes[1].plot(trainer.test_losses,  label=name, color=c, lw=2, ls=ls)

    for ax, title in zip(axes, ['Training Loss', 'Test Loss']):
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Relative L2 Loss', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(output_dir, 'ablation_loss_curves.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {p}')


def plot_contribution_heatmap(results: Dict[str, Dict], output_dir: str):
    """
    热力图：以 V1 为基准，展示各变体在各指标上的相对性能退化（%）。
    颜色越红 = 去掉该组件后性能下降越多 = 该组件贡献越大。
    """
    os.makedirs(output_dir, exist_ok=True)
    names    = list(results.keys())
    metrics  = ['pressure_rel_err', 'saturation_rel_err',
                 'combined_rel_err', 'inference_ms']
    mlabels  = ['P-RelErr', 'S-RelErr', 'Combined', 'Infer(ms)']

    v1    = results[names[0]]
    delta = np.zeros((len(names), len(metrics)))
    for i, n in enumerate(names):
        for j, m in enumerate(metrics):
            base = v1[m] + 1e-10
            delta[i, j] = (results[n][m] - base) / base * 100.0

    # 对 inference_ms 取绝对值（速度更快也是好事）
    delta[:, 3] = np.abs(delta[:, 3])

    fig, ax = plt.subplots(figsize=(9, 6))
    vmax = max(1.0, np.abs(delta).max())
    im   = ax.imshow(delta, cmap='RdYlGn_r', vmin=-vmax, vmax=vmax,
                     aspect='auto')
    plt.colorbar(im, ax=ax, label='Δ vs V1-Full (%)')

    ax.set_xticks(range(len(metrics)));  ax.set_xticklabels(mlabels, fontsize=10)
    ax.set_yticks(range(len(names)));    ax.set_yticklabels(names,    fontsize=9)
    ax.set_title('Component contribution heatmap\n'
                 '(red = removing component causes larger degradation)',
                 fontsize=11)

    for i in range(len(names)):
        for j in range(len(metrics)):
            ax.text(j, i, f'{delta[i,j]:+.1f}%',
                    ha='center', va='center', fontsize=8,
                    color='white' if abs(delta[i,j]) > vmax * 0.6 else 'black')

    plt.tight_layout()
    p = os.path.join(output_dir, 'ablation_heatmap.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {p}')


def plot_ablation_radar(results: Dict[str, Dict], output_dir: str):
    """雷达图（5 维），越小越好，归一化至 [0,1]。"""
    import matplotlib.patches as mpatches
    os.makedirs(output_dir, exist_ok=True)
    names   = list(results.keys())
    palette = ['#F5A623', '#1D9E75', '#E24B4A', '#534AB7',
               '#BA7517', '#0F6E56', '#712B13']

    keys   = ['pressure_rel_err', 'saturation_rel_err',
               'combined_rel_err', 'inference_ms', 'n_params']
    labels = ['P-RelErr', 'S-RelErr', 'Combined', 'Infer-Time', 'Params']

    raw    = np.array([[results[n].get(k, 0) for k in keys]
                        for n in names], dtype=float)
    normed = raw / (raw.max(axis=0) + 1e-12)

    N      = len(keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Ablation radar (smaller = better)',
                 fontsize=12, pad=18)

    patches = []
    for i, name in enumerate(names):
        c    = palette[i % len(palette)]
        vals = normed[i].tolist() + normed[i][:1].tolist()
        lw   = 2.5 if i == 0 else 1.5
        ax.plot(angles, vals, color=c, lw=lw)
        ax.fill(angles, vals, color=c, alpha=0.08 if i > 0 else 0.18)
        patches.append(mpatches.Patch(color=c, label=name))

    ax.legend(handles=patches, loc='upper right',
              bbox_to_anchor=(1.45, 1.15), fontsize=9)
    plt.tight_layout()
    p = os.path.join(output_dir, 'ablation_radar.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {p}')


def plot_prediction_grid(models_dict: Dict[str, nn.Module],
                          test_loader, dataset, device: str,
                          output_dir: str, n_samples: int = 2):
    """
    对 n_samples 个测试样本，展示所有变体的压力和饱和度预测。
    布局：每行一个变体，前两列=压力(真值/预测)，后两列=饱和度(真值/预测)
    """
    os.makedirs(output_dir, exist_ok=True)
    names = list(models_dict.keys())
    for m in models_dict.values(): m.eval()
    ds = dataset

    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            if bi >= n_samples: break
            x = batch['input'].to(device)
            y = batch['output'].to(device)

            # 反归一化
            def dn(arr, k): return ds._denorm(arr, k) if ds.normalize else arr

            truth = {
                'p':  dn(y[0, 0].cpu().numpy(), 'pressure'),
                'sw': dn(y[0, 1].cpu().numpy(), 'saturation'),
            }

            n_rows = len(names) + 1    # 第一行 = 真值
            n_cols = 4                 # p_true / p_pred / sw_true / sw_pred
            fig, axes = plt.subplots(n_rows, n_cols,
                                      figsize=(4 * n_cols, 3.2 * n_rows))
            fig.suptitle(f'Ablation prediction grid — test sample {bi+1}',
                         fontsize=12, fontweight='bold', y=1.01)

            def _show(ax, data, cmap, title, vmin=None, vmax=None):
                im = ax.imshow(data, cmap=cmap, origin='lower',
                               vmin=vmin, vmax=vmax)
                ax.set_title(title, fontsize=8)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # 行 0：真值
            _show(axes[0, 0], truth['p'],  'hot',      'Pressure (true)')
            _show(axes[0, 1], truth['p'],  'hot',      'Pressure (true)')
            _show(axes[0, 2], truth['sw'], 'coolwarm', 'Saturation (true)',
                  vmin=0, vmax=1)
            _show(axes[0, 3], truth['sw'], 'coolwarm', 'Saturation (true)',
                  vmin=0, vmax=1)
            axes[0, 0].set_title('Ground Truth — Pressure', fontsize=9)
            axes[0, 2].set_title('Ground Truth — Saturation', fontsize=9)
            axes[0, 1].axis('off'); axes[0, 3].axis('off')

            for ri, name in enumerate(names):
                model = models_dict[name]
                pred  = model(x)[0].cpu().numpy()
                pp    = dn(pred[0], 'pressure')
                ps    = dn(pred[1], 'saturation')

                p_mae  = np.abs(truth['p']  - pp).mean()
                s_mae  = np.abs(truth['sw'] - ps).mean()

                _show(axes[ri+1, 0], pp, 'hot',
                      f'{name}\nP-MAE={p_mae:.2e}')
                _show(axes[ri+1, 1], np.abs(truth['p'] - pp), 'plasma',
                      f'P error')
                _show(axes[ri+1, 2], ps, 'coolwarm',
                      f'{name}\nS-MAE={s_mae:.4f}', vmin=0, vmax=1)
                _show(axes[ri+1, 3], np.abs(truth['sw'] - ps), 'plasma',
                      f'S error')

            plt.tight_layout()
            p = os.path.join(output_dir, f'ablation_pred_sample{bi+1}.png')
            fig.savefig(p, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'  Saved → {p}')


# ═════════════════════════════════════════════════════════════════════════════
#  汇总打印 + 报告保存
# ═════════════════════════════════════════════════════════════════════════════

def print_ablation_summary(results: Dict[str, Dict]):
    w  = 16
    hd = f"{'Variant':<20}" + ''.join(f"{n:>{w}}" for n in results)
    sep = '─' * len(hd)

    print(f'\n{"="*len(hd)}')
    print('ABLATION STUDY SUMMARY')
    print(sep); print(hd); print(sep)

    rows = [
        ('n_params',           'Parameters',      '{:>14,.0f}'),
        ('pressure_mae',       'P-MAE (Pa)',       '{:>14.3e}'),
        ('pressure_rel_err',   'P-RelErr (%)',     '{:>14.3f}'),
        ('saturation_mae',     'S-MAE',            '{:>14.5f}'),
        ('saturation_rel_err', 'S-RelErr (%)',     '{:>14.3f}'),
        ('combined_rel_err',   'Combined (%)',     '{:>14.3f}'),
        ('inference_ms',       'Infer (ms)',       '{:>14.3f}'),
    ]

    names = list(results.keys())
    for key, label, fmt in rows:
        vals = [results[n].get(key, 0) for n in names]
        best = min(range(len(vals)), key=lambda i: vals[i])
        row  = f'{label:<20}'
        for i, v in enumerate(vals):
            cell = fmt.format(v)
            row += f"{'★'+cell if i == best else cell:>{w}}"
        print(row)

    print(sep)
    # 按综合误差排名
    ranked = sorted(names, key=lambda n: results[n]['combined_rel_err'])
    print('Ranking (by combined RelErr):')
    for rank, n in enumerate(ranked, 1):
        print(f'  {rank}. {n:<20}  {results[n]["combined_rel_err"]:.3f}%')

    # 组件贡献分析
    print('\nComponent contribution (degradation vs V1-Full):')
    v1 = results[names[0]]
    contrib_map = {
        'V2-w/o UNet':          'A. AdaptiveUNet (multi-scale)',
        'V3-w/o Residual':      'C. Residual connection in UNet',
        'V4-w/o Bias':          'B. Learnable bias in FNO layers',
        'V5-depth=1':           'D. UNet depth  (3 vs 1)',
        'V6-Independent UNet':  'E. Shared UNet (V1) vs Independent',
    }
    for vname, desc in contrib_map.items():
        if vname in results:
            delta = results[vname]['combined_rel_err'] - v1['combined_rel_err']
            print(f'  {desc:<38}  Δ={delta:+.3f}%')
    print('=' * len(hd))


def save_report(args, results, output_dir, total_time):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, 'ablation_results.json')
    with open(json_path, 'w') as fp:
        json.dump({'config': vars(args), 'results': results,
                   'total_time': total_time}, fp, indent=2)

    txt_path = os.path.join(output_dir, 'ablation_summary.txt')
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_ablation_summary(results)
    with open(txt_path, 'w') as fp:
        fp.write(buf.getvalue())

    print(f'\nFull results → {json_path}')
    print(f'Text summary → {txt_path}')


# ═════════════════════════════════════════════════════════════════════════════
#  主流程
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Ablation study for UNet-FNO components')
    p.add_argument('--h5',          required=True,
                   help='HDF5 数据集路径')
    p.add_argument('--outdir',      default='./ablation',
                   help='输出根目录')
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch',       type=int,   default=128)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--wd',          type=float, default=1e-4)
    p.add_argument('--patience',    type=int,   default=20)
    p.add_argument('--workers',     type=int,   default=12)
    p.add_argument('--modes',       type=int,   default=8)
    p.add_argument('--width',       type=int,   default=32)
    p.add_argument('--depth',       type=int,   default=4,
                   help='FNO 层数')
    p.add_argument('--eval_only',   action='store_true',
                   help='跳过训练，仅从 checkpoint 加载并评估')
    p.add_argument('--vis_samples', type=int,   default=2,
                   help='预测对比图中展示的测试样本数')
    p.add_argument('--variants',    nargs='+',  default=None,
                   help='仅跑指定变体，如 --variants V1-Full V2-w/o_UNet')
    return p.parse_args()


def main():
    args  = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_dir = os.path.join(args.outdir, 'checkpoints')
    fig_dir  = os.path.join(args.outdir, 'figures')
    rep_dir  = os.path.join(args.outdir, 'report')
    for d in (ckpt_dir, fig_dir, rep_dir): os.makedirs(d, exist_ok=True)

    t_start = time.time()
    print(f'\n{"#"*60}')
    print(f'  Ablation Study — UNet-FNO Component Analysis')
    print(f'  Device: {device}')
    print(f'  Output: {os.path.abspath(args.outdir)}')
    print(f'{"#"*60}')

    # ── 数据 ─────────────────────────────────────────────────────────────────
    print('\nLoading data...')
    train_loader, test_loader = create_data_loaders(
        args.h5, batch_size=args.batch,
        train_test_split=0.8, num_workers=args.workers)
    train_ds = train_loader.dataset

    # ── 模型 ─────────────────────────────────────────────────────────────────
    all_models = build_ablation_models(
        modes=args.modes, width=args.width,
        n_layers=args.depth, base_ch=32)

    # 如果指定了部分变体，过滤
    if args.variants:
        # 支持下划线或空格作为变体名中的连字符替代
        req = [v.replace('_', ' ') for v in args.variants]
        all_models = {k: v for k, v in all_models.items()
                      if k in req or k.replace('/', '_') in args.variants}

    print(f'\nVariants to run: {list(all_models.keys())}')
    print('\nParameter counts:')
    for name, m in all_models.items():
        print(f'  {name:<22}  {m.count_params():>12,} params')

    # ── 训练 + 评估 ───────────────────────────────────────────────────────────
    trainers: Dict[str, Trainer] = {}
    results:  Dict[str, Dict]    = {}

    for name, model in all_models.items():
        safe = name.lower().replace(' ', '_').replace('/', '_').replace('=', '')
        best_ckpt = os.path.join(ckpt_dir, f'best_{safe}.pth')

        trainer = Trainer(
            model, train_loader, test_loader,
            device=device,
            learning_rate=args.lr,
            weight_decay=args.wd,
            model_name=safe,
            output_dir=ckpt_dir,
        )

        if args.eval_only and os.path.exists(best_ckpt):
            print(f'\n  [{name}] Loading {best_ckpt}')
            trainer.load_model(best_ckpt)
        else:
            bar = '─' * 60
            print(f'\n{bar}\n  Training: {name}\n{bar}')
            trainer.train(num_epochs=args.epochs,
                          early_stopping_patience=args.patience)
            trainer.save_model(os.path.join(ckpt_dir, f'final_{safe}.pth'))

        trainers[name] = trainer

        print(f'  [{name}] Evaluating...')
        m_res = evaluate_model(model, test_loader, train_ds, device)
        m_res['n_params'] = model.count_params()
        results[name]     = m_res
        print(f'    P-Rel={m_res["pressure_rel_err"]:.2f}%  '
              f'S-Rel={m_res["saturation_rel_err"]:.2f}%  '
              f'Infer={m_res["inference_ms"]:.2f}ms')

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    print_ablation_summary(results)

    # ── 可视化 ────────────────────────────────────────────────────────────────
    print('\nGenerating figures...')
    plot_ablation_bar(results,       output_dir=fig_dir)
    plot_ablation_loss_curves(trainers, output_dir=fig_dir)
    plot_contribution_heatmap(results, output_dir=fig_dir)
    plot_ablation_radar(results,     output_dir=fig_dir)
    plot_prediction_grid(
        all_models, test_loader, train_ds, device,
        output_dir=fig_dir, n_samples=args.vis_samples)

    # ── 报告 ─────────────────────────────────────────────────────────────────
    total = _elapsed(t_start)
    save_report(args, results, rep_dir, total)

    print(f'\n{"="*60}')
    print(f'Ablation study complete.  Wall time: {total}')
    print(f'Figures  → {fig_dir}')
    print(f'Report   → {rep_dir}')
    print(f'{"="*60}')


def _elapsed(t0):
    s = time.time() - t0
    h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return f'{int(h)}h {int(m)}m {sec:.0f}s'


if __name__ == '__main__':
    main()
