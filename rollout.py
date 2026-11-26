#!/usr/bin/env python3
# =============================================================================
# rollout_test.py  —  多步递归预测测试
#
# 功能：
#   对 FNO / U-FNO / UNet-FNO 在测试集上做递归预测，
#   计算每步误差并生成误差积累曲线、场对比图、关键步数柱状图。
#
# 输出（--outdir 目录下）：
#   rollout_error_curves.png              误差积累曲线（均值 ± std）
#   rollout_mae_bar.png                   特定步数 MAE 柱状图
#   rollout_fields_pressure_sample0.png   压力场对比图
#   rollout_fields_saturation_sample0.png 饱和度场对比图
#   rollout_results.json                  所有数值结果
#
# 用法：
#   uv run rollout_test.py \
#       --h5       ./reservoir_fno_dataset_20x20/reservoir_fno_xxx.h5 \
#       --ckpt_dir ./experiments/checkpoints \
#       --steps 100 --n_samples 5 --outdir ./rollout
# =============================================================================

import os, sys, json, argparse
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_training import (
    ReservoirOperatorDataset, create_data_loaders,
    MultiStepPredictor, Trainer,
)
from ablation_study import build_ablation_models
from fno import FNO2d
from ufno import Net2d
# ── 论文字体 ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':    'serif',
    'font.serif':     ['Times New Roman', 'DejaVu Serif'],
    'font.size':       9,
    'axes.titlesize': 10,
    'axes.labelsize':  9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'savefig.dpi':   300,
    'savefig.bbox':  'tight',
})

COLORS = {
    'FNO':      '#85B7EB',
    'U-FNO':    '#E24B4A',
    'UNet-FNO': '#1D9E75',
}


# ═════════════════════════════════════════════════════════════════════════════
#  模型加载（与 run_all.py 保持一致）
# ═════════════════════════════════════════════════════════════════════════════

def load_models(args, train_loader, test_loader, device: str) -> Dict:
    """
    加载三个模型的 best checkpoint。
    UNet-FNO 取 ablation_study 的 V1-Full，保证架构与训练时完全一致。
    """
    all_ablation = build_ablation_models(
        modes=args.modes, width=args.width,
        n_layers=args.depth, base_ch=32)

    model_defs = {
        'FNO':      FNO2d(modes1=args.modes, modes2=args.modes,
                          width=args.width),
        'U-FNO':    Net2d(modes1=args.modes, modes2=args.modes,
                            width=args.width),
        'UNet-FNO': all_ablation['V1-Full'],
    }

    loaded = {}
    for name, model in model_defs.items():
        safe  = name.lower().replace('-', '_')
        ckpt  = os.path.join(args.ckpt_dir, f'best_{safe}.pth')
        if not os.path.exists(ckpt):
            print(f'  [SKIP] checkpoint not found: {ckpt}')
            continue

        trainer = Trainer(
            model, train_loader, test_loader,
            device=device, model_name=name,
            output_dir=args.ckpt_dir,
        )
        trainer.load_model(ckpt)
        model.eval()
        loaded[name] = model
        print(f'  [{name}] loaded  ({model.count_params():,} params)')

    return loaded


# ═════════════════════════════════════════════════════════════════════════════
#  测试样本加载（物理量，不归一化）
# ═════════════════════════════════════════════════════════════════════════════

def load_test_samples(h5_path: str, n_samples: int,
                       split: float = 0.8) -> List[Dict]:
    with h5py.File(h5_path, 'r') as f:
        all_keys = sorted(f['data'].keys())
        rng = np.random.default_rng(42)
        rng.shuffle(all_keys)
        cut       = int(len(all_keys) * split)
        test_keys = all_keys[cut:][:n_samples]

        samples = []
        for key in test_keys:
            sg     = f['data'][key]
            ny, nx = sg['permeability'].shape
            pres   = sg['pressure'][:].reshape(-1, ny, nx)
            sat    = sg['saturation'][:].reshape(-1, ny, nx)
            samples.append({
                'permeability': sg['permeability'][:],
                'porosity':     sg['porosity'][:],
                'pressure':     pres,
                'saturation':   sat,
                'time_steps':   sg['time_steps'][:],
                'sample_id':    int(sg.attrs.get('sample_id', 0)),
                'pattern':      str(sg.attrs.get('pattern', 'unknown')),
            })

    print(f'  Loaded {len(samples)} test samples')
    return samples


# ═════════════════════════════════════════════════════════════════════════════
#  单样本递归预测
# ═════════════════════════════════════════════════════════════════════════════

def rollout_one_sample(predictor: MultiStepPredictor,
                        sample: Dict,
                        num_steps: int,
                        start_step: int = 0) -> Dict:
    perm  = sample['permeability']
    poro  = sample['porosity']
    pres  = sample['pressure']
    sat   = sample['saturation']

    avail = pres.shape[0] - start_step - 1
    steps = min(num_steps, avail)

    init   = np.stack([pres[start_step], sat[start_step]], axis=0)
    preds  = predictor.predict_sequence(init, perm, poro, num_steps=steps)

    pred_p  = preds[:, 0]                              # (steps+1, ny, nx)
    pred_sw = preds[:, 1]
    true_p  = pres[start_step: start_step + steps + 1]
    true_sw = sat[ start_step: start_step + steps + 1]

    # 逐步误差（step 0 = 初始条件，从 step 1 开始）
    p_mae  = np.abs(pred_p[1:]  - true_p[1:] ).mean(axis=(-1, -2))
    s_mae  = np.abs(pred_sw[1:] - true_sw[1:]).mean(axis=(-1, -2))
    p_rel  = p_mae / (np.abs(true_p[1:] ).mean(axis=(-1,-2)) + 1e-8) * 100
    s_rel  = s_mae / (np.abs(true_sw[1:]).mean(axis=(-1,-2)) + 1e-8) * 100

    return {
        'pred_pressure':      pred_p,
        'pred_saturation':    pred_sw,
        'true_pressure':      true_p,
        'true_saturation':    true_sw,
        'pressure_mae':       p_mae,
        'saturation_mae':     s_mae,
        'pressure_rel_err':   p_rel,
        'saturation_rel_err': s_rel,
        'steps':              steps,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  多样本平均
# ═════════════════════════════════════════════════════════════════════════════

def rollout_all_samples(predictor: MultiStepPredictor,
                         samples: List[Dict],
                         num_steps: int,
                         model_name: str) -> Dict:
    all_p_rel, all_s_rel = [], []
    all_p_mae, all_s_mae = [], []
    sample0_res = None

    for i, sample in enumerate(samples):
        print(f'    [{model_name}] sample {i+1}/{len(samples)} '
              f"(id={sample['sample_id']}  {sample['pattern']})")
        res = rollout_one_sample(predictor, sample, num_steps)
        s   = res['steps']
        all_p_rel.append(res['pressure_rel_err'][:s])
        all_s_rel.append(res['saturation_rel_err'][:s])
        all_p_mae.append(res['pressure_mae'][:s])
        all_s_mae.append(res['saturation_mae'][:s])
        if i == 0:
            sample0_res = res

    min_s     = min(len(x) for x in all_p_rel)
    p_rel_arr = np.array([x[:min_s] for x in all_p_rel])
    s_rel_arr = np.array([x[:min_s] for x in all_s_rel])
    p_mae_arr = np.array([x[:min_s] for x in all_p_mae])
    s_mae_arr = np.array([x[:min_s] for x in all_s_mae])

    return {
        'pressure_rel_err_mean':   p_rel_arr.mean(0),
        'pressure_rel_err_std':    p_rel_arr.std(0),
        'saturation_rel_err_mean': s_rel_arr.mean(0),
        'saturation_rel_err_std':  s_rel_arr.std(0),
        'pressure_mae_mean':       p_mae_arr.mean(0),
        'saturation_mae_mean':     s_mae_arr.mean(0),
        'steps':                   min_s,
        'sample0':                 sample0_res,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  可视化
# ═════════════════════════════════════════════════════════════════════════════

def plot_error_curves(all_results: Dict[str, Dict], outdir: str):
    """
    误差曲线图。
    - 主图：FNO / UNet-FNO，y 轴聚焦在两者范围内
    - 右上角 inset 小图：U-FNO 完整曲线，供读者参考量级
    - (a)(b) 标注在各子图下方
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    fig.suptitle('Multi-step rollout error accumulation',
                 fontsize=10, fontweight='bold')

    for ax, mk, sk, title, lbl in zip(
            axes,
            ['pressure_rel_err_mean',   'saturation_rel_err_mean'],
            ['pressure_rel_err_std',    'saturation_rel_err_std'],
            ['Pressure', 'Water saturation'],
            ['(a)', '(b)']):

        # ── 主图：FNO + UNet-FNO ──────────────────────────────────────────
        for name, res in all_results.items():
            if name == 'U-FNO':
                continue
            c     = COLORS.get(name, 'gray')
            steps = np.arange(1, res['steps'] + 1)
            mean  = res[mk]; std = res[sk]
            ax.plot(steps, mean, color=c, lw=1.8, label=name)
            ax.fill_between(steps, np.clip(mean - std, 0, None),
                            mean + std, color=c, alpha=0.15)

        ax.set_title(title); ax.set_xlabel('Rollout step')
        ax.set_ylabel('Relative error (%)')
        ax.grid(True, alpha=0.3); ax.set_xlim(left=1)
        ax.legend(fontsize=7, loc='upper left')

        # ── Inset：U-FNO ─────────────────────────────────────────────────
        if 'U-FNO' in all_results:
            res_u = all_results['U-FNO']
            steps_u = np.arange(1, res_u['steps'] + 1)
            mean_u  = res_u[mk]; std_u = res_u[sk]
            axins = ax.inset_axes([0.38, 0.30, 0.58, 0.52])
            axins.plot(steps_u, mean_u, color=COLORS['U-FNO'], lw=1.4)
            axins.fill_between(steps_u,
                               np.clip(mean_u - std_u, 0, None),
                               mean_u + std_u,
                               color=COLORS['U-FNO'], alpha=0.15)
            axins.set_title('U-FNO', fontsize=6.5,
                            color=COLORS['U-FNO'], pad=2)
            axins.set_xlabel('Step', fontsize=6)
            axins.set_ylabel('Rel. err (%)', fontsize=6)
            axins.tick_params(labelsize=5)
            axins.grid(True, alpha=0.2)
            axins.set_xlim(left=1)
            # 细边框用 U-FNO 颜色
            for spine in axins.spines.values():
                spine.set_edgecolor(COLORS['U-FNO'])
                spine.set_linewidth(0.8)

        # (a)/(b) 标注在子图下方
        ax.text(0.5, -0.18, lbl, transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='center', va='top')

    plt.tight_layout()
    p = os.path.join(outdir, 'rollout_error_curves.png')
    fig.savefig(p); plt.close(); print(f'  Saved → {p}')


def plot_mae_bar(all_results: Dict[str, Dict], outdir: str,
                  checkpoints=(10, 25, 50, 100)):
    names  = list(all_results.keys())
    x      = np.arange(len(checkpoints))
    width  = 0.25
    colors = [COLORS.get(n, 'gray') for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    fig.suptitle('Rollout MAE at selected time steps', fontsize=10, fontweight='bold')

    for ax, mae_key, ylabel in zip(
            axes,
            ['pressure_mae_mean', 'saturation_mae_mean'],
            ['Pressure MAE (Pa)', 'Saturation MAE']):
        for i, (name, c) in enumerate(zip(names, colors)):
            res  = all_results[name]
            vals = [float(res[mae_key][min(cp-1, res['steps']-1)])
                    for cp in checkpoints]
            ax.bar(x + i*width, vals, width,
                   label=name, color=c, edgecolor='white', lw=0.5)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'Step {c}' for c in checkpoints])
        ax.set_ylabel(ylabel); ax.legend(); ax.grid(axis='y', alpha=0.3)

    for ax, lbl in zip(axes, ['(a)', '(b)']):
        ax.text(0.5, -0.22, lbl, transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='center', va='top')

    plt.tight_layout()
    p = os.path.join(outdir, 'rollout_mae_bar.png')
    fig.savefig(p); plt.close(); print(f'  Saved → {p}')


def plot_field_comparison(all_results: Dict[str, Dict],
                           outdir: str, dt_days: float = 0.125):
    names  = list(all_results.keys())
    s0_ref = all_results[names[0]]['sample0']
    steps  = s0_ref['steps']
    t_ids  = np.linspace(1, steps-1, 4, dtype=int)

    for var, pk, tk, cmap, unit, vmin, vmax in [
        ('pressure',   'pred_pressure',   'true_pressure',
         'hot_r',    'Pa', None, None),
        ('saturation', 'pred_saturation', 'true_saturation',
         'RdYlBu_r', '',  0.2,  1.0),
    ]:
        n_rows = 1 + len(names)
        fig, axes = plt.subplots(n_rows, 4, figsize=(3.2*4, 2.8*n_rows))
        fig.suptitle(
            f'{var.capitalize()} — ground truth vs predictions (sample 0)',
            fontsize=10, fontweight='bold', y=1.01)

        auto_vmin = s0_ref[tk][t_ids].min() if vmin is None else vmin
        auto_vmax = s0_ref[tk][t_ids].max() if vmax is None else vmax

        panel_idx = 0
        for col, ti in enumerate(t_ids):
            t_label = f't={ti*dt_days:.1f} d'
            ax = axes[0, col]
            im = ax.imshow(s0_ref[tk][ti], cmap=cmap, origin='lower',
                           vmin=auto_vmin, vmax=auto_vmax)
            ax.set_title(f'True\n{t_label}', fontsize=8); ax.axis('off')
            ax.text(0.5, -0.04, f'({chr(ord("a")+panel_idx)})',
                    transform=ax.transAxes, fontsize=9,
                    fontweight='bold', ha='center', va='top')
            panel_idx += 1
            if col == 3:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=unit)

            for row, name in enumerate(names):
                s0  = all_results[name]['sample0']
                ax  = axes[row+1, col]
                dat = s0[pk][ti]
                mae = np.abs(dat - s0_ref[tk][ti]).mean()
                im  = ax.imshow(dat, cmap=cmap, origin='lower',
                                vmin=auto_vmin, vmax=auto_vmax)
                ax.set_title(f'{name}\nMAE={mae:.2e}', fontsize=7)
                ax.axis('off')
                ax.text(0.5, -0.04, f'({chr(ord("a")+panel_idx)})',
                        transform=ax.transAxes, fontsize=9,
                        fontweight='bold', ha='center', va='top')
                panel_idx += 1
                if col == 3:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                                 label=unit)

        plt.tight_layout()
        p = os.path.join(outdir, f'rollout_fields_{var}_sample0.png')
        fig.savefig(p); plt.close(); print(f'  Saved → {p}')


# ═════════════════════════════════════════════════════════════════════════════
#  汇总打印 + JSON
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(all_results: Dict[str, Dict],
                   checkpoints=(10, 25, 50, 100)):
    print(f"\n{'='*65}")
    print('ROLLOUT SUMMARY  (relative error %)')
    print(f"{'Model':<14}", end='')
    for cp in checkpoints:
        print(f"  Step{cp:>4} P%   S%", end='')
    print(); print('─'*65)
    for name, res in all_results.items():
        print(f'{name:<14}', end='')
        for cp in checkpoints:
            if cp > res['steps']:
                print(f"  {'N/A':>9} {'N/A':>4}", end='')
            else:
                idx = cp - 1
                p   = res['pressure_rel_err_mean'][idx]
                s   = res['saturation_rel_err_mean'][idx]
                print(f"  {p:>9.3f} {s:>4.3f}", end='')
        print()
    print('='*65)


def save_json(all_results: Dict[str, Dict], outdir: str,
               checkpoints=(10, 25, 50, 100)):
    summary = {}
    for name, res in all_results.items():
        summary[name] = {
            'steps': res['steps'],
            'pressure_rel_err_mean':   res['pressure_rel_err_mean'].tolist(),
            'saturation_rel_err_mean': res['saturation_rel_err_mean'].tolist(),
            'pressure_mae_mean':       res['pressure_mae_mean'].tolist(),
            'saturation_mae_mean':     res['saturation_mae_mean'].tolist(),
            'at_steps': {
                str(cp): {
                    'pressure_rel_err_%':   float(
                        res['pressure_rel_err_mean'][min(cp-1, res['steps']-1)]),
                    'saturation_rel_err_%': float(
                        res['saturation_rel_err_mean'][min(cp-1, res['steps']-1)]),
                }
                for cp in checkpoints if cp <= res['steps']
            }
        }
    p = os.path.join(outdir, 'rollout_results.json')
    with open(p, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  Saved → {p}')


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--h5',        required=True)
    p.add_argument('--ckpt_dir',  default='./checkpoints')
    p.add_argument('--outdir',    default='./rollout')
    p.add_argument('--steps',     type=int,   default=100)
    p.add_argument('--n_samples', type=int,   default=5)
    p.add_argument('--batch',     type=int,   default=128)
    p.add_argument('--workers',   type=int,   default=4)
    p.add_argument('--modes',     type=int,   default=8)
    p.add_argument('--width',     type=int,   default=32)
    p.add_argument('--depth',     type=int,   default=4)
    p.add_argument('--dt_days',   type=float, default=0.125,
                   help='每步对应天数 (dt=10800 s → 0.125 day)')
    p.add_argument('--checkpoints', nargs='+', type=int,
                   default=[10, 25, 50, 100])
    return p.parse_args()


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.outdir, exist_ok=True)

    print(f'\n{"#"*60}')
    print(f'  Rollout Test  device={device}  '
          f'steps={args.steps}  n_samples={args.n_samples}')
    print(f'{"#"*60}')

    # DataLoader（仅用于获取归一化参数）
    print('\nLoading data...')
    train_loader, test_loader = create_data_loaders(
        args.h5, batch_size=args.batch,
        train_test_split=0.8, num_workers=args.workers)

    # 加载模型
    print('\nLoading checkpoints...')
    models = load_models(args, train_loader, test_loader, device)
    if not models:
        print('No checkpoints found. Exiting.'); return

    # 加载测试样本（物理量）
    print('\nLoading test samples...')
    test_samples = load_test_samples(args.h5, args.n_samples)

    # 逐模型做 rollout
    train_ds    = train_loader.dataset
    all_results = {}
    for name, model in models.items():
        print(f'\n[{name}] rollout...')
        predictor = MultiStepPredictor(model, train_ds, device=device)
        all_results[name] = rollout_all_samples(
            predictor, test_samples, args.steps, name)

    # 输出
    print_summary(all_results, checkpoints=tuple(args.checkpoints))
    print('\nGenerating figures...')
    plot_error_curves(all_results, args.outdir)
    plot_mae_bar(all_results, args.outdir,
                 checkpoints=tuple(args.checkpoints))
    plot_field_comparison(all_results, args.outdir, dt_days=args.dt_days)
    save_json(all_results, args.outdir,
              checkpoints=tuple(args.checkpoints))

    print(f'\nDone → {args.outdir}/')


if __name__ == '__main__':
    main()
