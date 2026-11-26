#!/usr/bin/env python3
# =============================================================================
# run_all.py  —  一键完整流水线
#
# 功能：
#   Step 1  生成数据集（可跳过，指定 --h5 直接使用已有数据）
#   Step 2  训练三个模型（FNO / U-FNO / UNet-FNO），完全相同的训练配置
#   Step 3  评估所有模型，计算逐通道精度指标
#   Step 4  自动保存所有对比可视化图
#
# 用法：
#   # 完整流水线（从生成数据开始）
#   python run_all.py
#
#   # 使用已有数据集（跳过数据生成）
#   python run_all.py --h5 ./reservoir_fno_dataset_20x20/reservoir_fno_xxx.h5
#
#   # 使用已有数据 + 跳过训练（只跑评估和可视化）
#   python run_all.py --h5 xxx.h5 --eval_only
#
#   # 快速调试（少样本、少 epoch）
#   python run_all.py --n_samples 10 --epochs 5 --batch 4
# =============================================================================

import os, sys, time, json, argparse, glob

# ── 确保当前目录在 path 中（import 同级模块）─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='One-click: generate data → train FNO/U-FNO/UNet-FNO → compare')

    # ── 输出根目录 ────────────────────────────────────────────────────────────
    p.add_argument('--outdir',    default='./experiments',
                   help='所有输出的根目录')

    # ── Step 1：数据生成 ──────────────────────────────────────────────────────
    p.add_argument('--h5',        default=None,
                   help='直接指定已有 HDF5 文件，跳过数据生成')
    p.add_argument('--nx',        type=int, default=20)
    p.add_argument('--ny',        type=int, default=20)
    p.add_argument('--n_samples', type=int, default=500,
                   help='生成样本数（调试用 10，论文用 500）')
    p.add_argument('--nt',        type=int, default=500,
                   help='每个样本的时间步数')

    # ── Step 2：训练 ──────────────────────────────────────────────────────────
    p.add_argument('--epochs',    type=int,   default=100)
    p.add_argument('--batch',     type=int,   default=128)
    p.add_argument('--lr',        type=float, default=1e-3)
    p.add_argument('--wd',        type=float, default=1e-4)
    p.add_argument('--patience',  type=int,   default=20,
                   help='早停耐心轮数')
    p.add_argument('--workers',   type=int,   default=4)

    # ── 模型超参 ──────────────────────────────────────────────────────────────
    p.add_argument('--modes',     type=int,   default=8,
                   help='Fourier 模式数')
    p.add_argument('--width',     type=int,   default=32,
                   help='隐藏通道数')
    p.add_argument('--depth',     type=int,   default=4,
                   help='UNet-FNO 的 FNO 层数')

    # ── 其他 ──────────────────────────────────────────────────────────────────
    p.add_argument('--eval_only', action='store_true',
                   help='跳过训练，仅从 checkpoint 加载后评估')
    p.add_argument('--vis_samples', type=int, default=3,
                   help='预测对比图中展示的测试样本数')
    p.add_argument('--models', nargs='+',
                default=['FNO', 'U-FNO', 'UNet-FNO'],
                choices=['FNO', 'U-FNO', 'UNet-FNO'],
                help='指定要训练的模型')
    return p.parse_args()


# =============================================================================
def section(title: str):
    bar = '=' * 60
    print(f'\n{bar}\n  {title}\n{bar}')


def elapsed(t0: float) -> str:
    s = time.time() - t0
    h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return f'{int(h)}h {int(m)}m {sec:.0f}s'


# =============================================================================
# Step 1  数据生成
# =============================================================================
def step1_generate(args) -> str:
    """返回 HDF5 文件路径"""
    if args.h5:
        if not os.path.exists(args.h5):
            raise FileNotFoundError(f'HDF5 not found: {args.h5}')
        print(f'  Using existing dataset: {args.h5}')
        return args.h5

    section('Step 1 / 4  —  Data Generation')
    from data_generation import ReservoirDatasetCreator

    t0 = time.time()
    creator = ReservoirDatasetCreator(
        output_dir=os.path.join(args.outdir, 'dataset'),
        nx=args.nx, ny=args.ny)
    h5_path = creator.create_dataset(
        n_samples=args.n_samples, nt=args.nt)
    print(f'  Done in {elapsed(t0)}  →  {h5_path}')
    return h5_path


# =============================================================================
# Step 2 + 3  训练 + 评估
# =============================================================================
def print_model_summary(results: dict):
    """
    汇总对比表，风格对齐 ablation_study.print_ablation_summary：
      - 每行指标标注 ★ 最优值
      - 末尾按 combined_rel_err 排名
    """
    names = list(results.keys())
    w  = 16
    hd = f"{'Metric':<22}" + ''.join(f'{n:>{w}}' for n in names)
    sep = '─' * len(hd)

    print(f'\n{"="*len(hd)}')
    print('MODEL COMPARISON SUMMARY')
    print(sep); print(hd); print(sep)

    rows = [
        ('n_params',           'Parameters',       '{:>14,.0f}'),
        ('pressure_mae',       'P-MAE (Pa)',        '{:>14.3e}'),
        ('pressure_rel_err',   'P-RelErr (%)',      '{:>14.3f}'),
        ('saturation_mae',     'S-MAE',             '{:>14.5f}'),
        ('saturation_rel_err', 'S-RelErr (%)',      '{:>14.3f}'),
        ('combined_rel_err',   'Combined (%)',      '{:>14.3f}'),
        ('inference_ms',       'Infer (ms)',        '{:>14.3f}'),
    ]

    for key, label, fmt in rows:
        vals = [results[n].get(key, 0) for n in names]
        best = min(range(len(vals)), key=lambda i: vals[i])
        row  = f'{label:<22}'
        for i, v in enumerate(vals):
            cell = fmt.format(v)
            row += f"{'★'+cell if i == best else cell:>{w}}"
        print(row)

    print(sep)
    ranked = sorted(names, key=lambda n: results[n]['combined_rel_err'])
    print('Ranking (by combined RelErr):')
    for rank, n in enumerate(ranked, 1):
        print(f'  {rank}. {n:<14}  {results[n]["combined_rel_err"]:.3f}%')
    print('=' * len(hd))


def step23_train_eval(args, h5_path: str):
    """
    加载三个模型并评估，返回 (trainers_dict, results_dict, models_dict)。
    加载逻辑：eval_only + checkpoint 存在 → 加载；否则跳过（不训练）。
    对比方式：print_model_summary（带 ★ 最优标注 + 排名，风格同 ablation）。
    """
    section('Step 2 / 4  —  Loading & Evaluating  (FNO / U-FNO / UNet-FNO)')

    from data_training import create_data_loaders, Trainer
    from compare_models import  evaluate_model
    from fno import FNO2d
    from ufno import Net2d
    from ablation_study import build_ablation_models
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'  Device: {device}')
    all_models = build_ablation_models(modes=args.modes, width=args.width, n_layers=args.depth, base_ch=32)


    ckpt_dir = os.path.join(args.outdir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── 共享 DataLoader ───────────────────────────────────────────────────────
    print('  Loading data...')
    train_loader, test_loader = create_data_loaders(
        h5_path,
        batch_size=args.batch,
        train_test_split=0.8,
        num_workers=args.workers,
    )
    train_ds = train_loader.dataset   # 保留用于反归一化

    # ── 模型实例 ──────────────────────────────────────────────────────────────
    model_defs = {
        'FNO':      FNO2d(modes1=args.modes, modes2=args.modes,
                          width=args.width),
        'U-FNO':    Net2d(modes1=args.modes, modes2=args.modes,
                           width=args.width),
        'UNet-FNO': all_models['V1-Full'],
    }

    print('\n  Model parameter counts:')
    for name, m in model_defs.items():
        print(f'    {name:<12}  {m.count_params():>12,} params')

    trainers, results = {}, {}

    for name, model in model_defs.items():
        safe_name = name.lower().replace('-', '_')
        best_ckpt = os.path.join(ckpt_dir, f'best_{safe_name}.pth')
        trainer = Trainer(
            model, train_loader, test_loader,
            device=device,
            learning_rate=args.lr,
            weight_decay=args.wd,
            model_name=name,
            output_dir=ckpt_dir,
        )
        # ── 仅 eval_only 模式：从 best checkpoint 加载 ───────────────────────
        if args.eval_only and os.path.exists(best_ckpt):
            print(f'\n  [{name}] Loading checkpoint: {best_ckpt}')
            trainer.load_model(best_ckpt)
        # else:
            # print(f'\n{"─"*60}')
            # print(f'  Training  {name}')
            # print(f'{"─"*60}')
            # t0 = time.time()
            # trainer.train(num_epochs=args.epochs,
            #               early_stopping_patience=args.patience)
            # trainer.save_model(
            #     os.path.join(ckpt_dir, f'final_{safe_name}.pth'))
            # print(f'  [{name}] Training done in {elapsed(t0)}')

        trainers[name] = trainer

        # ── 评估 ─────────────────────────────────────────────────────────────
        print(f'  [{name}] Evaluating...')
        metrics = evaluate_model(model, test_loader, train_ds, device)
        metrics['n_params'] = model.count_params()
        results[name] = metrics
        print(f'    P-MAE={metrics["pressure_mae"]:.3e} Pa  '
              f'P-Rel={metrics["pressure_rel_err"]:.2f}%  '
              f'S-MAE={metrics["saturation_mae"]:.5f}  '
              f'S-Rel={metrics["saturation_rel_err"]:.2f}%  '
              f'Infer={metrics["inference_ms"]:.2f} ms')

    # ── 汇总对比（★ 最优标注 + 排名，风格同 ablation_summary）────────────────
    print_model_summary(results)
    return trainers, results, model_defs, train_ds, test_loader, device


# =============================================================================
# Step 4  可视化
# =============================================================================
def step4_visualize(args, trainers, results, model_defs,
                    train_ds, test_loader, device, h5_path):
    section('Step 4 / 4  —  Visualization & Saving')

    from compare_models import (
        plot_comparison_bar,
        plot_loss_curves,
        plot_prediction_comparison,
        plot_metrics_radar,
    )
    from data_visualization import (
        ReservoirDataVisualizer,
        plot_kr_curves,
    )
    from data_generation import ReservoirProperties

    fig_dir = os.path.join(args.outdir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # ── 4-a  模型对比条形图 ───────────────────────────────────────────────────
    print('  [4a] Comparison bar chart...')
    plot_comparison_bar(results, output_dir=fig_dir)

    # ── 4-b  训练/测试损失曲线 ────────────────────────────────────────────────
    print('  [4b] Loss curves...')
    plot_loss_curves(trainers, output_dir=fig_dir)

    # ── 4-c  预测结果对比（真值 vs 三模型）───────────────────────────────────
    print('  [4c] Prediction comparison...')
    plot_prediction_comparison(
        model_defs, test_loader, train_ds, device,
        output_dir=fig_dir, n_samples=args.vis_samples)

    # ── 4-d  雷达图（5 维综合对比）───────────────────────────────────────────
    print('  [4d] Radar chart...')
    plot_metrics_radar(results, output_dir=fig_dir)

    # ── 4-e  数据集可视化 ─────────────────────────────────────────────────────
    print('  [4e] Dataset overview...')
    # ds_fig_dir = os.path.join(fig_dir, 'dataset')
    # vis = ReservoirDataVisualizer(h5_path, output_dir=ds_fig_dir)
    # vis.create_statistics_report(save_report=True)
    # vis.visualize_single_sample(sample_idx=0, save_fig=True)
    # vis.visualize_pressure_evolution(sample_idx=0, save_fig=True)
    # vis.visualize_saturation_evolution(sample_idx=0, save_fig=True)
    # if vis.n_samples > 1:
    #     vis.create_comparison_dashboard(
    #         sample_indices=list(range(min(4, vis.n_samples))),
    #         save_fig=True)

    # ── 4-f  Brooks-Corey 相渗曲线 ───────────────────────────────────────────
    # print('  [4f] Kr curves...')
    # kr_dir = os.path.join(fig_dir, 'kr_curves')
    # props  = ReservoirProperties()
    # plot_kr_curves(props, output_dir=kr_dir, save=True)

    print(f'\n  All figures saved → {fig_dir}/')
    return fig_dir


# =============================================================================
# 最终汇总报告
# =============================================================================
def save_final_report(args, results: dict, h5_path: str,
                       fig_dir: str, total_time: str):
    report_dir = os.path.join(args.outdir, 'report')
    os.makedirs(report_dir, exist_ok=True)

    # JSON
    report = {
        'config': vars(args),
        'h5_dataset': h5_path,
        'total_wall_time': total_time,
        'results': results,
        'figures_dir': fig_dir,
    }
    json_path = os.path.join(report_dir, 'final_report.json')
    with open(json_path, 'w') as fp:
        json.dump(report, fp, indent=2)

    # 纯文本摘要：复用 print_model_summary，捕获输出写入文件
    import io, contextlib
    txt_path = os.path.join(report_dir, 'summary.txt')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f'Dataset    : {h5_path}')
        print(f'Total time : {total_time}')
        print(f'Figures    : {fig_dir}')
        print_model_summary(results)
    summary_text = buf.getvalue()
    with open(txt_path, 'w') as fp:
        fp.write(summary_text)
    print(summary_text)
    print(f'Full report → {json_path}')
    print(f'Summary     → {txt_path}')


# =============================================================================
# 主函数
# =============================================================================
def main():
    args   = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    t_start = time.time()

    print(f'\n{"#"*60}')
    print(f'  Two-Phase Flow Neural Operator — Full Pipeline')
    print(f'  Output: {os.path.abspath(args.outdir)}')
    print(f'{"#"*60}')
    print(f'\nConfig:')
    for k, v in vars(args).items():
        print(f'  {k:<16} = {v}')

    # Step 1: 数据
    h5_path = step1_generate(args)

    # Step 2+3: 训练 + 评估
    trainers, results, model_defs, train_ds, test_loader, device = \
        step23_train_eval(args, h5_path)

    # Step 4: 可视化
    fig_dir = step4_visualize(
        args, trainers, results, model_defs,
        train_ds, test_loader, device, h5_path)

    # 最终报告
    total = elapsed(t_start)
    section('Done')
    save_final_report(args, results, h5_path, fig_dir, total)
    print(f'\nTotal wall time: {total}')


if __name__ == '__main__':
    main()
