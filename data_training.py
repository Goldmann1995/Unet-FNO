# =============================================================================
# data_training.py  —  模块二：模型训练
#
# 包含：
#   LpLoss                     相对 L2 损失函数
#   ReservoirOperatorDataset   HDF5 数据集（state-to-state mapping）
#   create_data_loaders        创建 train/test DataLoader
#   _EncoderBlock              U-Net 编码块（GroupNorm + GELU）
#   _DecoderBlock              U-Net 解码块（双线性上采样 + skip concat）
#   SpectralConv2d             2D 傅里叶谱卷积层
#   AdaptiveUNet               自适应深度 U-Net
#   UNetFNO2d                  核心模型：每层 FNO 前置 U-Net 特征提取
#   Trainer                    训练器（AdamW + CosineAnnealing + 早停）
#   MultiStepPredictor         递归多步预测器
#
# 用法：
#   from data_training import UNetFNO2d, Trainer, create_data_loaders
#   train_loader, test_loader = create_data_loaders('dataset.h5')
#   model = UNetFNO2d(modes1=8, modes2=8, width=32)
#   trainer = Trainer(model, train_loader, test_loader, output_dir='./runs')
#   trainer.train(num_epochs=100)
# =============================================================================

import os, time
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple

torch.manual_seed(42)
np.random.seed(42)

torch.backends.cudnn.benchmark = True
# ──────────────────────────────── 损失函数 ────────────────────────────────────

class LpLoss:
    """相对 Lp 损失，默认 p=2"""

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        self.d, self.p = d, p
        self.size_average = size_average
        self.reduction    = reduction

    def rel(self, x, y):
        num_examples = x.size(0)
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


# ─────────────────────────── 数据集 ──────────────────────────────────────────

class ReservoirOperatorDataset(Dataset):
    """
    油藏算子学习数据集。
    输入：[K, φ, p^t, S_w^t]  (4, ny, nx)
    输出：[p^{t+1}, S_w^{t+1}] (2, ny, nx)
    """

    def __init__(self, h5_file_path: str, mode: str = 'train',
                 train_test_split: float = 0.8, time_window: int = 1,
                 normalize: bool = True, stride: int = 1,
                 input_steps: int = 2, output_steps: int = 2):
        super().__init__()
        assert mode in ('train', 'test')
        self.h5_file_path   = h5_file_path
        self.mode           = mode
        self.time_window    = time_window
        self.normalize      = normalize
        self.stride         = stride
        self.input_steps    = input_steps
        self.output_steps   = output_steps

        self.data           = self._load_data()
        self.sample_indices = self._split_train_test(train_test_split)
        if normalize:
            self._compute_normalization_params()

        print(f"[{mode}] {len(self.sample_indices)} pairs  "
              f"(time_window={time_window}  normalize={normalize})")

    # ── 加载 HDF5 ─────────────────────────────────────────────────────────────

    def _load_data(self) -> Dict:
        data = {}
        with h5py.File(self.h5_file_path, 'r') as f:
            for sname in f['data'].keys():
                sg = f['data'][sname]
                ny, nx = sg['permeability'].shape
                pressure    = sg['pressure'][:].reshape(-1, ny, nx)
                saturation  = sg['saturation'][:].reshape(-1, ny, nx)
                data[sname] = {
                    'permeability': sg['permeability'][:],
                    'porosity':     sg['porosity'][:],
                    'pressure':     pressure,
                    'saturation':   saturation,
                    'time_steps':   sg['time_steps'][:],
                    'nx': nx, 'ny': ny,
                    'sample_id': sg.attrs['sample_id'],
                    'pattern':   sg.attrs['pattern'],
                }
        return data

    # ── 划分 ─────────────────────────────────────────────────────────────────

    def _split_train_test(self, split_ratio: float) -> List[Tuple]:
        all_names = list(self.data.keys())
        np.random.shuffle(all_names)
        cut = int(len(all_names) * split_ratio)
        chosen = all_names[:cut] if self.mode == 'train' else all_names[cut:]

        indices = []
        for name in chosen:
            T = self.data[name]['pressure'].shape[0]
            for t in range(0, T - self.time_window, self.stride):
                indices.append((name, t))
        return indices

    # ── 归一化 ────────────────────────────────────────────────────────────────

    def _compute_normalization_params(self):
        def _cat(key, sub=None):
            arrays = []
            for s in self.data.values():
                a = s[key] if sub is None else s[key]
                arrays.append(a.ravel())
            return np.concatenate(arrays)

        for key, attr in [('permeability', 'perm'), ('porosity', 'poro'),
                           ('pressure', 'press'), ('saturation', 'sat')]:
            arr = _cat(key)
            setattr(self, f'{attr}_mean', float(arr.mean()))
            setattr(self, f'{attr}_std',  float(arr.std()) + 1e-10)

        print(f"  Norm  K: μ={self.perm_mean:.2e} σ={self.perm_std:.2e}  "
              f"p: μ={self.press_mean:.2e} σ={self.press_std:.2e}  "
              f"Sw: μ={self.sat_mean:.3f} σ={self.sat_std:.3f}")

    def _norm(self, arr, key):
        if not self.normalize: return arr
        m = {'permeability': (self.perm_mean, self.perm_std),
             'porosity':     (self.poro_mean, self.poro_std),
             'pressure':     (self.press_mean, self.press_std),
             'saturation':   (self.sat_mean,   self.sat_std)}[key]
        return (arr - m[0]) / m[1]

    def _denorm(self, arr, key):
        if not self.normalize: return arr
        m = {'permeability': (self.perm_mean, self.perm_std),
             'porosity':     (self.poro_mean, self.poro_std),
             'pressure':     (self.press_mean, self.press_std),
             'saturation':   (self.sat_mean,   self.sat_std)}[key]
        return arr * m[1] + m[0]

    # ── __getitem__ ───────────────────────────────────────────────────────────

    def __len__(self): return len(self.sample_indices)

    def __getitem__(self, idx):
        name, t = self.sample_indices[idx]
        s       = self.data[name]
        t1      = min(t + self.time_window, s['pressure'].shape[0] - 1)

        K   = self._norm(s['permeability'],    'permeability')
        phi = self._norm(s['porosity'],        'porosity')
        p_k = self._norm(s['pressure'][t],     'pressure')
        sw_k= self._norm(s['saturation'][t],   'saturation')
        p_t = self._norm(s['pressure'][t1],    'pressure')
        sw_t= self._norm(s['saturation'][t1],  'saturation')

        x = torch.FloatTensor(np.stack([K, phi, p_k, sw_k], axis=0))
        y = torch.FloatTensor(np.stack([p_t, sw_t],          axis=0))

        ts = s['time_steps']
        dt = (ts[t1] - ts[t]) if t1 < len(ts) else 0.0

        return {'input': x, 'output': y,
                'sample_id': s['sample_id'], 'pattern': s['pattern'],
                'time_step': ts[t], 'delta_time': dt,
                'start_idx': t, 'grid_shape': (s['ny'], s['nx'])}

    def get_full_sequence(self, sample_name: str) -> Dict:
        s   = self.data[sample_name]
        return {
            'permeability': self._norm(s['permeability'], 'permeability'),
            'porosity':     self._norm(s['porosity'],     'porosity'),
            'pressure':     self._norm(s['pressure'],     'pressure'),
            'saturation':   self._norm(s['saturation'],   'saturation'),
            'time_steps':   s['time_steps'],
            'nx': s['nx'], 'ny': s['ny'],
            'sample_id': s['sample_id'], 'pattern': s['pattern'],
        }


def create_data_loaders(h5_file_path: str, batch_size: int = 16,
                        train_test_split: float = 0.8, time_window: int = 1,
                        normalize: bool = True, num_workers: int = 4,
                        shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader]:
    """创建训练和测试 DataLoader"""
    kw = dict(h5_file_path=h5_file_path, train_test_split=train_test_split,
              time_window=time_window, normalize=normalize,
              stride=1, input_steps=2, output_steps=2)
    tr = ReservoirOperatorDataset(mode='train', **kw)
    te = ReservoirOperatorDataset(mode='test',  **kw)
    loader_kw = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
    )
    return (DataLoader(tr, shuffle=shuffle_train, drop_last=True,  **loader_kw),
            DataLoader(te, shuffle=False,         drop_last=False, **loader_kw))


# ────────────────────────── 模型架构 ─────────────────────────────────────────

class _EncoderBlock(nn.Module):
    """GroupNorm + GELU，可选 AvgPool2d 下采样"""

    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        ng = min(32, out_ch)
        self.encode = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(ng, out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(ng, out_ch), nn.GELU(),
        )
        self.pool = nn.AvgPool2d(2, 2) if downsample else None

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        return self.encode(x)


class _DecoderBlock(nn.Module):
    """双线性上采样（分辨率不变性）+ skip concat"""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        ng = min(32, out_ch)
        self.decode = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.GroupNorm(ng, out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(ng, out_ch), nn.GELU(),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                          align_corners=False)
        return self.decode(torch.cat([x, skip], dim=1))


class SpectralConv2d(nn.Module):
    """2D Fourier 谱卷积层（rfft2 + 低频模式线性变换 + irfft2）"""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1, self.modes2 = modes1, modes2
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(
            in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(
            in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def _mul(self, inp, w):
        return torch.einsum("bixy,ioxy->boxy", inp, w)

    def forward(self, x):
        orig_dtype = x.dtype

        # 👉 强制 FFT 用 float32
        x = x.float()

        xf = torch.fft.rfft2(x)

        out = torch.zeros(
            x.shape[0], self.out_channels,
            x.size(-2), x.size(-1)//2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        out[:, :, :self.modes1, :self.modes2] = self._mul(
            xf[:, :, :self.modes1, :self.modes2], self.weights1)

        out[:, :, -self.modes1:, :self.modes2] = self._mul(
            xf[:, :, -self.modes1:, :self.modes2], self.weights2)

        out = torch.fft.irfft2(out, s=(x.size(-2), x.size(-1)))

        # 👉 转回 AMP dtype（float16 / bfloat16）
        return out.to(orig_dtype)


class AdaptiveUNet(nn.Module):
    """
    自适应深度 U-Net（默认 depth=3，避免 20×20 → 1×1 坍塌）。

    改进点：
    - F.interpolate 替换 ConvTranspose2d → 分辨率不变
    - GroupNorm 替换 BatchNorm → 小批量稳定
    - 残差连接防止信息丢失
    """

    def __init__(self, in_channels, out_channels, base_channels=32, max_depth=3):
        super().__init__()
        self.max_depth = max_depth
        enc_ch = [in_channels]
        ch     = base_channels
        self.encoders = nn.ModuleList()
        for d in range(max_depth):
            self.encoders.append(_EncoderBlock(enc_ch[-1], ch, downsample=(d > 0)))
            enc_ch.append(ch)
            ch = min(ch * 2, 256)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_ch[-1], ch, 3, padding=1),
            nn.GroupNorm(min(32, ch), ch), nn.GELU())

        self.decoders = nn.ModuleList()
        dec_in = ch
        for d in reversed(range(max_depth)):
            skip_ch = enc_ch[d + 1]; out_ch = enc_ch[d + 1]
            self.decoders.append(_DecoderBlock(dec_in, skip_ch, out_ch))
            dec_in = out_ch

        self.output_conv  = nn.Conv2d(enc_ch[1], out_channels, 1)
        self.residual_proj = (nn.Conv2d(in_channels, out_channels, 1)
                              if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        residual = self.residual_proj(x)
        enc_feats, out = [], x
        for enc in self.encoders:
            out = enc(out); enc_feats.append(out)
        out = self.bottleneck(out)
        for dec, skip in zip(self.decoders, reversed(enc_feats)):
            out = dec(out, skip)
        return self.output_conv(out) + residual


class UNetFNO2d(nn.Module):
    """
    UNet-FNO：每个 FNO 谱卷积层前置 AdaptiveUNet 特征提取。

    层结构：
        x → fc0 (Linear lift) → [UNet → SpectralConv + W + bias] × L → fc1 → fc2 → out
    """

    def __init__(self, modes1=8, modes2=8, width=32, n_fno_layers=4,
                 in_channels=4, out_channels=2,
                 unet_base_ch=32, unet_depth=3):
        super().__init__()
        self.width        = width
        self.n_fno_layers = n_fno_layers

        self.fc0   = nn.Linear(in_channels, width)
        self.unet  = AdaptiveUNet(width, width, unet_base_ch, unet_depth)

        self.convs  = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(n_fno_layers)])
        self.ws     = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_fno_layers)])
        self.biases = nn.Parameter(torch.zeros(n_fno_layers, 1, width, 1, 1))

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x: (B, 4, ny, nx)
        x = x.permute(0, 2, 3, 1)          # → (B, ny, nx, 4)
        x = self.fc0(x)                      # → (B, ny, nx, W)
        x = x.permute(0, 3, 1, 2)           # → (B, W, ny, nx)

        for i, (conv, w) in enumerate(zip(self.convs, self.ws)):
            x = self.unet(x)
            xw = w(x.view(x.size(0), self.width, -1)).view_as(x)
            x  = conv(x) + xw + self.biases[i]
            if i < self.n_fno_layers - 1:
                x = F.gelu(x)

        x = x.permute(0, 2, 3, 1)           # → (B, ny, nx, W)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)                      # → (B, ny, nx, 2)
        return x.permute(0, 3, 1, 2)         # → (B, 2, ny, nx)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────── 训练器 ─────────────────────────────────────────

class Trainer:
    """
    AdamW + CosineAnnealingLR + 早停。

    用法：
        trainer = Trainer(model, train_loader, test_loader, output_dir='./runs/exp1')
        trainer.train(num_epochs=100)
    """

    def __init__(self, model, train_loader, test_loader,
                 device=None, learning_rate=1e-3, weight_decay=1e-4,
                 model_name='ufno', output_dir='./output'):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model        = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.model_name   = model_name
        self.output_dir   = output_dir
        self.scaler = torch.amp.GradScaler(enabled=False)
        os.makedirs(output_dir, exist_ok=True)

        self.criterion = LpLoss(size_average=False)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate,
            weight_decay=weight_decay, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=1e-6)

        self.train_losses: List[float] = []
        self.test_losses:  List[float] = []

        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 单 epoch 训练 ─────────────────────────────────────────────────────────

    def train_epoch(self) -> float:
        self.model.train()
        total, n = 0.0, 0
        for i, batch in enumerate(self.train_loader):
            x = batch['input'].to(self.device, non_blocking=True)
            y = batch['output'].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(x)
            loss = self.criterion(pred, y)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            loss.backward()
            self.optimizer.step()
            total += loss.item(); n += 1
            if (i + 1) % 20 == 0:
                print(f"    batch {i+1}/{len(self.train_loader)}  loss={loss.item():.6f}")
        avg = total / n
        self.train_losses.append(avg)
        return avg

    # ── 评估 ──────────────────────────────────────────────────────────────────

    def evaluate(self) -> float:
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in self.test_loader:
                x = batch['input'].to(self.device, non_blocking=True)
                y = batch['output'].to(self.device, non_blocking=True)

                pred = self.model(x)
                total += self.criterion(pred, y).item()
                n += 1
        avg = total / n
        self.test_losses.append(avg)
        return avg

    # ── 完整训练循环 ──────────────────────────────────────────────────────────

    def train(self, num_epochs=100, early_stopping_patience=20):
        best_loss, patience_cnt = float('inf'), 0
        t0 = time.time()

        for epoch in range(num_epochs):
            te = time.time()
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            tr_loss = self.train_epoch()
            ts_loss = self.evaluate()
            self.scheduler.step()
            print(f"  train={tr_loss:.6f}  test={ts_loss:.6f}  "
                  f"lr={self.optimizer.param_groups[0]['lr']:.2e}  "
                  f"time={time.time()-te:.1f}s")

            if ts_loss < best_loss:
                best_loss    = ts_loss
                patience_cnt = 0
                self.save_model(f'{self.output_dir}/best_{self.model_name}.pth')
                print(f"  ★ saved best  loss={best_loss:.6f}")
            else:
                patience_cnt += 1
                # if patience_cnt >= early_stopping_patience:
                #     print(f"  Early stopping at epoch {epoch+1}")
                #     break

        total = time.time() - t0
        h, r  = divmod(total, 3600); m, s = divmod(r, 60)
        print(f"\n{'='*60}")
        print(f"Done  best_test={best_loss:.6f}  "
              f"time={int(h)}h {int(m)}m {s:.0f}s")
        print(f"{'='*60}")
        self.plot_losses()

    # ── 保存 / 加载 ───────────────────────────────────────────────────────────

    def save_model(self, path: str):
        torch.save({
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'test_losses':  self.test_losses,
        }, path)
        print(f"  Saved → {path}")

    def load_model(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.train_losses = ckpt.get('train_losses', [])
        self.test_losses  = ckpt.get('test_losses',  [])
        print(f"  Loaded ← {path}")

    # ── 损失曲线 ──────────────────────────────────────────────────────────────

    def plot_losses(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, losses, title in zip(
                axes, [self.train_losses, self.test_losses],
                ['Training Loss', 'Test Loss']):
            ax.plot(losses)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
            ax.set_title(title); ax.grid(True)
        plt.tight_layout()
        path = f'{self.output_dir}/losses_{self.model_name}.png'
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Loss curve → {path}")


# ───────────────────────── 多步递归预测器 ────────────────────────────────────

class MultiStepPredictor:
    """
    使用训练好的 UNet-FNO 进行递归多步预测。

    示例：
        predictor = MultiStepPredictor(model, dataset)
        preds = predictor.predict_sequence(initial_state, K, phi, num_steps=50)
        # preds.shape = (51, 2, ny, nx)  [0]=初始，[1..50]=预测
    """

    def __init__(self, model, dataset: ReservoirOperatorDataset,
                 device: str = 'cpu'):
        self.model   = model.to(device)
        self.model.eval()
        self.device  = device
        self.dataset = dataset

    def predict_sequence(self, initial_state, permeability, porosity,
                         num_steps: int = 10) -> np.ndarray:
        """
        Args:
            initial_state : (2, ny, nx) ndarray  [pressure, saturation]
            permeability  : (ny, nx)
            porosity      : (ny, nx)
            num_steps     : 递归步数
        Returns:
            ndarray (num_steps+1, 2, ny, nx)
        """
        def _to_np(x):
            return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

        init   = _to_np(initial_state)
        K      = _to_np(permeability)
        phi    = _to_np(porosity)
        ds     = self.dataset

        if ds.normalize:
            K_n   = ds._norm(K,       'permeability')
            phi_n = ds._norm(phi,     'porosity')
            p_n   = ds._norm(init[0], 'pressure')
            s_n   = ds._norm(init[1], 'saturation')
        else:
            K_n, phi_n, p_n, s_n = K, phi, init[0], init[1]

        preds = [init]
        with torch.no_grad():
            for _ in range(num_steps):
                x = torch.FloatTensor(
                    np.stack([K_n, phi_n, p_n, s_n], axis=0)
                ).unsqueeze(0).to(self.device)

                out = self.model(x)[0].cpu().numpy()  # (2, ny, nx)

                if ds.normalize:
                    p_out = ds._denorm(out[0], 'pressure')
                    s_out = ds._denorm(out[1], 'saturation')
                else:
                    p_out, s_out = out[0], out[1]

                preds.append(np.stack([p_out, s_out], axis=0))
                p_n, s_n = out[0], out[1]   # 下一步用归一化值

        return np.array(preds)


# ──────────────────────────── 入口 ───────────────────────────────────────────

if __name__ == '__main__':
    import argparse, sys

    parser = argparse.ArgumentParser(description='Train UNet-FNO')
    parser.add_argument('--h5',     required=True, help='HDF5 数据集路径')
    parser.add_argument('--outdir', default='./runs/exp1')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch',  type=int, default=128)
    parser.add_argument('--modes',  type=int, default=8)
    parser.add_argument('--width',  type=int, default=32)
    parser.add_argument('--depth',  type=int, default=4)
    args = parser.parse_args()

    print("Loading data...")
    train_loader, test_loader = create_data_loaders(
        args.h5, batch_size=args.batch)

    grid_shape = next(iter(train_loader))['grid_shape']
    print(f"Grid: {grid_shape[0][0].item()}×{grid_shape[1][0].item()}")

    model = UNetFNO2d(modes1=args.modes, modes2=args.modes,
                      width=args.width, n_fno_layers=args.depth)
    print(f"Model params: {model.count_params():,}")

    trainer = Trainer(model, train_loader, test_loader,
                      output_dir=args.outdir)
    trainer.train(num_epochs=args.epochs)
    trainer.save_model(f'{args.outdir}/final_model.pth')
