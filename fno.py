import torch
import torch.nn as nn
import torch.nn.functional as F



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

        return out.to(orig_dtype)


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
