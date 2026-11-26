"""
plot_kr_curves.py
-----------------
直接从 main_claude_fixed.py 读取 ReservoirProperties，
生成 Brooks-Corey 相对渗透率曲线并保存为 PNG + CSV。

用法（与 main_claude_fixed.py 放在同一目录）：
    python plot_kr_curves.py
"""

import sys, os, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── 导入 ReservoirProperties（与 main_claude_fixed.py 同目录运行）──────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class ReservoirProperties:
    def __init__(self):
        # 流体性质
        self.mu_o = 1.8e-3  # 油粘度 (Pa·s)
        self.mu_w = 1e-3    # 水粘度 (Pa·s)
        self.Bo = 1.2       # 油地层体积系数
        self.Bw = 1.0       # 水地层体积系数

        # 岩石性质
        self.poro = 0.2     # 孔隙度
        self.Siw = 0.2      # 束缚水饱和度

        # 压缩系数 (1/Pa)
        self.Cr = 10 * 1e-6 / 6894   # 岩石压缩系数
        self.Cw = 4 * 1e-6 / 6894    # 水压缩系数
        self.Co = 100 * 1e-6 / 6894  # 油压缩系数

        # 压力条件
        self.p_init = 20e6  # 初始压力 (Pa)
        self.p_e = 20e6     # 边界压力 (Pa)

        # 井参数
        self.q_in = 5.0 / 86400  # 注入流量 (m³/s)
        self.re = 0.14 * math.sqrt(2 * 5.0**2)  # 等效半径
        self.rw = 0.05      # 井半径
        self.SS = 3         # 表皮系数
        # 相对渗透率参数（Brooks-Corey模型）
        self.Swr = 0.2      # 残余水饱和度
        self.Sor = 0.2      # 残余油饱和度
        self.nw = 4.0       # 水相指数
        self.no = 2.0       # 油相指数
        # 流体密度 (kg/m³)
        self.rho_o = 800.0  # 原油密度
        self.rho_w = 1000.0 # 水密度


# ── 读取参数 ─────────────────────────────────────────────────────────────────
props  = ReservoirProperties()
Swr    = props.Swr          # 残余水饱和度
Sor    = props.Sor          # 残余油饱和度
nw     = props.nw           # 水相 Corey 指数
no     = props.no           # 油相 Corey 指数
mu_w   = props.mu_w         # 水粘度  (Pa·s)
mu_o   = props.mu_o         # 油粘度  (Pa·s)
Bw     = props.Bw
Bo     = props.Bo
denom  = 1.0 - Swr - Sor   # 可动饱和度范围

print(f"ReservoirProperties 参数")
print(f"  Swr={Swr},  Sor={Sor},  nw={nw},  no={no}")
print(f"  mu_w={mu_w:.3e} Pa·s,  mu_o={mu_o:.3e} Pa·s")
print(f"  可动区间: Sw ∈ [{Swr:.3f}, {1-Sor:.3f}]  (denom={denom:.3f})")

# ── 计算曲线 ─────────────────────────────────────────────────────────────────
Sw  = np.linspace(0.0, 1.0, 1000)
krw = np.zeros_like(Sw)
kro = np.zeros_like(Sw)

mobile = (Sw > Swr) & (Sw < 1.0 - Sor)
Swn         = np.clip((Sw[mobile] - Swr) / denom, 0.0, 1.0)
krw[mobile] = Swn ** nw
kro[mobile] = (1.0 - Swn) ** no
krw[Sw >= 1.0 - Sor] = 1.0   # 端点
kro[Sw <= Swr]        = 1.0   # 端点

# 流度 & 分流系数
lam_w = krw / mu_w / Bw
lam_o = kro / mu_o / Bo
fw    = np.where((lam_w + lam_o) > 0, lam_w / (lam_w + lam_o), 0.0)

# dfw/dSw（数值微分）
dfw_dSw = np.gradient(fw, Sw)

# ── Buckley-Leverett 前缘（切线法）──────────────────────────────────────────
# 切点满足: dfw/dSw * (Sw - Swr) = fw
err = np.abs(dfw_dSw * (Sw - Swr) - fw)
err[Sw < Swr + 0.01] = np.inf
idx_front = np.argmin(err)
Sw_front  = Sw[idx_front]
fw_front  = fw[idx_front]

# ── 交叉点 krw = kro ─────────────────────────────────────────────────────────
diff = np.abs(krw - kro)
diff[~mobile] = np.inf
idx_cross  = np.argmin(diff)
Sw_cross   = Sw[idx_cross]
kr_cross   = (krw[idx_cross] + kro[idx_cross]) / 2

# ── 关键值打印 ────────────────────────────────────────────────────────────────
print(f"\n关键点")
print(f"  krw-kro 交叉点 : Sw={Sw_cross:.4f},  kr={kr_cross:.4f}")
print(f"  BL 前缘饱和度  : Sw={Sw_front:.4f},  fw={fw_front:.4f}")
print(f"  端点 krw(1-Sor): {krw[np.argmin(np.abs(Sw-(1-Sor)))]:.4f}  (应=1.000)")
print(f"  端点 kro(Swr)  : {kro[np.argmin(np.abs(Sw-Swr))]:.4f}   (应=1.000)")

# ── 绘图 ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# fig.suptitle(
#     f"Brooks-Corey Relative Permeability  |  "
#     f"$S_{{wr}}$={Swr:.2f},  $S_{{or}}$={Sor:.2f},  "
#     f"$n_w$={nw:.1f},  $n_o$={no:.1f}",
#     fontsize=13, fontweight="bold"
# )

# ── 图 1: krw & kro ───────────────────────────────────────────────────────────
ax1 = axes[0]
ax1.plot(Sw, krw, color="#1D9E75", lw=2.5, label=r"$k_{rw}$  (water)")
ax1.plot(Sw, kro, color="#E24B4A", lw=2.5, label=r"$k_{ro}$  (oil)")

ax1.axvspan(0.0,       Swr,       alpha=0.08, color="#1D9E75", label=f"irreducible water  $S_{{wr}}$={Swr}")
ax1.axvspan(1.0 - Sor, 1.0,       alpha=0.08, color="#E24B4A", label=f"residual oil  $S_{{or}}$={Sor}")
ax1.axvline(Swr,       color="#1D9E75", lw=1, ls="--", alpha=0.7)
ax1.axvline(1.0 - Sor, color="#E24B4A", lw=1, ls="--", alpha=0.7)

ax1.plot(Sw_cross, kr_cross, "ko", ms=7, zorder=5,
         label=f"crossover  $S_w$={Sw_cross:.3f},  $k_r$={kr_cross:.3f}")

ax1.set_xlabel("Water Saturation  $S_w$", fontsize=11)
ax1.set_ylabel("Relative Permeability  $k_r$", fontsize=11)
# ax1.set_title("$k_{rw}$ and $k_{ro}$", fontsize=11)
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=8.5, loc="upper left")
ax1.grid(True, alpha=0.25)
ax1.text(0.5, -0.18, "(a)", transform=ax1.transAxes, fontsize=13, fontweight="bold", va="top", ha="center")

# ── 图 2: 分流系数 fw ──────────────────────────────────────────────────────────
ax2 = axes[1]
ax2.plot(Sw, fw, color="#534AB7", lw=2.5, label=r"$f_w$")
ax2.axvline(Swr,       color="#1D9E75", lw=1, ls="--", alpha=0.7)
ax2.axvline(1.0 - Sor, color="#E24B4A", lw=1, ls="--", alpha=0.7)

# BL 切线
ax2.plot([Swr, Sw_front, 1.0],
         [0,   fw_front, 1.0],
         color="orange", lw=1.5, ls="--", alpha=0.85,
         label=f"BL tangent  $S_{{front}}$={Sw_front:.3f}")
ax2.plot(Sw_front, fw_front, "o", color="orange", ms=7, zorder=5)

ax2.set_xlabel("Water Saturation  $S_w$", fontsize=11)
ax2.set_ylabel("Fractional Flow  $f_w$", fontsize=11)
# ax2.set_title("Fractional Flow Curve", fontsize=11)
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=8.5, loc="upper left")
ax2.grid(True, alpha=0.25)
ax2.text(0.5, -0.18, "(b)", transform=ax2.transAxes, fontsize=13, fontweight="bold", va="top", ha="center")

# ── 图 3: dfw/dSw（前缘速度）──────────────────────────────────────────────────
ax3 = axes[2]
# 只画可动区间
mask_plot = (Sw >= Swr) & (Sw <= 1 - Sor)
ax3.plot(Sw[mask_plot], dfw_dSw[mask_plot], color="#BA7517", lw=2.5,
         label=r"$\partial f_w / \partial S_w$")
ax3.axvline(Sw_front, color="orange", lw=1.5, ls="--",
            label=f"BL front  $S_w$={Sw_front:.3f}")
ax3.axvline(Swr,       color="#1D9E75", lw=1, ls="--", alpha=0.5)
ax3.axvline(1.0 - Sor, color="#E24B4A", lw=1, ls="--", alpha=0.5)

ax3.set_xlabel("Water Saturation  $S_w$", fontsize=11)
ax3.set_ylabel(r"Fractional Flow Derivative $\partial f_w / \partial S_w$", fontsize=11)
# ax3.set_title("Fractional Flow Derivative\n(shock front velocity)", fontsize=11)
ax3.set_xlim(Swr - 0.02, 1 - Sor + 0.02)
ax3.legend(fontsize=8.5, loc="upper left")
ax3.grid(True, alpha=0.25)
ax3.text(0.5, -0.18, "(c)", transform=ax3.transAxes, fontsize=13, fontweight="bold", va="top", ha="center")
ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

plt.tight_layout()

# ── 保存 ──────────────────────────────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(__file__))
png_path = os.path.join(out_dir, "kr_curves.png")
csv_path = os.path.join(out_dir, "kr_curves.csv")

fig.savefig(png_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n图像已保存: {png_path}")

# CSV
header = "Sw,krw,kro,fw,dfw_dSw,lambda_w,lambda_o"
np.savetxt(csv_path,
           np.column_stack([Sw, krw, kro, fw, dfw_dSw, lam_w, lam_o]),
           delimiter=",", header=header, comments="", fmt="%.8f")
print(f"数据已保存: {csv_path}")
