import torch
import math
import time
import numpy as np
import h5py
import os
import json
from datetime import datetime
from enum import Enum
from utilities3 import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from typing import Dict, List, Tuple, Optional

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

class WellType(Enum):
    INJECTOR = 0
    PRODUCER = 1

class NODE:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0

class CELL:
    def __init__(self, cell_id=0):
        self.id = cell_id
        self.vertices = [-1, -1, -1, -1, -1, -1, -1, -1]
        self.neighbors = [-1, -1, -1, -1, -1, -1]
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.volume = 0
        self.xc = 0
        self.yc = 0
        self.zc = 0
        self.porosity = 0
        self.kx = 0
        self.ky = 0
        self.kz = 0
        self.trans = [0, 0, 0, 0, 0, 0]
        self.transw = [0, 0, 0, 0, 0, 0]
        self.markbc = -2
        self.press = 0
        self.Sw = 0
        self.markbc_Sw = 0
        self.markwell = -1
        self.well_type = None
        self.mobiw = 0
        self.mobio = 0
        self.mobit = 0

class ReservoirGrid:
    def __init__(self, nx=20, ny=20, nz=1, dx=5.0, dy=5.0, dz=1.0, total_length=None):
        """
        初始化网格

        参数:
        nx, ny, nz: 网格数量 (默认 20, 20, 1)
        dx, dy, dz: 网格单元尺寸 (默认 5.0, 5.0, 5.0)
        total_length: 保持的总尺寸 (x_total, y_total, z_total)，如果提供则重新计算dx,dy,dz
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.ncell = nx * ny * nz

        # 如果有指定总尺寸，重新计算单元尺寸
        if total_length is not None:
            x_total, y_total, z_total = total_length
            self.dx = x_total / nx if nx > 0 else 0
            self.dy = y_total / ny if ny > 0 else 0
            self.dz = z_total / nz if nz > 0 else 0
        else:
            self.dx = dx
            self.dy = dy
            self.dz = dz

        self.nodelist = []
        self.celllist = []

        # 验证和显示信息
        self._validate_and_display()

    def _validate_and_display(self):
        """验证网格参数并显示基本信息"""
        if self.dx <= 0 or self.dy <= 0 or self.dz <= 0:
            raise ValueError("网格单元尺寸必须大于0")

        total_x, total_y, total_z = self.get_total_dimensions()
        print(f"创建网格: {self.nx}×{self.ny}×{self.nz}")
        print(f"单元尺寸: dx={self.dx:.2f}, dy={self.dy:.2f}, dz={self.dz:.2f}")
        print(f"总尺寸: x={total_x:.2f}, y={total_y:.2f}, z={total_z:.2f}")

    def get_total_dimensions(self):
        """获取网格的总尺寸"""
        total_x = self.nx * self.dx
        total_y = self.ny * self.dy
        total_z = self.nz * self.dz
        return total_x, total_y, total_z

    def build_grid(self):
        """构建网格"""
        print("  Building grid...")

        # 创建节点
        llz = 0
        for k in range(self.nz + 1):
            lly = 0
            for j in range(self.ny + 1):
                llx = 0
                for i in range(self.nx + 1):
                    node = NODE()
                    node.x = llx
                    node.y = lly
                    node.z = llz
                    self.nodelist.append(node)
                    llx += self.dx if i < self.nx else 0
                lly += self.dy if j < self.ny else 0
            llz += self.dz if k < self.nz else 0

        # 创建单元格
        cell_id = 0
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    cell = CELL(cell_id)

                    # 设置邻居
                    if i > 0:
                        cell.neighbors[0] = cell_id - 1
                    if i < self.nx - 1:
                        cell.neighbors[1] = cell_id + 1
                    if j > 0:
                        cell.neighbors[2] = cell_id - self.nx
                    if j < self.ny - 1:
                        cell.neighbors[3] = cell_id + self.nx
                    if k > 0:
                        cell.neighbors[4] = cell_id - self.nx * self.ny
                    if k < self.nz - 1:
                        cell.neighbors[5] = cell_id + self.nx * self.ny

                    # 设置顶点
                    i0 = k * (self.nx + 1) * (self.ny + 1) + j * (self.nx + 1) + i
                    i1 = i0 + 1
                    i2 = i0 + (self.nx + 1)
                    i3 = i2 + 1
                    i4 = i0 + (self.nx + 1) * (self.ny + 1)
                    i5 = i4 + 1
                    i6 = i4 + (self.nx + 1)
                    i7 = i6 + 1

                    cell.vertices = [i0, i1, i2, i3, i4, i5, i6, i7]

                    # 计算几何属性
                    cell.dx = self.nodelist[i1].x - self.nodelist[i0].x
                    cell.dy = self.nodelist[i2].y - self.nodelist[i0].y
                    cell.dz = self.nodelist[i4].z - self.nodelist[i0].z
                    cell.volume = cell.dx * cell.dy * cell.dz

                    # 计算中心点
                    cell.xc = 0.125 * sum(self.nodelist[v].x for v in cell.vertices)
                    cell.yc = 0.125 * sum(self.nodelist[v].y for v in cell.vertices)
                    cell.zc = 0.125 * sum(self.nodelist[v].z for v in cell.vertices)

                    self.celllist.append(cell)
                    cell_id += 1

        print(f"  Grid built with {len(self.nodelist)} nodes and {len(self.celllist)} cells")

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

class ReservoirSimulator:
    """简化的油藏模拟器 - 用于生成训练数据"""

    def __init__(self, grid, properties):
        self.grid = grid
        self.props = properties
        self.ncell = grid.ncell

        # 初始化张量
        self.neiborvec_w = torch.zeros(self.ncell, dtype=torch.long)
        self.neiborvec_e = torch.zeros(self.ncell, dtype=torch.long)
        self.neiborvec_s = torch.zeros(self.ncell, dtype=torch.long)
        self.neiborvec_n = torch.zeros(self.ncell, dtype=torch.long)

        self.transvec_w = torch.zeros(self.ncell)
        self.transvec_e = torch.zeros(self.ncell)
        self.transvec_s = torch.zeros(self.ncell)
        self.transvec_n = torch.zeros(self.ncell)

        self.presslast = torch.zeros(self.ncell)
        self.qtvec = torch.zeros(self.ncell)
        self.alphavec = torch.zeros(self.ncell)
        self.geometric_trans= {}
        self.geometric_trans_computed =False
        # 初始化邻居张量（只做一次）
        for i, cell in enumerate(self.grid.celllist):
            self.neiborvec_w[i] = max(cell.neighbors[0], 0)
            self.neiborvec_e[i] = max(cell.neighbors[1], 0)
            self.neiborvec_s[i] = max(cell.neighbors[2], 0)
            self.neiborvec_n[i] = max(cell.neighbors[3], 0)

    def set_properties(self, permeability_field):
        """设置属性到网格"""
        for i, cell in enumerate(self.grid.celllist):
            cell.porosity = self.props.poro
            cell.kx = permeability_field[i]
            cell.ky = permeability_field[i]
            cell.kz = permeability_field[i]
            cell.Sw = self.props.Siw
            cell.press = self.props.p_init

            self.presslast[i] = cell.press

    def set_well_conditions(self):
        """设置井条件"""
        nx, ny = self.grid.nx, self.grid.ny
        ncell = self.grid.ncell

        # 注入井 (左上角)
        injector_cell = self.grid.celllist[0]
        injector_cell.markwell = 0
        injector_cell.markbc = -1
        injector_cell.markbc_Sw = 1
        injector_cell.Sw = 1.0
        injector_cell.well_type = WellType.INJECTOR
        self.qtvec[0] = self.props.q_in

        # 生产井 (右下角)
        producer_cell = self.grid.celllist[ncell - 1]
        producer_cell.markwell = 1
        producer_cell.markbc = -1
        producer_cell.well_type = WellType.PRODUCER
        self.qtvec[ncell - 1] = -self.props.q_in

        # 边界条件
        # self.grid.celllist[nx - 1].markbc = 1
        # self.grid.celllist[nx - 1].press = self.props.p_e
        # self.grid.celllist[nx * ny - nx].markbc = 1
        # self.grid.celllist[nx * ny - nx].press = self.props.p_e

    def compute_mobility(self):
        """计算流度 — Brooks-Corey 标准公式，由 ReservoirProperties 参数驱动"""
        Swr   = self.props.Swr
        Sor   = self.props.Sor
        nw    = self.props.nw
        no    = self.props.no
        denom = 1.0 - Swr - Sor          # 可动饱和度范围，如 0.6
        for cell in self.grid.celllist:
            sw  = max(Swr, min(1.0 - Sor, cell.Sw))
            Swn = (sw - Swr) / denom     # 归一化饱和度 ∈ [0, 1]
            krw = Swn ** nw
            kro = (1.0 - Swn) ** no
            cell.mobiw = krw / self.props.mu_w / self.props.Bw
            cell.mobio = kro / self.props.mu_o / self.props.Bo
            cell.mobit = cell.mobiw + cell.mobio

    def compute_transmissibility(self):
        """计算传导率"""
        for i, cell in enumerate(self.grid.celllist):
            for j in range(4):  # 只处理4个水平方向邻居
                je = cell.neighbors[j]
                if je >= 0:
                    mt1 = cell.mobit
                    mt2 = self.grid.celllist[je].mobit
                    k1 = cell.kx
                    k2 = self.grid.celllist[je].kx
                    t1 = mt1 * k1 * self.grid.dx * self.grid.dy / (self.grid.dx / 2)
                    t2 = mt2 * k2 * self.grid.dx * self.grid.dy / (self.grid.dx / 2)
                    tt = 1 / (1 / t1 + 1 / t2) if t1 > 0 and t2 > 0 else 0
                    cell.trans[j] = tt

    def update_trans_tensors(self):
        """更新传导率张量"""
        for i, cell in enumerate(self.grid.celllist):
            self.transvec_w[i] = cell.trans[0]
            self.transvec_e[i] = cell.trans[1]
            self.transvec_s[i] = cell.trans[2]
            self.transvec_n[i] = cell.trans[3]

    def update_alpha_vector(self, dt): #dt=36000
        """更新alpha系数向量"""
        for i, cell in enumerate(self.grid.celllist):
            alpha = cell.porosity * (1 - cell.Sw) * (self.props.Cr + self.props.Co) / self.props.Bo
            alpha += cell.porosity * cell.Sw * (self.props.Cr + self.props.Cw) / self.props.Bw
            alpha *= cell.volume / dt
            self.alphavec[i] = alpha

    def solve_pressure_implicit(self, dt):
        """使用直接法求解压力方程（更精确）"""
        ncell = self.grid.ncell

        # 构建稀疏矩阵
        A = sp.lil_matrix((ncell, ncell))
        b = np.zeros(ncell)

        for idx, cell in enumerate(self.grid.celllist):
            # if cell.markbc == 1:
            #     A[idx, :] = 0
            #     A[idx, idx] = 1.0
            #     b[idx] = self.props.p_e
            #     continue
            # elif cell.markwell == 1:
            #     # 生产井通常有固定流压
            #     A[idx, :] = 0
            #     A[idx, idx] = 1.0
            #     b[idx] = -self.props.p_e  # 生产井流压
            #     continue
            # 对角项（累积项）
            alpha = cell.porosity * ((1 - cell.Sw) * (self.props.Cr + self.props.Co) / self.props.Bo +
                                    cell.Sw * (self.props.Cr + self.props.Cw) / self.props.Bw)
            accum = alpha * cell.volume / dt

            A[idx, idx] = accum
            b[idx] = accum * cell.press + self.qtvec[idx]

            # 邻居项
            for direction, neighbor_idx in enumerate(cell.neighbors[:4]):  # 仅水平方向
                if neighbor_idx >= 0:

                    # 计算传导率（使用已经计算好的传导率）
                    if direction == 0:  # West
                        trans = cell.trans[0]
                    elif direction == 1:  # East
                        trans = cell.trans[1]
                    elif direction == 2:  # South
                        trans = cell.trans[2]
                    elif direction == 3:  # North
                        trans = cell.trans[3]
                    else:
                        trans = 0.0

                    # 添加传导项到矩阵
                    if trans > 0:
                        A[idx, idx] += trans
                        A[idx, neighbor_idx] = -trans

        # 处理边界条件

        # 求解线性系统
        A_csr = A.tocsr()
        p_new = splinalg.spsolve(A_csr, b)

        return p_new

    def update_saturation(self, dt):
        """基于水相质量守恒方程的饱和度显式更新"""
        # 物理参数
        poro = self.props.poro
        Siw = self.props.Siw
        Bw = self.props.Bw
        Cr = self.props.Cr
        Cw = self.props.Cw
        mu_w = self.props.mu_w
        mu_o = self.props.mu_o
        Bo = self.props.Bo
        q_in = self.props.q_in
        cfl_violations = 0
        # # 保存当前压力用于压缩性项计算
        # current_press = torch.tensor([cell.press for cell in self.grid.celllist])

        for i, cell in enumerate(self.grid.celllist):
            # if cell.markbc_Sw == 1:  # 饱和度固定边界（如注水井）
            #     continue

            # 初始化水相总流量
            tfluxsw = 0.0  # 水相流量
            tfluxin = 0.0  # 总流入量

            # 当前网格压力和饱和度
            pi = cell.press
            Sw_i = cell.Sw

            # 计算当前网格的流度 — Brooks-Corey 标准公式
            _Swr   = self.props.Swr
            _Sor   = self.props.Sor
            _denom = 1.0 - _Swr - _Sor
            _sw_i  = max(_Swr, min(1.0 - _Sor, Sw_i))
            _Swn_i = (_sw_i - _Swr) / _denom
            krw_i  = _Swn_i ** self.props.nw
            kro_i  = (1.0 - _Swn_i) ** self.props.no
            mobiw_i = krw_i / mu_w / Bw
            mobio_i = kro_i / mu_o / Bo
            mobit_i = mobiw_i + mobio_i

            # 遍历四个方向的邻居
            for direction in range(4):
                neighbor_idx = cell.neighbors[direction]
                if neighbor_idx >= 0:
                    neighbor_cell = self.grid.celllist[neighbor_idx]
                    pj = neighbor_cell.press
                    # 如果邻居压力高，流体从邻居流入当前网格
                    if pj > pi:
                        # 传导率（使用计算好的传导率）
                        trans = cell.trans[direction]

                        # 总流量 = 传导率 × 压差
                        fluxin = trans * (pj - pi)
                        tfluxin += fluxin
                        upstream_idx = neighbor_idx
                        upstream_cell = self.grid.celllist[upstream_idx]
                        # 计算邻居网格的水相分流系数 f_w = λ_w/λ_t
                        # Sw_j = upstream_cell.Sw
                        # a_j = (1 - Sw_j) / (1 - Siw)
                        # b_j = (Sw_j - Siw) / (1 - Siw)
                        # kro_j = a_j * a_j * (1 - b_j * b_j)
                        # krw_j = b_j * b_j * b_j * b_j
                        # mobiw_j = krw_j / mu_w / Bw
                        # mobio_j = kro_j / mu_o / Bo
                        # mobit_j = mobiw_j + mobio_j
                        # fw_j = mobiw_j / mobit_j
                        fw_up = upstream_cell.mobiw / upstream_cell.mobit
                        # 水相流量 = 总流量 × 水相分流系数
                        tfluxsw += fluxin * fw_up

            # 总流出量 = -总流入量
            tfluxout = -tfluxin

            # 流出部分的水相流量
            fw_i = mobiw_i / mobit_i  # 当前网格的水相分流系数
            tfluxsw = tfluxsw + tfluxout * fw_i
            # 井点源汇处理
            if cell.markwell == 0:  # 注水井
                tfluxsw += q_in  # 注水增加水相流量
            elif cell.markwell == 1:  # 生产井
                # 生产井：总产量 × 水相分流系数
                tfluxsw += -q_in * fw_i  # q_in为正值，生产井为负值

            # 压缩性项（考虑压力和饱和度的耦合）
            # 来自：- (∂p/∂t) * φ * S_w * (c_r + c_w) / B_w
            # 离散形式：- (p_i^{n+1} - p_i^n)/Δt * φ * S_w * (c_r + c_w) / B_w
            dp_dt = (cell.press - self.presslast[i].item()) / dt
            compress_term = -dp_dt * poro * Sw_i * (Cr + Cw) / Bw
            # compress_term = -Sw_i * (Cr + Cw) *(cell.press - self.presslast[i].item())
            tfluxsw += compress_term

            # ==================== CFL检查 ====================
            # 计算当前网格的最大允许时间步

            # 1. 计算归一化饱和度
            denominator = 1 - Siw - (1 - self.props.Sor)  # 1 - Siw - Sor
            if denominator > 1e-10:
                s = max(0.0, min(1.0, (Sw_i - Siw) / denominator))
            else:
                s = 0.0

            # 2. 计算分相流函数及其导数
            fw = fw_i  # 当前网格的分相流系数

            # 计算导数 fw' (对归一化饱和度s)
            # 使用简单的有限差分
            epsilon = 1e-6
            Sw_plus = min(1.0, Sw_i + epsilon)
            Sw_minus = max(0.0, Sw_i - epsilon)

            # 计算Sw_plus的fw — Brooks-Corey
            _swp   = max(_Swr, min(1.0 - _Sor, Sw_plus))
            _Swn_p = (_swp - _Swr) / _denom
            krw_plus  = _Swn_p ** self.props.nw
            kro_plus  = (1.0 - _Swn_p) ** self.props.no
            mobiw_plus = krw_plus / mu_w / Bw
            mobio_plus = kro_plus / mu_o / Bo
            fw_plus = mobiw_plus / (mobiw_plus + mobio_plus)

            # 计算Sw_minus的fw — Brooks-Corey
            _swm   = max(_Swr, min(1.0 - _Sor, Sw_minus))
            _Swn_m = (_swm - _Swr) / _denom
            krw_minus  = _Swn_m ** self.props.nw
            kro_minus  = (1.0 - _Swn_m) ** self.props.no
            mobiw_minus = krw_minus / mu_w / Bw
            mobio_minus = kro_minus / mu_o / Bo
            fw_minus = mobiw_minus / (mobiw_minus + mobio_minus)

            # 导数 ∂fw/∂Sw
            fw_prime_Sw = (fw_plus - fw_minus) / (Sw_plus - Sw_minus)

            # 转换为 ∂fw/∂s
            if denominator > 1e-10:
                fw_prime_s = fw_prime_Sw * denominator
            else:
                fw_prime_s = fw_prime_Sw

            # 3. 计算总流入量 (已经计算了tfluxin)
            total_influx = tfluxin

            # 加上井的贡献
            if cell.markwell == 0:  # 注入井
                total_influx += q_in

            # 4. CFL条件检查
            if total_influx > 0 and fw_prime_s > 0:
                # 按照论文公式：Δt ≤ φ_i V_i / (η_m × max[f_w'(s)])
                dt_cell = poro * cell.volume / (total_influx * fw_prime_s)

                # 添加安全系数
                dt_cell_safe = dt_cell * 0.8

                # 检查是否违反CFL
                if dt > dt_cell_safe:
                    cfl_violations += 1
                    if cfl_violations <= 5:  # 只打印前5个违规
                        print(f"CFL warning - Cell {i}: dt_current={dt:.1f}s, "
                            f"dt_max_allowed={dt_cell_safe:.1f}s, "
                            f"total_influx={total_influx:.3e}, fw'={fw_prime_s:.3e}")
            # ==================== CFL检查结束 ====================

            # 饱和度显式更新
            # 水相质量守恒方程：φ * V * ∂S_w/∂t = B_w * ∑(f_w * q)
            # 离散形式：S_w^{n+1} = S_w^n + (Δt * B_w / (φ * V)) * ∑(f_w * q)
            delta_Sw = tfluxsw * dt * Bw / (poro * cell.volume) # + compress_term/poro
            cell.Sw += delta_Sw

            # 确保饱和度在物理范围内
            cell.Sw = max(Siw, min(1.0, cell.Sw))

    def update_saturation_paper_style(self, dt):
        """按照论文Eq.15实现饱和度更新"""
        # 确保传导率已经计算
        # if not hasattr(self, 'geometric_trans_computed') or not self.geometric_trans_computed:
        #     self.compute_geometric_transmissibility()
        #     self.geometric_trans_computed = True

        for i, cell in enumerate(self.grid.celllist):
            # 1. 计算水相流量项
            water_flux = 0.0
            # 四个方向：W, E, S, N
            for direction in range(4):
                j = cell.neighbors[direction]
                if j >= 0:  # 有效邻居
                    neighbor_cell = self.grid.celllist[j]

                    # 确定上游网格
                    if neighbor_cell.press > cell.press:
                        upstream_idx = j
                    else:
                        upstream_idx = i

                    upstream_cell = self.grid.celllist[upstream_idx]
                    Sw_up = upstream_cell.Sw

                    # 计算上游水相流度 λ_w = k_rw / μ_w
                    krw_up = self.compute_krw(Sw_up)
                    lambda_w_up = krw_up / self.props.mu_w

                    if direction == 0:
                        T = cell.trans[0]
                    elif direction == 1:
                        T = cell.trans[1]
                    elif direction == 2:
                        T = cell.trans[2]
                    elif direction == 3:
                        T = cell.trans[3]
                    else:
                        T = 0.0

                    # 势差（包含重力）
                    phi_i = cell.press # - self.props.rho_w * self.props.g * cell.depth
                    phi_j = self.grid.celllist[j].press # - self.props.rho_w * self.props.g * self.grid.celllist[j].depth

                    water_flux += lambda_w_up * T * (phi_j - phi_i)

            # 2. 源汇项
            if cell.markwell == 0:  # 注入井
                water_flux += self.props.q_in
            elif cell.markwell == 1:  # 生产井
                # 生产井：总流量 × 水相分流系数
                fw = self.compute_fw(cell.Sw)
                water_flux += -self.props.q_in * fw

            # 3. 压缩性项
            dp = cell.press - self.presslast[i].item()
            compress_term = -cell.Sw * (self.props.Cw + self.props.Cr) * dp

            # 4. 更新饱和度（论文Eq.15）
            delta_Sw = (water_flux * dt *self.props.Bw / (self.props.poro * cell.volume) +
                        compress_term / self.props.poro)
            cell.Sw += delta_Sw

            # 5. 物理限制
            cell.Sw = max(self.props.Siw, min(1.0, cell.Sw))

    def compute_geometric_transmissibility(self):
        """计算几何传导率（不含流度） - 论文Eq.13，只计算一次"""
        for i, cell in enumerate(self.grid.celllist):
            for direction in range(4):
                j = cell.neighbors[direction]
                if j >= 0:
                    neighbor_cell = self.grid.celllist[j]

                    # 几何参数
                    if direction in [0, 1]:  # 东西方向
                        A = self.grid.dy * self.grid.dz
                        d1 = self.grid.dx / 2
                        d2 = self.grid.dx / 2
                    else:  # 南北方向
                        A = self.grid.dx * self.grid.dz
                        d1 = self.grid.dy / 2
                        d2 = self.grid.dy / 2

                    # 调和平均渗透率
                    k1 = cell.kx
                    k2 = neighbor_cell.kx
                    if k1 > 0 and k2 > 0:
                        k_harm = 2 * k1 * k2 / (k1 + k2)
                    else:
                        k_harm = 0.0

                    # 几何传导率 T_geo = A * k / d
                    T_geo = A * k_harm / (d1 + d2)

                    # 存储到字典中，键为 (i, j) 对
                    key = (i, j) if i < j else (j, i)
                    self.geometric_trans[key] = T_geo

        self.geometric_trans_computed = True

    def get_geometric_transmissibility(self, i, j):
        """获取两个网格间的几何传导率"""
        if not hasattr(self, 'geometric_trans'):
            self.compute_geometric_transmissibility()

        key = (i, j)  if i < j else (j, i)
        return self.geometric_trans.get(key, 0.0)

    def compute_total_transmissibility(self):
        """计算总传导率（包含流度），用于压力方程"""
        for i, cell in enumerate(self.grid.celllist):
            for direction in range(4):
                j = cell.neighbors[direction]
                if j >= 0:
                    # 获取几何传导率
                    T_geo = self.get_geometric_transmissibility(i, j)

                    # 获取上游总流度
                    if self.grid.celllist[j].press > cell.press:
                        upstream_idx = j
                    else:
                        upstream_idx = i

                    upstream_cell = self.grid.celllist[upstream_idx]
                    mobit_up = upstream_cell.mobit

                    # 总传导率 = 几何传导率 × 上游总流度
                    T_total = T_geo * mobit_up

                    # 存储到 cell.trans 中（用于压力方程）
                    cell.trans[direction] = T_total

    def compute_krw(self, Sw):
            """计算水相相对渗透率 - 按照论文Fig.5b的曲线"""
            Siw = self.props.Siw  # 不可动水饱和度
            Sor = 1.0 - self.props.Sor  # 残余油饱和度

            # 归一化饱和度
            if Sw <= Siw:
                return 0.0
            elif Sw >= 1.0 - Sor:
                return 1.0
            else:
                # 使用论文的相对渗透率模型（假设是Corey类型）
                # 归一化水饱和度
                Swn = (Sw - Siw) / (1.0 - Siw - Sor)

                # 论文Fig.5b显示水相相对渗透率大约是 Swn^4
                # 你可以根据论文精确调整
                krw = Swn ** 4
                return krw

    def compute_kro(self, Sw):
            """计算油相相对渗透率 - 按照论文Fig.5b的曲线"""
            Siw = self.props.Siw
            Sor = 1.0 - self.props.Sor

            if Sw <= Siw:
                return 1.0
            elif Sw >= 1.0 - Sor:
                return 0.0
            else:
                Swn = (Sw - Siw) / (1.0 - Siw - Sor)
                # 油相相对渗透率大约是 (1 - Swn)^2
                kro = (1.0 - Swn) ** 2
                return kro

    def compute_fw(self, Sw):
            """计算水相分流系数 f_w = λ_w / (λ_w + λ_o)"""
            krw = self.compute_krw(Sw)
            kro = self.compute_kro(Sw)

            # 流度 = 相对渗透率 / 粘度
            lambda_w = krw / self.props.mu_w
            lambda_o = kro / self.props.mu_o

            if lambda_w + lambda_o == 0:
                return 0.0
            return lambda_w / (lambda_w + lambda_o)

    def run_simulation(self, permeability_field, nt=10, dt=43200):
        """运行简化模拟"""
        # 设置属性
        self.set_properties(permeability_field)
        self.set_well_conditions()
        self.compute_geometric_transmissibility()
        # 存储结果
        time_steps = []
        pressure_history = []
        saturation_history = []

        # # 初始状态
        # time_steps.append(0.0)
        # pressure_history.append([cell.press for cell in self.grid.celllist])
        # saturation_history.append([cell.Sw for cell in self.grid.celllist])

        for t in range(nt):
            current_time = t * dt / 86400  # 转换为天
            # 计算流度和传导率
            self.compute_mobility()
            # self.compute_transmissibility()
            self.compute_total_transmissibility()
            self.update_trans_tensors()
            # 更新alpha系数
            self.update_alpha_vector(dt)

            # 求解压力
            press_result = self.solve_pressure_implicit(dt)

            # 更新压力
            for i, cell in enumerate(self.grid.celllist):
                cell.press = press_result[i].item()
                # if cell.markbc == 1:
                #     cell.press = self.props.p_e

            # 更新饱和度
            self.update_saturation(dt)

            # 更新上一时间步压力
            self.presslast = press_result.copy()

            # 存储结果
            time_steps.append(current_time)
            pressure_history.append([cell.press for cell in self.grid.celllist])
            saturation_history.append([cell.Sw for cell in self.grid.celllist])

            # 显示进度
            if (t + 1) % 5 == 0:
                print(f"Time step {t+1}/{nt}")

        return {
            'time_steps': np.array(time_steps),
            'pressure': np.array(pressure_history),
            'saturation': np.array(saturation_history),
            'permeability': np.array(permeability_field),
            'porosity': np.full(self.ncell, self.props.poro),

            'grid_info': {
                'nx': self.grid.nx,
                'ny': self.grid.ny,
                'nz': self.grid.nz,
                'dx': self.grid.dx,
                'dy': self.grid.dy,
                'dz': self.grid.dz
            }
        }

class PermeabilityGenerator:
    """快速渗透率场生成器 - 支持多种地质统计模型"""

    def __init__(self, nx=20, ny=20, nz=1):
        """
        初始化渗透率生成器

        Parameters:
        -----------
        nx : int
            x方向网格数
        ny : int
            y方向网格数
        nz : int
            z方向网格数
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.ncells = nx * ny * nz

    def generate_uniform(self, mean_perm=1e-15, variation=0.3):
        """
        生成均匀分布的渗透率

        Parameters:
        -----------
        mean_perm : float
            平均渗透率 (m²)
        variation : float
            变化幅度 (0-1)

        Returns:
        --------
        numpy.ndarray
            渗透率数组，形状为(ncells,)
        """
        low = mean_perm * (1 - variation)
        high = mean_perm * (1 + variation)
        permvec = np.random.uniform(low, high, self.ncells)
        return permvec

    def generate_lognormal(self, mean_perm=1e-15, std_factor=0.5):
        """
        生成对数正态分布的渗透率（最常用的地质统计模型）

        Parameters:
        -----------
        mean_perm : float
            对数平均值
        std_factor : float
            对数标准差因子

        Returns:
        --------
        numpy.ndarray
            渗透率数组，形状为(ncells,)
        """
        # 转换为对数空间
        mean_log = np.log(mean_perm)
        std_log = np.log(1 + std_factor)

        # 生成对数正态分布
        permvec = np.random.lognormal(mean_log, std_log, self.ncells)
        return permvec

    def generate_gaussian_field(self, mean_perm=1e-15, correlation_length=3, sigma=1.0):
        """
        生成高斯随机场（具有空间相关性）

        Parameters:
        -----------
        mean_perm : float
            平均渗透率 (m²)
        correlation_length : float
            相关长度（网格数）
        sigma : float
            高斯滤波器的标准差

        Returns:
        --------
        numpy.ndarray
            渗透率数组，形状为(ncells,)
        """
        # 快速生成：使用numpy的随机函数
        noise = np.random.randn(self.ncells)

        # 重塑为3D网格
        if self.nz == 1:
            noise_2d = noise.reshape(self.ny, self.nx)
            smoothed = gaussian_filter(noise_2d, sigma=sigma)
            smoothed_flat = smoothed.flatten()
        else:
            noise_3d = noise.reshape(self.nz, self.ny, self.nx)
            smoothed = gaussian_filter(noise_3d, sigma=sigma)
            smoothed_flat = smoothed.flatten()

        # 调整到目标渗透率范围
        mean_val = np.mean(smoothed_flat)
        std_val = np.std(smoothed_flat)

        # 标准化并映射到渗透率范围
        normalized = (smoothed_flat - mean_val) / std_val
        permvec = mean_perm * np.exp(normalized * 0.5)

        # 确保正值
        permvec = np.abs(permvec)

        return permvec

    def generate_channel_field(self, background=5e-15, channel=50e-15, channel_width=3):
        """
        生成河道砂体渗透率场

        Parameters:
        -----------
        background : float
            背景渗透率 (m²)
        channel : float
            河道渗透率 (m²)
        channel_width : int
            河道宽度（网格数）

        Returns:
        --------
        numpy.ndarray
            渗透率数组，形状为(ncells,)
        """
        field = np.full(self.ncells, background)

        # 创建简单的直线河道
        channel_center = self.ny // 2
        half_width = channel_width // 2

        for k in range(self.nz):
            for i in range(self.nx):
                for j in range(max(0, channel_center - half_width),
                              min(self.ny, channel_center + half_width + 1)):
                    idx = k * self.nx * self.ny + j * self.nx + i
                    field[idx] = channel

        return field

    def generate_layered_field(self, n_layers=3, perm_range=(5e-15, 25e-15)):
        """
        生成层状渗透率场

        Parameters:
        -----------
        n_layers : int
            层数
        perm_range : tuple
            渗透率范围 (min, max)

        Returns:
        --------
        numpy.ndarray
            渗透率数组，形状为(ncells,)
        """
        field = np.zeros(self.ncells)

        layer_thickness = self.ny // n_layers
        # 生成每层的渗透率
        layer_values = np.random.uniform(perm_range[0], perm_range[1], n_layers)

        for layer in range(n_layers):
            layer_k = layer_values[layer]
            start_j = layer * layer_thickness
            end_j = min((layer + 1) * layer_thickness, self.ny)

            for k in range(self.nz):
                for j in range(start_j, end_j):
                    for i in range(self.nx):
                        idx = k * self.nx * self.ny + j * self.nx + i
                        field[idx] = layer_k

        return field

    def generate_heterogeneous_field(self, pattern_type='gaussian', **kwargs):
        """
        生成各种模式的渗透率场（向后兼容）

        Parameters:
        -----------
        pattern_type : str
            模式类型: 'gaussian', 'channel', 'layered', 'uniform', 'lognormal'
        **kwargs : dict
            各生成方法的关键参数

        Returns:
        --------
        numpy.ndarray
            渗透率数组，形状为(ncells,)
        """
        if pattern_type == 'gaussian':
            return self.generate_gaussian_field(
                mean_perm=kwargs.get('mean_perm', 15e-15),
                correlation_length=kwargs.get('correlation_length', 3),
                sigma=kwargs.get('sigma', 1.0)
            )
        elif pattern_type == 'channel':
            return self.generate_channel_field(
                background=kwargs.get('background', 5e-15),
                channel=kwargs.get('channel', 50e-15),
                channel_width=kwargs.get('channel_width', 3)
            )
        elif pattern_type == 'layered':
            return self.generate_layered_field(
                n_layers=kwargs.get('n_layers', 3),
                perm_range=kwargs.get('perm_range', (5e-15, 25e-15))
            )
        elif pattern_type == 'uniform':
            return self.generate_uniform(
                mean_perm=kwargs.get('mean_perm', 15e-15),
                variation=kwargs.get('variation', 0.3)
            )
        elif pattern_type == 'lognormal':
            return self.generate_lognormal(
                mean_perm=kwargs.get('mean_perm', 15e-15),
                std_factor=kwargs.get('std_factor', 0.5)
            )
        else:
            raise ValueError(f"未知的模式类型: {pattern_type}")

    def generate_advanced_heterogeneous(self, background_perm=15e-15,
                                        high_perm_zones=3,
                                        high_perm_value=50e-15,
                                        zone_radius_range=(2, 5)):
        """
        生成带有高渗透条带的异质性渗透率

        Parameters:
        -----------
        background_perm : float
            背景渗透率 (m²)
        high_perm_zones : int
            高渗透条带数量
        high_perm_value : float
            高渗透条带的渗透率值
        zone_radius_range : tuple
            高渗透区半径范围

        Returns:
        --------
        numpy.ndarray
            渗透率数组，形状为(ncells,)
        """
        permvec = np.ones(self.ncells) * background_perm

        # 随机放置高渗透条带
        for _ in range(high_perm_zones):
            # 随机选择条带位置
            center_x = np.random.randint(0, self.nx)
            center_y = np.random.randint(0, self.ny)
            radius = np.random.randint(zone_radius_range[0], zone_radius_range[1] + 1)

            # 应用高渗透率
            for k in range(self.nz):
                for j in range(max(0, center_y - radius), min(self.ny, center_y + radius + 1)):
                    for i in range(max(0, center_x - radius), min(self.nx, center_x + radius + 1)):
                        distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                        if distance <= radius:
                            idx = k * self.nx * self.ny + j * self.nx + i
                            # 距离越近，渗透率越高
                            factor = 1 - (distance / radius) * 0.5
                            permvec[idx] = (background_perm +
                                          (high_perm_value - background_perm) * factor)

        return permvec

    def reshape_to_grid(self, permvec):
        """
        将一维渗透率数组重塑为网格形状

        Parameters:
        -----------
        permvec : numpy.ndarray
            一维渗透率数组

        Returns:
        --------
        numpy.ndarray
            三维渗透率网格
        """
        if self.nz == 1:
            return permvec.reshape(self.ny, self.nx)
        else:
            return permvec.reshape(self.nz, self.ny, self.nx)

    def get_statistics(self, permvec):
        """
        获取渗透率统计信息

        Parameters:
        -----------
        permvec : numpy.ndarray
            渗透率数组

        Returns:
        --------
        dict
            包含统计信息的字典
        """
        return {
            'mean': np.mean(permvec),
            'median': np.median(permvec),
            'min': np.min(permvec),
            'max': np.max(permvec),
            'std': np.std(permvec),
            'p10': np.percentile(permvec, 10),
            'p90': np.percentile(permvec, 90),
            'd90_d10': np.percentile(permvec, 90) / np.percentile(permvec, 10),
            'cv': np.std(permvec) / np.mean(permvec)  # 变异系数
        }

    def generate_homogeneous_anisotropic(self, kx=1e-15, ky=None, kz=None):
        """
        生成均匀各项均值渗透率场

        Parameters:
        -----------
        kx : float
            x方向渗透率 (m²)
        ky : float, optional
            y方向渗透率。如果为None，则与kx相同（各向同性）
        kz : float, optional
            z方向渗透率。如果为None，则与kx相同（各向同性）

        Returns:
        --------
        numpy.ndarray
            渗透率数组，形状为(ncells,)
        """
        # 设置默认值
        if ky is None:
            ky = kx
        if kz is None:
            kz = kx

        # 创建均匀场
        permvec = np.ones(self.ncells) * kx  # 所有网格使用kx值

        # 打印各向异性信息
        if not (kx == ky == kz):
            print(f"各向异性渗透率场生成完成")
            print(f"  kx = {kx:.2e} m²")
            print(f"  ky = {ky:.2e} m²")
            print(f"  kz = {kz:.2e} m²")
            print(f"  各向异性比例: kx/ky = {kx/ky:.2f}, ky/kz = {ky/kz:.2f}")
        else:
            print(f"各向同性均匀场生成完成: k = {kx:.2e} m²")

        return permvec


class ReservoirDatasetCreator:
    """油藏数据集创建器 - 简化版本"""

    def __init__(self, output_dir='./reservoir_fno_dataset', nx=20, ny=20):
        self.output_dir = f"{output_dir}_{nx}x{ny}"
        self.nx = nx
        self.ny = ny

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化网格和属性
        print("Initializing reservoir simulator...")
        self.grid = ReservoirGrid(nx=nx, ny=ny, total_length=(100,100,1))
        self.grid.build_grid()
        self.props = ReservoirProperties()

        # 初始化模拟器
        self.simulator = ReservoirSimulator(self.grid, self.props)
        self.perm_generator = PermeabilityGenerator(nx=nx, ny=ny)

        # 数据集信息
        self.dataset_info = {
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'grid_shape': [nx, ny, 1],
            'cell_size': [self.grid.dx, self.grid.dy, self.grid.dz],
            'n_samples': 0,
            'simulation_settings': {
                'nt': 10,  # 减少时间步数
                'dt': 43200,  # 12小时时间步
                'total_time_days': 10  # 总共10天
            }
        }

    def generate_single_sample(self, sample_id, perm_pattern='gaussian', nt=10, dt=43200):
        """生成单个样本"""
        print(f"Generating sample {sample_id} with pattern: {perm_pattern}")

        # 生成渗透率场
        # perm_field = self.perm_generator.generate_heterogeneous_field(perm_pattern)
        perm_field = self.perm_generator.generate_homogeneous_anisotropic()

        # 运行模拟
        start_time = time.time()
        results = self.simulator.run_simulation(perm_field, nt=nt, dt=dt)
        sim_time = time.time() - start_time

        # 添加样本信息
        sample_data = {
            'sample_id': sample_id,
            'permeability_pattern': perm_pattern,
            'simulation_time': sim_time,
            **results
        }
        self.dataset_info["simulation_settings"]["nt"] = nt
        self.dataset_info["simulation_settings"]["dt"] = dt
        print(f"  Sample {sample_id} completed in {sim_time:.2f}s")

        return sample_data

    def create_dataset(self, n_samples=5, patterns=None):
        """创建简化数据集"""
        if patterns is None:
            patterns = ['gaussian',   'uniform', 'lognormal']

        print(f"Creating reservoir dataset with {n_samples} samples...")
        print("=" * 60)

        # 统计信息
        pattern_counts = {pattern: 0 for pattern in patterns}
        total_time = 0

        # 创建HDF5文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        h5_filename = os.path.join(self.output_dir, f'reservoir_fno_{timestamp}_samples_{n_samples}.h5')

        with h5py.File(h5_filename, 'w') as f:
            # 创建数据组
            data_group = f.create_group('data')

            # 生成样本
            for sample_idx in range(n_samples):
                # 随机选择模式
                pattern = np.random.choice(patterns)
                # pattern = "gaussian"  # 仅使用高斯模式以减少计算时间
                pattern_counts[pattern] += 1

                # 生成样本
                sample_data = self.generate_single_sample(
                    sample_idx,
                    perm_pattern=pattern,
                    nt=500,  # 500个时间步
                    dt=43200/4  # 12小时时间步
                )

                total_time += sample_data['simulation_time']

                # 创建样本组
                sample_name = f'sample_{sample_idx:03d}'
                sample_group = data_group.create_group(sample_name)

                # 保存数据
                sample_group.create_dataset('permeability',
                                          data=sample_data['permeability'].reshape(self.ny, self.nx))
                sample_group.create_dataset('porosity',
                                          data=sample_data['porosity'].reshape(self.ny, self.nx))
                sample_group.create_dataset('pressure',
                                          data=sample_data['pressure'])
                sample_group.create_dataset('saturation',
                                          data=sample_data['saturation'])
                sample_group.create_dataset('time_steps',
                                          data=sample_data['time_steps'])

                # 保存元数据
                sample_group.attrs['sample_id'] = sample_idx
                sample_group.attrs['pattern'] = str(pattern)
                sample_group.attrs['simulation_time'] = float(sample_data['simulation_time'])

                # 进度显示
                if (sample_idx + 1) % 1 == 0:  # 每个样本都显示
                    print(f"  Completed {sample_idx + 1}/{n_samples} samples")

        # 更新数据集信息
        self.dataset_info['n_samples'] = n_samples
        self.dataset_info['pattern_distribution'] = pattern_counts
        self.dataset_info['total_generation_time'] = total_time

        print("=" * 60)
        print(f"Dataset creation completed!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average per sample: {total_time/n_samples:.2f}s")
        print(f"Saved to: {h5_filename}")

        # 保存数据集信息到JSON文件
        info_file = os.path.join(self.output_dir, f'dataset_info_{timestamp}.json')
        with open(info_file, 'w') as f:
            # 转换numpy类型以便JSON序列化
            def convert(obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                else:
                    return str(obj)

            import json
            # 递归转换
            def convert_dict(d):
                new_dict = {}
                for k, v in d.items():
                    if isinstance(v, dict):
                        new_dict[k] = convert_dict(v)
                    elif isinstance(v, (list, tuple)):
                        new_dict[k] = [convert(item) for item in v]
                    else:
                        new_dict[k] = convert(v)
                return new_dict

            json.dump(convert_dict(self.dataset_info), f, indent=2)

        return h5_filename


class ReservoirDataVisualizer:
    """油藏数据可视化器"""

    def __init__(self, h5_file_path, output_dir='./visualizations'):
        """
        初始化可视化器

        Parameters:
        -----------
        h5_file_path : str
            HDF5文件路径
        output_dir : str
            输出目录
        """
        self.h5_file_path = h5_file_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 加载数据
        self.load_data()

        # 设置颜色映射
        self.cmap_pressure = 'hot'
        self.cmap_saturation = 'coolwarm'
        self.cmap_permeability = 'viridis'
        self.cmap_porosity = 'Blues'

    def load_data(self):
        """加载HDF5文件中的数据"""
        with h5py.File(self.h5_file_path, 'r') as f:
            # 获取样本列表
            self.sample_names = list(f['data'].keys())
            self.n_samples = len(self.sample_names)

            # 加载第一个样本的数据结构作为参考
            sample_group = f['data'][self.sample_names[0]]
            self.grid_shape = sample_group['permeability'].shape  # (ny, nx)
            self.ny, self.nx = self.grid_shape
            self.n_cells = self.nx * self.ny

            # 获取时间步信息
            if 'time_steps' in sample_group:
                self.time_steps = sample_group['time_steps'][:]
                self.nt = len(self.time_steps) -1
            else:
                self.nt = sample_group['pressure'].shape[0] - 1
                self.time_steps = np.arange(self.nt + 1)

            print(f"Loaded {self.n_samples} samples")
            print(f"Grid size: {self.nx} x {self.ny}")
            print(f"Time steps: {self.nt}")

    def visualize_single_sample(self, sample_idx=0, save_fig=False):
        """
        可视化单个样本的所有数据

        Parameters:
        -----------
        sample_idx : int
            样本索引
        save_fig : bool
            是否保存图像
        """
        with h5py.File(self.h5_file_path, 'r') as f:
            sample_name = self.sample_names[sample_idx]
            sample_group = f['data'][sample_name]

            # 获取数据
            permeability = sample_group['permeability'][:]  # (ny, nx)
            porosity = sample_group['porosity'][:]          # (ny, nx)
            pressure = sample_group['pressure'][:]          # (nt+1, n_cells)
            saturation = sample_group['saturation'][:]      # (nt+1, n_cells)
            time_steps = sample_group['time_steps'][:] if 'time_steps' in sample_group else self.time_steps

            # 获取元数据
            pattern = sample_group.attrs.get('pattern', 'unknown')
            sample_id = sample_group.attrs.get('sample_id', sample_idx)

            # 重塑压力场和饱和度场为网格形式
            pressure_grid = pressure.reshape(-1, self.ny, self.nx)
            saturation_grid = saturation.reshape(-1, self.ny, self.nx)

            # 创建大图
            fig = plt.figure(figsize=(20, 16))
            fig.suptitle(f'Reservoir Data Visualization - Sample {sample_id} (Pattern: {pattern})',
                        fontsize=16, fontweight='bold', y=0.98)

            # 子图1: 渗透率场
            ax1 = plt.subplot(3, 4, 1)
            im1 = ax1.imshow(permeability, cmap=self.cmap_permeability, origin='lower')
            ax1.set_title('Permeability Field', fontsize=12, fontweight='bold')
            ax1.set_xlabel('X Coordinate', fontsize=10)
            ax1.set_ylabel('Y Coordinate', fontsize=10)
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            # 子图2: 孔隙度场
            ax2 = plt.subplot(3, 4, 2)
            im2 = ax2.imshow(porosity, cmap=self.cmap_porosity, origin='lower')
            ax2.set_title('Porosity Field', fontsize=12, fontweight='bold')
            ax2.set_xlabel('X Coordinate', fontsize=10)
            # ax2.set_ylabel('Y Coordinate', fontsize=10)
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            # 子图3: 渗透率直方图
            ax3 = plt.subplot(3, 4, 3)
            ax3.hist(permeability.flatten(), bins=30, edgecolor='black', alpha=0.7)
            ax3.set_title('Permeability Distribution', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Permeability (m²)', fontsize=10)
            ax3.set_ylabel('Frequency', fontsize=10)
            ax3.grid(True, alpha=0.3)

            # 子图4: 孔隙度直方图
            ax4 = plt.subplot(3, 4, 4)
            ax4.hist(porosity.flatten(), bins=30, edgecolor='black', alpha=0.7)
            ax4.set_title('Porosity Distribution', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Porosity', fontsize=10)
            ax4.set_ylabel('Frequency', fontsize=10)
            ax4.grid(True, alpha=0.3)

            # 选择关键时间步可视化压力和饱和度
            time_indices = [0, self.nt//2, self.nt]
            time_labels = ['Initial', f'Time {time_steps[time_indices[1]]:.0f}', 'Final']

            # 压力场在不同时间步
            for i, (t_idx, t_label) in enumerate(zip(time_indices, time_labels)):
                ax = plt.subplot(3, 4, 5 + i)
                im = ax.imshow(pressure_grid[t_idx], cmap=self.cmap_pressure, origin='lower')
                ax.set_title(f'Pressure at {t_label} Time', fontsize=12, fontweight='bold')
                if i == 0:
                    ax.set_ylabel('Y Coordinate', fontsize=10)
                # ax.set_xlabel('X Coordinate', fontsize=10)
                # ax.set_ylabel('Y Coordinate', fontsize=10)
                if i == 2:
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Pressure (Pa)', fontsize=9)
                else:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # 饱和度场在不同时间步
            for i, (t_idx, t_label) in enumerate(zip(time_indices, time_labels)):
                ax = plt.subplot(3, 4, 9 + i)
                im = ax.imshow(saturation_grid[t_idx], cmap=self.cmap_saturation,
                              origin='lower', vmin=0, vmax=1)
                ax.set_title(f'Water Saturation at {t_label} Time', fontsize=12, fontweight='bold')
                if i == 0:
                    ax.set_ylabel('Y Coordinate', fontsize=10)
                ax.set_xlabel('X Coordinate', fontsize=10)
                # ax.set_ylabel('Y Coordinate', fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # 子图12: 平均压力和饱和度随时间变化
            ax12 = plt.subplot(3, 4, 12)

            # 计算平均压力和饱和度
            mean_pressure = np.mean(pressure_grid, axis=(1, 2))
            mean_saturation = np.mean(saturation_grid, axis=(1, 2))

            # 绘制平均压力
            ax12a = ax12
            color1 = 'tab:red'
            ax12a.plot(time_steps, mean_pressure, color=color1, linewidth=2, label='Mean Pressure')
            ax12a.set_xlabel('Time', fontsize=10)
            ax12a.set_ylabel('Mean Pressure (Pa)', color=color1, fontsize=10)
            ax12a.tick_params(axis='y', labelcolor=color1)
            ax12a.grid(True, alpha=0.3)

            # 添加第二个y轴用于饱和度
            ax12b = ax12a.twinx()
            color2 = 'tab:blue'
            ax12b.plot(time_steps, mean_saturation, color=color2, linewidth=2,
                      linestyle='--', label='Mean Saturation')
            ax12b.set_ylabel('Mean Water Saturation', color=color2, fontsize=10)
            ax12b.tick_params(axis='y', labelcolor=color2)

            # 合并图例
            lines1, labels1 = ax12a.get_legend_handles_labels()
            lines2, labels2 = ax12b.get_legend_handles_labels()
            ax12a.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            ax12.set_title('Temporal Evolution of Mean Values', fontsize=12, fontweight='bold')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            if save_fig:
                output_path = os.path.join(self.output_dir, f'sample_{sample_id}_visualization.png')
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"Saved visualization to: {output_path}")

            plt.show()

            return fig

    def visualize_pressure_evolution(self, sample_idx=0, save_fig=False):
        """
        可视化压力场随时间演化过程

        Parameters:
        -----------
        sample_idx : int
            样本索引
        save_fig : bool
            是否保存图像
        """
        with h5py.File(self.h5_file_path, 'r') as f:
            sample_name = self.sample_names[sample_idx]
            sample_group = f['data'][sample_name]

            # 获取数据
            pressure = sample_group['pressure'][:]  # (nt+1, n_cells)
            time_steps = sample_group['time_steps'][:] if 'time_steps' in sample_group else self.time_steps
            pattern = sample_group.attrs.get('pattern', 'unknown')
            sample_id = sample_group.attrs.get('sample_id', sample_idx)

            # 重塑为网格形式
            pressure_grid = pressure.reshape(-1, self.ny, self.nx)

            # 选择要显示的时间步
            if self.nt <= 8:
                display_times = np.arange(self.nt + 1)
            else:
                display_times = np.linspace(0, self.nt, 9, dtype=int)

            n_rows = 3
            n_cols = 3

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
            fig.suptitle(f'Pressure Field Evolution - Sample {sample_id} (Pattern: {pattern})',
                        fontsize=16, fontweight='bold', y=0.95)

            axes = axes.flatten()

            for i, t_idx in enumerate(display_times):
                if i >= len(axes):
                    break

                ax = axes[i]
                im = ax.imshow(pressure_grid[t_idx], cmap=self.cmap_pressure, origin='lower')
                ax.set_title(f'Time Step {t_idx}\nTime = {time_steps[t_idx]:.1f}', fontsize=10)
                ax.set_xlabel('X Coordinate', fontsize=8)
                ax.set_ylabel('Y Coordinate', fontsize=8)
                ax.tick_params(labelsize=8)

                # 添加颜色条到每个子图
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                plt.colorbar(im, cax=cax).set_label('Pressure (Pa)', fontsize=8)

            # 隐藏多余的子图
            for i in range(len(display_times), len(axes)):
                axes[i].axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            if save_fig:
                output_path = os.path.join(self.output_dir, f'sample_{sample_id}_pressure_evolution.png')
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"Saved pressure evolution to: {output_path}")

            plt.show()

            return fig

    def visualize_saturation_evolution(self, sample_idx=0, save_fig=False):
        """
        可视化饱和度场随时间演化过程

        Parameters:
        -----------
        sample_idx : int
            样本索引
        save_fig : bool
            是否保存图像
        """
        with h5py.File(self.h5_file_path, 'r') as f:
            sample_name = self.sample_names[sample_idx]
            sample_group = f['data'][sample_name]

            # 获取数据
            saturation = sample_group['saturation'][:]  # (nt+1, n_cells)
            time_steps = sample_group['time_steps'][:] if 'time_steps' in sample_group else self.time_steps
            pattern = sample_group.attrs.get('pattern', 'unknown')
            sample_id = sample_group.attrs.get('sample_id', sample_idx)

            # 重塑为网格形式
            saturation_grid = saturation.reshape(-1, self.ny, self.nx)

            # 选择要显示的时间步
            if self.nt <= 8:
                display_times = np.arange(self.nt + 1)
            else:
                display_times = np.linspace(0, self.nt, 9, dtype=int)

            n_rows = 3
            n_cols = 3

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
            fig.suptitle(f'Water Saturation Evolution - Sample {sample_id} (Pattern: {pattern})',
                        fontsize=16, fontweight='bold', y=0.95)

            axes = axes.flatten()

            for i, t_idx in enumerate(display_times):
                if i >= len(axes):
                    break

                ax = axes[i]
                im = ax.imshow(saturation_grid[t_idx], cmap=self.cmap_saturation,
                              origin='lower', vmin=0, vmax=1)
                ax.set_title(f'Time Step {t_idx}\nTime = {time_steps[t_idx]:.1f}', fontsize=10)
                ax.set_xlabel('X Coordinate', fontsize=8)
                ax.set_ylabel('Y Coordinate', fontsize=8)
                ax.tick_params(labelsize=8)

                # 添加颜色条到每个子图
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                plt.colorbar(im, cax=cax).set_label('Water Saturation', fontsize=8)

            # 隐藏多余的子图
            for i in range(len(display_times), len(axes)):
                axes[i].axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            if save_fig:
                output_path = os.path.join(self.output_dir, f'sample_{sample_id}_saturation_evolution.png')
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"Saved saturation evolution to: {output_path}")

            plt.show()

            return fig

    def create_comparison_dashboard(self, sample_indices=None, save_fig=False):
        """
        创建多个样本的对比仪表板

        Parameters:
        -----------
        sample_indices : list
            要对比的样本索引列表
        save_fig : bool
            是否保存图像
        """
        if sample_indices is None:
            sample_indices = list(range(min(4, self.n_samples)))

        n_samples = len(sample_indices)

        # 创建大图
        fig = plt.figure(figsize=(20, 5 * n_samples))
        fig.suptitle('Reservoir Data Comparison Dashboard', fontsize=18, fontweight='bold', y=0.98)

        with h5py.File(self.h5_file_path, 'r') as f:
            for row_idx, sample_idx in enumerate(sample_indices):
                sample_name = self.sample_names[sample_idx]
                sample_group = f['data'][sample_name]

                # 获取数据
                permeability = sample_group['permeability'][:]
                porosity = sample_group['porosity'][:]
                pressure = sample_group['pressure'][:]
                saturation = sample_group['saturation'][:]
                pattern = sample_group.attrs.get('pattern', 'unknown')
                sample_id = sample_group.attrs.get('sample_id', sample_idx)

                # 重塑压力和饱和度
                pressure_grid = pressure.reshape(-1, self.ny, self.nx)
                saturation_grid = saturation.reshape(-1, self.ny, self.nx)

                # 获取最终时间步
                final_pressure = pressure_grid[-1]
                final_saturation = saturation_grid[-1]

                # 子图1: 渗透率
                ax1 = plt.subplot(n_samples, 4, row_idx * 4 + 1)
                im1 = ax1.imshow(permeability, cmap=self.cmap_permeability, origin='lower')
                ax1.set_title(f'Sample {sample_id}\nPermeability ({pattern})', fontsize=12)
                ax1.set_xlabel('X', fontsize=10)
                ax1.set_ylabel('Y', fontsize=10)
                plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

                # 子图2: 孔隙度
                ax2 = plt.subplot(n_samples, 4, row_idx * 4 + 2)
                im2 = ax2.imshow(porosity, cmap=self.cmap_porosity, origin='lower')
                ax2.set_title(f'Porosity', fontsize=12)
                ax2.set_xlabel('X', fontsize=10)
                ax2.set_ylabel('Y', fontsize=10)
                plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

                # 子图3: 最终压力
                ax3 = plt.subplot(n_samples, 4, row_idx * 4 + 3)
                im3 = ax3.imshow(final_pressure, cmap=self.cmap_pressure, origin='lower')
                ax3.set_title('Final Pressure', fontsize=12)
                ax3.set_xlabel('X', fontsize=10)
                ax3.set_ylabel('Y', fontsize=10)
                plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

                # 子图4: 最终饱和度
                ax4 = plt.subplot(n_samples, 4, row_idx * 4 + 4)
                im4 = ax4.imshow(final_saturation, cmap=self.cmap_saturation,
                                origin='lower', vmin=0, vmax=1)
                ax4.set_title('Final Water Saturation', fontsize=12)
                ax4.set_xlabel('X', fontsize=10)
                ax4.set_ylabel('Y', fontsize=10)
                plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])

        if save_fig:
            output_path = os.path.join(self.output_dir, 'comparison_dashboard.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison dashboard to: {output_path}")

        plt.show()

        return fig

    def create_statistics_report(self, save_report=False):
        """
        创建数据集统计报告

        Parameters:
        -----------
        save_report : bool
            是否保存报告
        """
        statistics = []

        with h5py.File(self.h5_file_path, 'r') as f:
            for sample_name in self.sample_names:
                sample_group = f['data'][sample_name]

                # 获取数据
                permeability = sample_group['permeability'][:]
                porosity = sample_group['porosity'][:]
                pressure = sample_group['pressure'][:]
                saturation = sample_group['saturation'][:]
                pattern = sample_group.attrs.get('pattern', 'unknown')
                sample_id = sample_group.attrs.get('sample_id', 0)

                # 计算统计量
                stats = {
                    'sample_id': sample_id,
                    'pattern': pattern,
                    'permeability': {
                        'mean': np.mean(permeability),
                        'std': np.std(permeability),
                        'min': np.min(permeability),
                        'max': np.max(permeability),
                        'median': np.median(permeability)
                    },
                    'porosity': {
                        'mean': np.mean(porosity),
                        'std': np.std(porosity),
                        'min': np.min(porosity),
                        'max': np.max(porosity),
                        'median': np.median(porosity)
                    }
                }
                statistics.append(stats)

        # 打印报告
        print("=" * 80)
        print("RESERVOIR DATASET STATISTICAL REPORT")
        print("=" * 80)

        for stats in statistics:
            print(f"\nSample {stats['sample_id']} - Pattern: {stats['pattern']}")
            print("-" * 40)
            print(f"Permeability:")
            print(f"  Mean: {stats['permeability']['mean']:.2e} m²")
            print(f"  Std: {stats['permeability']['std']:.2e} m²")
            print(f"  Range: [{stats['permeability']['min']:.2e}, {stats['permeability']['max']:.2e}] m²")
            print(f"  Median: {stats['permeability']['median']:.2e} m²")
            print(f"Porosity:")
            print(f"  Mean: {stats['porosity']['mean']:.3f}")
            print(f"  Std: {stats['porosity']['std']:.3f}")
            print(f"  Range: [{stats['porosity']['min']:.3f}, {stats['porosity']['max']:.3f}]")
            print(f"  Median: {stats['porosity']['median']:.3f}")

        # 保存报告
        if save_report:
            import json
            report_file = os.path.join(self.output_dir, 'statistics_report.json')

            # 转换numpy类型为Python原生类型
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj

            with open(report_file, 'w') as f:
                json.dump(convert_numpy(statistics), f, indent=2)
            print(f"\nStatistics report saved to: {report_file}")

        return statistics


# 使用示例
def visualize_dataset(h5_file_path, output_dir='./visualizations'):
    """
    主函数：可视化油藏数据集

    Parameters:
    -----------
    h5_file_path : str
        HDF5文件路径
    output_dir : str
        输出目录
    """
    # 创建可视化器
    visualizer = ReservoirDataVisualizer(h5_file_path, output_dir)

    print(f"Dataset Visualization Tool")
    print("=" * 60)

    # 1. 显示数据集统计报告
    print("\n1. Generating statistical report...")
    visualizer.create_statistics_report(save_report=True)

    # 2. 可视化第一个样本
    print("\n2. Visualizing first sample...")
    visualizer.visualize_single_sample(sample_idx=0, save_fig=True)

    # 3. 可视化压力演化
    print("\n3. Visualizing pressure evolution...")
    visualizer.visualize_pressure_evolution(sample_idx=0, save_fig=True)

    # 4. 可视化饱和度演化
    print("\n4. Visualizing saturation evolution...")
    visualizer.visualize_saturation_evolution(sample_idx=0, save_fig=True)

    # 5. 创建对比仪表板（如果有多个样本）
    if visualizer.n_samples > 1:
        print("\n5. Creating comparison dashboard...")
        n_compare = min(4, visualizer.n_samples)
        sample_indices = list(range(n_compare))
        visualizer.create_comparison_dashboard(sample_indices=sample_indices, save_fig=True)

    print("\n" + "=" * 60)
    print("Visualization completed!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)




class ReservoirOperatorDataset(Dataset):
    """
    油藏算子学习数据集
    """
    def __init__(
        self,
        h5_file_path: str,
        mode: str = 'train',
        train_test_split: float = 0.8,
        input_steps: int = 3,    # 输入时间步数 (k+1)
        output_steps: int = 3,   # 输出时间步数 (k+1)
        stride: int = 1,         # 滑动步长
        time_window: int = 1,
        normalize: bool = True,
        sequence_length: int = 10,  # 总时间步数
        use_all_time_steps: bool = False  # 是否使用所有时间步作为训练对
    ):
        """
        初始化数据集

        Args:
            h5_file_path: HDF5数据文件路径
            mode: 数据集模式 'train' 或 'test'
            train_test_split: 训练测试分割比例
            time_window: 时间窗口大小（state_k 到 state_k+time_window）
            normalize: 是否进行标准化
            sequence_length: 每个样本的时间序列长度
            use_all_time_steps: 是否使用所有连续时间步作为训练对
        """
        super().__init__()

        self.h5_file_path = h5_file_path
        self.mode = mode
        self.time_window = time_window
        self.normalize = normalize
        self.sequence_length = sequence_length
        self.use_all_time_steps = use_all_time_steps
        # 检查参数
        assert input_steps >= 1, "输入步数至少为1"
        assert output_steps >= 1, "输出步数至少为1"
        self.input_steps = input_steps  # 输入序列长度 (k+1)
        self.output_steps = output_steps  # 输出序列长度 (k+1)
        self.stride = stride
        # 加载数据
        self.data = self._load_data()

        # 划分训练测试集
        self.sample_indices = self._split_train_test(train_test_split)

        # 如果需要，计算标准化参数
        if normalize:
            self._compute_normalization_params()

        print(f"Loaded {len(self.sample_indices)} {mode} samples")
        print(f"Total time steps per sample: {sequence_length}")
        print(f"Time window: {time_window}")
        print(f"Loaded {len(self.sample_indices)} {mode} samples")
        print(f"Input sequence length: {input_steps} steps")
        print(f"Output sequence length: {output_steps} steps")
        print(f"Sliding stride: {stride}")

    def _load_data(self) -> Dict:
        """从HDF5文件加载数据"""
        with h5py.File(self.h5_file_path, 'r') as f:
            data = {}

            # 获取所有样本
            data_group = f['data']
            sample_names = list(data_group.keys())

            # 加载每个样本的数据
            for sample_name in sample_names:
                sample_group = data_group[sample_name]

                # 读取数据
                permeability = sample_group['permeability'][:]  # (ny, nx)
                porosity = sample_group['porosity'][:]          # (ny, nx)
                pressure = sample_group['pressure'][:]          # (nt+1, ncell)
                saturation = sample_group['saturation'][:]      # (nt+1, ncell)
                time_steps = sample_group['time_steps'][:]      # (nt+1,)

                # 获取网格信息
                ny, nx = permeability.shape

                # 重塑压力和饱和度为网格形式
                pressure_grid = pressure.reshape(pressure.shape[0], ny, nx)  # (nt+1, ny, nx)
                saturation_grid = saturation.reshape(saturation.shape[0], ny, nx)  # (nt+1, ny, nx)

                # 存储样本数据
                data[sample_name] = {
                    'permeability': permeability,      # (ny, nx)
                    'porosity': porosity,              # (ny, nx)
                    'pressure': pressure_grid,         # (nt+1, ny, nx)
                    'saturation': saturation_grid,     # (nt+1, ny, nx)
                    'time_steps': time_steps,          # (nt+1,)
                    'nx': nx,
                    'ny': ny,
                    'sample_id': sample_group.attrs['sample_id'],
                    'pattern': sample_group.attrs['pattern']
                }

        return data

    def _split_train_test(self, split_ratio: float) -> List[Tuple]:
        """划分训练测试集"""
        all_samples = list(self.data.keys())
        n_samples = len(all_samples)

        # 随机打乱
        np.random.shuffle(all_samples)

        # 分割
        split_idx = int(n_samples * split_ratio)

        if self.mode == 'train':
            selected_samples = all_samples[:split_idx]
        else:
            selected_samples = all_samples[split_idx:]

        # 为每个样本生成索引（样本名，起始时间步）
        sample_indices = []
        for sample_name in selected_samples:
            sample_data = self.data[sample_name]
            total_time_steps = sample_data['pressure'].shape[0]
            if 1:
                # 使用所有连续时间步作为训练对
                for start_idx in range(total_time_steps - self.time_window):
                    sample_indices.append((sample_name, start_idx))
            else:
                # 只使用相邻时间步
                for start_idx in range(total_time_steps - 1):
            # min_required_steps = self.input_steps + self.output_steps - 1
            # if total_time_steps < min_required_steps:
            #     print(f"Warning: Sample {sample_name} has only {total_time_steps} steps, "
            #         f"but requires at least {min_required_steps} steps. Skipping.")
            #     continue

            # # 生成滑动窗口
            # # 最后一个输入窗口的结束位置是 total_time_steps - output_steps
            # max_start_idx = total_time_steps - self.output_steps - self.input_steps + 1

            # for start_idx in range(0, max_start_idx + 1, self.stride):
            #     # 输入窗口: [start_idx, start_idx + input_steps - 1]
            #     # 输出窗口: [start_idx + input_steps - 1, start_idx + input_steps + output_steps - 2]
            #     # 注意：时刻 start_idx + input_steps - 1 同时出现在输入和输出中
            #     end_idx = start_idx + self.input_steps + self.output_steps - 2

            #     if end_idx < total_time_steps:
                    sample_indices.append((sample_name, start_idx))

        return sample_indices

    def _compute_normalization_params(self):
        """计算标准化参数"""
        all_permeability = []
        all_porosity = []
        all_pressure = []
        all_saturation = []

        for sample_name in self.data.keys():
            sample_data = self.data[sample_name]
            all_permeability.append(sample_data['permeability'])
            all_porosity.append(sample_data['porosity'])
            all_pressure.append(sample_data['pressure'])
            all_saturation.append(sample_data['saturation'])

        # 合并所有数据
        self.permeability_mean = np.mean(np.concatenate([p.ravel() for p in all_permeability]))
        self.permeability_std = np.std(np.concatenate([p.ravel() for p in all_permeability]))

        self.porosity_mean = np.mean(np.concatenate([p.ravel() for p in all_porosity]))
        self.porosity_std = np.std(np.concatenate([p.ravel() for p in all_porosity]))

        self.pressure_mean = np.mean(np.concatenate([p.ravel() for p in all_pressure]))
        self.pressure_std = np.std(np.concatenate([p.ravel() for p in all_pressure]))

        self.saturation_mean = np.mean(np.concatenate([s.ravel() for s in all_saturation]))
        self.saturation_std = np.std(np.concatenate([s.ravel() for s in all_saturation]))

        print(f"Normalization parameters:")
        print(f"  Permeability: mean={self.permeability_mean:.2e}, std={self.permeability_std:.2e}")
        print(f"  Porosity: mean={self.porosity_mean:.3f}, std={self.porosity_std:.3f}")
        print(f"  Pressure: mean={self.pressure_mean:.2e}, std={self.pressure_std:.2e}")
        print(f"  Saturation: mean={self.saturation_mean:.3f}, std={self.saturation_std:.3f}")

    def _normalize(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """标准化数据"""
        if not self.normalize:
            return data

        if data_type == 'permeability':
            return (data-self.permeability_mean) / self.permeability_std
        elif data_type == 'porosity':
            return (data-self.porosity_mean) / self.porosity_std
        elif data_type == 'pressure':
            return (data-self.pressure_mean) / self.pressure_std
        elif data_type == 'saturation':
            return (data-self.saturation_mean) / self.saturation_std
        else:
            return data

    def _denormalize(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """反标准化数据"""
        if not self.normalize:
            return data

        if data_type == 'permeability':
            return data * self.permeability_std + self.permeability_mean
        elif data_type == 'porosity':
            return data* self.porosity_std + self.porosity_mean
        elif data_type == 'pressure':
            return data* self.pressure_std + self.pressure_mean
        elif data_type == 'saturation':
            return data* self.saturation_std + self.saturation_mean
        else:
            return data

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个训练样本"""
        sample_name, start_idx = self.sample_indices[idx]
        sample_data = self.data[sample_name]

        # 确定结束时间步
        end_idx = start_idx + self.time_window

        # 获取输入数据（状态k）
        permeability = sample_data['permeability']  # (ny, nx)
        porosity = sample_data['porosity']          # (ny, nx)
        pressure_k = sample_data['pressure'][start_idx]  # (ny, nx)
        saturation_k = sample_data['saturation'][start_idx]  # (ny, nx)

        # 获取目标数据（状态k+1或状态k+time_window）
        if end_idx < sample_data['pressure'].shape[0]:
            pressure_target = sample_data['pressure'][end_idx]  # (ny, nx)
            saturation_target = sample_data['saturation'][end_idx]  # (ny, nx)
        else:
            # 如果超出范围，使用最后一个时间步
            pressure_target = sample_data['pressure'][-1]
            saturation_target = sample_data['saturation'][-1]

        # 标准化
        if self.normalize:
            permeability = self._normalize(permeability, 'permeability')
            porosity = self._normalize(porosity, 'porosity')
            pressure_k = self._normalize(pressure_k, 'pressure')
            saturation_k = self._normalize(saturation_k, 'saturation')
            pressure_target = self._normalize(pressure_target, 'pressure')
            saturation_target = self._normalize(saturation_target, 'saturation')

        # 堆叠输入特征
        # 方式1: 将所有特征堆叠在通道维度上
        input_features = np.stack([
            permeability,      # 通道0: 渗透率
            porosity,          # 通道1: 孔隙度
            pressure_k,        # 通道2: 当前压力
            saturation_k       # 通道3: 当前饱和度
        ], axis=0)  # (4, ny, nx)

        # 堆叠输出特征
        output_features = np.stack([
            pressure_target,   # 通道0: 下一时间步压力
            saturation_target  # 通道1: 下一时间步饱和度
        ], axis=0)  # (2, ny, nx)

        # 转换为PyTorch张量
        input_tensor = torch.FloatTensor(input_features)
        output_tensor = torch.FloatTensor(output_features)

        # 获取时间步信息
        time_step = sample_data['time_steps'][start_idx]
        delta_time = sample_data['time_steps'][end_idx] - time_step if end_idx < len(sample_data['time_steps']) else 0

        return {
            'input': input_tensor,           # (4, ny, nx)
            'output': output_tensor,         # (2, ny, nx)
            'sample_id': sample_data['sample_id'],
            'pattern': sample_data['pattern'],
            'time_step': time_step,
            'delta_time': delta_time,
            'start_idx': start_idx,
            'grid_shape': (sample_data['ny'], sample_data['nx'])
        }

    def get_full_sequence(self, sample_name: str) -> Dict[str, np.ndarray]:
        """获取完整的时间序列用于测试"""
        sample_data = self.data[sample_name]

        # 获取完整序列
        permeability = sample_data['permeability']  # (ny, nx)
        porosity = sample_data['porosity']          # (ny, nx)
        pressure_seq = sample_data['pressure']      # (T, ny, nx)
        saturation_seq = sample_data['saturation']  # (T, ny, nx)
        time_steps = sample_data['time_steps']      # (T,)

        # 标准化
        if self.normalize:
            permeability = self._normalize(permeability, 'permeability')
            porosity = self._normalize(porosity, 'porosity')
            pressure_seq = self._normalize(pressure_seq, 'pressure')
            saturation_seq = self._normalize(saturation_seq, 'saturation')

        return {
            'permeability': permeability,
            'porosity': porosity,
            'pressure': pressure_seq,
            'saturation': saturation_seq,
            'time_steps': time_steps,
            'nx': sample_data['nx'],
            'ny': sample_data['ny'],
            'sample_id': sample_data['sample_id'],
            'pattern': sample_data['pattern']
        }


def create_data_loaders(
    h5_file_path: str,
    batch_size: int = 8,
    train_test_split: float = 0.8,
    time_window: int = 1,
    normalize: bool = True,
    num_workers: int = 4,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和测试数据加载器

    Args:
        h5_file_path: HDF5数据文件路径
        batch_size: 批量大小
        train_test_split: 训练测试分割比例
        time_window: 时间窗口大小
        normalize: 是否标准化
        num_workers: 数据加载工作进程数
        shuffle_train: 是否打乱训练数据

    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 创建训练数据集
    train_dataset = ReservoirOperatorDataset(
        h5_file_path=h5_file_path,
        mode='train',
        train_test_split=train_test_split,
        time_window=time_window,
        normalize=normalize,
        stride=1,
        input_steps=2,
        output_steps=2
    )

    # 创建测试数据集
    test_dataset = ReservoirOperatorDataset(
        h5_file_path=h5_file_path,
        mode='test',
        train_test_split=train_test_split,
        time_window=time_window,
        normalize=normalize,
        stride=1,
        input_steps=2,
        output_steps=2
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")

    return train_loader, test_loader


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.fft



class _EncoderBlock(nn.Module):
    """Encoder block: GroupNorm + GELU, optional downsampling via AvgPool2d"""
    def __init__(self, in_ch, out_ch, downsample=True, dropout=False):
        super().__init__()
        ng = min(32, out_ch)
        # self.pool = nn.AvgPool2d(2, 2) if downsample else nn.Identity()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(ng, out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(ng, out_ch), nn.GELU(),
        ]
        if dropout: layers.append(nn.Dropout2d(0.1))
        self.encode = nn.Sequential(*layers)
        self.pool = None
        if downsample:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        return self.encode(x)


class _DecoderBlock(nn.Module):
    """
    Decoder block:
    - ConvTranspose2d: 学习上采样（可训练）
    - F.interpolate:   对齐 skip 尺寸（处理奇数分辨率的边界误差）
    - skip concat + Conv: 融合多尺度特征
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        ng = min(32, out_ch)
        # ★ 先用 ConvTranspose2d 学习上采样（in_ch -> in_ch，尺寸 x2）
        # self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        # 再卷积融合 skip
        self.decode = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.GroupNorm(ng, out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(ng, out_ch), nn.GELU(),
        )

    def forward(self, x, skip):
        # x = self.up(x)                     # ConvTranspose2d 学习上采样（x2）
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        # F.interpolate 只做微小的尺寸校正（如 5->10 vs 5->11 的1像素差）
        return self.decode(torch.cat([x, skip], dim=1))


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class AdaptiveUNet(nn.Module):
    """
    Adaptive-depth U-Net.

    Improvements over original U_net in main_ufno.py:
    - F.interpolate replaces ConvTranspose2d / manual crop -> resolution-invariant
    - GroupNorm replaces BatchNorm -> stable for small batches
    - Residual connection: output = UNet(x) + residual_proj(x)
    - Adaptive depth: default depth=3 prevents 20x20 -> 1x1 collapse

    Pooling path for 20x20 grid with depth=3:
      enc1 (no pool)  : 20x20 -> 20x20
      enc2 (pool)     : 20x20 -> 10x10
      enc3 (pool)     : 10x10 ->  5x5
      bottleneck      :  5x5  ->  5x5
      dec3 (up)       :  5x5  -> 10x10
      dec2 (up)       : 10x10 -> 20x20
      output          : 20x20 -> 20x20
    """
    def __init__(self, in_channels, out_channels, base_channels=32, max_depth=3):
        super().__init__()
        self.max_depth = max_depth
        enc_ch = [in_channels]; ch = base_channels
        self.encoders = nn.ModuleList()
        for d in range(max_depth):
            self.encoders.append(_EncoderBlock(enc_ch[-1], ch, downsample=(d > 0)))
            enc_ch.append(ch); ch = min(ch*2, 256)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_ch[-1], ch, 3, padding=1),
            nn.GroupNorm(min(32, ch), ch), nn.GELU())
        self.decoders = nn.ModuleList()
        dec_in = ch
        for d in reversed(range(max_depth)):
            skip_ch = enc_ch[d+1]; out_ch = enc_ch[d+1]
            self.decoders.append(_DecoderBlock(dec_in, skip_ch, out_ch))
            dec_in = out_ch
        self.output_conv = nn.Conv2d(enc_ch[1], out_channels, 1)
        self.residual_proj = (nn.Conv2d(in_channels, out_channels, 1)
                              if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        residual = self.residual_proj(x)
        enc_feats = []; out = x
        for enc in self.encoders:
            out = enc(out); enc_feats.append(out)
        out = self.bottleneck(out)             # broadcast 加法
        for dec, skip in zip(self.decoders, reversed(enc_feats)):
            out = dec(out, skip)
        return self.output_conv(out)  + residual   # residual connection

class UNetFNO2d(nn.Module):
    """
    UNet-FNO (recommended): AdaptiveUNet as feature preprocessor,
    followed by FNO spectral layers for global long-range dependencies.

    Fixes over original UnetFNO2d:
    1. Depth limited to 3 (no 20->1 collapse)
    2. Skip connections via F.interpolate (resolution-invariant)
    3. Residual connection in UNet (no information loss)
    """
    def __init__(self, modes1, modes2, width, n_fno_layers=4,
                 in_channels=4, out_channels=2,
                 unet_base_ch=32, unet_depth=3):
        super().__init__()
        self.width = width; self.n_fno_layers = n_fno_layers
        self.fc0 = nn.Linear(in_channels, width)
        self.unet = AdaptiveUNet(width, width, unet_base_ch, unet_depth)

        self.convs = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(n_fno_layers)])
        self.ws = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_fno_layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        self.bias3 = nn.Parameter(torch.zeros(1, 1, 1,128))
        self.biases = nn.Parameter(torch.zeros(n_fno_layers, 1, self.width, 1, 1))
        # self.refine = nn.Sequential(
        #     nn.Conv2d(width, width, 3, padding=1),
        #     nn.GroupNorm(min(32, width), width), nn.GELU(),
        #     nn.Conv2d(width, width, 3, padding=1)
        #     )

    def forward(self, x):
        x = x.permute(0,2,3,1); x = self.fc0(x); x = x.permute(0,3,1,2)
            # multi-scale feature extraction (residual, res-invariant)
        for i, (conv, w) in enumerate(zip(self.convs, self.ws)):
            # x = conv(x) + w(x)
            x = self.unet(x)
            x = conv(x) + w(x.view(x.size(0), self.width, -1)).view(x.size(0), self.width, x.size(2), x.size(3)) + self.biases[i]
            if i < self.n_fno_layers-1: x = F.gelu(x)
        x = x.permute(0,2,3,1)
        x = F.gelu(self.fc1(x) ); x = self.fc2(x)
        return x.permute(0,3,1,2)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())



class TwoPhasePhysicsConstraint:
    """
    使用具体油藏参数的两相流物理约束计算器
    """

    def __init__(self, props: ReservoirProperties, grid_size=(20, 20),
                 dx=5.0, dy=5.0, dt=86400.0, device='cuda'):
        """
        初始化物理约束参数

        Args:
            props: 油藏物性参数
            grid_size: 网格大小 (ny, nx)
            dx, dy: 网格间距 (m)
            dt: 时间步长 (秒)
            device: 计算设备
        """
        self.props = props
        self.grid_size = grid_size
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.device = device

        # 物理常数
        self.g = 9.81  # 重力加速度 m/s²
        self.Pa_to_MPa = 1e-6  # Pa到MPa的转换

        # 创建网格坐标
        self._create_grid()

        # 预计算井索引（假设井在网格中心）
        self._setup_well_locations()

        print(f"Physics Constraint Initialized:")
        print(f"  Grid: {grid_size}, dx={dx}m, dy={dy}m")
        print(f"  Time step: {dt}s ({dt/86400:.2f} days)")
        print(f"  Porosity: {props.poro}")
        print(f"  Oil viscosity: {props.mu_o:.3e} Pa·s")
        print(f"  Water viscosity: {props.mu_w:.3e} Pa·s")

    def _create_grid(self):
        """创建网格坐标系统"""
        ny, nx = self.grid_size
        x_coords = torch.linspace(0, (nx-1)*self.dx, nx, device=self.device)
        y_coords = torch.linspace(0, (ny-1)*self.dy, ny, device=self.device)

        self.X, self.Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        self.X = self.X.T  # (ny, nx)
        self.Y = self.Y.T  # (ny, nx)

        # 计算网格中心深度（假设简单水平层）
        self.D = torch.zeros_like(self.X)  # 深度，简化处理

        # 计算网格体积
        self.cell_volume = self.dx * self.dy * 1.0  # 假设厚度为1m

    def _setup_well_locations(self):
        """设置井位置（示例：注入井在左下角，生产井在右上角）"""
        ny, nx = self.grid_size

        # 注入井位置（左上）
        self.injector_i = 0  # 第一行
        self.injector_j = 0       # 第一列

        # 生产井位置（右下角）
        self.producer_i = ny - 1       # 最后一行
        self.producer_j = nx - 1  # 最后一列

    def relative_permeability(self, Sw):
        """
        计算相对渗透率（Brooks-Corey模型）

        Args:
            Sw: 水饱和度 (0-1)

        Returns:
            krw: 水相相对渗透率
            kro: 油相相对渗透率
        """
        # 归一化饱和度
        Swn = (Sw - self.props.Swr) / (1 - self.props.Swr - self.props.Sor)
        Swn = torch.clamp(Swn, 0, 1)

        # Brooks-Corey模型
        krw = Swn ** self.props.nw
        kro = (1 - Swn) ** self.props.no

        return krw, kro

    def compute_formation_volume_factor(self, P, phase='water'):
        """
        计算地层体积系数（考虑压缩性）

        Args:
            P: 压力 (Pa)
            phase: 'water' 或 'oil'

        Returns:
            B: 地层体积系数
        """
        if phase == 'water':
            B_ref = self.props.Bw
            C = self.props.Cw
        else:  # oil
            B_ref = self.props.Bo
            C = self.props.Co

        # 使用压缩系数公式: B = B_ref * exp[C * (P - P_ref)]
        # 简化处理：假设压缩系数很小，使用线性近似
        P_ref = self.props.p_init
        B = B_ref * (1 + C * (P - P_ref))

        return B

    def compute_transmissibility(self, permeability):
        """
        计算网格间的传导率（考虑单位转换）

        Args:
            permeability: 渗透率张量 (batch, ny, nx) 单位：mD

        Returns:
            Tx: x方向传导率 (batch, ny, nx-1) 单位：m³/(Pa·s)
            Ty: y方向传导率 (batch, ny-1, nx) 单位：m³/(Pa·s)
        """
        if permeability.dim() == 2:
            permeability = permeability.unsqueeze(0)  # 添加batch维度

        batch_size, ny, nx = permeability.shape

        # 转换单位：mD → m² (1 mD = 9.869e-16 m²)
        k_m2 = permeability * 9.869e-16

        # x方向传导率（在界面处求调和平均）
        kx_left = k_m2[:, :, :-1]
        kx_right = k_m2[:, :, 1:]
        kx_harmonic = 2 * kx_left * kx_right / (kx_left + kx_right + 1e-20)

        # 传导率公式: T = k * A / (μ * L)
        # 这里使用简化形式，后面会在流动项中考虑粘度
        Tx = kx_harmonic * self.dy / self.dx  # (batch, ny, nx-1)

        # y方向传导率
        ky_top = k_m2[:, :-1, :]
        ky_bottom = k_m2[:, 1:, :]
        ky_harmonic = 2 * ky_top * ky_bottom / (ky_top + ky_bottom + 1e-20)
        Ty = ky_harmonic * self.dx / self.dy  # (batch, ny-1, nx)

        return Tx, Ty

    def compute_phase_potential(self, P, phase='water'):
        """
        计算相势（考虑重力和毛管压力）

        Args:
            P: 压力 (Pa)
            phase: 'water' 或 'oil'

        Returns:
            phase_potential: 相势 (Pa)
        """
        if phase == 'water':
            rho = self.props.rho_w
        else:  # oil
            rho = self.props.rho_o

        # 重力项: ρ * g * D
        gravity_term = rho * self.g * self.D

        # 相势: Φ = P - ρ * g * D
        # 注意：这里没有考虑毛管压力，如果需要可以添加
        phase_potential = P - gravity_term

        return phase_potential

    def compute_well_index(self, permeability, location_type='injector'):
        """
        计算井指数（Peaceman模型）

        Args:
            permeability: 渗透率 (m²)
            location_type: 'injector' 或 'producer'

        Returns:
            WI: 井指数 (m³/(Pa·s))
        """
        # 获取井位置
        if location_type == 'injector':
            i, j = self.injector_i, self.injector_j
        else:
            i, j = self.producer_i, self.producer_j

        # 井所在网格的渗透率
        k = permeability[:, i, j] if permeability.dim() == 3 else permeability[i, j]

        # Peaceman井模型公式
        # WI = 2πkh / [μ * (ln(re/rw) + S)]
        # 这里h=1m（假设厚度）
        h = 1.0  # 地层厚度 (m)

        if location_type == 'injector':
            mu = self.props.mu_w  # 注入井注入水
        else:
            # 生产井：根据流度加权平均
            # 简化处理，使用油粘度
            mu = self.props.mu_o

        # 计算井指数
        ln_term = math.log(self.props.re / self.props.rw) + self.props.SS
        WI = 2 * math.pi * k * h / (mu * ln_term + 1e-20)

        return WI

    def compute_well_source_term(self, permeability, Sw, P, P_well):
        """
        计算井源汇项（修正版，处理维度匹配）

        Args:
            permeability: 渗透率 (mD) [batch, ny, nx]
            Sw: 水饱和度 [batch, ny, nx]
            P: 网格压力 (Pa) [batch, ny, nx]
            P_well: 井底流压 (Pa)

        Returns:
            qw: 水相产量 (m³/s) [batch, ny, nx]
            qo: 油相产量 (m³/s) [batch, ny, nx]
        """
        batch_size, ny, nx = P.shape

        # 初始化输出张量
        qw = torch.zeros_like(P)
        qo = torch.zeros_like(P)

        # 转换渗透率单位（如果单位是mD）
        k_m2 = permeability * 9.869e-16

        # 计算相对渗透率
        krw, kro = self.relative_permeability(Sw)

        # 计算流度
        lambda_w = krw / self.props.mu_w
        lambda_o = kro / self.props.mu_o
        lambda_t = lambda_w + lambda_o

        # 计算分流率
        fw = lambda_w / (lambda_t + 1e-20)  # 水相分流率

        # 井底流压张量化（与P同形状）
        P_well_tensor = torch.full_like(P, P_well)

        for b in range(batch_size):
            # 1. 计算注入井项（左下角）
            i_inj, j_inj = self.injector_i, self.injector_j

            # 注入井的渗透率
            k_inj = k_m2[b, i_inj, j_inj]

            # 计算注入井指数
            ln_term_inj = math.log(self.props.re / self.props.rw) + self.props.SS
            WI_inj = 2 * math.pi * k_inj * 1.0 / (self.props.mu_w * ln_term_inj + 1e-20)

            # 注入井流量（固定注入速率，简化处理）
            q_inj_rate = self.props.q_in  # 总注入速率 (m³/s)

            # 分配到水相（注入井只注水）
            qw[b, i_inj, j_inj] = q_inj_rate

            # 2. 计算生产井项（右上角）
            i_prod, j_prod = self.producer_i, self.producer_j

            # 生产井位置的参数
            k_prod = k_m2[b, i_prod, j_prod]
            lambda_w_prod = lambda_w[b, i_prod, j_prod]
            lambda_o_prod = lambda_o[b, i_prod, j_prod]
            lambda_t_prod = lambda_t[b, i_prod, j_prod]
            fw_prod = fw[b, i_prod, j_prod]
            P_prod = P[b, i_prod, j_prod]

            # 计算生产井指数
            ln_term_prod = math.log(self.props.re / self.props.rw) + self.props.SS
            WI_prod = 2 * math.pi * k_prod * 1.0 / (self.props.mu_o * ln_term_prod + 1e-20)

            # 压力差（生产井：P > P_well时流体流入井筒）
            delta_P = P_prod - P_well

            # 确保生产井只生产（delta_P为正时才生产）
            if delta_P > 0:
                # 总产量与压力差和井指数成正比
                q_total_prod = WI_prod * lambda_t_prod * delta_P

                # 按分流率分配
                qw_prod = q_total_prod * fw_prod
                qo_prod = q_total_prod * (1 - fw_prod)

                # 生产为负值（流出油藏）
                qw[b, i_prod, j_prod] = -qw_prod
                qo[b, i_prod, j_prod] = -qo_prod

        return qw, qo

    def compute_pressure_residual(self, permeability, porosity, P_k, Sw_k,
                                  P_k1_pred, Sw_k1_pred):
        """
        计算压力方程残差（基于IMPES方法）

        Args:
            permeability: 渗透率 (mD)
            porosity: 孔隙度
            P_k: 时间步k的压力 (Pa)
            Sw_k: 时间步k的水饱和度
            P_k1_pred: 预测的时间步k+1压力 (Pa)
            Sw_k1_pred: 预测的时间步k+1水饱和度

        Returns:
            residual: 压力残差 (Pa)
        """
        batch_size = P_k.shape[0] if P_k.dim() == 3 else 1
        ny, nx = self.grid_size

        # 确保所有输入有batch维度
        tensors = [permeability, porosity, P_k, Sw_k, P_k1_pred, Sw_k1_pred]
        tensors = [t.unsqueeze(0) if t.dim() == 2 else t for t in tensors]
        permeability, porosity, P_k, Sw_k, P_k1_pred, Sw_k1_pred = tensors

        # 1. 计算饱和度相关量
        So_k = 1.0 - Sw_k

        # 2. 计算相对渗透率和流度（向量化）
        krw_k, kro_k = self.relative_permeability(Sw_k)
        krw_k1, kro_k1 = self.relative_permeability(Sw_k1_pred)

        lambda_w_k = krw_k / self.props.mu_w
        lambda_o_k = kro_k / self.props.mu_o
        lambda_t_k = lambda_w_k + lambda_o_k

        lambda_w_k1 = krw_k1 / self.props.mu_w
        lambda_o_k1 = kro_k1 / self.props.mu_o
        lambda_t_k1 = lambda_w_k1 + lambda_o_k1

        # 3. 计算传导率
        Tx, Ty = self.compute_transmissibility(permeability)

        # 4. 计算相势（向量化）
        Phi_w_k = self.compute_phase_potential(P_k, 'water')
        Phi_o_k = self.compute_phase_potential(P_k, 'oil')
        Phi_w_k1 = self.compute_phase_potential(P_k1_pred, 'water')
        Phi_o_k1 = self.compute_phase_potential(P_k1_pred, 'oil')

        # 5. 计算累积项（向量化）
        C_total = porosity * (Sw_k * self.props.Cw + So_k * self.props.Co + self.props.Cr)
        dP_dt = (P_k1_pred - P_k) / self.dt
        accumulation = C_total * self.cell_volume * dP_dt

        # 6. 计算流动项（完全向量化）--------------------------------

        # x方向流动（使用切片操作）
        # 获取相邻网格的势
        Phi_w_k1_left = Phi_w_k1[:, :, :-1]  # 所有行，第0列到倒数第2列
        Phi_w_k1_right = Phi_w_k1[:, :, 1:]   # 所有行，第1列到最后一列

        Phi_o_k1_left = Phi_o_k1[:, :, :-1]
        Phi_o_k1_right = Phi_o_k1[:, :, 1:]

        # 计算势差
        dPhi_w_x = Phi_w_k1_right - Phi_w_k1_left
        dPhi_o_x = Phi_o_k1_right - Phi_o_k1_left
        dPhi_total_x = dPhi_w_x + dPhi_o_x

        # 上游加权流度
        lambda_t_left = lambda_t_k1[:, :, :-1]
        lambda_t_right = lambda_t_k1[:, :, 1:]

        # 使用torch.where进行向量化条件选择
        lambda_t_interface_x = torch.where(dPhi_total_x > 0, lambda_t_left, lambda_t_right)

        # 计算x方向流量（批量、行、列-1）
        flux_x = lambda_t_interface_x * Tx * dPhi_total_x

        # y方向流动（类似）
        Phi_w_k1_top = Phi_w_k1[:, :-1, :]    # 第0行到倒数第2行，所有列
        Phi_w_k1_bottom = Phi_w_k1[:, 1:, :]  # 第1行到最后一行，所有列

        Phi_o_k1_top = Phi_o_k1[:, :-1, :]
        Phi_o_k1_bottom = Phi_o_k1[:, 1:, :]

        dPhi_w_y = Phi_w_k1_bottom - Phi_w_k1_top
        dPhi_o_y = Phi_o_k1_bottom - Phi_o_k1_top
        dPhi_total_y = dPhi_w_y + dPhi_o_y

        lambda_t_top = lambda_t_k1[:, :-1, :]
        lambda_t_bottom = lambda_t_k1[:, 1:, :]
        lambda_t_interface_y = torch.where(dPhi_total_y > 0, lambda_t_top, lambda_t_bottom)

        flux_y = lambda_t_interface_y * Ty * dPhi_total_y

        # 7. 计算散度（向量化）
        # 为散度分配空间
        div_flux = torch.zeros_like(P_k)

        # x方向散度
        # 左边界：流入（正）
        div_flux[:, :, 0] += flux_x[:, :, 0] / self.dx
        # 右边界：流出（负）
        div_flux[:, :, -1] -= flux_x[:, :, -1] / self.dx

        # 内部网格：相邻网格流量差
        if nx > 2:
            # (flux_right - flux_left) / dx
            div_flux[:, :, 1:-1] += (flux_x[:, :, 1:] - flux_x[:, :, :-1]) / self.dx

        # y方向散度
        div_flux[:, 0, :] += flux_y[:, 0, :] / self.dy
        div_flux[:, -1, :] -= flux_y[:, -1, :] / self.dy

        if ny > 2:
            div_flux[:, 1:-1, :] += (flux_y[:, 1:, :] - flux_y[:, :-1, :]) / self.dy

        # 8. 计算井源汇项
        qw, qo = self.compute_well_source_term(permeability, Sw_k, P_k, self.props.p_e)
        total_source = qw + qo

        # 9. 计算残差
        residual = accumulation - div_flux * self.cell_volume + total_source * self.cell_volume



        return residual

    def compute_saturation_residual(self, permeability, porosity, Sw_k, Sw_k1_pred,
                                    P_k, P_k1_pred):
        """
        计算饱和度方程残差
        """
        batch_size = Sw_k.shape[0] if Sw_k.dim() == 3 else 1

        # 确保batch维度
        tensors = [permeability, porosity, Sw_k, Sw_k1_pred, P_k, P_k1_pred]
        tensors = [t.unsqueeze(0) if t.dim() == 2 else t for t in tensors]
        permeability, porosity, Sw_k, Sw_k1_pred, P_k, P_k1_pred = tensors

        # 1. 计算累积项
        accumulation = porosity * (Sw_k1_pred - Sw_k) * self.cell_volume / self.dt

        # 2. 计算压缩性项
        compressibility_term = Sw_k * porosity * (self.props.Cw + self.props.Cr) * (P_k1_pred - P_k) * self.cell_volume / self.dt

        # 3. 计算流动项（简化）
        # 这里可以调用更精细的流动计算，但为简化先使用0
        flow_term = torch.zeros_like(accumulation)

        # 4. 计算井项（水相）
        qw, _ = self.compute_well_source_term(permeability, Sw_k, P_k, self.props.p_e)
        well_term = qw * self.cell_volume

        # 5. 计算残差
        residual = accumulation - flow_term - compressibility_term + well_term

        # 归一化
        # reference = torch.abs(accumulation) + torch.abs(well_term) + 1e-10
        # normalized_residual = residual / reference

        return residual

    def compute_physics_loss(self, inputs, predictions):
        """
        计算物理约束损失

        Args:
            inputs: 模型输入 (batch, 4, ny, nx)
            predictions: 模型输出 (batch, 2, ny, nx)

        Returns:
            total_loss: 总物理损失
            loss_breakdown: 损失分解
        """
        # 解析输入
        permeability = inputs[:, 0, :, :]  # 渗透率 (mD)
        porosity = inputs[:, 1, :, :]      # 孔隙度
        P_k = inputs[:, 2, :, :]           # 当前压力 (Pa)
        Sw_k = inputs[:, 3, :, :]          # 当前饱和度

        # 解析预测输出
        P_k1_pred = predictions[:, 0, :, :]  # 预测压力 (Pa)
        Sw_k1_pred = predictions[:, 1, :, :]  # 预测饱和度

        # 计算压力残差
        pressure_residual = self.compute_pressure_residual(
            permeability, porosity, P_k, Sw_k, P_k1_pred, Sw_k1_pred
        )

        # 计算饱和度残差
        saturation_residual = self.compute_saturation_residual(
            permeability, porosity, Sw_k, Sw_k1_pred, P_k, P_k1_pred
        )

        # 计算损失（均方残差）
        pressure_loss = torch.mean(pressure_residual ** 2)
        saturation_loss = torch.mean(saturation_residual ** 2)

        # 总损失
        total_loss = pressure_loss + saturation_loss

        # 损失分解
        loss_breakdown = {
            'pressure_loss': pressure_loss.item(),
            'saturation_loss': saturation_loss.item(),
            'pressure_residual_mean': torch.mean(torch.abs(pressure_residual)).item(),
            'saturation_residual_mean': torch.mean(torch.abs(saturation_residual)).item(),
        }

        return total_loss, loss_breakdown

class Trainer:
    """FNO训练器"""

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=1e-3,
        weight_decay=1e-4,
        model_name = 'fno_reservoir',
        output_dir = './output'
    ):
        """
        初始化训练器

        Args:
            model: FNO模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            device: 训练设备
            learning_rate: 学习率
            weight_decay: 权重衰减
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # 损失函数和优化器
        self.criterion = LpLoss(size_average=False)
        # self.optimizer = torch.optim.Adam(
        #     model.parameters(),
        #     lr=learning_rate,
        #     weight_decay=weight_decay
        # )

        # # 学习率调度器
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode='min', patience=5, factor=0.5
        # )
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)

        # 训练历史
        self.train_losses = []
        self.test_losses = []

        print(f"Training on device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train_epoch(self):
        """训练一个周期"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # 获取数据
            inputs = batch['input'].to(self.device)
            targets = batch['output'].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            predictions= self.model(inputs) #, physics_loss, loss_info
            # 计算损失
            data_loss = self.criterion(predictions, targets)
            loss = data_loss #+ physics_loss
            # 反向传播
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 更新参数
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)}, Loss: {loss.item():.6f}")

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)

        return avg_loss

    def evaluate(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.test_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['output'].to(self.device)

                predictions = self.model(inputs) #, physics_loss, loss_info

                # 计算损失
                data_loss = self.criterion(predictions, targets)
                loss = data_loss #+ physics_loss

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        self.test_losses.append(avg_loss)

        return avg_loss

    def train(self, num_epochs=100, early_stopping_patience=20):
        """训练模型"""
        print(f"Starting training for {num_epochs} epochs...")

        best_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            start_time_epoch = time.time()

            # 训练一个周期
            train_loss = self.train_epoch()
            print(f"  Training Loss: {train_loss:.6f}")

            # 评估
            test_loss = self.evaluate()
            print(f"  Test Loss: {test_loss:.6f}")

            # 更新学习率
            self.scheduler.step(test_loss)
            end_time_epoch = time.time()
            print(f"  Epoch time: {end_time_epoch - start_time_epoch:.2f}s")
            # 保存最佳模型
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
                self.save_model(f'{self.output_dir}/best_UnetFNO_model.pth')
                print(f"  Saved best model with test loss: {best_loss:.6f}")
            else:
                patience_counter += 1

            # 提前停止
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"\nTraining completed. Best test loss: {best_loss:.6f}")
        end_time = time.time()
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best test loss: {best_loss:.6f}")
        print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        print(f"{'='*60}")
        # 绘制损失曲线
        self.plot_losses()

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.test_losses = checkpoint['test_losses']
        print(f"Model loaded from {path}")

    def plot_losses(self):
        """绘制损失曲线"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Test Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'training_losses_{self.model_name}.png')
        plt.show()

    def visualize_predictions(self, num_samples=3):
        """可视化预测结果，包含绝对误差和相对误差"""
        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                if batch_idx >= num_samples:
                    break

                inputs = batch['input'].to(self.device)
                targets = batch['output'].to(self.device)
                predictions= self.model(inputs) #, physics_loss, loss_info

                # 选择第一个样本
                input_sample = inputs[0].cpu().numpy()
                target_sample = targets[0].cpu().numpy()
                prediction_sample = predictions[0].cpu().numpy()

                # 计算误差
                absolute_errors = []
                relative_errors = []
                error_stats = []

                output_names = ['Pressure', 'Saturation']

                for i in range(2):
                    # 绝对误差
                    abs_error = np.abs(target_sample[i] - prediction_sample[i])
                    absolute_errors.append(abs_error)

                    # 相对误差（避免除以零）
                    relative_error = np.zeros_like(abs_error)
                    mask = target_sample[i] != 0  # 避免除以零
                    relative_error[mask] = abs_error[mask] / np.abs(target_sample[i][mask])
                    relative_errors.append(relative_error)

                    # 统计信息
                    stats = {
                        'name': output_names[i],
                        'abs_mean': np.mean(abs_error),
                        'abs_max': np.max(abs_error),
                        'rel_mean': np.mean(relative_error[mask]) if np.any(mask) else 0,
                        'rel_max': np.max(relative_error[mask]) if np.any(mask) else 0,
                    }
                    error_stats.append(stats)

                # 创建可视化 - 使用4行布局
                # 第1行：输入特征 (4个子图)
                # 第2行：压力相关 (4个子图)
                # 第3行：饱和度相关 (4个子图)
                fig = plt.figure(figsize=(20, 12))

                # 第1行：输入特征
                input_titles = ['Permeability', 'Porosity', 'Pressure_k', 'Saturation_k']
                for i in range(4):
                    ax = plt.subplot(3, 4, i + 1)
                    im = ax.imshow(input_sample[i], cmap='viridis')
                    ax.set_title(f"Input: {input_titles[i]}", fontsize=12)
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # 第2行：压力相关
                # 子图5：真实压力
                ax = plt.subplot(3, 4, 5)
                im = ax.imshow(target_sample[0], cmap='viridis')
                ax.set_title("Pressure (True)", fontsize=12)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # 子图6：预测压力
                ax = plt.subplot(3, 4, 6)
                im = ax.imshow(prediction_sample[0], cmap='viridis')
                ax.set_title("Pressure (Predicted)", fontsize=12)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # 子图7：压力绝对误差
                ax = plt.subplot(3, 4, 7)
                im = ax.imshow(absolute_errors[0], cmap='hot', vmin=0, vmax=np.max(absolute_errors[0]))
                ax.set_title(f"Pressure Abs Error", fontsize=12)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # 在图上添加统计信息
                stats_text = f"Mean: {error_stats[0]['abs_mean']:.2e}\nMax: {error_stats[0]['abs_max']:.2e}"
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                # 子图8：压力相对误差
                ax = plt.subplot(3, 4, 8)

                # 找到合适的vmax（排除异常值）
                valid_data = relative_errors[0][relative_errors[0] > 0]
                if len(valid_data) > 0:
                    vmax = np.percentile(valid_data, 95)  # 使用95百分位数作为上限
                else:
                    vmax = np.max(relative_errors[0])

                im = ax.imshow(relative_errors[0], cmap='plasma',
                            vmin=0, vmax=vmax if vmax > 0 else 1)
                ax.set_title("Pressure Rel Error", fontsize=12)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # 在图上添加统计信息
                stats_text = f"Mean: {error_stats[0]['rel_mean']:.2%}\nMax: {error_stats[0]['rel_max']:.2%}"
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                # 第3行：饱和度相关
                # 子图9：真实饱和度
                ax = plt.subplot(3, 4, 9)
                im = ax.imshow(target_sample[1], cmap='viridis')
                ax.set_title("Saturation (True)", fontsize=12)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # 子图10：预测饱和度
                ax = plt.subplot(3, 4, 10)
                im = ax.imshow(prediction_sample[1], cmap='viridis')
                ax.set_title("Saturation (Predicted)", fontsize=12)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # 子图11：饱和度绝对误差
                ax = plt.subplot(3, 4, 11)
                im = ax.imshow(absolute_errors[1], cmap='hot', vmin=0, vmax=np.max(absolute_errors[1]))
                ax.set_title(f"Saturation Abs Error", fontsize=12)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # 在图上添加统计信息
                stats_text = f"Mean: {error_stats[1]['abs_mean']:.2e}\nMax: {error_stats[1]['abs_max']:.2e}"
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                # 子图12：饱和度相对误差
                ax = plt.subplot(3, 4, 12)

                # 找到合适的vmax（排除异常值）
                valid_data = relative_errors[1][relative_errors[1] > 0]
                if len(valid_data) > 0:
                    vmax = np.percentile(valid_data, 95)  # 使用95百分位数作为上限
                else:
                    vmax = np.max(relative_errors[1])

                im = ax.imshow(relative_errors[1], cmap='plasma',
                            vmin=0, vmax=vmax if vmax > 0 else 1)
                ax.set_title("Saturation Rel Error", fontsize=12)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # 在图上添加统计信息
                stats_text = f"Mean: {error_stats[1]['rel_mean']:.2%}\nMax: {error_stats[1]['rel_max']:.2%}"
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                # 添加总标题
                plt.suptitle(f"Sample {batch_idx+1} Visualization - Pressure and Saturation Predictions\n"
                            f"Pressure: Abs Mean={error_stats[0]['abs_mean']:.2e}, "
                            f"Rel Mean={error_stats[0]['rel_mean']:.2%} | "
                            f"Saturation: Abs Mean={error_stats[1]['abs_mean']:.2e}, "
                            f"Rel Mean={error_stats[1]['rel_mean']:.2%}",
                            fontsize=14, y=1.02)

                plt.tight_layout()
                plt.show()

                # 打印详细的误差统计
                print(f"\n{'='*60}")
                print(f"详细误差统计 - 样本 {batch_idx+1}")
                print(f"{'='*60}")
                for stats in error_stats:
                    print(f"\n{stats['name']}:")
                    print(f"  绝对误差 - 均值: {stats['abs_mean']:.2e}, 最大值: {stats['abs_max']:.2e}")
                    print(f"  相对误差 - 均值: {stats['rel_mean']:.2%}, 最大值: {stats['rel_max']:.2%}")


class MultiStepPredictor:
    """多步预测器：使用训练好的FNO模型进行递归预测"""

    def __init__(self, model, dataset, device='cpu'):
        """
        初始化多步预测器

        Args:
            model: 训练好的FNO模型
            dataset: 数据集（用于获取归一化参数）
            device: 设备
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.dataset = dataset

    def predict_sequence(self, initial_state, permeability, porosity, num_steps=10):
        """
        递归预测多个时间步

        Args:
            initial_state: 初始状态 [pressure_0, saturation_0]，形状为 (2, ny, nx)
            permeability: 渗透率场，形状为 (ny, nx)
            porosity: 孔隙度场，形状为 (ny, nx)
            num_steps: 预测步数

        Returns:
            预测序列，形状为 (num_steps+1, 2, ny, nx)
        """
        # 确保输入是numpy数组
        if isinstance(initial_state, torch.Tensor):
            initial_state = initial_state.cpu().numpy()
        if isinstance(permeability, torch.Tensor):
            permeability = permeability.cpu().numpy()
        if isinstance(porosity, torch.Tensor):
            porosity = porosity.cpu().numpy()

        # 归一化
        if self.dataset.normalize:
            permeability_norm = self.dataset._normalize(permeability, 'permeability')
            porosity_norm = self.dataset._normalize(porosity, 'porosity')
            pressure_norm = self.dataset._normalize(initial_state[0], 'pressure')
            saturation_norm = self.dataset._normalize(initial_state[1], 'saturation')
        else:
            permeability_norm = permeability
            porosity_norm = porosity
            pressure_norm = initial_state[0]
            saturation_norm = initial_state[1]

        # 存储预测序列
        predictions = [initial_state]
        current_pressure = pressure_norm
        current_saturation = saturation_norm

        with torch.no_grad():
            for step in range(num_steps):
                # 构建输入
                input_features = np.stack([
                    permeability_norm,
                    porosity_norm,
                    current_pressure,
                    current_saturation
                ], axis=0)  # (4, ny, nx)

                # 转换为张量
                input_tensor = torch.FloatTensor(input_features).unsqueeze(0).to(self.device)

                # 预测
                output= self.model(input_tensor)

                # 获取预测结果
                pred_pressure = output[0, 0].cpu().numpy()
                pred_saturation = output[0, 1].cpu().numpy()

                # 反归一化
                if self.dataset.normalize:
                    pred_pressure_denorm = self.dataset._denormalize(pred_pressure, 'pressure')
                    pred_saturation_denorm = self.dataset._denormalize(pred_saturation, 'saturation')
                else:
                    pred_pressure_denorm = pred_pressure
                    pred_saturation_denorm = pred_saturation

                # 存储预测结果
                predictions.append(np.stack([pred_pressure_denorm, pred_saturation_denorm], axis=0))

                # 更新当前状态（用于下一步预测）
                current_pressure = pred_pressure
                current_saturation = pred_saturation

        return np.array(predictions)


def create_data():
    """创建数据集的示例函数"""
    creater = ReservoirDatasetCreator(nx=200, ny=200)
    creater.create_dataset(n_samples=50)

def train_data(file_name, nx=20, ny=20, output_dir="./UnetFNO/reservoir_fno"):
    # file_name = 'reservoir_fno_20251220_132502'
    config = {
        # 模型参数
        'input_channels': 4,
        'output_channels': 2,
        'width': 32,
        'modes': 8,
        'depth': 4,

        # 训练参数
        'batch_size': 16,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_epochs': 100,

        # 数据参数
        'data_path': f'./reservoir_fno_dataset_{nx}x{ny}/{file_name}.h5',
        'train_test_split': 0.8,
        'normalize': True
    }

    # 创建数据加载器
    print("Loading data...")
    train_loader, test_loader = create_data_loaders(
        h5_file_path=config['data_path'],
        batch_size=config['batch_size'],
        train_test_split=config['train_test_split'],
        normalize=config['normalize']
    )

    # 获取网格大小
    sample_batch = next(iter(train_loader))
    grid_shape = sample_batch['grid_shape'][0]
    ny, nx = grid_shape[0].item(), grid_shape[1].item()
    print(f"Grid resolution: {ny}x{nx}")
    print("Creating FNO model...")

    model = UNetFNO2d(modes1=config['modes'], modes2=config['modes'], width=config['width'],n_fno_layers=config['depth'])
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        model_name=file_name,
        output_dir=output_dir
    )

    # 训练模型
    trainer.train(num_epochs=config['num_epochs'])

    # 可视化预测结果
    # print("\nVisualizing predictions...")
    # trainer.visualize_predictions(num_samples=3)

    # 保存最终模型
    trainer.save_model(f'{output_dir}/final_claudefno_model.pth')

    print("\nTraining completed!")
# 主程序：训练和评估FNO模型
#Model created with 8,318,754 parameters.
#Model created with 2,024,482 parameters. ufno大小
if __name__ == "__main__":
    file_name = "reservoir_fno_20260101_110527_samples_10"
    nx = 20
    ny = 20
    if 1:
        train_data(file_name,nx=nx, ny=ny,output_dir=f"./UnetFNO/{file_name}")
    else:
        if 1 :
            creater = ReservoirDatasetCreator(nx=nx, ny=ny)
            creater.create_dataset(n_samples=10)
        # 设置参数
        else:
            # file_name = "reservoir_fno_20251220_214501"
            visualize_dataset(f'./reservoir_fno_dataset_{nx}x{ny}/{file_name}.h5',output_dir=f"./visualizations/{file_name}")
