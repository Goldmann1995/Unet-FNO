# =============================================================================
# data_generation.py  —  模块一：数据生成
#
# 包含：
#   WellType / NODE / CELL         基础数据结构
#   ReservoirGrid                  结构化笛卡尔网格
#   ReservoirProperties            物理参数（Brooks-Corey，修正版）
#   ReservoirSimulator             IMPES 数值模拟器
#   PermeabilityGenerator          渗透率场生成（gaussian/lognormal/uniform）
#   ReservoirDatasetCreator        数据集创建与 HDF5 保存
#
# 用法：
#   from data_generation import ReservoirDatasetCreator
#   creator = ReservoirDatasetCreator(nx=20, ny=20)
#   creator.create_dataset(n_samples=500)
# =============================================================================

import math, time, json, os
import numpy as np
import h5py
import torch
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from datetime import datetime
from enum import Enum
from scipy.ndimage import gaussian_filter

np.random.seed(42)
torch.manual_seed(42)


# ─────────────────────────── 基础数据结构 ────────────────────────────────────

class WellType(Enum):
    INJECTOR = 0
    PRODUCER = 1


class NODE:
    def __init__(self):
        self.x = self.y = self.z = 0.0


class CELL:
    def __init__(self, cell_id=0):
        self.id        = cell_id
        self.vertices  = [-1] * 8
        self.neighbors = [-1] * 6
        self.dx = self.dy = self.dz = 0.0
        self.volume    = 0.0
        self.xc = self.yc = self.zc = 0.0
        self.porosity  = 0.0
        self.kx = self.ky = self.kz = 0.0
        self.trans     = [0.0] * 6
        self.transw    = [0.0] * 6
        self.markbc    = -2
        self.press     = 0.0
        self.Sw        = 0.0
        self.markbc_Sw = 0
        self.markwell  = -1
        self.well_type = None
        self.mobiw = self.mobio = self.mobit = 0.0


# ────────────────────────────── 网格 ─────────────────────────────────────────

class ReservoirGrid:
    """结构化笛卡尔网格"""

    def __init__(self, nx=20, ny=20, nz=1, dx=5.0, dy=5.0, dz=1.0,
                 total_length=None):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.ncell = nx * ny * nz
        if total_length is not None:
            self.dx = total_length[0] / nx
            self.dy = total_length[1] / ny
            self.dz = total_length[2] / nz
        else:
            self.dx, self.dy, self.dz = dx, dy, dz
        self.nodelist, self.celllist = [], []
        if self.dx <= 0 or self.dy <= 0 or self.dz <= 0:
            raise ValueError("网格尺寸必须 > 0")
        tx, ty, tz = nx*self.dx, ny*self.dy, nz*self.dz
        print(f"Grid {nx}×{ny}×{nz}  "
              f"dx={self.dx:.2f} dy={self.dy:.2f} dz={self.dz:.2f}  "
              f"domain {tx:.0f}×{ty:.0f}×{tz:.0f} m")

    def build_grid(self):
        print("  Building grid...")
        llz = 0.0
        for k in range(self.nz + 1):
            lly = 0.0
            for j in range(self.ny + 1):
                llx = 0.0
                for i in range(self.nx + 1):
                    nd = NODE(); nd.x, nd.y, nd.z = llx, lly, llz
                    self.nodelist.append(nd)
                    if i < self.nx: llx += self.dx
                if j < self.ny: lly += self.dy
            if k < self.nz: llz += self.dz

        cid = 0
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    c = CELL(cid)
                    if i > 0:            c.neighbors[0] = cid - 1
                    if i < self.nx - 1:  c.neighbors[1] = cid + 1
                    if j > 0:            c.neighbors[2] = cid - self.nx
                    if j < self.ny - 1:  c.neighbors[3] = cid + self.nx
                    if k > 0:            c.neighbors[4] = cid - self.nx*self.ny
                    if k < self.nz - 1:  c.neighbors[5] = cid + self.nx*self.ny

                    i0 = k*(self.nx+1)*(self.ny+1) + j*(self.nx+1) + i
                    i1, i2 = i0+1, i0+(self.nx+1)
                    i4 = i0 + (self.nx+1)*(self.ny+1)
                    c.vertices = [i0, i1, i2, i2+1, i4, i4+1, i4+(self.nx+1), i4+(self.nx+1)+1]

                    nl = self.nodelist
                    c.dx = nl[i1].x - nl[i0].x
                    c.dy = nl[i2].y - nl[i0].y
                    c.dz = nl[i4].z - nl[i0].z
                    c.volume = c.dx * c.dy * c.dz
                    c.xc = 0.125 * sum(nl[v].x for v in c.vertices)
                    c.yc = 0.125 * sum(nl[v].y for v in c.vertices)
                    c.zc = 0.125 * sum(nl[v].z for v in c.vertices)
                    self.celllist.append(c); cid += 1

        print(f"  {len(self.nodelist)} nodes, {len(self.celllist)} cells")


# ──────────────────────────── 物理参数 ───────────────────────────────────────

class ReservoirProperties:
    """两相流物性参数（修正版 Brooks-Corey，nw=4, no=2）"""

    def __init__(self):
        # 流体
        self.mu_o, self.mu_w = 1.8e-3, 1.0e-3      # Pa·s
        self.Bo,   self.Bw   = 1.2, 1.0
        self.rho_o, self.rho_w = 800.0, 1000.0     # kg/m³
        # 岩石
        self.poro = 0.20
        self.Siw  = 0.20
        # 压缩系数 (1/Pa)，由 psi^-1 换算
        _p = 6894.76
        self.Cr = 10e-6  / _p
        self.Cw = 4e-6   / _p
        self.Co = 100e-6 / _p
        # 压力
        self.p_init = 20e6
        self.p_e    = 20e6
        # 井
        self.q_in = 5.0 / 86400            # m³/s
        self.re   = 0.14 * math.sqrt(2 * 5.0**2)  # ≈ 0.990 m
        self.rw   = 0.05
        self.SS   = 3
        # Brooks-Corey（修正：分母 = 1-Swr-Sor = 0.6）
        self.Swr  = 0.20
        self.Sor  = 0.20
        self.nw   = 4.0   # 水相 Corey 指数
        self.no   = 2.0   # 油相 Corey 指数


# ─────────────────────────── IMPES 模拟器 ────────────────────────────────────

class ReservoirSimulator:
    """IMPES 两相流模拟器（隐式压力 + 显式饱和度）"""

    def __init__(self, grid: ReservoirGrid, props: ReservoirProperties):
        self.grid, self.props = grid, props
        self.ncell = grid.ncell
        self.presslast       = torch.zeros(self.ncell)
        self.qtvec           = torch.zeros(self.ncell)
        self.alphavec        = torch.zeros(self.ncell)
        self.geometric_trans = {}
        self.geometric_trans_computed = False
        # 邻居索引缓存
        for i, c in enumerate(grid.celllist):
            pass   # 邻居通过 cell.neighbors 直接访问

    # ── 初始化 ────────────────────────────────────────────────────────────────

    def set_properties(self, perm_field):
        for i, c in enumerate(self.grid.celllist):
            c.porosity = self.props.poro
            c.kx = c.ky = c.kz = perm_field[i]
            c.Sw    = self.props.Siw
            c.press = self.props.p_init
            self.presslast[i] = c.press

    def set_well_conditions(self):
        ncell = self.ncell
        inj = self.grid.celllist[0]
        inj.markwell, inj.markbc, inj.markbc_Sw = 0, -1, 1
        inj.Sw = 1.0; inj.well_type = WellType.INJECTOR
        self.qtvec[0] = self.props.q_in

        prod = self.grid.celllist[ncell - 1]
        prod.markwell, prod.markbc = 1, -1
        prod.well_type = WellType.PRODUCER
        self.qtvec[ncell - 1] = -self.props.q_in

    # ── Brooks-Corey 流度（修正版，分母 = 1-Swr-Sor）────────────────────────

    def compute_mobility(self):
        Swr, Sor = self.props.Swr, self.props.Sor
        nw,  no  = self.props.nw,  self.props.no
        denom    = 1.0 - Swr - Sor          # 0.6
        for c in self.grid.celllist:
            sw  = max(Swr, min(1.0 - Sor, c.Sw))
            Swn = (sw - Swr) / denom
            c.mobiw = Swn**nw / self.props.mu_w / self.props.Bw
            c.mobio = (1.0-Swn)**no / self.props.mu_o / self.props.Bo
            c.mobit = c.mobiw + c.mobio

    # ── 几何传导率（只算一次）────────────────────────────────────────────────

    def compute_geometric_transmissibility(self):
        for i, c in enumerate(self.grid.celllist):
            for d in range(4):
                j = c.neighbors[d]
                if j < 0: continue
                nb = self.grid.celllist[j]
                A  = self.grid.dy * self.grid.dz if d in (0,1) else self.grid.dx * self.grid.dz
                L  = self.grid.dx                if d in (0,1) else self.grid.dy
                k1, k2 = c.kx, nb.kx
                k_h = 2*k1*k2/(k1+k2) if k1 > 0 and k2 > 0 else 0.0
                key = (i,j) if i < j else (j,i)
                self.geometric_trans[key] = A * k_h / L
        self.geometric_trans_computed = True

    def _geo_trans(self, i, j):
        key = (i,j) if i < j else (j,i)
        return self.geometric_trans.get(key, 0.0)

    def compute_total_transmissibility(self):
        for i, c in enumerate(self.grid.celllist):
            for d in range(4):
                j = c.neighbors[d]
                if j < 0: continue
                T_geo = self._geo_trans(i, j)
                up = j if self.grid.celllist[j].press > c.press else i
                c.trans[d] = T_geo * self.grid.celllist[up].mobit

    # ── 隐式压力方程 ──────────────────────────────────────────────────────────

    def solve_pressure_implicit(self, dt):
        A = sp.lil_matrix((self.ncell, self.ncell))
        b = np.zeros(self.ncell)
        for idx, c in enumerate(self.grid.celllist):
            alpha = (c.porosity * ((1-c.Sw)*(self.props.Cr+self.props.Co)/self.props.Bo
                                  +   c.Sw *(self.props.Cr+self.props.Cw)/self.props.Bw))
            accum = alpha * c.volume / dt
            A[idx, idx] = accum
            b[idx]      = accum * c.press + self.qtvec[idx]
            for d, nb in enumerate(c.neighbors[:4]):
                if nb < 0: continue
                t = c.trans[d]
                if t > 0:
                    A[idx, idx] += t
                    A[idx, nb]  -= t
        return splinalg.spsolve(A.tocsr(), b)

    # ── 显式饱和度更新 ────────────────────────────────────────────────────────

    def update_saturation(self, dt):
        Swr, Sor = self.props.Swr, self.props.Sor
        nw,  no  = self.props.nw,  self.props.no
        denom    = 1.0 - Swr - Sor
        mu_w, Bw = self.props.mu_w, self.props.Bw
        mu_o, Bo = self.props.mu_o, self.props.Bo
        poro     = self.props.poro
        Cr, Cw   = self.props.Cr, self.props.Cw
        q_in     = self.props.q_in

        for i, c in enumerate(self.grid.celllist):
            pi, Sw_i = c.press, c.Sw

            # 当前格流度（Brooks-Corey 修正版）
            _sw  = max(Swr, min(1.0-Sor, Sw_i))
            _Swn = (_sw - Swr) / denom
            krw_i = _Swn**nw;  kro_i = (1.0-_Swn)**no
            mobiw_i = krw_i / mu_w / Bw
            mobio_i = kro_i / mu_o / Bo
            mobit_i = mobiw_i + mobio_i
            fw_i    = mobiw_i / mobit_i if mobit_i > 0 else 0.0

            tfluxsw = tfluxin = 0.0
            for d in range(4):
                nb_idx = c.neighbors[d]
                if nb_idx < 0: continue
                nb = self.grid.celllist[nb_idx]
                if nb.press > pi:
                    fluxin = c.trans[d] * (nb.press - pi)
                    tfluxin  += fluxin
                    fw_up = nb.mobiw / nb.mobit if nb.mobit > 0 else 0.0
                    tfluxsw += fluxin * fw_up

            tfluxsw += (-tfluxin) * fw_i   # 流出项

            if c.markwell == 0:    tfluxsw += q_in
            elif c.markwell == 1:  tfluxsw -= q_in * fw_i

            # 压缩性项
            dp_dt = (c.press - self.presslast[i].item()) / dt
            tfluxsw += -dp_dt * poro * Sw_i * (Cr + Cw) / Bw

            delta_Sw = tfluxsw * dt * Bw / (poro * c.volume)
            c.Sw = max(self.props.Siw, min(1.0, c.Sw + delta_Sw))

    # ── 主运行 ────────────────────────────────────────────────────────────────

    def run_simulation(self, perm_field, nt=500, dt=43200/4):
        """
        运行 nt 步 IMPES 模拟。
        dt 默认 10800 s（3 小时步长）。
        """
        self.set_properties(perm_field)
        self.set_well_conditions()
        self.compute_geometric_transmissibility()

        time_steps, pressure_hist, saturation_hist = [], [], []

        for t in range(nt):
            self.compute_mobility()
            self.compute_total_transmissibility()

            p_new = self.solve_pressure_implicit(dt)
            for i, c in enumerate(self.grid.celllist):
                c.press = p_new[i].item()

            self.update_saturation(dt)
            self.presslast = p_new.copy()

            time_steps.append(t * dt / 86400)
            pressure_hist.append([c.press for c in self.grid.celllist])
            saturation_hist.append([c.Sw   for c in self.grid.celllist])

            if (t + 1) % 100 == 0:
                print(f"    step {t+1}/{nt}")

        return {
            'time_steps':   np.array(time_steps),
            'pressure':     np.array(pressure_hist),
            'saturation':   np.array(saturation_hist),
            'permeability': np.array(perm_field),
            'porosity':     np.full(self.ncell, self.props.poro),
            'grid_info':    {'nx': self.grid.nx, 'ny': self.grid.ny,
                             'nz': self.grid.nz, 'dx': self.grid.dx,
                             'dy': self.grid.dy, 'dz': self.grid.dz},
        }


# ─────────────────────── 渗透率场生成器 ──────────────────────────────────────

class PermeabilityGenerator:
    """支持 gaussian / lognormal / uniform 三种地质统计模型"""

    def __init__(self, nx=20, ny=20, nz=1):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.ncells = nx * ny * nz

    def generate_uniform(self, mean_perm=15e-15, variation=0.3):
        lo, hi = mean_perm*(1-variation), mean_perm*(1+variation)
        return np.random.uniform(lo, hi, self.ncells)

    def generate_lognormal(self, mean_perm=15e-15, std_factor=0.5):
        return np.random.lognormal(np.log(mean_perm),
                                   np.log(1+std_factor), self.ncells)

    def generate_gaussian_field(self, mean_perm=15e-15, sigma=1.0, **_):
        noise = np.random.randn(self.ncells)
        if self.nz == 1:
            smoothed = gaussian_filter(noise.reshape(self.ny, self.nx),
                                        sigma=sigma).flatten()
        else:
            smoothed = gaussian_filter(noise.reshape(self.nz, self.ny, self.nx),
                                        sigma=sigma).flatten()
        norm = (smoothed - smoothed.mean()) / (smoothed.std() + 1e-10)
        return np.abs(mean_perm * np.exp(norm * 0.5))

    def generate_heterogeneous_field(self, pattern_type='gaussian', **kwargs):
        """统一接口"""
        mp = kwargs.get('mean_perm', 15e-15)
        if   pattern_type == 'gaussian':
            return self.generate_gaussian_field(mp, kwargs.get('sigma', 1.0))
        elif pattern_type == 'lognormal':
            return self.generate_lognormal(mp, kwargs.get('std_factor', 0.5))
        elif pattern_type == 'uniform':
            return self.generate_uniform(mp, kwargs.get('variation', 0.3))
        raise ValueError(f"未知模式: {pattern_type}")


# ───────────────────────── 数据集创建器 ──────────────────────────────────────

class ReservoirDatasetCreator:
    """
    生成油藏数据集并保存为 HDF5。

    示例：
        creator = ReservoirDatasetCreator(nx=20, ny=20)
        h5_path = creator.create_dataset(n_samples=500)
    """

    def __init__(self, output_dir='./reservoir_fno_dataset', nx=20, ny=20):
        self.nx, self.ny = nx, ny
        self.output_dir  = f"{output_dir}_{nx}x{ny}"
        os.makedirs(self.output_dir, exist_ok=True)

        print("Initializing reservoir simulator...")
        self.grid      = ReservoirGrid(nx=nx, ny=ny, total_length=(100,100,1))
        self.grid.build_grid()
        self.props     = ReservoirProperties()
        self.simulator = ReservoirSimulator(self.grid, self.props)
        self.perm_gen  = PermeabilityGenerator(nx=nx, ny=ny)

        self._info = {
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'grid_shape':   [nx, ny, 1],
            'cell_size':    [self.grid.dx, self.grid.dy, self.grid.dz],
        }

    # ── 单样本 ────────────────────────────────────────────────────────────────

    def generate_single_sample(self, idx, pattern='gaussian',
                                nt=500, dt=43200/4):
        print(f"  Sample {idx:03d}  pattern={pattern}")
        perm = self.perm_gen.generate_heterogeneous_field(pattern)
        t0   = time.time()
        res  = self.simulator.run_simulation(perm, nt=nt, dt=dt)
        elapsed = time.time() - t0
        print(f"    done in {elapsed:.1f}s")
        return {'idx': idx, 'pattern': pattern, 'sim_time': elapsed, **res}

    # ── 完整数据集 ────────────────────────────────────────────────────────────

    def create_dataset(self, n_samples=500,
                       patterns=('gaussian', 'uniform', 'lognormal'),
                       nt=500, dt=43200/4):
        print(f"\n{'='*60}")
        print(f"Dataset: {n_samples} samples  patterns={patterns}")
        print(f"Simulator: nt={nt}  dt={dt:.0f}s ({dt/86400:.2f} days/step)")
        print(f"{'='*60}")

        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        h5p  = os.path.join(self.output_dir,
                            f'reservoir_fno_{ts}_samples_{n_samples}.h5')
        pcnt = {p: 0 for p in patterns}
        ttot = 0.0

        with h5py.File(h5p, 'w') as f:
            dg = f.create_group('data')
            for idx in range(n_samples):
                pat = np.random.choice(patterns)
                pcnt[pat] += 1
                s = self.generate_single_sample(idx, pat, nt=nt, dt=dt)
                ttot += s['sim_time']

                sg = dg.create_group(f'sample_{idx:03d}')
                sg.create_dataset('permeability',
                    data=s['permeability'].reshape(self.ny, self.nx))
                sg.create_dataset('porosity',
                    data=s['porosity'].reshape(self.ny, self.nx))
                sg.create_dataset('pressure',   data=s['pressure'])
                sg.create_dataset('saturation', data=s['saturation'])
                sg.create_dataset('time_steps', data=s['time_steps'])
                sg.attrs['sample_id']  = idx
                sg.attrs['pattern']    = str(pat)
                sg.attrs['sim_time']   = float(s['sim_time'])
                print(f"  [{idx+1}/{n_samples}] saved")

        # JSON 元信息
        info = {**self._info, 'n_samples': n_samples,
                'simulation_settings': {'nt': nt, 'dt': dt},
                'pattern_distribution': pcnt,
                'total_generation_time': ttot}
        info_p = os.path.join(self.output_dir, f'dataset_info_{ts}.json')
        with open(info_p, 'w') as fp:
            json.dump(self._to_json(info), fp, indent=2)

        print(f"\nSaved: {h5p}")
        print(f"Total: {ttot:.0f}s  avg: {ttot/n_samples:.1f}s/sample")
        return h5p

    @staticmethod
    def _to_json(obj):
        if isinstance(obj, dict):
            return {k: ReservoirDatasetCreator._to_json(v) for k,v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [ReservoirDatasetCreator._to_json(v) for v in obj]
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.ndarray):   return obj.tolist()
        if isinstance(obj, np.bool_):     return bool(obj)
        return obj


# ─────────────────────────────── 入口 ────────────────────────────────────────

if __name__ == '__main__':
    creator = ReservoirDatasetCreator(nx=80, ny=80)
    # 快速测试：n_samples=10；正式生成改为 500
    h5_file = creator.create_dataset(n_samples=10)
    print(f"\nDone. Dataset: {h5_file}")
