import numpy as np
from numba import njit
import config as cfg

@njit
def generate_velocity_map(nx, ny, dx, dy):
    """
    生成二维声速分布场 c(x,y)
    使用 Numba 加速循环构建
    """
    c = np.ones((nx, ny)) * cfg.C_BG
    
    for i in range(nx):
        for j in range(ny):
            # 计算物理坐标
            x = i * dx
            y = j * dy
            
            # 计算到圆心的距离
            dist = np.sqrt((x - cfg.CENTER_X)**2 + (y - cfg.CENTER_Y)**2)
            
            if dist <= cfg.R_INNER:
                c[i, j] = cfg.C_HARD
            elif dist <= cfg.R_OUTER:
                c[i, j] = cfg.C_NORMAL
            # else: 保持背景声速
            
    return c

@njit
def ricker_wavelet(t, f0):
    """
    计算 t 时刻的 Ricker 子波振幅
    """
    t0 = 1.0 / f0
    # 简单的截断，避免过长时间的无效计算
    if t > 2.5 * t0:
        return 0.0
        
    term = (np.pi * f0 * (t - t0)) ** 2
    val = (1 - 2 * term) * np.exp(-term)
    return val

class LiverModel:
    def __init__(self, nx=None, ny=None, h=None):
        """
        初始化支持两种模式：
        1. 指定网格数: LiverModel(nx=65, ny=65) -> 此时 dx, dy 可能不相等
        2. 指定步长 h: LiverModel(h=0.5e-3)    -> 自动计算 nx, ny，保证 dx≈dy≈h
        """
        if h is not None:
            # 模式 1: 基于步长 h 自动计算最近似的整数网格点
            # N = L / h + 1
            self.nx = int(np.round(cfg.L_X / h)) + 1
            self.ny = int(np.round(cfg.L_Y / h)) + 1
        else:
            # 模式 2: 手动指定 nx, ny
            if nx is None or ny is None:
                raise ValueError("Must specify either 'h' or both 'nx' and 'ny'")
            self.nx = nx
            self.ny = ny

        # 反算真实的 dx, dy (确保网格精确覆盖 L_X, L_Y)
        self.dx = cfg.L_X / (self.nx - 1)
        self.dy = cfg.L_Y / (self.ny - 1)
        
        # 打印一下实际的网格信息，方便确认
        # print(f"Model initialized: {self.nx}x{self.ny}, dx={self.dx:.2e}, dy={self.dy:.2e}")
        
        # 初始化声速场
        self.c_map = generate_velocity_map(self.nx, self.ny, self.dx, self.dy)
        self.c_max = np.max(self.c_map)
        
        # 计算源在网格中的索引
        self.src_i = int(cfg.SRC_X / self.dx)
        self.src_j = int(cfg.SRC_Y / self.dy)

    def get_source_term(self, t):
        """获取当前时刻的源项场矩阵"""
        s = np.zeros((self.nx, self.ny))
        val = ricker_wavelet(t, cfg.F0)
        s[self.src_i, self.src_j] = val * cfg.A0
        return s