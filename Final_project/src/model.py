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
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.dx = cfg.L_X / (nx - 1)
        self.dy = cfg.L_Y / (ny - 1)
        
        # 初始化声速场
        self.c_map = generate_velocity_map(nx, ny, self.dx, self.dy)
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