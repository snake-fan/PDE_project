"""
Crank-Nicolson implicit scheme for wave equation
验证其在大时间步长下的稳定性（可超过CFL条件限制）
"""

import numpy as np
from scipy.sparse import diags, csr_matrix, eye
from scipy.sparse.linalg import spsolve
import time
from typing import Tuple, Callable


class CrankNicolsonSolver:
    """Crank-Nicolson隐式格式求解器"""
    
    def __init__(self, domain_size: Tuple[float, float], 
                 grid_points: Tuple[int, int],
                 c: float = 1.0,
                 source_func: Callable = None):
        """
        初始化C-N求解器
        
        Args:
            domain_size: (Lx, Ly) 计算域大小
            grid_points: (nx, ny) 网格点数
            c: 波速（常数）
            source_func: 源项函数 s(x, y, t)
        """
        self.Lx, self.Ly = domain_size
        self.nx, self.ny = grid_points
        self.c = c
        self.source_func = source_func if source_func else lambda x, y, t: 0.0
        
        # 计算网格步长
        self.hx = self.Lx / (self.nx - 1)
        self.hy = self.Ly / (self.ny - 1)
        self.h = self.hx  # 假设等间距
        
        # 坐标
        self.x = np.linspace(0, self.Lx, self.nx)
        self.y = np.linspace(0, self.Ly, self.ny)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        
    def build_cn_matrix(self, dt: float) -> csr_matrix:
        """
        构造C-N格式的矩阵
        p_{i,j}^{n+1} - α(∇²p^{n+1}) = RHS
        其中 α = c²Δt²/(2h²)
        
        Returns:
            稀疏矩阵A （已转换为CSR格式）
        """
        alpha = (self.c ** 2) * (dt ** 2) / (2 * self.h ** 2)
        n = self.nx * self.ny
        
        # 使用COO格式构造矩阵
        rows = []
        cols = []
        data = []
        
        # 遍历每个网格点
        for i in range(self.ny):
            for j in range(self.nx):
                idx = i * self.nx + j
                
                # 边界条件：Dirichlet边界为0
                if i == 0 or i == self.ny - 1 or j == 0 or j == self.nx - 1:
                    rows.append(idx)
                    cols.append(idx)
                    data.append(1.0)
                else:
                    # 内部点
                    # 主对角线：1 + 4α
                    rows.append(idx)
                    cols.append(idx)
                    data.append(1.0 + 4 * alpha)
                    
                    # 右邻居（j+1）
                    if j < self.nx - 1:
                        rows.append(idx)
                        cols.append(idx + 1)
                        data.append(-alpha)
                    
                    # 左邻居（j-1）
                    if j > 0:
                        rows.append(idx)
                        cols.append(idx - 1)
                        data.append(-alpha)
                    
                    # 下邻居（i+1）
                    if i < self.ny - 1:
                        rows.append(idx)
                        cols.append(idx + self.nx)
                        data.append(-alpha)
                    
                    # 上邻居（i-1）
                    if i > 0:
                        rows.append(idx)
                        cols.append(idx - self.nx)
                        data.append(-alpha)
        
        # 转换为CSR格式
        from scipy.sparse import coo_matrix
        A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        
        return A
    
    def solve_step(self, p_n: np.ndarray, p_nm1: np.ndarray, 
                   dt: float, t: float, A: csr_matrix) -> np.ndarray:
        """
        执行一个时间步长
        
        Args:
            p_n: 当前时刻解
            p_nm1: 前一时刻解
            dt: 时间步长
            t: 当前时间
            A: C-N矩阵
            
        Returns:
            下一时刻的解 p_{n+1}
        """
        n_points = self.nx * self.ny
        alpha = (self.c ** 2) * (dt ** 2) / (2 * self.h ** 2)
        
        # 计算右侧向量
        rhs = np.zeros(n_points)
        
        for i in range(self.ny):
            for j in range(self.nx):
                idx = i * self.nx + j
                
                # 边界条件
                if i == 0 or i == self.ny - 1 or j == 0 or j == self.nx - 1:
                    rhs[idx] = 0
                else:
                    # 内部点
                    # RHS = 2p_n - p_{n-1} + (c²Δt²/2)∇²p_n + Δt²s_n
                    p_val = 2 * p_n[idx] - p_nm1[idx]
                    
                    # 拉普拉斯算子
                    laplace_p_n = (p_n[idx + 1] + p_n[idx - 1] + 
                                   p_n[idx + self.nx] + p_n[idx - self.nx] - 4 * p_n[idx]) / (self.h ** 2)
                    
                    source = self.source_func(self.x[j], self.y[i], t)
                    
                    rhs[idx] = p_val + (self.c ** 2) * (dt ** 2) * 0.5 * laplace_p_n + (dt ** 2) * source
        
        # 求解线性系统
        p_np1 = spsolve(A, rhs)
        
        return p_np1
    
    def solve(self, initial_condition: Callable, 
              T: float, dt: float,
              exact_solution: Callable = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        求解波方程
        
        Args:
            initial_condition: 初始条件 u(x, y, 0)
            T: 求解时间
            dt: 时间步长
            exact_solution: 精确解（用于误差分析）
            
        Returns:
            (x, y, t, p_all): 解向量、网格、时间、所有解
            统计信息 dict
        """
        # 初始化
        num_steps = int(T / dt)
        dt = T / num_steps  # 调整dt使得整除
        
        p_all = []
        time_steps = []
        
        # 初始条件 p(x, y, 0)
        p_0 = np.zeros((self.ny, self.nx))
        for i in range(self.ny):
            for j in range(self.nx):
                p_0[i, j] = initial_condition(self.x[j], self.y[i])
        p_n_flat = p_0.flatten()
        
        # p(x, y, -dt) ≈ p(x, y, 0) - dt * ∂p/∂t|_0 = p(x, y, 0) （零初始速度）
        p_nm1_flat = p_n_flat.copy()
        
        p_all.append(p_n_flat.copy())
        time_steps.append(0.0)
        
        # 构造矩阵
        A = self.build_cn_matrix(dt)
        
        # 时间积分
        errors = []
        times = []
        
        for n in range(num_steps):
            t_n = n * dt
            
            # 执行时间步
            p_np1_flat = self.solve_step(p_n_flat, p_nm1_flat, dt, t_n, A)
            
            # 保存
            p_all.append(p_np1_flat.copy())
            time_steps.append((n + 1) * dt)
            
            # 计算误差（如果有精确解）
            if exact_solution is not None:
                t_np1 = (n + 1) * dt
                p_exact = np.zeros((self.ny, self.nx))
                for i in range(self.ny):
                    for j in range(self.nx):
                        p_exact[i, j] = exact_solution(self.x[j], self.y[i], t_np1)
                
                error = np.linalg.norm(p_np1_flat - p_exact.flatten()) / np.linalg.norm(p_exact.flatten())
                errors.append(error)
                times.append(t_np1)
            
            # 更新
            p_nm1_flat = p_n_flat.copy()
            p_n_flat = p_np1_flat.copy()
        
        stats = {
            'num_steps': num_steps,
            'dt': dt,
            'cfl_ratio': self.c * dt / self.h,  # 实际CFL比值（可能>1，显示CN的无条件稳定性）
            'errors': np.array(errors) if errors else None,
            'error_times': np.array(times) if times else None,
        }
        
        return self.xx, self.yy, np.array(time_steps), np.array(p_all), stats
