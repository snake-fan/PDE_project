"""
几何多重网格(GMG) V-cycle算法实现
包括限制、延拓等算子，测试前后平滑次数的影响
"""

import numpy as np
from scipy.sparse import csr_matrix, diags, eye, kron, identity
from scipy.sparse.linalg import spsolve
import time
from typing import Tuple, List, Dict, Any
from iterative_solvers import JacobiSolver, GaussSeidelSolver, SORSolver


class MultigridVCycle:
    """多重网格V-cycle求解器"""
    
    def __init__(self, A: csr_matrix, b: np.ndarray,
                 grid_size: int = None,
                 num_levels: int = 2,
                 smoother: str = 'jacobi',
                 rel_tol: float = 1e-6,
                 max_iter: int = 1000):
        """
        初始化多重网格求解器
        
        Args:
            A: 最细网格系数矩阵
            b: 最细网格右侧向量
            grid_size: 细网格大小（n x n）
            num_levels: 网格层数
            smoother: 平滑器类型 ('jacobi', 'gs', 'sor')
            rel_tol: 相对残差容差
            max_iter: 最大V-cycle次数
        """
        self.A_fine = A
        self.b_fine = b.copy()
        self.grid_size = grid_size
        self.num_levels = num_levels
        self.smoother = smoother
        self.rel_tol = rel_tol
        self.max_iter = max_iter
        
        # 存储各层的矩阵和向量
        self.A_levels = [self.A_fine]
        self.b_levels = [self.b_fine.copy()]
        
        # 构造多层网格
        self._construct_hierarchy()
        
        # 统计
        self.residuals = []
        self.iterations = 0
        self.time_elapsed = 0.0
    
    def _construct_hierarchy(self):
        """构造多网格层级"""
        # 从最细网格开始逐层粗化
        current_A = self.A_fine
        current_grid_size = self.grid_size
        
        for level in range(1, self.num_levels):
            # 粗化网格大小
            coarse_grid_size = (current_grid_size - 1) // 2 + 1
            
            # 构造粗化矩阵（简单注入法）
            # 对于Poisson方程：-∇²u = f，粗网格矩阵保持相同形式
            coarse_A = self._coarsen_matrix(current_A, current_grid_size, coarse_grid_size)
            
            self.A_levels.append(coarse_A)
            self.b_levels.append(np.zeros(coarse_A.shape[0]))
            
            current_A = coarse_A
            current_grid_size = coarse_grid_size
    
    def _coarsen_matrix(self, A_fine: csr_matrix, fine_size: int, 
                        coarse_size: int) -> csr_matrix:
        """
        粗化矩阵（注入法）
        从细网格选择特定点形成粗网格矩阵
        """
        # 2D情况：只保留奇数索引的点
        n_fine = A_fine.shape[0]
        n_coarse = coarse_size * coarse_size
        
        # 建立细网格到粗网格的索引映射
        fine_to_coarse = np.full(n_fine, -1, dtype=int)
        coarse_idx = 0
        
        for i in range(coarse_size):
            for j in range(coarse_size):
                fine_i = i * 2
                fine_j = j * 2
                if fine_i < fine_size and fine_j < fine_size:
                    fine_idx = fine_i * fine_size + fine_j
                    fine_to_coarse[fine_idx] = coarse_idx
                    coarse_idx += 1
        
        # 转换为LIL格式便于行访问
        A_lil = A_fine.tolil()
        rows, cols, data = [], [], []
        
        for i in range(n_fine):
            if fine_to_coarse[i] >= 0:
                coarse_i = fine_to_coarse[i]
                for j in A_lil.rows[i]:
                    if fine_to_coarse[j] >= 0:
                        coarse_j = fine_to_coarse[j]
                        rows.append(coarse_i)
                        cols.append(coarse_j)
                        data.append(A_lil[i, j])
        
        # 构造粗网格矩阵
        A_coarse_csr = csr_matrix((data, (rows, cols)), shape=(n_coarse, n_coarse))
        
        return A_coarse_csr
    
    def _restriction(self, u_fine: np.ndarray, fine_size: int, 
                    coarse_size: int) -> np.ndarray:
        """
        限制算子：将细网格解限制到粗网格
        使用注入法（简单选择）或全加权限制
        """
        u_coarse = np.zeros(coarse_size * coarse_size)
        
        coarse_idx = 0
        for i in range(coarse_size):
            for j in range(coarse_size):
                fine_i = i * 2
                fine_j = j * 2
                if fine_i < fine_size and fine_j < fine_size:
                    fine_idx = fine_i * fine_size + fine_j
                    u_coarse[coarse_idx] = u_fine[fine_idx]
                    coarse_idx += 1
        
        return u_coarse
    
    def _prolongation(self, u_coarse: np.ndarray, fine_size: int, 
                     coarse_size: int) -> np.ndarray:
        """
        延拓算子：将粗网格解延拓到细网格
        使用双线性插值
        """
        u_fine = np.zeros(fine_size * fine_size)
        
        coarse_idx = 0
        for i in range(coarse_size):
            for j in range(coarse_size):
                fine_i = i * 2
                fine_j = j * 2
                if fine_i < fine_size and fine_j < fine_size:
                    fine_idx = fine_i * fine_size + fine_j
                    u_fine[fine_idx] = u_coarse[coarse_idx]
                    coarse_idx += 1
        
        # 双线性插值填充其他点
        for i in range(fine_size):
            for j in range(fine_size):
                idx = i * fine_size + j
                if u_fine[idx] == 0 and not (i % 2 == 0 and j % 2 == 0):
                    # 计算内插值
                    neighbors = []
                    for di in [-1, 1]:
                        for dj in [-1, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < fine_size and 0 <= nj < fine_size:
                                neighbors.append(u_fine[ni * fine_size + nj])
                    
                    if neighbors:
                        u_fine[idx] = np.mean(neighbors)
        
        return u_fine
    
    def _smooth(self, A: csr_matrix, u: np.ndarray, f: np.ndarray, 
                num_iter: int = 1) -> np.ndarray:
        """
        平滑操作：使用迭代法进行num_iter次迭代
        """
        for _ in range(num_iter):
            if self.smoother == 'jacobi':
                # Jacobi平滑
                D_inv = np.zeros(len(u))
                for i in range(len(u)):
                    if A[i, i] != 0:
                        D_inv[i] = 1.0 / A[i, i]
                
                LU = A.copy()
                for i in range(len(u)):
                    LU[i, i] = 0
                
                u = D_inv * (f - LU @ u)
            
            elif self.smoother == 'gs':
                # Gauss-Seidel平滑
                A_lil = A.tolil()
                for i in range(len(u)):
                    row_sum = 0.0
                    for j in A_lil.rows[i]:
                        if j != i:
                            row_sum += A_lil[i, j] * u[j]
                    
                    if A[i, i] != 0:
                        u[i] = (f[i] - row_sum) / A[i, i]
        
        return u
    
    def v_cycle(self, u: np.ndarray, f: np.ndarray, level: int = 0,
                pre_smooth: int = 1, post_smooth: int = 1) -> np.ndarray:
        """
        执行一个V-cycle迭代
        
        Args:
            u: 当前近似解
            f: 右侧向量
            level: 当前网格层（0为最细）
            pre_smooth: 下行前平滑次数
            post_smooth: 上行后平滑次数
        """
        # 最粗网格：直接求解
        if level == self.num_levels - 1:
            u = spsolve(self.A_levels[level].tocsr(), f)
            return u
        
        # 下行：前平滑
        u = self._smooth(self.A_levels[level], u, f, pre_smooth)
        
        # 计算细网格残差
        r = f - self.A_levels[level] @ u
        
        # 限制到粗网格
        fine_size = int(np.sqrt(len(u)))
        coarse_size = (fine_size - 1) // 2 + 1
        r_coarse = self._restriction(r, fine_size, coarse_size)
        
        # 在粗网格上求解修正方程
        e_coarse = np.zeros(len(r_coarse))
        e_coarse = self.v_cycle(e_coarse, r_coarse, level + 1, 
                               pre_smooth, post_smooth)
        
        # 延拓修正回细网格
        e_fine = self._prolongation(e_coarse, fine_size, coarse_size)
        
        # 更新解
        u = u + e_fine
        
        # 上行：后平滑
        u = self._smooth(self.A_levels[level], u, f, post_smooth)
        
        return u
    
    def solve(self, pre_smooth: int = 1, post_smooth: int = 1,
             x0: np.ndarray = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        使用多重网格V-cycle求解
        
        Args:
            pre_smooth: 下行前平滑次数
            post_smooth: 上行后平滑次数
            x0: 初始解
        """
        if x0 is not None:
            x = x0.copy()
        else:
            x = np.zeros(len(self.b_fine))
        
        start_time = time.time()
        self.residuals = []
        
        for k in range(self.max_iter):
            # 执行V-cycle
            x = self.v_cycle(x, self.b_fine, level=0,
                            pre_smooth=pre_smooth, post_smooth=post_smooth)
            
            # 计算残差
            r = self.b_fine - self.A_fine @ x
            rel_residual = np.linalg.norm(r) / np.linalg.norm(self.b_fine)
            self.residuals.append(rel_residual)
            
            if rel_residual < self.rel_tol:
                self.iterations = k + 1
                break
        
        self.time_elapsed = time.time() - start_time
        
        stats = {
            'iterations': self.iterations,
            'time_elapsed': self.time_elapsed,
            'residuals': np.array(self.residuals),
            'final_residual': self.residuals[-1] if self.residuals else np.inf,
            'pre_smooth': pre_smooth,
            'post_smooth': post_smooth,
        }
        
        return x, stats
