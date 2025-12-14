"""
迭代求解器：Jacobi、Gauss-Seidel、SOR、Conjugate Gradient
"""

import numpy as np
from scipy.sparse import csr_matrix, diags
import time
from typing import Tuple, Dict, Any


class IterativeSolver:
    """基类：迭代求解器"""
    
    def __init__(self, A: csr_matrix, b: np.ndarray, 
                 rel_tol: float = 1e-6, max_iter: int = 10000):
        """
        初始化迭代求解器
        
        Args:
            A: 系数矩阵（稀疏）
            b: 右侧向量
            rel_tol: 相对残差容差
            max_iter: 最大迭代次数
        """
        self.A = A
        self.b = b.copy()
        self.rel_tol = rel_tol
        self.max_iter = max_iter
        self.n = len(b)
        
        # 初始解
        self.x = np.zeros(self.n)
        
        # 收敛历史
        self.residuals = []
        self.iterations = 0
        self.time_elapsed = 0.0
        
    def compute_residual(self, x: np.ndarray) -> float:
        """计算相对残差"""
        r = self.b - self.A @ x
        rel_residual = np.linalg.norm(r) / np.linalg.norm(self.b)
        return rel_residual
    
    def solve(self, x0: np.ndarray = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """求解（由子类实现）"""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """返回统计信息"""
        return {
            'iterations': self.iterations,
            'time_elapsed': self.time_elapsed,
            'residuals': np.array(self.residuals),
            'final_residual': self.residuals[-1] if self.residuals else np.inf,
        }


class JacobiSolver(IterativeSolver):
    """Jacobi迭代法"""
    
    def solve(self, x0: np.ndarray = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Jacobi迭代：使用前一次迭代的所有邻居值
        x^{k+1} = D^{-1}(b - (L+U)x^k)
        """
        if x0 is not None:
            self.x = x0.copy()
        else:
            self.x = np.zeros(self.n)
        
        start_time = time.time()
        self.residuals = []
        
        # 提取对角线和非对角线部分
        D_inv = np.zeros(self.n)
        for i in range(self.n):
            D_inv[i] = 1.0 / self.A[i, i] if self.A[i, i] != 0 else 0
        
        LU = self.A.copy()
        for i in range(self.n):
            LU[i, i] = 0
        
        for k in range(self.max_iter):
            # x_{k+1} = D^{-1}(b - (L+U)x_k)
            x_new = D_inv * (self.b - LU @ self.x)
            
            # 计算残差
            rel_residual = self.compute_residual(x_new)
            self.residuals.append(rel_residual)
            
            self.x = x_new
            
            if rel_residual < self.rel_tol:
                self.iterations = k + 1
                break
        
        self.time_elapsed = time.time() - start_time
        return self.x, self.get_stats()


class GaussSeidelSolver(IterativeSolver):
    """Gauss-Seidel迭代法"""
    
    def solve(self, x0: np.ndarray = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Gauss-Seidel迭代：使用最新计算的邻居值（就地更新）
        """
        if x0 is not None:
            self.x = x0.copy()
        else:
            self.x = np.zeros(self.n)
        
        start_time = time.time()
        self.residuals = []
        
        # 转换为LIL格式便于高效访问
        A_lil = self.A.tolil()
        
        for k in range(self.max_iter):
            x_new = self.x.copy()
            
            for i in range(self.n):
                # 计算第i行的贡献
                row_sum = 0.0
                a_ii = A_lil[i, i]
                
                for j in A_lil.rows[i]:
                    if j != i:
                        row_sum += A_lil[i, j] * x_new[j] if j < i else A_lil[i, j] * self.x[j]
                
                if a_ii != 0:
                    x_new[i] = (self.b[i] - row_sum) / a_ii
            
            # 计算残差
            rel_residual = self.compute_residual(x_new)
            self.residuals.append(rel_residual)
            
            self.x = x_new
            
            if rel_residual < self.rel_tol:
                self.iterations = k + 1
                break
        
        self.time_elapsed = time.time() - start_time
        return self.x, self.get_stats()


class SORSolver(IterativeSolver):
    """逐次超松弛(SOR)迭代法"""
    
    def __init__(self, A: csr_matrix, b: np.ndarray, 
                 rel_tol: float = 1e-6, max_iter: int = 10000,
                 omega: float = 1.2):
        """
        初始化SOR求解器
        
        Args:
            omega: 松弛因子（1 < omega < 2）
        """
        super().__init__(A, b, rel_tol, max_iter)
        self.omega = omega
    
    def solve(self, x0: np.ndarray = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        SOR迭代：x_{k+1} = (1-ω)x_k + ω*x_{GS}
        """
        if x0 is not None:
            self.x = x0.copy()
        else:
            self.x = np.zeros(self.n)
        
        start_time = time.time()
        self.residuals = []
        
        A_lil = self.A.tolil()
        
        for k in range(self.max_iter):
            x_new = self.x.copy()
            
            # Gauss-Seidel步
            for i in range(self.n):
                row_sum = 0.0
                a_ii = A_lil[i, i]
                
                for j in A_lil.rows[i]:
                    if j != i:
                        row_sum += A_lil[i, j] * x_new[j] if j < i else A_lil[i, j] * self.x[j]
                
                x_gs = (self.b[i] - row_sum) / a_ii if a_ii != 0 else 0.0
                
                # 松弛
                x_new[i] = (1 - self.omega) * self.x[i] + self.omega * x_gs
            
            # 计算残差
            rel_residual = self.compute_residual(x_new)
            self.residuals.append(rel_residual)
            
            self.x = x_new
            
            if rel_residual < self.rel_tol:
                self.iterations = k + 1
                break
        
        self.time_elapsed = time.time() - start_time
        return self.x, self.get_stats()


class ConjugateGradientSolver(IterativeSolver):
    """共轭梯度法(CG)"""
    
    def solve(self, x0: np.ndarray = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        标准CG算法（假设A为对称正定）
        """
        if x0 is not None:
            self.x = x0.copy()
        else:
            self.x = np.zeros(self.n)
        
        start_time = time.time()
        self.residuals = []
        
        # 初始残差
        r = self.b - self.A @ self.x
        p = r.copy()
        
        rel_residual = np.linalg.norm(r) / np.linalg.norm(self.b)
        self.residuals.append(rel_residual)
        
        if rel_residual < self.rel_tol:
            self.iterations = 1
            self.time_elapsed = time.time() - start_time
            return self.x, self.get_stats()
        
        for k in range(self.max_iter):
            # α_k = (r_k^T r_k) / (p_k^T A p_k)
            r_norm_sq = np.dot(r, r)
            Ap = self.A @ p
            alpha = r_norm_sq / np.dot(p, Ap)
            
            # x_{k+1} = x_k + α_k p_k
            self.x = self.x + alpha * p
            
            # r_{k+1} = r_k - α_k A p_k
            r = r - alpha * Ap
            
            # 计算残差
            rel_residual = np.linalg.norm(r) / np.linalg.norm(self.b)
            self.residuals.append(rel_residual)
            
            if rel_residual < self.rel_tol:
                self.iterations = k + 1
                break
            
            # γ_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
            r_norm_sq_new = np.dot(r, r)
            gamma = r_norm_sq_new / r_norm_sq
            
            # p_{k+1} = r_{k+1} + γ_k p_k
            p = r + gamma * p
        
        self.time_elapsed = time.time() - start_time
        return self.x, self.get_stats()


def solve_poisson(A: csr_matrix, b: np.ndarray, 
                  method: str = 'cg', **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    使用指定方法求解Poisson方程
    
    Args:
        A: 系数矩阵
        b: 右侧向量
        method: 求解方法 ('jacobi', 'gs', 'sor', 'cg')
        **kwargs: 方法特定参数
        
    Returns:
        (x, stats): 解和统计信息
    """
    rel_tol = kwargs.get('rel_tol', 1e-6)
    max_iter = kwargs.get('max_iter', 10000)
    
    if method.lower() == 'jacobi':
        solver = JacobiSolver(A, b, rel_tol, max_iter)
    elif method.lower() == 'gs' or method.lower() == 'gauss_seidel':
        solver = GaussSeidelSolver(A, b, rel_tol, max_iter)
    elif method.lower() == 'sor':
        omega = kwargs.get('omega', 1.2)
        solver = SORSolver(A, b, rel_tol, max_iter, omega)
    elif method.lower() == 'cg' or method.lower() == 'conjugate_gradient':
        solver = ConjugateGradientSolver(A, b, rel_tol, max_iter)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    x0 = kwargs.get('x0', None)
    return solver.solve(x0)
