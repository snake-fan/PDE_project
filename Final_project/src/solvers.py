import numpy as np
import time
import kernels as kn

class LinearSolver:
    def __init__(self, alpha_map, method='mg', **kwargs):
        """
        初始化线性求解器
        method: 'jacobi', 'gs', 'sor', 'cg', 'mg'
        kwargs: 包含 omega, pre_smooth, post_smooth 等参数
        """
        self.alpha = alpha_map
        self.method = method.lower()
        self.kwargs = kwargs
        self.history = [] # 记录残差历史

    def solve(self, u_guess, b, tol=1e-6, max_iter=1000):
        """求解 Ax = b"""
        u = u_guess.copy()
        self.history = []
        
        # 计算初始残差
        r0 = kn.compute_residual(u, b, self.alpha)
        norm_r0 = kn.compute_norm(r0)
        if norm_r0 < 1e-12: norm_r0 = 1.0
        
        start_time = time.time()
        
        # CG 方法有独特的逻辑，单独处理
        if self.method == 'cg':
            return self._solve_cg(u, b, tol, max_iter, norm_r0, start_time)

        # 迭代法通用循环
        for k in range(max_iter):
            if self.method == 'jacobi':
                u = kn.jacobi_step(u, b, self.alpha)
            elif self.method == 'gs':
                u = kn.gs_rb_step(u, b, self.alpha)
            elif self.method == 'sor':
                omega = self.kwargs.get('omega', 1.5)
                u = kn.sor_rb_step(u, b, self.alpha, omega)
            elif self.method == 'mg':
                pre = self.kwargs.get('pre', 2)
                post = self.kwargs.get('post', 2)
                u = self._v_cycle(u, b, self.alpha, pre, post)
            
            # 每隔几步检查收敛，减少开销（这里每步检查以便画收敛图）
            r = kn.compute_residual(u, b, self.alpha)
            rel_res = kn.compute_norm(r) / norm_r0
            self.history.append(rel_res)
            
            if rel_res < tol:
                return u, k+1, time.time() - start_time
                
        return u, max_iter, time.time() - start_time

    def _solve_cg(self, u, b, tol, max_iter, norm_r0, start_time):
        """共轭梯度法"""
        r = b - kn.apply_A(u, self.alpha)
        p = r.copy()
        rsold = np.sum(r * r)
        
        for k in range(max_iter):
            Ap = kn.apply_A(p, self.alpha)
            alpha_cg = rsold / (np.sum(p * Ap) + 1e-15)
            u = u + alpha_cg * p
            r = r - alpha_cg * Ap
            rsnew = np.sum(r * r)
            
            rel_res = np.sqrt(rsnew) / norm_r0
            self.history.append(rel_res)
            
            if rel_res < tol:
                return u, k+1, time.time() - start_time
                
            p = r + (rsnew / rsold) * p
            rsold = rsnew
            
        return u, max_iter, time.time() - start_time

    def _v_cycle(self, u, b, alpha, pre, post):
        """几何多重网格 V-Cycle 递归"""
        nx, ny = u.shape
        
        # 1. 最粗层直接求解 (或迭代多次)
        if nx <= 5:
            for _ in range(20):
                u = kn.gs_rb_step(u, b, alpha)
            return u
            
        # 2. 前平滑 (Pre-smoothing)
        for _ in range(pre):
            u = kn.gs_rb_step(u, b, alpha)
            
        # 3. 计算残差并限制
        r = kn.compute_residual(u, b, alpha)
        r_c = kn.restriction(r)
        
        # 4. 粗网格系数处理
        # 物理意义: alpha = c^2 dt^2 / (2 h^2)
        # 粗网格 h_c = 2 * h_f => alpha_c = alpha_f / 4
        # 对于变系数，进行下采样
        alpha_c = alpha[::2, ::2] * 0.25
        
        # 5. 递归求解误差方程 Ae = r
        e_c = np.zeros_like(r_c)
        e_c = self._v_cycle(e_c, r_c, alpha_c, pre, post)
        
        # 6. 延拓并修正
        e_f = kn.prolongation(e_c, u.shape)
        u = u + e_f
        
        # 7. 后平滑 (Post-smoothing)
        for _ in range(post):
            u = kn.gs_rb_step(u, b, alpha)
            
        return u