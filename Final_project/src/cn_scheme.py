import numpy as np
from solvers import LinearSolver

class CNSimulation:
    def __init__(self, model, dt_factor=2.0):
        self.model = model
        
        # 计算时间步长
        # 显式 CFL 极限: dt <= h / (c_max * sqrt(2))
        self.dt_explicit = model.dx / (model.c_max * np.sqrt(2))
        self.dt = dt_factor * self.dt_explicit
        
        # 预计算系数矩阵 Alpha
        # alpha(x,y) = c(x,y)^2 * dt^2 / (2 * h^2)
        self.alpha_map = (model.c_map**2 * self.dt**2) / (2 * model.dx**2)
        
        # 初始化波场
        self.p_prev = np.zeros((model.nx, model.ny)) # p^{n-1}
        self.p_curr = np.zeros((model.nx, model.ny)) # p^n
        
        self.time = 0.0

    def step(self, method='mg', **solver_kwargs):
        """
        执行一个时间步: p^n -> p^{n+1}
        求解方程: (I - A) p^{n+1} = RHS
        在我们的定义中，求解的是 Ax = b，其中 A 对应 kernels 里的 (1+4a)u - a(neighbors)
        """
        nx, ny = self.p_curr.shape
        
        # 1. 计算 RHS (显式部分)
        # b = 2*p^n - p^{n-1} + alpha * Laplacian(p^n) + dt^2 * s
        
        # 简易拉普拉斯计算 (内部点)
        lap_p = np.zeros_like(self.p_curr)
        p = self.p_curr
        lap_p[1:-1, 1:-1] = p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4*p[1:-1, 1:-1]
        
        # 源项
        s_field = self.model.get_source_term(self.time)
        
        # 组装 RHS b
        b = 2 * self.p_curr - self.p_prev + \
            self.alpha_map * lap_p + \
            (self.dt**2) * s_field
            
        # 2. 隐式求解 p^{n+1}
        # 提取 tol 和 max_iter，避免传入 Solver 的 init
        tol = solver_kwargs.pop('tol', 1e-6)
        max_iter = solver_kwargs.pop('max_iter', 1000)
        
        # 实例化求解器 (method='mg', 'cg', etc.)
        # 注意: 我们的 LHS 矩阵系数也是 alpha_map
        solver = LinearSolver(self.alpha_map, method=method, **solver_kwargs)
        
        # 使用 p_curr 作为初值 (Warm Start)
        p_next, iters, t_solve = solver.solve(self.p_curr, b, tol=tol, max_iter=max_iter)
        
        # 3. 更新状态
        self.p_prev = self.p_curr
        self.p_curr = p_next
        self.time += self.dt
        
        return p_next, iters, t_solve