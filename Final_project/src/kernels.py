import numpy as np
from numba import njit, prange

# ==========================================
# 基础辅助函数
# ==========================================

@njit(parallel=True)
def compute_residual(u, b, alpha):
    """
    计算残差 r = b - A*u
    A 是 Crank-Nicolson 离散后的算子: (1+4a)u_ij - a(u_neighbors)
    """
    nx, ny = u.shape
    r = np.zeros_like(u)
    
    # 并行计算内部点
    for i in prange(1, nx-1):
        for j in range(1, ny-1):
            Au = (1 + 4 * alpha[i,j]) * u[i,j] - \
                 alpha[i,j] * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1])
            r[i,j] = b[i,j] - Au
            
    # 边界处理：假设 Dirichlet 边界残差为 0 (或根据具体边界条件修改)
    return r

@njit
def compute_norm(r):
    """计算 L2 范数"""
    return np.linalg.norm(r.flatten())

@njit(parallel=True)
def apply_A(u, alpha):
    """计算 v = A * u (用于 CG 方法)"""
    nx, ny = u.shape
    v = np.zeros_like(u)
    for i in prange(1, nx-1):
        for j in range(1, ny-1):
            v[i,j] = (1 + 4 * alpha[i,j]) * u[i,j] - \
                     alpha[i,j] * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1])
    return v

# ==========================================
# 迭代平滑器 (Smoother)
# ==========================================

@njit(parallel=True)
def jacobi_step(u, b, alpha):
    """Jacobi 迭代一步"""
    nx, ny = u.shape
    u_new = np.zeros_like(u)
    for i in prange(1, nx-1):
        for j in range(1, ny-1):
            neighbors = u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1]
            u_new[i,j] = (b[i,j] + alpha[i,j] * neighbors) / (1 + 4 * alpha[i,j])
    return u_new

@njit(parallel=True)
def gs_rb_step(u, b, alpha):
    """红黑 Gauss-Seidel 迭代 (Red-Black GS) - 适合并行化"""
    nx, ny = u.shape
    
    # Red Pass (i+j is even)
    for i in prange(1, nx-1):
        for j in range(1, ny-1):
            if (i + j) % 2 == 0:
                neighbors = u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1]
                u[i,j] = (b[i,j] + alpha[i,j] * neighbors) / (1 + 4 * alpha[i,j])
                
    # Black Pass (i+j is odd)
    for i in prange(1, nx-1):
        for j in range(1, ny-1):
            if (i + j) % 2 == 1:
                neighbors = u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1]
                u[i,j] = (b[i,j] + alpha[i,j] * neighbors) / (1 + 4 * alpha[i,j])
    return u

@njit(parallel=True)
def sor_rb_step(u, b, alpha, omega):
    """红黑 SOR 迭代"""
    nx, ny = u.shape
    # Red
    for i in prange(1, nx-1):
        for j in range(1, ny-1):
            if (i + j) % 2 == 0:
                neighbors = u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1]
                u_gs = (b[i,j] + alpha[i,j] * neighbors) / (1 + 4 * alpha[i,j])
                u[i,j] = (1 - omega) * u[i,j] + omega * u_gs
    # Black
    for i in prange(1, nx-1):
        for j in range(1, ny-1):
            if (i + j) % 2 == 1:
                neighbors = u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1]
                u_gs = (b[i,j] + alpha[i,j] * neighbors) / (1 + 4 * alpha[i,j])
                u[i,j] = (1 - omega) * u[i,j] + omega * u_gs
    return u

# ==========================================
# 多重网格算子
# ==========================================

@njit
def restriction(r):
    """Full Weighting 限制算子 (Fine -> Coarse)"""
    nx_f, ny_f = r.shape
    nx_c = (nx_f - 1) // 2 + 1
    ny_c = (ny_f - 1) // 2 + 1
    r_c = np.zeros((nx_c, ny_c))
    
    for i in range(1, nx_c - 1):
        for j in range(1, ny_c - 1):
            if2 = 2 * i
            jf2 = 2 * j
            # Full Weighting stencil
            # 1/16 * [1 2 1]
            #        [2 4 2]
            #        [1 2 1]
            center = 4.0 * r[if2, jf2]
            cross  = 2.0 * (r[if2+1,jf2] + r[if2-1,jf2] + r[if2,jf2+1] + r[if2,jf2-1])
            corner = 1.0 * (r[if2+1,jf2+1] + r[if2+1,jf2-1] + r[if2-1,jf2+1] + r[if2-1,jf2-1])
            r_c[i,j] = (center + cross + corner) / 16.0
    return r_c

@njit
def prolongation(e_c, shape_f):
    """双线性插值延拓算子 (Coarse -> Fine)"""
    nx_f, ny_f = shape_f
    e_f = np.zeros(shape_f)
    nx_c, ny_c = e_c.shape
    
    for i in range(nx_c - 1):
        for j in range(ny_c - 1):
            if2 = 2 * i
            jf2 = 2 * j
            val = e_c[i, j]
            
            # 填充细网格点
            e_f[if2, jf2]     = val
            e_f[if2+1, jf2]   = 0.5 * (val + e_c[i+1, j])
            e_f[if2, jf2+1]   = 0.5 * (val + e_c[i, j+1])
            e_f[if2+1, jf2+1] = 0.25 * (val + e_c[i+1, j] + e_c[i, j+1] + e_c[i+1, j+1])
    return e_f