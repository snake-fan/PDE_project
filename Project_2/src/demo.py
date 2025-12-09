import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time
from pathlib import Path
import os
from numba import jit

# ==========================================
# Part 1: Numba 加速内核
# ==========================================

@jit(nopython=True)
def get_k_numba(x, y):
    """扩散系数 k(x,y)"""
    return 1.0 + 0.9 * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.02)

@jit(nopython=True)
def tdma_solve(a, b, c, d, x):
    """三对角矩阵求解器 (Thomas Algorithm)"""
    n = len(d)
    c[0] /= b[0]
    d[0] /= b[0]
    for i in range(1, n):
        temp = 1.0 / (b[i] - a[i] * c[i-1])
        c[i] *= temp
        d[i] = (d[i] - a[i] * d[i-1]) * temp
    x[n-1] = d[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d[i] - c[i] * x[i+1]

@jit(nopython=True)
def compute_residual_norm(u, coef_h, coef_v, source):
    """计算残差 L2 范数 ||b - Ax||"""
    N = u.shape[0] - 1
    sum_sq = 0.0
    for i in range(1, N):
        for j in range(1, N):
            a_E = coef_h[i, j]; a_W = coef_h[i-1, j]
            a_N = coef_v[i, j]; a_S = coef_v[i, j-1]
            a_P = a_E + a_W + a_N + a_S
            
            Au = a_P * u[i, j] - (a_E * u[i+1, j] + a_W * u[i-1, j] + 
                                  a_N * u[i, j+1] + a_S * u[i, j-1])
            r = source[i, j] - Au
            sum_sq += r * r
    return np.sqrt(sum_sq)

@jit(nopython=True)
def line_sor_kernel(u, coef_h, coef_v, source, max_iter, tol, omega):
    """Line SOR 求解器"""
    N = u.shape[0] - 1
    n_inner = N - 1
    vec_a = np.zeros(n_inner); vec_b = np.zeros(n_inner)
    vec_c = np.zeros(n_inner); vec_d = np.zeros(n_inner); vec_x = np.zeros(n_inner)
    residuals = []
    
    r0 = compute_residual_norm(u, coef_h, coef_v, source)
    if r0 == 0: r0 = 1.0
    residuals.append(r0)
    
    for it in range(max_iter):
        for j in range(1, N): 
            for k in range(n_inner):
                i = k + 1 
                a_E = coef_h[i, j]; a_W = coef_h[i-1, j]
                a_N = coef_v[i, j]; a_S = coef_v[i, j-1]
                a_P = a_E + a_W + a_N + a_S
                
                vec_b[k] = a_P
                if k > 0: vec_a[k] = -a_W
                if k < n_inner - 1: vec_c[k] = -a_E
                vec_d[k] = source[i, j] + a_N * u[i, j+1] + a_S * u[i, j-1]
            
            tdma_solve(vec_a, vec_b, vec_c, vec_d, vec_x)
            for k in range(n_inner):
                u[k+1, j] += omega * (vec_x[k] - u[k+1, j])
        
        rk = compute_residual_norm(u, coef_h, coef_v, source)
        residuals.append(rk)
        if rk / r0 < tol: break
            
    return u, residuals

@jit(nopython=True)
def red_black_gs_smoother(u, coef_h, coef_v, source, steps):
    """红黑 GS 平滑器"""
    N = u.shape[0] - 1
    for _ in range(steps):
        # Red Pass
        for j in range(1, N):
            start_i = 1 if (1 + j) % 2 == 0 else 2
            for i in range(start_i, N, 2):
                a_E = coef_h[i, j]; a_W = coef_h[i-1, j]
                a_N = coef_v[i, j]; a_S = coef_v[i, j-1]
                a_P = a_E + a_W + a_N + a_S
                rhs = source[i, j] + a_E*u[i+1,j] + a_W*u[i-1,j] + a_N*u[i,j+1] + a_S*u[i,j-1]
                u[i, j] = rhs / a_P
        # Black Pass
        for j in range(1, N):
            start_i = 1 if (1 + j) % 2 != 0 else 2
            for i in range(start_i, N, 2):
                a_E = coef_h[i, j]; a_W = coef_h[i-1, j]
                a_N = coef_v[i, j]; a_S = coef_v[i, j-1]
                a_P = a_E + a_W + a_N + a_S
                rhs = source[i, j] + a_E*u[i+1,j] + a_W*u[i-1,j] + a_N*u[i,j+1] + a_S*u[i,j-1]
                u[i, j] = rhs / a_P
    return u

@jit(nopython=True)
def mg_restrict_full(rf, N_fine):
    """Full Weighting Restriction"""
    N_coarse = N_fine // 2
    rc = np.zeros((N_coarse + 1, N_coarse + 1))
    for i in range(1, N_coarse):
        for j in range(1, N_coarse):
            fi, fj = 2 * i, 2 * j
            val = 4.0 * rf[fi, fj]
            val += 2.0 * (rf[fi+1, fj] + rf[fi-1, fj] + rf[fi, fj+1] + rf[fi, fj-1])
            val += 1.0 * (rf[fi+1, fj+1] + rf[fi-1, fj+1] + rf[fi+1, fj-1] + rf[fi-1, fj-1])
            rc[i, j] = val / 16.0
    return rc

@jit(nopython=True)
def mg_prolong_full(ec, N_coarse):
    """Bilinear Interpolation"""
    N_fine = N_coarse * 2
    ef = np.zeros((N_fine + 1, N_fine + 1))
    for i in range(N_coarse):
        for j in range(N_coarse):
            val = ec[i, j]
            ef[2*i, 2*j] = val
            if 2*i+1 <= N_fine: 
                ef[2*i+1, 2*j] = 0.5 * (val + ec[i+1, j])
            if 2*j+1 <= N_fine: 
                ef[2*i, 2*j+1] = 0.5 * (val + ec[i, j+1])
            if 2*i+1 <= N_fine and 2*j+1 <= N_fine:
                ef[2*i+1, 2*j+1] = 0.25 * (val + ec[i+1, j] + ec[i, j+1] + ec[i+1, j+1])
    return ef

@jit(nopython=True)
def calculate_residual_field(u, coef_h, coef_v, source):
    """计算残差场"""
    N = u.shape[0] - 1
    r = np.zeros_like(u)
    for i in range(1, N):
        for j in range(1, N):
            a_E = coef_h[i, j]; a_W = coef_h[i-1, j]
            a_N = coef_v[i, j]; a_S = coef_v[i, j-1]
            a_P = a_E + a_W + a_N + a_S
            Au = a_P * u[i, j] - (a_E * u[i+1, j] + a_W * u[i-1, j] + 
                                  a_N * u[i, j+1] + a_S * u[i, j-1])
            r[i, j] = source[i, j] - Au
    return r

# ==========================================
# Part 2: Python 辅助与主逻辑
# ==========================================

def get_f_func(x, y):
    return 8 * np.pi**2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

def prepare_grid_data(N):
    h = 1.0 / N
    coef_h = np.zeros((N, N+1)) 
    coef_v = np.zeros((N+1, N))
    source = np.zeros((N+1, N+1))
    for i in range(N):
        for j in range(1, N): coef_h[i, j] = get_k_numba((i + 0.5) * h, j * h)
    for i in range(1, N):
        for j in range(N): coef_v[i, j] = get_k_numba(i * h, (j + 0.5) * h)
    for i in range(1, N):
        for j in range(1, N): source[i, j] = get_f_func(i * h, j * h) * h**2
    return coef_h, coef_v, source

def build_sparse_matrix(N):
    h = 1.0 / N
    n_inner = N - 1
    num_unknowns = n_inner * n_inner
    data, row_ind, col_ind = [], [], []
    b = np.zeros(num_unknowns)
    coef_h, coef_v, source_grid = prepare_grid_data(N)
    
    for i in range(n_inner):
        for j in range(n_inner):
            row = i * n_inner + j
            pi, pj = i + 1, j + 1
            k_E = coef_h[pi, pj]; k_W = coef_h[pi-1, pj]
            k_N = coef_v[pi, pj]; k_S = coef_v[pi, pj-1]
            diag = k_E + k_W + k_N + k_S
            data.append(diag); row_ind.append(row); col_ind.append(row)
            if i < n_inner - 1:
                col = (i + 1) * n_inner + j; data.append(-k_E); row_ind.append(row); col_ind.append(col)
            if i > 0:
                col = (i - 1) * n_inner + j; data.append(-k_W); row_ind.append(row); col_ind.append(col)
            if j < n_inner - 1:
                col = i * n_inner + (j + 1); data.append(-k_N); row_ind.append(row); col_ind.append(col)
            if j > 0:
                col = i * n_inner + (j - 1); data.append(-k_S); row_ind.append(row); col_ind.append(col)
            b[row] = source_grid[pi, pj]
    return sp.csr_matrix((data, (row_ind, col_ind)), shape=(num_unknowns, num_unknowns)), b

def mg_v_cycle(u, N, coef_h, coef_v, rhs):
    if N <= 8:
        A_base, _ = build_sparse_matrix(N)
        n_inner = N - 1
        b_flat = rhs[1:N, 1:N].flatten()
        u_flat = spla.spsolve(A_base, b_flat)
        u[1:N, 1:N] = u_flat.reshape((n_inner, n_inner))
        return u

    # 1. Pre-smoothing
    u = red_black_gs_smoother(u, coef_h, coef_v, rhs, steps=3)
    
    # 2. Residual
    r = calculate_residual_field(u, coef_h, coef_v, rhs)
    
    # 3. Restriction (WITH SCALING FIX 4.0)
    # 因为 2D 问题，h -> 2h，算子缩放需要乘 4
    rc = mg_restrict_full(r, N) * 4.0 
    
    # 4. Coarse Solve
    N_c = N // 2
    coef_h_c, coef_v_c, _ = prepare_grid_data(N_c)
    ec = np.zeros((N_c + 1, N_c + 1))
    ec = mg_v_cycle(ec, N_c, coef_h_c, coef_v_c, rc)
    
    # 5. Prolongation
    ef = mg_prolong_full(ec, N_c)
    u += ef
    
    # 6. Post-smoothing
    u = red_black_gs_smoother(u, coef_h, coef_v, rhs, steps=3)
    return u

def solve_line_sor(N, tol=1e-6):
    coef_h, coef_v, source = prepare_grid_data(N)
    u = np.zeros((N+1, N+1))
    t0 = time.time()
    u, res = line_sor_kernel(u, coef_h, coef_v, source, 3000, tol, 1.75)
    print(f"Line SOR: {len(res)-1} iters, {(time.time()-t0):.4f}s")
    return u, np.array(res)/res[0]

def solve_cg(N, tol=1e-6):
    A, b = build_sparse_matrix(N)
    residuals = []
    r0 = np.linalg.norm(b); residuals.append(r0)
    def cb(xk): residuals.append(np.linalg.norm(b - A@xk))
    t0 = time.time()
    u_flat, _ = spla.cg(A, b, rtol=tol, maxiter=1000, callback=cb)
    print(f"CG: {len(residuals)-1} iters, {(time.time()-t0):.4f}s")
    u = np.zeros((N+1, N+1))
    u[1:N, 1:N] = u_flat.reshape((N-1, N-1))
    return u, np.array(residuals)/r0

def solve_mg(N, tol=1e-6):
    coef_h, coef_v, source = prepare_grid_data(N)
    u = np.zeros((N+1, N+1))
    residuals = []
    r0 = compute_residual_norm(u, coef_h, coef_v, source)
    if r0 == 0: r0 = 1.0
    residuals.append(r0)
    t0 = time.time()
    for i in range(50):
        u = mg_v_cycle(u, N, coef_h, coef_v, source)
        rk = compute_residual_norm(u, coef_h, coef_v, source)
        residuals.append(rk)
        if rk / r0 < tol:
            print(f"Multigrid: Converged at iter {i+1}")
            break
    print(f"Multigrid: {len(residuals)-1} iters, {(time.time()-t0):.4f}s")
    return u, np.array(residuals)/r0

def main():
    N = 64
    print(f"Solving N={N}, tol=1e-6...")
    u1, r1 = solve_line_sor(N)
    u2, r2 = solve_cg(N)
    u3, r3 = solve_mg(N)
    

    project_path = Path(__file__).resolve().parent.parent
    target_path = os.path.join(project_path, 'plots')

    plt.figure(figsize=(7, 5))
    plt.semilogy(r1, label='Line SOR', linestyle='--', color='blue')
    plt.semilogy(r2, label='CG', linestyle='-', color='orange')
    plt.semilogy(r3, label='Multigrid', marker='o', color='green', markersize=4)
    plt.ylabel(r'Relative Residual $\|r_k\|_2 / \|r_0\|_2$')
    plt.xlabel('Iterations')
    plt.title(f'Convergence Comparison (N={N})')
    plt.axhline(1e-6, color='gray', linestyle=':')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(target_path + '/Convergence_Comparison.jpg', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.imshow(u3.T, origin='lower', extent=[0,1,0,1], cmap='jet')
    plt.colorbar(label='Potential $u$')
    plt.title('Potential Distribution (Multigrid)')
    plt.xlabel('x'); plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(target_path + '/Potential Distribution.jpg')
    plt.close()

if __name__ == "__main__":
    main()