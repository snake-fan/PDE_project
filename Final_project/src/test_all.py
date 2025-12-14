"""
完整测试和性能分析代码
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import norm
import time

from cn_scheme import CrankNicolsonSolver
from iterative_solvers import solve_poisson
from multigrid import MultigridVCycle


def create_laplacian_2d(n: int) -> tuple:
    """
    创建2D Laplacian矩阵 (n*n网格)
    边界条件：Dirichlet (u=0 on boundary)
    """
    # 1D三对角矩阵
    diag_main = np.ones(n) * (-4)
    diag_off = np.ones(n - 1) * 1
    T_1d = diags([diag_off, diag_main, diag_off], 
                 offsets=[-1, 0, 1], shape=(n, n), format='csr')
    
    # 2D Laplacian: L = I ⊗ T_1d + T_1d ⊗ I
    I_n = eye(n, format='csr')
    L = kron(I_n, T_1d, format='csr') + kron(T_1d, I_n, format='csr')
    
    return L / 1.0  # h² = 1


def test_iterative_solvers():
    """
    任务b：比较四种迭代求解器的性能
    """
    print("=" * 80)
    print("任务B：迭代求解器比较")
    print("=" * 80)
    
    results_by_size = {}
    grid_sizes = [32, 64, 128]
    
    for n in grid_sizes:
        print(f"\n网格大小: {n}x{n} (总计 {n*n} 个点)")
        
        # 创建测试问题
        A = create_laplacian_2d(n)
        b = np.ones(n * n)
        
        rel_tol = 1e-6
        max_iter = 10000
        
        results = {}
        methods = {
            'jacobi': {'method': 'jacobi'},
            'gs': {'method': 'gs'},
            'sor': {'method': 'sor', 'omega': 1.5},
            'cg': {'method': 'cg'},
        }
        
        for method_name, kwargs in methods.items():
            x, stats = solve_poisson(A, b, rel_tol=rel_tol, max_iter=max_iter, **kwargs)
            results[method_name] = stats
            
            print(f"\n  {method_name.upper()}:")
            print(f"    迭代次数: {stats['iterations']}")
            print(f"    计算时间: {stats['time_elapsed']:.4f} 秒")
            print(f"    最终残差: {stats['final_residual']:.2e}")
        
        results_by_size[n] = results
    
    return results_by_size


def test_multigrid_smoothing():
    """
    任务c：测试多重网格中前后平滑次数的影响
    """
    print("\n" + "=" * 80)
    print("任务C：多重网格V-cycle - 平滑次数影响")
    print("=" * 80)
    
    n = 64  # 网格大小
    A = create_laplacian_2d(n)
    b = np.ones(n * n)
    
    print(f"\n网格大小: {n}x{n}")
    print(f"网格层数: 2")
    
    smoothing_configs = [
        (1, 1),  # (前平滑, 后平滑)
        (1, 2),
        (2, 1),
        (2, 2),
        (3, 3),
    ]
    
    results = {}
    
    for pre, post in smoothing_configs:
        print(f"\n前平滑={pre}, 后平滑={post}:")
        
        mg = MultigridVCycle(A, b, grid_size=n, num_levels=2, 
                            smoother='jacobi', rel_tol=1e-6, max_iter=100)
        
        x, stats = mg.solve(pre_smooth=pre, post_smooth=post)
        results[(pre, post)] = stats
        
        print(f"  迭代次数: {stats['iterations']}")
        print(f"  计算时间: {stats['time_elapsed']:.4f} 秒")
        print(f"  最终残差: {stats['final_residual']:.2e}")
    
    return results


def test_multigrid_vs_iterative():
    """
    任务d：不同网格尺寸下比较多重网格与单层迭代法的性能
    """
    print("\n" + "=" * 80)
    print("任务D：多重网格 vs 单层迭代法 - 加速比和复杂度分析")
    print("=" * 80)
    
    grid_sizes = [32, 64, 128]  # 256 might be too large
    comparison = {}
    
    for n in grid_sizes:
        print(f"\n网格大小: {n}x{n} (总计 {n*n} 个点)")
        
        A = create_laplacian_2d(n)
        b = np.ones(n * n)
        rel_tol = 1e-6
        
        # 1. 单层迭代法 - Jacobi
        print(f"  单层迭代法 (Jacobi):")
        x_jacobi, stats_jacobi = solve_poisson(A, b, method='jacobi', 
                                              rel_tol=rel_tol, max_iter=10000)
        print(f"    迭代次数: {stats_jacobi['iterations']}")
        print(f"    计算时间: {stats_jacobi['time_elapsed']:.4f} 秒")
        
        # 2. 多重网格
        print(f"  多重网格 V-cycle:")
        mg = MultigridVCycle(A, b, grid_size=n, num_levels=2,
                            smoother='jacobi', rel_tol=rel_tol, max_iter=100)
        x_mg, stats_mg = mg.solve(pre_smooth=1, post_smooth=1)
        print(f"    迭代次数: {stats_mg['iterations']}")
        print(f"    计算时间: {stats_mg['time_elapsed']:.4f} 秒")
        
        # 计算加速比
        speedup = stats_jacobi['time_elapsed'] / stats_mg['time_elapsed']
        iter_reduction = stats_jacobi['iterations'] / stats_mg['iterations']
        
        print(f"  加速比 (时间): {speedup:.2f}x")
        print(f"  迭代次数减少: {iter_reduction:.2f}x")
        
        # 计算复杂度
        # 理论复杂度：Jacobi O(n²) 迭代，每次迭代 O(n²) 操作
        # 多重网格 O(n²) 复杂度
        jacobi_total_ops = stats_jacobi['iterations'] * (n * n)
        mg_total_ops = stats_mg['iterations'] * (n * n) * 4  # 粗略估计
        
        print(f"  估计总操作数:")
        print(f"    Jacobi: {jacobi_total_ops:.2e}")
        print(f"    多重网格: {mg_total_ops:.2e}")
        
        comparison[n] = {
            'jacobi': stats_jacobi,
            'mg': stats_mg,
            'speedup': speedup,
            'iter_reduction': iter_reduction,
        }
    
    return comparison


def test_cn_stability():
    """
    任务a：验证Crank-Nicolson隐式格式的无条件稳定性
    """
    print("\n" + "=" * 80)
    print("任务A：Crank-Nicolson隐式格式 - 稳定性验证")
    print("=" * 80)
    
    # 参数设置
    domain_size = (1.0, 1.0)
    grid_points = (33, 33)  # 32x32 内部点 + 1 边界
    c = 1.0
    T = 1.0
    
    # 初始条件：高斯脉冲
    def initial_condition(x, y):
        x0, y0 = 0.5, 0.5
        sigma = 0.1
        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    
    print(f"\n域大小: {domain_size}")
    print(f"网格点数: {grid_points}")
    print(f"波速: {c}")
    print(f"求解时间: {T}")
    
    # 测试不同的时间步长
    time_steps = [0.01, 0.05, 0.1, 0.2]  # 包括违反CFL的情况
    
    cn_results = {}
    
    for dt in time_steps:
        # CFL数
        h = domain_size[0] / (grid_points[0] - 1)
        cfl = c * dt / h
        
        print(f"\n时间步长: {dt}, CFL数: {cfl:.3f}", end="")
        
        if cfl > 1.0:
            print(" (超过CFL条件)")
        else:
            print(" (满足CFL条件)")
        
        # 求解
        solver = CrankNicolsonSolver(domain_size, grid_points, c)
        xx, yy, t_steps, p_all, stats = solver.solve(initial_condition, T, dt)
        
        # 检查解的有界性
        max_val = np.max(np.abs(p_all))
        final_energy = np.linalg.norm(p_all[-1])
        
        print(f"  最大值: {max_val:.4f}")
        print(f"  最终能量: {final_energy:.4f}")
        print(f"  收敛步数: {len(t_steps)}")
        
        cn_results[dt] = {
            'cfl': cfl,
            'max_val': max_val,
            'final_energy': final_energy,
            'num_steps': len(t_steps),
            'solution': p_all,
            'times': t_steps,
        }
    
    return cn_results


def main():
    """主测试程序"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "PDE数值模拟：波方程求解器完整测试".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    # 任务A：Crank-Nicolson稳定性
    cn_results = test_cn_stability()
    
    # 任务B：迭代求解器比较
    iterative_results = test_iterative_solvers()
    
    # 任务C：多重网格平滑次数
    mg_smoothing = test_multigrid_smoothing()
    
    # 任务D：加速比和复杂度分析
    comparison = test_multigrid_vs_iterative()
    
    print("\n" + "=" * 80)
    print("所有测试完成")
    print("=" * 80 + "\n")
    
    return {
        'cn': cn_results,
        'iterative': iterative_results,
        'mg_smoothing': mg_smoothing,
        'comparison': comparison,
    }


if __name__ == "__main__":
    results = main()
