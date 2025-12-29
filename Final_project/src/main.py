import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import LiverModel
from cn_scheme import CNSimulation
from solvers import LinearSolver # 显式导入用于 Task B/D
import config as cfg
from pathlib import Path
import os

# 设置绘图风格
plt.style.use('default')

# --- Task A: 稳定性验证 ---
def task_a_stability():
    print("\n[Task A] 验证大时间步长稳定性...")
    N = 65
    h = 0.1 / N
    model = LiverModel(h=h)
    
    # 对比 0.5倍 CFL (显式稳定) 和 2.0倍 CFL (隐式稳定，显式会炸)
    factors = [0.5, 2.0]
    results = {}
    
    for f in factors:
        sim = CNSimulation(model, dt_factor=f)
        center_p = []
        times = []
        # 运行 200 步
        for _ in range(200):
            sim.step(method='mg', tol=1e-4)
            # 记录中心点声压
            center_p.append(sim.p_curr[int(N/2), int(N/2)])
            times.append(sim.time)
        results[f] = (times, center_p)
        
    plt.figure(figsize=(10, 5))
    for f, (t, p) in results.items():
        plt.plot(t, p, label=f'dt = {f} * CFL_limit')
    plt.title("Stability Verification: Center Pressure History")
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (Pa)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    rootpath = Path(__file__).resolve().parent.parent / 'figs'
    rootpath.mkdir(exist_ok=True)
    fig_dir = rootpath / "Task2_A_Stability.png"
    plt.savefig(fig_dir)
    print("  -> Saved 'Task2_A_Stability.png'")

# --- Task B: 求解器对比 ---
def task_b_solvers_comparison():
    print("\n[Task B] 对比四种迭代求解器 (Jacobi, GS, SOR, CG)...")
    N = 65
    h = 0.1 / N
    model = LiverModel(h=h)
    sim = CNSimulation(model, dt_factor=2.0)
    print("  Running explicit steps to generate wave field...")
    for _ in range(50):
        sim.step(method='jacobi', max_iter=1) # 用简单方法先跑几步
    
    methods = ['jacobi', 'gs', 'sor', 'cg', 'mg']
    plt.figure(figsize=(10, 6))
    
    data = []
    
    for m in methods:
        print(f"  Testing {m}...")
        # 为了公平，每次重置仿真对象
        temp_sim = CNSimulation(model, dt_factor=2.0)
        temp_sim.p_curr = sim.p_curr.copy()
        
        # 准备参数
        solve_params = {'tol': 1e-6, 'max_iter': 5000}
        init_params = {}
        if m == 'sor': init_params['omega'] = 1.4
        
        # 运行一步
        combined_params = {**solve_params, **init_params}
        _, iters, t_elapsed = temp_sim.step(method=m, **combined_params)
        
        data.append({'Method': m, 'Iterations': iters, 'Time': t_elapsed})
        
        # 为了画收敛曲线，我们需要 Solver 对象的 history
        # 手动实例化 Solver 再跑一次 (为了获取 history list)
        solver = LinearSolver(temp_sim.alpha_map, method=m, **init_params)
        solver.solve(np.zeros_like(temp_sim.p_curr), temp_sim.p_curr, **solve_params) # Mock RHS as p_curr
        plt.semilogy(solver.history, label=m.upper())

    plt.title("Solver Convergence History (Residual < 1e-6)")
    plt.xlabel("Iterations")
    plt.ylabel("Relative Residual")
    plt.legend()
    plt.grid(True, alpha=0.3)

    rootpath = Path(__file__).resolve().parent.parent / 'figs'
    rootpath.mkdir(exist_ok=True)
    fig_dir = rootpath / "Task2_B_Convergence.png"
    plt.savefig(fig_dir)
    
    print("\nSolver Performance:")
    print(pd.DataFrame(data))
    print("  -> Saved 'Task2_B_Convergence.png'")

# --- Task C: MG 平滑次数分析 ---
def task_c_mg_smoothing():
    print("\n[Task C] 分析 MG 平滑次数影响...")
    N = 129
    h = 0.1 / N
    model = LiverModel(h=h)
    sim = CNSimulation(model, dt_factor=2.0)
    print("  Running explicit steps to generate wave field...")
    for _ in range(50):
        sim.step(method='jacobi', max_iter=1) # 用简单方法先跑几步
    
    configs = [(1,1), (2,2), (3,3)]
    plt.figure(figsize=(8, 6))
    
    for pre, post in configs:
        solver = LinearSolver(sim.alpha_map, method='mg', pre=pre, post=post)
        solver.solve(np.zeros_like(sim.p_curr), sim.p_curr, tol=1e-8)
        plt.semilogy(solver.history, label=f'MG ({pre}, {post})')
        
    plt.title("Effect of Smoothing Steps on MG Convergence")
    plt.xlabel("V-Cycles")
    plt.ylabel("Relative Residual")
    plt.legend()
    plt.grid(True, alpha=0.3)

    rootpath = Path(__file__).resolve().parent.parent / 'figs'
    rootpath.mkdir(exist_ok=True)
    fig_dir = rootpath / "Task2_C_Smoothing.png"
    plt.savefig(fig_dir)

    print("  -> Saved 'Task2_C_Smoothing.png'")

# --- Task D: 复杂度分析 ---
def task_d_complexity():
    print("\n[Task D] 网格复杂度与加速比分析")
    
    # 1. 预热 (Warm-up) - 极其重要，消除首次编译抖动
    print("  Performing JIT warm-up...")
    warmup_sim = CNSimulation(LiverModel(h=0.1/33))
    warmup_sim.step(method='mg', max_iter=1)
    warmup_sim.step(method='jacobi', max_iter=1)
    warmup_sim.step(method='cg', max_iter=1)

    # 2. 定义测试配置
    # 剔除了 33，从 65 开始测，避免并行开销占主导
    # 因为本问题的特殊性 (CFL约束)，Jacobi 收敛很快，可以测更大的网格
    test_configs = {
        'jacobi': {'sizes': [65, 129, 257, 513, 1025], 'max_iter': 50000000}, 
        'cg':     {'sizes': [65, 129, 257, 513, 1025], 'max_iter': 20000000},
        'mg':     {'sizes': [65, 129, 257, 513, 1025], 'max_iter': 20000000}
    }
    
    results = {m: {'sizes': [], 'times': []} for m in test_configs}
    
    print(f"{'Method':<10} | {'Grid':<5} | {'Time (s)':<10} | {'Iters':<8} | {'Slope approx.'}")
    print("-" * 65)
    
    for m, config in test_configs.items():
        prev_time = None
        prev_N = None
        
        for N in config['sizes']:
            h = 0.1 / N
            model = LiverModel(h=h)
            model = LiverModel(N, N)
            sim = CNSimulation(model, dt_factor=2.0)
            sim.p_curr[int(N/2), int(N/2)] = 1.0 
            
            # 运行求解
            solve_params = {'tol': 1e-6, 'max_iter': config['max_iter']}
            _, iters, t_solve = sim.step(method=m, **solve_params)
            
            # 计算局部斜率 (Empirical Order of Complexity)
            slope_str = "-"
            if prev_time is not None:
                # Slope = log(T2/T1) / log(N2/N1)
                slope = np.log(t_solve/prev_time) / np.log(N/prev_N)
                slope_str = f"{slope:.2f}"
            
            if iters < config['max_iter']:
                results[m]['sizes'].append(N)
                results[m]['times'].append(t_solve)
                print(f"{m:<10} | {N:<5} | {t_solve:.4f}     | {iters:<8} | {slope_str}")
            else:
                print(f"{m:<10} | {N:<5} | {t_solve:.4f}     | {iters:<8} | Not Converged")
            
            prev_time = t_solve
            prev_N = N
            
    # --- 3. 绘图 ---
    plt.figure(figsize=(9, 7))
    markers = {'mg': 'o-', 'cg': 's-', 'jacobi': '^-'}
    
    for m in results:
        sizes = results[m]['sizes']
        times = results[m]['times']
        if not sizes: continue
        plt.loglog(sizes, times, markers[m], label=f"{m.upper()}")

    # 添加参考线
    if len(results['mg']['sizes']) > 0:
        base_N = results['mg']['sizes'][0]
        base_T = results['mg']['times'][0]
        ref_N = np.array([base_N, 1025])
        
        # O(N^2) - 理想情况 (线性工作量，因为未知数是 N^2)
        plt.loglog(ref_N, base_T * (ref_N/base_N)**2, 'k--', alpha=0.4, label=r'$O(N_{side}^2)$ (Linear)')
        
        # O(N^3) - 次优情况
        plt.loglog(ref_N, base_T * (ref_N/base_N)**3, 'k:', alpha=0.3, label=r'$O(N_{side}^3)$')

    plt.xlabel(r"Grid Size $N$ ($N \times N$ unknowns)")
    plt.ylabel("Time to Solution (s)")
    plt.title("Solver Complexity Analysis (Wave Eq + CFL)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    rootpath = Path(__file__).resolve().parent.parent / 'figs'
    rootpath.mkdir(exist_ok=True)
    fig_dir = rootpath / "Task2_D_Complexity.png"
    plt.savefig(fig_dir)
    
    print("  -> Saved 'Task2_D_Complexity.png'")

def task_visualization():
    print("\n[Visualization] 生成波场快照...")
    N = 257
    h = 0.1 / N
    model = LiverModel(h=h)
    sim = CNSimulation(model, dt_factor=2.0)
    
    # 运行一段时间并截图
    snapshots = []
    target_times = [1.0e-5, 2.0e-5, 3.0e-5]
    idx = 0
    
    for _ in range(1000):
        sim.step(method='mg', tol=1e-5, max_iter=20)
        if idx < len(target_times) and sim.time >= target_times[idx]:
            snapshots.append(sim.p_curr.copy())
            idx += 1
        if idx >= len(target_times): break
            
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, t in zip(axes, snapshots, target_times):
        im = ax.imshow(data.T, origin='lower', cmap='seismic', 
                       extent=[0, cfg.L_X, 0, cfg.L_Y])
        ax.set_title(f"t = {t*1e6:.1f} us")
        # 画同心圆辅助线
        c1 = plt.Circle((cfg.CENTER_X, cfg.CENTER_Y), cfg.R_INNER, fill=False, color='k', ls='--')
        c2 = plt.Circle((cfg.CENTER_X, cfg.CENTER_Y), cfg.R_OUTER, fill=False, color='k', ls=':')
        ax.add_patch(c1); ax.add_patch(c2)
        
    plt.suptitle("Wave Propagation Snapshots (Implicit MG Solver)")

    rootpath = Path(__file__).resolve().parent.parent / 'figs'
    rootpath.mkdir(exist_ok=True)
    fig_dir = rootpath / "Visualization.png"
    plt.savefig(fig_dir)

    print("  -> Saved 'Visualization.png'")

def main():
    task_a_stability()
    task_b_solvers_comparison()
    task_c_mg_smoothing()
    task_d_complexity()
    task_visualization()
    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()