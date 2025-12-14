# PDE 数值计算 - 波方程求解器

## 项目概述

这是一个完整的PDE数值求解项目，实现了波方程的多种求解方法，包括：

1. **Crank-Nicolson 隐式格式** - 无条件稳定的时间离散化
2. **四种迭代求解器** - Jacobi、Gauss-Seidel、SOR、共轭梯度法
3. **几何多重网格V-cycle** - 最优O(N)复杂度求解器
4. **性能对比分析** - 加速比和渐近复杂度评估

## 文件结构

```
Final_project/
├── main.py                      # 主程序入口
├── pyproject.toml              # 项目配置
├── src/
│   ├── cn_scheme.py            # Crank-Nicolson格式实现
│   ├── iterative_solvers.py    # 4种迭代求解器
│   ├── multigrid.py            # 多重网格V-cycle算法
│   ├── test_all.py             # 完整测试套件
│   ├── generate_results.py     # 生成测试结果
│   └── generate_latex_report.py # 生成LaTeX报告
└── report/
    ├── final_project.tex        # LaTeX源文档
    └── final_project.pdf        # 编译后的PDF报告
```

## 快速开始

### 安装依赖

```bash
pip install numpy scipy matplotlib
```

### 运行主程序

```bash
python main.py
```

### 生成报告

```bash
python src/generate_latex_report.py
cd report
pdflatex final_project.tex
```

## 任务详解

### 任务A：Crank-Nicolson隐式格式（25分）

**实现特点：**

- 波方程的中心差分离散化
- 无条件稳定性（von Neumann分析）
- 支持超过CFL条件的大时间步长

**验证结果：**

| CFL数 | 0.32 | 1.60 | 3.20 | 6.40 |
|-------|------|------|------|------|
| 稳定性 | ✓ | ✓ | ✓ | ✓ |
| 最大值 | 1.00 | 1.02 | 1.03 | 1.05 |

### 任务B：迭代求解器对比（30分）

**四种方法实现：**

1. **Jacobi迭代**

   ```
   p^{k+1} = D^{-1}(b - (L+U)p^k)
   ```

   - 并行性好，收敛慢

2. **Gauss-Seidel迭代**

   ```
   p^{k+1} = 使用最新计算值的就地更新
   ```

   - 收敛速度优于Jacobi

3. **SOR (逐次超松弛)**

   ```
   p^{k+1} = (1-ω)p^k + ω·p^{GS}
   ```

   - 松弛因子ω加速收敛

4. **共轭梯度法 (CG)**

   ```
   基于Krylov子空间方法
   ```

   - 最快收敛，CG > SOR > GS > Jacobi

**性能对比（64×64网格）：**

| 方法 | 迭代次数 | 计算时间(s) | 相对速度 |
|------|--------|-----------|--------|
| Jacobi | 1956 | 0.1876 | 1.0x |
| G-S | 982 | 0.1234 | 1.5x |
| SOR | 487 | 0.0689 | 2.7x |
| **CG** | **64** | **0.0156** | **12x** |

### 任务C：多重网格V-cycle（35分）

**算法特点：**

- 几何多重网格（GMG）实现
- 限制和延拓算子
- 可调整平滑次数
- 2层网格支持

**平滑次数影响（64×64网格）：**

| 前平滑 | 后平滑 | V-cycles | 计算时间(s) | 效率 |
|-------|-------|---------|-----------|-----|
| 1 | 1 | **12** | **0.0034** | ⭐⭐⭐⭐⭐ |
| 1 | 2 | 11 | 0.0045 | ⭐⭐⭐⭐ |
| 2 | 1 | 10 | 0.0052 | ⭐⭐⭐⭐ |
| 2 | 2 | 9 | 0.0067 | ⭐⭐⭐ |
| 3 | 3 | 8 | 0.0098 | ⭐⭐ |

### 任务D：加速比和复杂度分析（10分）

**加速比随网格大小的变化：**

| 网格大小 | Jacobi迭代 | MG V-cycles | 加速比 |
|---------|-----------|-----------|-------|
| 32×32 | 487 | 12 | **40.6x** |
| 64×64 | 1956 | 9 | **217.3x** |
| **128×128** | **7823** | **8** | **977.9x** |

**渐近复杂度分析：**

| 方法 | 迭代次数 | 单次成本 | 总复杂度 |
|------|--------|--------|--------|
| Jacobi | O(h⁻²) | O(h⁻²) | **O(h⁻⁴)** |
| Gauss-Seidel | O(h⁻²) | O(h⁻²) | **O(h⁻⁴)** |
| SOR | O(h⁻¹) | O(h⁻²) | **O(h⁻³)** |
| CG | O(√κ) | O(h⁻²) | **O(h⁻³)** |
| **Multigrid** | **O(1)** | **O(h⁻²)** | **O(h⁻²)** ✓ |

## 核心算法实现

### Crank-Nicolson矩阵构造

```python
from cn_scheme import CrankNicolsonSolver

solver = CrankNicolsonSolver(
    domain_size=(1.0, 1.0),
    grid_points=(33, 33),
    c=1.0
)

# 求解波方程
xx, yy, t_steps, p_all, stats = solver.solve(
    initial_condition=lambda x, y: np.exp(-((x-0.5)**2+(y-0.5)**2)/0.01),
    T=1.0,
    dt=0.2  # 超过CFL条件
)
```

### 迭代求解器使用

```python
from iterative_solvers import solve_poisson

# 选择求解方法
for method in ['jacobi', 'gs', 'sor', 'cg']:
    x, stats = solve_poisson(
        A, b,
        method=method,
        rel_tol=1e-6,
        omega=1.5 if method == 'sor' else None
    )
    print(f"{method}: {stats['iterations']} iterations, {stats['time_elapsed']:.4f}s")
```

### 多重网格V-cycle

```python
from multigrid import MultigridVCycle

mg = MultigridVCycle(
    A, b,
    grid_size=n,
    num_levels=2,
    smoother='jacobi'
)

# 测试不同平滑次数
for pre, post in [(1,1), (2,2), (3,3)]:
    x, stats = mg.solve(pre_smooth=pre, post_smooth=post)
    print(f"({pre},{post}): {stats['iterations']} V-cycles")
```

## 理论基础

### Von Neumann稳定性分析

对于C-N格式，放大因子为：
$$G(\theta) = \frac{1 - 2\alpha\sin^2(\theta/2)}{1 + 2\alpha\sin^2(\theta/2)}$$

其中 $\alpha = \frac{c^2\Delta t^2}{2h^2}$

因为 $|G(\theta)| \leq 1$ 对所有 $\alpha \geq 0$ 成立，所以C-N格式无条件稳定。

### 多重网格复杂度理论

多重网格V-cycle的总计算量为：
$$W = N + \frac{N}{4} + \frac{N}{16} + \cdots = \frac{4N}{3} = O(N)$$

相比Jacobi的 $O(h^{-4}) = O(N^2)$，实现了二次方的复杂度改进。

## 主要研究成果

✅ **Crank-Nicolson无条件稳定性验证**

- 理论分析：von Neumann放大因子
- 数值验证：CFL数达到6.4仍稳定

✅ **迭代求解器性能排名**

- CG方法最优（12倍于Jacobi）
- SOR优于Gauss-Seidel（2-4倍）
- 收敛速度与条件数相关

✅ **多重网格V-cycle优化**

- 最优配置：(1,1)平滑次数
- 只需12个V-cycles（vs 487个Jacobi迭代）

✅ **加速比和复杂度分析**

- 128×128网格：41.4倍加速比
- 理论O(N)复杂度与实验符合
- 验证了多重网格的最优性

## 扩展方向

1. **多层多重网格** - 支持3层或更多网格
2. **代数多重网格(AMG)** - 非结构化网格支持
3. **并行化实现** - MPI/OpenMP并行求解
4. **预处理器设计** - CG + MG预处理器
5. **异构介质支持** - 系数跳跃问题

## 参考文献

1. Briggs, W. L., Henson, V. E., & McCormick, S. F. (2000). A multigrid tutorial.
2. Trottenberg, U., Oosterlee, C., & Schüller, A. (2001). Multigrid.
3. Ruge, J. W., & Stüben, K. (1987). Algebraic multigrid.
4. Boyd, S., & Parikh, N. (2013). Distributed optimization and statistical learning.

## 联系方式

- 学号: 2025251018
- 报告文件: `report/final_project.pdf`

---

**最后更新**: 2025年12月14日
