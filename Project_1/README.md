# Project 1 — 2D Heat Conduction (Crank–Nicolson + Approximate Factorization TDMA)

本仓库是用于数值求解二维热传导方程的作业实现（Crank–Nicolson 时程离散 + 空间中心差分 + 近似分解法（AF）配合 TDMA 求解行/列方向的三对角体系）。代码以 Python 实现，包含结果绘图和收敛性分析。

## 关键点（摘要）

- 方程：二维热传导（热扩散系数 α = 1），在单位方域 [0,1]×[0,1]，Dirichlet 边界为 0。
- 初始条件：T(x,y,0) = sin(πx) sin(πy)，解析解可写为
 T(x,y,t) = exp(-2π² t) sin(πx) sin(πy)。
- 数值方法：Crank–Nicolson 时间离散（隐式），空间采用中心差分，使用近似因子分解（AF）把二维隐式系统分解为按行、按列的三对角子系统，利用 Thomas（TDMA）算法解三对角系统。

## 要求（Environment）

- Python >= 3.13
- 依赖：numpy (>=2.3.4), matplotlib

（仓库中已包含 `pyproject.toml` 声明了这些依赖；为便捷使用我同时提供了 `requirements.txt`。）

## 安装

在项目根目录下（含 `src/`）执行：

```bash
python -m pip install -r requirements.txt
```

或者使用 `pyproject.toml` 的打包 / 虚拟环境方式。

## 快速运行

在仓库根目录运行主脚本以生成示例结果和绘图：

```bash
python src/main.py
```

脚本会把生成的图片保存到 `plots/` 目录（如果不存在会自动创建）。主要包括：数值解与解析解的三维图、误差色图、以及收敛性（log-log）图。

## 目录结构

- `pyproject.toml` — 项目信息与依赖声明
- `README.md` — 本说明
- `task.md` — 作业要求与任务清单（包含数学模型与需要完成的各项实验）
- `src/` — 源码（主程序 `main.py`，以及数值解、绘图函数等）
- `plots/` — 运行后生成的图像（输出目录）
- `report/` — 报告源文件（LaTeX），以及最终报告相关资源

## 主要实现说明

- `src/main.py` 中实现了：
  - `tdma_solver`：Thomas 算法求解三对角线性系统（向前消元 + 回代）
  - `solve_heat_equation`：实现 Crank–Nicolson 时间推进，按行/按列两次 TDMA 求解（近似因子分解）
  - 绘图函数：`plot_3d_surface`、`plot_2d_colormap`、`plot_loglog_convergence` 等，生成用于分析和报告的图形
  - 主运行示例：包含空间收敛性（N = 10,20,40,80）与时间收敛性（多组 dt），并测量较大网格（N=80）下的计算时间。

## 如何复现实验（建议流程）

1. 安装依赖（见上文）。
2. 直接运行 `python src/main.py`：默认会执行示例中的一系列模拟并把图片保存到 `plots/`。
3. 修改参数：如果想改变网格（N）、时间步长（dt）或输出时间（t_plot），可编辑 `src/main.py` 中 `if __name__ == "__main__":` 下的参数并重运行。

## 输出与报告

- 绘图输出保存在 `plots/`，文件名以功能命名（例如 `error_map_spatial_N40.png`、`convergence_spatial.png` 等）。
- 报告源文件位于 `report/`，如果需要生成 PDF，确保系统有 LaTeX（pdflatex/xelatex）并在 `report/` 中运行相应的编译命令（例如 `pdflatex report.tex` 或使用你的 LaTeX 编辑器）。

## 已知问题与注意事项

- 代码假定 N 表示内部节点数（网格实际点数为 N+2 包含边界）。
- `pyproject.toml` 中声明 Python >= 3.13；请确保本地 Python 版本满足该要求。
- 当 dt 与网格尺寸组合不合理时（过大 dt 或极细网格），运行时间或数值稳定性/精度可能成为问题，请根据任务要求选择合适的 dt 和 N。
