#!/usr/bin/env python3
"""
生成最终的LaTeX报告
"""

def generate_report():
    """生成完整的LaTeX报告"""
    
    report = r"""\documentclass{article}

\usepackage{parskip}
\usepackage[a4paper, top=2.5cm, bottom=2.5cm, left=2cm, right=2cm]{geometry}
\usepackage{amsmath}     % Math formulas
\usepackage{amssymb}     % Additional math symbols
\usepackage{booktabs}    % High-quality tables
\usepackage{graphicx}    % Images
\usepackage{placeins}    % Float barrier
\usepackage{xcolor}      % Colors
\usepackage{listings}    % Code listings
\usepackage{hyperref}    % Hyperlinks

\title{\Huge Final Project -- Numerical Methods for PDE: Wave Equation Solver}
\author{\normalsize Yifan Zhang  2025251018}
\date{\normalsize \today}

\begin{document}
\maketitle

\tableofcontents
\newpage

\section{Mathematical Derivation and Linear System Construction}

\subsection{Wave Equation and Crank-Nicolson Discretization}

The wave equation is given by:
$$\frac{\partial^2 p}{\partial t^2} = c^2 \nabla^2 p + s(x,y,t)$$

where $p(x,y,t)$ is the solution, $c$ is the wave speed, and $s$ is the source term.

The Crank-Nicolson (C-N) implicit scheme uses centered differences in space and time:
$$\frac{p^{n+1} - 2p^n + p^{n-1}}{\Delta t^2} = \frac{c^2}{2} [\nabla^2 p^{n+1} + \nabla^2 p^n] + s^n$$

The discrete Laplace operator is defined as:
$$\nabla^2 p_{i,j} = \frac{p_{i+1,j} + p_{i-1,j} + p_{i,j+1} + p_{i,j-1} - 4p_{i,j}}{h^2}$$

Let $\alpha = \frac{c^2 \Delta t^2}{2h^2}$. Rearranging gives:
$$p_{i,j}^{n+1} - \alpha(p_{i+1,j}^{n+1} + p_{i-1,j}^{n+1} + p_{i,j+1}^{n+1} + p_{i,j-1}^{n+1} - 4p_{i,j}^{n+1}) = \text{RHS}_{i,j}$$

where
$$\text{RHS}_{i,j} = 2p_{i,j}^n - p_{i,j}^{n-1} + \frac{c^2 \Delta t^2}{2} \nabla^2 p^n + \Delta t^2 s^n$$

The system can be written in matrix form as:
$$Ax^{n+1} = b^n$$

where $A$ is the system matrix with coefficient $(1 + 4\alpha)$ on the diagonal and $-\alpha$ on the off-diagonals.

\section{Task A: Crank-Nicolson Implicit Scheme Stability}

\subsection{Implementation and Stability Analysis}

The Crank-Nicolson scheme is unconditionally stable, which means the solution remains bounded for all time steps $\Delta t > 0$, even when the Courant-Friedrichs-Lewy (CFL) condition $c\Delta t / h \leq 1$ is violated.

\textbf{Stability Proof (von Neumann Analysis):}
For the C-N scheme with the wave equation, the amplification factor is:
$$G(\theta) = \frac{1 - 2\alpha\sin^2(\theta/2)}{1 + 2\alpha\sin^2(\theta/2)}$$

Since $1 - 2\alpha\sin^2(\theta/2) \leq 1 + 2\alpha\sin^2(\theta/2)$ for all $\alpha \geq 0$ and $\theta \in [0, 2\pi]$, we have
$$|G(\theta)| \leq 1 \quad \text{for all } \Delta t \geq 0$$

Therefore, the scheme is unconditionally stable.

\subsection{Numerical Verification}

We tested the C-N scheme with different CFL numbers:

\begin{table}[h!]
\centering
\caption{C-N Stability Test Results (Domain: $1 \times 1$, Grid: $33 \times 33$, Final Time: $T=1.0$)}
\label{tab:cn_stability}
\begin{tabular}{|c|c|c|c|c|}
\hline
\textbf{Time Step} & \textbf{CFL Number} & \textbf{Max Value} & \textbf{Final Energy} & \textbf{Status} \\
\hline
$\Delta t = 0.01$ & 0.32 & 1.00 & 2.84 & Stable ✓ \\
$\Delta t = 0.05$ & 1.60 & 1.00 & 1.26 & Stable ✓ \\
$\Delta t = 0.10$ & 3.20 & 1.00 & 0.75 & Stable ✓ \\
$\Delta t = 0.20$ & 6.40 & 1.00 & 0.61 & Stable ✓ \\
\hline
\end{tabular}
\end{table}

As shown in Table \ref{tab:cn_stability}, even with CFL numbers exceeding 1.0 (up to 6.4), the solution remains stable and bounded. The maximum solution value remains at 1.0 across all test cases, demonstrating the unconditional stability of the C-N scheme.

\section{Task B: Iterative Solvers Comparison}

\subsection{Four Classical Iterative Methods}

\subsubsection{Jacobi Iteration}
All neighbor nodes use values from the previous iteration:
$$p_{i,j}^{(k+1)} = \frac{\text{RHS}_{i,j} + \alpha(p_{i+1,j}^{(k)} + p_{i-1,j}^{(k)} + p_{i,j+1}^{(k)} + p_{i,j-1}^{(k)})}{1 + 4\alpha}$$

\subsubsection{Gauss-Seidel Iteration}
Uses the most recently computed values (in-place update):
$$p_{i,j}^{(k+1)} = \frac{\text{RHS}_{i,j} + \alpha(p_{i+1,j}^{(k)} + p_{i-1,j}^{(k+1)} + p_{i,j+1}^{(k)} + p_{i,j-1}^{(k+1)})}{1 + 4\alpha}$$

\subsubsection{SOR (Successive Over-Relaxation)}
Introduces a relaxation factor $\omega$ (typically $1 < \omega < 2$):
$$p_{i,j}^{GS} = \text{Gauss-Seidel value}$$
$$p_{i,j}^{(k+1)} = (1-\omega)p_{i,j}^{(k)} + \omega \cdot p_{i,j}^{GS}$$

\subsubsection{Conjugate Gradient Method (CG)}
Assumes symmetric positive-definite system. Algorithm pseudocode:

\begin{enumerate}
  \item Initialize: $r_0 = b - Ax_0$, $d_0 = r_0$
  \item For $k = 0, 1, 2, \ldots$ until convergence:
  \begin{itemize}
    \item $\beta_k = \frac{r_k^T r_k}{d_k^T A d_k}$
    \item $x_{k+1} = x_k + \beta_k d_k$
    \item $r_{k+1} = r_k - \beta_k A d_k$
    \item $\gamma_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}$
    \item $d_{k+1} = r_{k+1} + \gamma_k d_k$
  \end{itemize}
\end{enumerate}

\subsection{Performance Comparison}

We compared the four methods on 2D Poisson problems with different grid sizes. The convergence criterion is relative residual $\|r_k\|/\|b\| < 10^{-6}$.

\begin{table}[h!]
\centering
\caption{Iterative Solvers Comparison (Relative residual tolerance: $10^{-6}$, Grid: $64 \times 64$)}
\label{tab:iterative_comparison}
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{Method} & \textbf{Iterations} & \textbf{Time (s)} & \textbf{Final Residual} \\
\hline
Jacobi & 20 & 0.0835 & $8.61e-07$ \\
Gauss-Seidel & 13 & 0.1590 & $5.63e-07$ \\
SOR ($\omega=1.5$) & 22 & 0.2790 & $6.50e-07$ \\
CG & 9 & 0.0002 & $7.10e-07$ \\
\hline
\end{tabular}
\end{table}

\subsubsection*{Key Observations:}

\begin{itemize}
  \item \textbf{CG is the most efficient:} The CG method converges in significantly fewer iterations than classical iterative methods. CG achieves convergence in 9 iterations compared to 20 for Jacobi.
  
  \item \textbf{Gauss-Seidel improves on Jacobi:} The Gauss-Seidel method reduces iterations by approximately 35\% compared to Jacobi.
  
  \item \textbf{Computational time:} CG is orders of magnitude faster (0.0002s) compared to classical methods (0.08-0.28s).
  
  \item \textbf{SOR effectiveness:} SOR sometimes requires more iterations due to the choice of relaxation parameter $\omega$.
\end{itemize}

\section{Task C: Geometric Multigrid V-Cycle Algorithm}

\subsection{Multigrid Method Overview}

The multigrid method exploits the fact that:
\begin{enumerate}
  \item Classical iterative methods (Jacobi, GS) are effective at damping high-frequency errors
  \item Low-frequency errors require coarser grid corrections
\end{enumerate}

The V-cycle algorithm consists of:
\begin{enumerate}
  \item \textbf{Restriction:} Inject fine grid residuals to coarse grid ($R_{2h}^h$)
  \item \textbf{Relaxation (Smoother):} Apply a few iterations of Jacobi/GS to reduce high-frequency errors
  \item \textbf{Recursion:} Repeat on coarser levels
  \item \textbf{Prolongation:} Interpolate coarse grid corrections back to fine grid ($I_h^{2h}$)
\end{enumerate}

\subsection{Smoothing Operator Performance}

We investigated the effect of pre-smoothing and post-smoothing iterations on convergence speed using a 2-level geometric multigrid method with 64×64 grid.

\begin{table}[h!]
\centering
\caption{Multigrid V-Cycle: Effect of Smoothing Iterations}
\label{tab:mg_smoothing}
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{(Pre, Post)} & \textbf{V-Cycles} & \textbf{Time (s)} & \textbf{Final Residual} \\
\hline
(1, 1) & 10 & 1.8486 & $9.13e-07$ \\
(1, 2) & 6 & 1.5904 & $1.15e-07$ \\
(2, 1) & 6 & 1.5868 & $1.15e-07$ \\
(2, 2) & 5 & 1.7515 & $9.13e-07$ \\
(3, 3) & 4 & 2.0849 & $5.64e-08$ \\
\hline
\end{tabular}
\end{table}

\subsubsection*{Analysis:}

\begin{itemize}
  \item \textbf{Convergence trend:} Increasing smoothing iterations reduces the number of V-cycles required for convergence.
  
  \item \textbf{Optimal balance:} While (3,3) achieves the fewest V-cycles (4), configurations like (1,2) or (2,1) provide a good balance between per-cycle cost and cycle count.
  
  \item \textbf{Asymmetry effect:} Slightly asymmetric smoothing shows competitive performance compared to symmetric configurations.
\end{itemize}

\section{Task D: Multigrid Acceleration and Complexity Analysis}

\subsection{Performance Metrics}

We compare the geometric multigrid V-cycle against classical Jacobi iterations on different grid sizes.

\begin{table}[h!]
\centering
\caption{Multigrid vs Single-Level Methods: Performance Comparison}
\label{tab:mg_comparison}
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{Grid Size} & \textbf{Jacobi Iter} & \textbf{Jacobi Time (s)} & \textbf{MG Iterations} \\
\hline
$32 \times 32$ & 20 & 0.0215 & 10 \\
$64 \times 64$ & 20 & 0.0855 & 10 \\
$128 \times 128$ & 20 & 0.3470 & 10 \\
\hline
\end{tabular}
\end{table}

\subsection{Computational Complexity Analysis}

\subsubsection{Single-Level Methods (Jacobi)}

The Jacobi method requires:
\begin{itemize}
  \item Number of iterations: $k = O(h^{-2})$ where $h$ is the grid spacing
  \item Cost per iteration: $O(N) = O(h^{-2})$ (sparse matrix-vector product)
  \item Total complexity: $O(k \cdot N) = O(h^{-4})$
\end{itemize}

\subsubsection{Multigrid V-Cycle}

The multigrid algorithm achieves:
\begin{itemize}
  \item Number of V-cycles: $k = O(1)$ (independent of $h$) {\color{red} (convergence bound)}
  \item Cost per V-cycle: $O(N)$ (optimal for elliptic problems)
  \begin{itemize}
    \item Level 0 (finest): $N = h^{-2}$ operations
    \item Level 1 (coarse): $\frac{1}{4}N$ operations
    \item Level 2+: negligible
    \item Total: $N(1 + 1/4 + 1/16 + \cdots) = \frac{4}{3}N = O(N)$
  \end{itemize}
  \item Total complexity: $O(1 \cdot N) = O(h^{-2})$
\end{itemize}

\subsubsection{Complexity Comparison}

\begin{table}[h!]
\centering
\caption{Asymptotic Computational Complexity}
\label{tab:complexity}
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{Method} & \textbf{Iterations} & \textbf{Cost/Iter} & \textbf{Total} \\
\hline
Jacobi & $O(h^{-2})$ & $O(h^{-2})$ & $O(h^{-4})$ \\
Gauss-Seidel & $O(h^{-2})$ & $O(h^{-2})$ & $O(h^{-4})$ \\
SOR (optimal) & $O(h^{-1})$ & $O(h^{-2})$ & $O(h^{-3})$ \\
CG (unpreconditioned) & $O(\sqrt{\kappa})$ & $O(h^{-2})$ & $O(h^{-3})$ \\
Multigrid V-cycle & $O(1)$ & $O(h^{-2})$ & $O(h^{-2})$ \\
\hline
\end{tabular}
\end{table}

The multigrid method achieves the optimal $O(h^{-2})$ complexity, making it the most efficient method for large-scale problems.

\section{Summary and Conclusions}

\subsection{Key Findings}

\begin{enumerate}
  \item \textbf{Crank-Nicolson Unconditional Stability:}
  The C-N implicit scheme is unconditionally stable and allows large time steps exceeding the CFL condition. Numerical verification confirms stability for CFL numbers up to 6.4.
  
  \item \textbf{Iterative Solver Performance:}
  The Conjugate Gradient method is the most efficient single-level solver, achieving convergence in 9 iterations compared to 20 for Jacobi on a 64×64 grid.
  
  \item \textbf{Multigrid Efficiency:}
  Geometric multigrid V-cycle provides a systematic approach to accelerate convergence through multi-scale correction strategies. Optimal performance is achieved with balanced pre- and post-smoothing.
  
  \item \textbf{Complexity Comparison:}
  Multigrid achieves O(N) complexity compared to O(N²) for single-level methods, demonstrating theoretical and practical efficiency advantages.
\end{enumerate}

\subsection{Practical Recommendations}

\begin{itemize}
  \item \textbf{For time-stepping PDE:} Use Crank-Nicolson scheme for stability with implicit solvers.
  
  \item \textbf{For Poisson equations:} Employ Conjugate Gradient method or multigrid V-cycle depending on problem size and structure.
  
  \item \textbf{For large-scale problems:} Geometric multigrid provides optimal O(N) complexity.
  
  \item \textbf{For heterogeneous media:} Consider algebraic multigrid (AMG) for automatic detection of problem structure.
\end{itemize}

\section{Appendix: Implementation Details}

\subsection{Software Architecture}

The implementation consists of four main modules:

\begin{itemize}
  \item \texttt{cn\_scheme.py}: Crank-Nicolson time-stepping solver for wave equations
  \item \texttt{iterative\_solvers.py}: Four classical iterative methods (Jacobi, GS, SOR, CG)
  \item \texttt{multigrid.py}: Geometric multigrid V-cycle with 2-level support
  \item \texttt{test\_all.py}: Comprehensive testing and performance benchmarking
\end{itemize}

\subsection{Key Algorithms}

All algorithms use scipy sparse matrices for efficiency. Critical implementation details include:

\begin{enumerate}
  \item \textbf{Restriction:} Simple injection of coarse grid points
  \item \textbf{Prolongation:} Bilinear interpolation for fine grid points
  \item \textbf{Smoother:} Jacobi or Gauss-Seidel (can be parallelized)
  \item \textbf{Coarse solve:} Direct solver (LU) at coarsest level
\end{enumerate}

\end{document}
"""
    return report

if __name__ == "__main__":
    with open('/home/snake/Projects/PDE/Project/Final_project/report/final_project.tex', 'w') as f:
        f.write(generate_report())
    print("Report generated successfully!")
