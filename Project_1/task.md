# Homework #1: Numerical Solution of 2D Heat Conduction

Consider the 2D heat conduction equation:
$$
\frac{\partial T}{\partial t} = \alpha \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} \right)
$$
where $T(x, y, t)$ is the temperature, $\alpha = 1$ is the thermal diffusivity, and the domain is $0 \le x, y \le 1$ with Dirichlet boundary conditions $T(0, y, t) = T(1, y, t) = T(x, 0, t) = T(x, 1, t) = 0$ for $t > 0$. The initial condition is:
$$
T(x, y, 0) = \sin(\pi x) \sin(\pi y)
$$
The exact solution is:
$$
T(x, y, t) = e^{-2\pi^2 t} \sin(\pi x) \sin(\pi y)
$$

## Tasks

* Discretize the spatial domain using a uniform grid with $N_x = N_y = N$ interior points.
* Use the Crank-Nicolson implicit scheme for time discretization.
* For the spatial derivatives, employ central finite differences.
* Solve the resulting system of linear equations at each time step using the Approximate Factorization method using TDMA (**you must implement yourself**).
* Simulate up to $t = 0.1$ and plot the temperature field at $t = 0.05$ for visualization.
* Compute the numerical solution for varying spatial grid sizes: $N = 10, 20, 40, 80$. For each spatial resolution, use a fixed $\Delta t = 0.001$ and plot the absolute error map at $t = 0.05$.
* Repeat with varying time steps: $\Delta t = 0.01, 0.005, 0.0025, 0.00125$ while fixing $N = 40$, and plot the error map.
* Estimate the observed order of convergence for spatial and temporal errors, and compare with expected second-order accuracy.
* Include plots of RMSE vs. $\Delta t$ on log-log scales to visualize the convergence rates.

## Requirements

* Provide your code (in Python or MATLAB) for the full scheme, including TDMA.
* Report computational time for the largest grid and discuss efficiency.
* Submit your report with code, plots, error tables, and analysis.

