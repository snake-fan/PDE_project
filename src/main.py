import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
import os

plt.rcParams.update({'font.size': 12})

def tdma_solver(a, b, c, d):
    """
    Solve the tridiagonal system Ax = d using the Thomas algorithm.
    
    Parameters:
    a : 1D numpy array
        Sub-diagonal (length n-1)
    b : 1D numpy array
        Main diagonal (length n)
    c : 1D numpy array
        Super-diagonal (length n-1)
    d : 1D numpy array
        Right-hand side (length n)
    
    Returns:
    x : 1D numpy array
        Solution vector (length n)
    """
    n = len(b)
    cp = np.zeros(n-1)
    dp = np.zeros(n)

    # Forward sweep
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n-1):
        denom = b[i] - a[i-1] * cp[i-1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i-1] * dp[i-1]) / denom
    dp[n-1] = (d[n-1] - a[n-2] * dp[n-2]) / (b[n-1] - a[n-2] * cp[n-2])

    # Back substitution
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]

    return x

def exact_heat_equation(xx, yy, t):
    return np.exp(-2 * np.pi**2 * t) * np.sin(np.pi * xx) * np.sin(np.pi * yy)

def solve_heat_equation(N, final_t, dt, t_plot):
    """
    Solve the 2D heat equation using the Crank-Nicolson method.
    
    Parameters:
    N : int
        Number of spatial grid number in each dimension (total grid points = (N+1)x(N+1)).
    final_t : float
        The final time to which the solution is computed.
    dt : float
        The spatial step in the time.
    t_plot : float
        The time at which to store the temperature distribution for plotting.

    Returns:
    T : 2D numpy array
        The temperature distribution at final_t.
    T_at_t_plot: 2D numpy array
        The temperature distribution at t for plotting.
    xx, yy : 2D numpy arrays
        The meshgrid arrays for x and y coordinates.
    """
    # Adopt the implementation from the demo solver: treat N as the number of INTERNAL nodes
    # and use a (N+2)x(N+2) array including Dirichlet boundaries at 0 and 1.
    # This version has been verified in `src/demo.py` and ensures consistent indexing.
    h = 1.0 / (N + 1)

    x = np.linspace(0, 1, N + 2)
    y = np.linspace(0, 1, N + 2)
    xx, yy = np.meshgrid(x, y)
    xx_int, yy_int = np.meshgrid(x[1:-1], y[1:-1])

    # Temperature array including boundaries
    T = np.zeros((N + 2, N + 2))
    # initial condition on internal nodes
    T[1:-1, 1:-1] = np.sin(np.pi * xx_int) * np.sin(np.pi * yy_int)

    r = dt / (2.0 * h**2)
    n_steps = int(round(final_t / dt))

    T_at_t_plot = None
    plot_time_step = int(round(t_plot / dt))

    # TDMA diagonals for internal system size N
    a = np.full(N - 1, -r)
    b = np.full(N, 1 + 2 * r)
    c = np.full(N - 1, -r)

    for step in range(n_steps):
        current_t = step * dt
        if T_at_t_plot is None and abs(current_t - t_plot) < (dt / 2.0):
            T_at_t_plot = T.copy()

        T_old = T.copy()
        T_int = T_old[1:-1, 1:-1]

        Tx = T_old[1:-1, 0:-2] - 2 * T_int + T_old[1:-1, 2:]
        Ty = T_old[0:-2, 1:-1] - 2 * T_int + T_old[2:, 1:-1]

        RHS = T_int + r * (Tx + Ty)

        # X-sweep: solve along rows (for each y)
        T_star = np.zeros((N, N))
        for j in range(N):
            d = RHS[j, :]
            sol = tdma_solver(a, b, c, d)
            if sol is None:
                print(f"TDMA X-sweep failed at step {step}, row {j}")
                return T, None, xx, yy
            T_star[j, :] = sol

        # Y-sweep: solve along columns
        T_new_int = np.zeros((N, N))
        for i in range(N):
            d = T_star[:, i]
            sol = tdma_solver(a, b, c, d)
            if sol is None:
                print(f"TDMA Y-sweep failed at step {step}, col {i}")
                return T, None, xx, yy
            T_new_int[:, i] = sol

        T[1:-1, 1:-1] = T_new_int

    # capture final if needed
    if T_at_t_plot is None and abs(final_t - t_plot) < (dt / 2.0):
        T_at_t_plot = T.copy()

    if T_at_t_plot is None and plot_time_step < n_steps:
        # if never set during stepping but plot_time_step is inside the integration range
        # capture at last step
        T_at_t_plot = T.copy()

    return T, T_at_t_plot, xx, yy

def plot_3d_surface(xx, yy, T, title, filename=None):
    """draw 3D surface plot"""
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, T, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('T')
    ax.set_title(title)
    if filename:
        plt.savefig(filename)
        print(f"Figure successfully saved as {filename}")
    else:
        plt.show()


def plot_2d_colormap(xx, yy, Z, title, filename=None):
    """draw 2D colormap plot"""
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(xx, yy, Z, shading='auto', cmap='Reds', vmin=0)
    plt.colorbar(label='Absolute Error')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.axis('scaled')
    if filename:
        plt.savefig(filename)
        print(f"Figure successfully saved as {filename}")
    else:
        plt.show()

def plot_line(x, y, title, xlabel, ylabel, filename=None):
    """draw 2D line plot"""
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'o-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename)
        print(f"Figure successfully saved as {filename}")
    else:
        plt.show()

def compute_error(T_numeric, t):
    N = T_numeric.shape[0] - 1
    x = np.linspace(0, 1, N + 1)
    y = np.linspace(0, 1, N + 1)
    xx, yy = np.meshgrid(x, y)
    T_exact = exact_heat_equation(xx, yy, t)
    error = np.abs(T_numeric - T_exact)
    return error


def plot_loglog_convergence(h_values, errors, h_label, title, filename=None):
    """draw log-log convergence plot"""
    h_values = np.array(h_values)
    errors = np.array(errors)
    
    slope = "N/A"
    try:
        if len(h_values) > 1 and len(errors) > 1:
            # log(error) = p * log(h) + log(C)
            valid_indices = np.where((errors > 0) & (h_values > 0))
            if len(valid_indices[0]) > 1:
                log_h = np.log10(h_values[valid_indices])
                log_err = np.log10(errors[valid_indices])
                slope_val, _ = np.polyfit(log_h, log_err, 1)
                slope = f"{slope_val:.2f}"
    except np.linalg.LinAlgError:
        pass # 保持 slope = "N/A"

    plt.figure(figsize=(7, 5))
    plt.loglog(h_values, errors, 'o-', label=f'RMSE (Slope={slope})')
    
    # draw reference lines for order 1 and order 2
    if len(h_values) > 0 and errors[0] > 0:
        order2_line = errors[0] * (h_values / h_values[0])**2
        plt.loglog(h_values, order2_line, 'k--', label='Order 2 Ref.')
        
    plt.xlabel(f'log({h_label})')
    plt.ylabel('log(RMSE)')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    if filename:
        plt.savefig(filename)
        print(f"Figure successfully saved as {filename}")
    else:
        plt.show()


if __name__ == "__main__":
    output_dir = "plots"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def getfilename(name):
        return os.path.join(output_dir, name)

    # Task 5: simulate up to t=0.1 and plot the temperature field at t=0.05 for visualization
    N = 100
    final_t = 0.1
    dt = 0.0005
    t_plot = 0.05
    _, T_at_t_plot, xx, yy = solve_heat_equation(N, final_t, dt, t_plot)
    if T_at_t_plot is not None:
        plot_3d_surface(xx, yy, T_at_t_plot, title=f'Numerical Temperature Distribution at t={t_plot}', 
                        filename=getfilename(f'Numerical_temperature_t{t_plot}.png'))

    exact_T_at_t_plot = exact_heat_equation(xx, yy, t_plot)
    plot_3d_surface(xx, yy, exact_T_at_t_plot, title=f'Exact Temperature Distribution at t={t_plot}', 
                     filename=getfilename(f'exact_temperature_t{t_plot}.png'))

    # Task 6: Compute the numerical solution for varying spatial grid sizes: $N = 10, 20, 40, 80$. 
    # For each spatial resolution, use a fixed $\Delta t = 0.001$ and plot the absolute error map at $t = 0.05$
    print("| N   | dx       | RMSE         | Order |")
    print("|-----|----------|--------------|-------|")
    final_t = 0.1
    t_plot = 0.05
    dt = 0.001
    N = [10, 20, 40, 80]
    errors = []
    h_values = []

    last_rmse = -1
    last_h = -1
    for n in N:
        h = 1.0 / n
        _, T_at_t_plot, xx, yy = solve_heat_equation(n, final_t, dt, t_plot)

        if T_at_t_plot is None:
            print(f"Skipping N={n} due to solver failure.")
            continue

        error = compute_error(T_at_t_plot, t_plot)
        rmse = np.sqrt(np.mean(error**2))

        errors.append(rmse)
        h_values.append(h)
        plot_2d_colormap(xx, yy, error, title=f'Absolute Error Map (N={n}) at t={t_plot}', 
                         filename=getfilename(f'error_map_spatial_N{n}.png'))
        
        order = "-"
        if last_rmse > 0:
            try:
                order_val = np.log(last_rmse / rmse) / np.log(last_h / h)
                order = f"{order_val:.2f}"
            except Exception:
                pass # 保持 "-"
        last_rmse = rmse
        last_h = h
            
        print(f"| {n:<3} | {h:<8.4f} | {rmse:<12.3e} | {order:<5} |")
    
    plot_loglog_convergence(h_values, errors, h_label='dx', 
                          title='Spatial Convergence Plot', 
                          filename=getfilename('convergence_spatial.png'))
    
    # task 7: Repeat with varying time steps: $\Delta t = 0.01, 0.005, 0.0025, 0.00125$ while fixing $N = 40$, and plot the error map.
    print("| dt       | RMSE         | Order |")
    print("|----------|--------------|-------|")
    final_t = 0.1
    t_plot = 0.05
    dt = 0.001
    N = 40
    errors = []
    dt_list = [0.01, 0.005, 0.0025, 0.00125]

    last_rmse = -1
    last_h = -1
    h = 1.0 / N
    for dt in dt_list:
        _, T_at_t_plot, xx, yy = solve_heat_equation(N, final_t, dt, t_plot)

        if T_at_t_plot is None:
            print(f"Skipping dt={dt} due to solver failure.")
            continue

        error = compute_error(T_at_t_plot, t_plot)
        rmse = np.sqrt(np.mean(error**2))

        errors.append(rmse)
        plot_2d_colormap(xx, yy, error, title=f'Absolute Error Map (dt={dt}) at t={t_plot}', 
                         filename=getfilename(f'error_map_temporal_dt{dt}.png'))

        order = "-"
        if last_rmse > 0:
            try:
                order_val = np.log(last_rmse / rmse) / np.log(last_dt / dt)
                order = f"{order_val:.2f}"
            except Exception:
                pass # 保持 "-"
        last_rmse = rmse
        last_dt = dt

        print(f"| {dt:<8.4f} | {rmse:<12.3e} | {order:<5} |")

    plot_loglog_convergence(dt_list, errors, h_label='dt', 
                          title='Temporal Convergence Plot', 
                          filename=getfilename('convergence_temporal.png'))

    # Task 9: Measure the computation time for the largest grid (N=80) with dt=0.001 up to t=0.1
    N_time = 80
    dt_time = 0.001
    t_final_time = 0.1
    n_steps_time = int(round(t_final_time / dt_time))

    print(f"Measure the computation time (N={N_time}, dt={dt_time}, {n_steps_time} steps) ...")

    start_time = time.time()
    solve_heat_equation(N_time, dt_time, t_final_time, t_plot=t_plot)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Computation time: {elapsed_time:.4f} seconds")

