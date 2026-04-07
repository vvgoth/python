import numpy as np
import matplotlib.pyplot as plt

def run_simulation(N, dt=0.0008, save_interval=0.1):
    # Physical parameters
    g = 9.81           # gravity (m/s^2)
    W = 1.0            # channel width (m)
    L = 20.0           # channel length (m)
    h0 = 12.0          # initial water depth (m)
    u0 = 0.0           # initial velocity (m/s)
    Cf = 0.005         # friction coefficient
    T = 2.0            # wave period (s)
    H = 1.0            # wave amplitude parameter from the sheet (m)

    # Numerical parameters
    dx = L / (N - 1)
    n_cycles = 50
    t_end = n_cycles * T
    nt = int(t_end / dt)

    # Grid
    x = np.linspace(0.0, L, N)

    # Initial conditions
    h = np.full(N, h0, dtype=float)
    u = np.full(N, u0, dtype=float)

    A = h * u
    B = h * u**2 + 0.5 * g * h**2
    pw = W + 2.0 * h
    C = Cf * u * np.abs(u) * pw / (2.0 * W)

    # Storage for homework plots
    time_history = np.zeros(nt + 1)
    dt_allow_history = np.zeros(nt + 1)

    h_left_history = np.zeros(nt + 1)
    h_right_history = np.zeros(nt + 1)

    time_history[0] = 0.0
    dt_allow_history[0] = np.min(dx / (np.abs(u) + np.sqrt(g * h)))

    h_left_history[0] = h[0]
    h_right_history[0] = h[-1]

    # Storage for Tecplot export
    save_every = int(save_interval / dt)
    if save_every < 1:
        save_every = 1

    saved_times = [0.0]
    saved_h = [h.copy()]
    saved_u = [u.copy()]

    # Time loop
    for n in range(nt):
        t = n * dt

        # Intermediate step arrays
        h_half = np.zeros_like(h)
        A_half = np.zeros_like(A)

        # Lax-Wendroff first half-step for interior nodes
        h_half[1:-1] = 0.5 * (h[2:] + h[:-2]) - (dt / (4.0 * dx)) * (A[2:] - A[:-2])
        A_half[1:-1] = 0.5 * (A[2:] + A[:-2]) - (dt / (4.0 * dx)) * (B[2:] - B[:-2]) - 0.5 * dt * C[1:-1]

        # Half-step boundary conditions
        A_half[0] = 0.0
        h_half[0] = h[0] - (dt / (2.0 * dx)) * (A[1] - A[0])

        h_half[-1] = h0 + 0.5 * H * np.sin(2.0 * np.pi * (t + 0.5 * dt) / T)
        A_half[-1] = A[-1] - (dt / dx) * (B[-1] - B[-2]) - 0.5 * dt * C[-1]

        # Half-step variables
        u_half = A_half / h_half
        B_half = h_half * u_half**2 + 0.5 * g * h_half**2
        pw_half = W + 2.0 * h_half
        C_half = Cf * u_half * np.abs(u_half) * pw_half / (2.0 * W)

        # Full-step arrays
        h_new = np.zeros_like(h)
        A_new = np.zeros_like(A)

        # Lax-Wendroff full step for interior nodes
        h_new[1:-1] = h[1:-1] - (dt / (2.0 * dx)) * (A_half[2:] - A_half[:-2])
        A_new[1:-1] = A[1:-1] - (dt / (2.0 * dx)) * (B_half[2:] - B_half[:-2]) - dt * C_half[1:-1]

        # Full-step boundary conditions
        A_new[0] = 0.0
        h_new[0] = h[0] - (dt / dx) * (A_half[1] - A_half[0])

        h_new[-1] = h0 + 0.5 * H * np.sin(2.0 * np.pi * (t + dt) / T)
        A_new[-1] = A[-1] - (dt / dx) * (B_half[-1] - B_half[-2]) - dt * C_half[-1]

        # Updated velocity
        u_new = A_new / h_new

        # Advance solution
        h = h_new.copy()
        A = A_new.copy()
        u = u_new.copy()

        B = h * u**2 + 0.5 * g * h**2
        pw = W + 2.0 * h
        C = Cf * u * np.abs(u) * pw / (2.0 * W)

        # Save homework results
        time_history[n + 1] = t + dt
        dt_allow_history[n + 1] = np.min(dx / (np.abs(u) + np.sqrt(g * h)))
        h_left_history[n + 1] = h[0]
        h_right_history[n + 1] = h[-1]

        # Save Tecplot snapshots
        if (n + 1) % save_every == 0:
            saved_times.append(t + dt)
            saved_h.append(h.copy())
            saved_u.append(u.copy())

    return x, h, time_history, dt_allow_history, h_left_history, h_right_history, saved_times, saved_h, saved_u, h0


def export_to_tecplot_water(filename, x, saved_times, saved_h, saved_u, h0):
    with open(filename, 'w') as f:
        f.write('TITLE = "HW5 Shallow Water Basin"\n')
        f.write('VARIABLES = "X", "Y", "H", "ETA", "U"\n')

        for k, t in enumerate(saved_times):
            h = saved_h[k]
            u = saved_u[k]
            eta = h - h0

            f.write(
                f'ZONE T="t={t:.4f}s", I={len(x)}, J=2, '
                f'ZONETYPE=Ordered, DATAPACKING=POINT, '
                f'STRANDID=1, SOLUTIONTIME={t:.6f}\n'
            )

            # Bottom row: basin floor
            for i in range(len(x)):
                f.write(f'{x[i]:.8f} {0.0:.8f} {h[i]:.8f} {eta[i]:.8f} {u[i]:.8f}\n')

            # Top row: free surface
            for i in range(len(x)):
                f.write(f'{x[i]:.8f} {h[i]:.8f} {h[i]:.8f} {eta[i]:.8f} {u[i]:.8f}\n')


# Base run for N = 2001
x, h_final, time_history, dt_allow_history, h_left_history, h_right_history, saved_times, saved_h, saved_u, h0 = run_simulation(
    2001, dt=0.0008, save_interval=0.1
)

# Tecplot export
export_to_tecplot_water("hw5_basin_water.dat", x, saved_times, saved_h, saved_u, h0)
print("Tecplot file written: hw5_basin_water.dat")