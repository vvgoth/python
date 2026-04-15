import numpy as np
import matplotlib.pyplot as plt

def run_simulation(N, dt=0.0008):
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
    n_cycles = 10
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

    # Storage for plots
    time_history = np.zeros(nt + 1)
    dt_allow_history = np.zeros(nt + 1)

    h_left_history = np.zeros(nt + 1)
    h_right_history = np.zeros(nt + 1)

    time_history[0] = 0.0
    dt_allow_history[0] = np.min(dx / (np.abs(u) + np.sqrt(g * h)))

    h_left_history[0] = h[0]
    h_right_history[0] = h[-1]

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

        # Save results
        time_history[n + 1] = t + dt
        dt_allow_history[n + 1] = np.min(dx / (np.abs(u) + np.sqrt(g * h)))
        h_left_history[n + 1] = h[0]
        h_right_history[n + 1] = h[-1]

    return x, h, time_history, dt_allow_history, h_left_history, h_right_history


# Base run for N = 2001
x, h_final, time_history, dt_allow_history, h_left_history, h_right_history = run_simulation(2001)

# Allowable time-step size as a function of time
plt.figure(figsize=(10, 6))
plt.plot(time_history, dt_allow_history, linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Allowable time step (s)')
plt.title('Allowable time-step size as a function of time')
plt.grid(True)
plt.tight_layout()

# Water surface level at both ends of the basin
plt.figure(figsize=(10, 6))
plt.plot(time_history, h_left_history, linewidth=1.5, label='x = 0 m')
plt.plot(time_history, h_right_history, linewidth=1.5, label='x = 20 m')
plt.xlabel('Time (s)')
plt.ylabel('Water surface level h (m)')
plt.title('Water surface level at both ends of the basin')
plt.ylim(0, 14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Water surface profile at the end of the 10th wave cycle
eta_final = h_final - 12.0

plt.figure(figsize=(10, 6))
plt.plot(x, eta_final, linewidth=1.5)
plt.xlabel('x (m)')
plt.ylabel(r'Water surface fluctuation $\eta$ (m)')
plt.title('Water surface profile at the end of the 10th wave cycle')
plt.grid(True)
plt.tight_layout()

# Comparison of final water surface profiles for different mesh sizes
mesh_sizes = [201, 501, 1001, 2001]

plt.figure(figsize=(10, 6))

for N in mesh_sizes:
    x_mesh, h_mesh, _, _, _, _ = run_simulation(N)
    eta_mesh = h_mesh - 12.0
    plt.plot(x_mesh, eta_mesh, linewidth=1.5, label=f'N = {N}')

plt.xlabel('x (m)')
plt.ylabel(r'Water surface fluctuation $\eta$ (m)')
plt.title('Final water surface profiles for different mesh sizes')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()