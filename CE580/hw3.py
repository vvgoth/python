import numpy as np
import matplotlib.pyplot as plt

# Physical parameters
H = 0.02              # half-channel height [m]
mu = 0.001            # dynamic viscosity [N.s/m^2]
rho = 1000.0          # density [kg/m^3]
nu = mu / rho         # kinematic viscosity [m^2/s]

# Numerical parameters
N = 101               # number of grid points
max_iter = 100        # number of iteration cycles

# Cases
dpdx_cases = [-100.0, -1000.0, -10000.0]   # pressure gradients [N/m^3]
grid_ratios = [0.96, 0.94, 0.93]           # corresponding grid ratios

# Boundary conditions
u_wall = 0.0            # u(y=0) = 0

def generate_grid(H, N, r):
    delta_min = H * (1 - r) / (1 - r**(N - 1))

    delta_rev = np.zeros(N - 1)
    for i in range(N - 1):
        delta_rev[i] = delta_min * r**i

    delta = delta_rev[::-1]

    y = np.zeros(N)
    for i in range(1, N):
        y[i] = y[i - 1] + delta[i - 1]

    return y, delta

def laminar_initial_velocity(y, H, mu, Cp):
    return (Cp / (2 * mu)) * y**2 - (Cp * H / mu) * y

def compute_dudy(y, u):
    dudy = np.zeros_like(u)

    # forward difference at the wall
    dudy[0] = (u[1] - u[0]) / (y[1] - y[0])

    # central difference for interior nodes
    for i in range(1, len(u) - 1):
        dudy[i] = (u[i + 1] - u[i - 1]) / (y[i + 1] - y[i - 1])

    # backward difference at the centerline
    dudy[-1] = (u[-1] - u[-2]) / (y[-1] - y[-2])

    return dudy

def solve_tridiagonal(A, B, C, D):
    E = np.zeros(len(B))
    F = np.zeros(len(B))
    u = np.zeros(len(B))

    E[0] = A[0] / B[0]
    F[0] = D[0] / B[0]

    for i in range(1, len(B)):
        denom = B[i] - C[i] * E[i - 1]
        E[i] = A[i] / denom
        F[i] = (D[i] + C[i] * F[i - 1]) / denom

    u[-1] = F[-1]

    for i in range(len(B) - 2, -1, -1):
        u[i] = E[i] * u[i + 1] + F[i]

    return u

results = {}

for Cp, r in zip(dpdx_cases, grid_ratios):
    # Generate grid
    y, delta = generate_grid(H, N, r)

    # Initial velocity from laminar analytical solution
    u = laminar_initial_velocity(y, H, mu, Cp)

    # Initial turbulent viscosity guess
    mu_t = np.zeros(N)

    errors = []

    for n in range(max_iter):
        dudy = compute_dudy(y, u)

        tau_w = -Cp * H
        u_star = np.sqrt(tau_w / rho)
        y_plus = y * u_star / nu

        f_mu = 1.0 - np.exp(-y_plus / 26.0)
        eta = 1.0 - y / H
        lm = H * (0.14 - 0.08 * eta**2 - 0.06 * eta**4) * f_mu

        mu_t_new = rho * lm**2 * np.abs(dudy)
        mu_t = 0.5 * (mu_t + mu_t_new)
        mu_e = mu + mu_t

        A = np.zeros(N)
        B = np.zeros(N)
        C = np.zeros(N)
        D = np.zeros(N)

        for i in range(1, N - 1):
            Delta_i = y[i] - y[i - 1]
            Delta_ip1 = y[i + 1] - y[i]

            mu_e_iphalf = 0.5 * (mu_e[i] + mu_e[i + 1])
            mu_e_imhalf = 0.5 * (mu_e[i - 1] + mu_e[i])

            A[i] = mu_e_iphalf / (Delta_ip1 * ((Delta_ip1 + Delta_i) / 2.0))
            C[i] = mu_e_imhalf / (Delta_i   * ((Delta_ip1 + Delta_i) / 2.0))
            B[i] = A[i] + C[i]
            D[i] = -Cp

        # Wall BC: u = 0
        A[0] = 0.0
        B[0] = 1.0
        C[0] = 0.0
        D[0] = 0.0

        # Centerline BC: du/dy = 0  ->  u_N - u_(N-1) = 0
        A[-1] = 0.0
        B[-1] = 1.0
        C[-1] = 1.0
        D[-1] = 0.0

        u_new = solve_tridiagonal(A, B, C, D)

        denom = (N - 1) * max(abs(u_new[-1]), 1e-12)
        error = np.sum(np.abs(u[1:] - u_new[1:])) / denom
        
        errors.append(error)

        u = u_new.copy()
    
    results[Cp] = {
        "y": y.copy(),
        "u": u.copy(),
        "mu_t": mu_t.copy(),
        "errors": np.array(errors),
        "u_star": u_star,
        "y_plus": y_plus.copy(),
        "u_plus": (u / u_star).copy()
    }

for Cp in dpdx_cases:
    umax = results[Cp]["u"][-1]
    print(f"dp/dx = {Cp:8.1f} N/m^3   U_max = {umax:.6f} m/s")

plt.figure(1)

for Cp in dpdx_cases:
    y_plus = results[Cp]["y_plus"]
    u_plus = results[Cp]["u_plus"]

    mask = y_plus > 0
    plt.semilogx(y_plus[mask], u_plus[mask], label=f'dp/dx = {Cp:.0f} N/m^3')

# Log-law reference line
y_log = np.linspace(30, 500, 300)
kappa = 0.41
B_log = 5.0
u_log = (1.0 / kappa) * np.log(y_log) + B_log

plt.semilogx(y_log, u_log, 'k--', label=r'$u^+ = \frac{1}{0.41}\ln(y^+) + 5.0$')

plt.xlabel(r'$y^+$')
plt.ylabel(r'$u^+$')
plt.title(r'Velocity Profiles in Wall Coordinates')
plt.legend()
plt.grid(True, which='both')

plt.figure(2)

for Cp in dpdx_cases:
    err = results[Cp]["errors"]
    iters = np.arange(1, len(err) + 1)

    mask = np.isfinite(err) & (err > 0)
    plt.loglog(iters[mask], err[mask], label=f'dp/dx = {Cp:.0f} N/m^3')

plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error vs Iteration')
plt.legend()
plt.grid(True, which='both')
plt.show()