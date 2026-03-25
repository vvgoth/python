import numpy as np
import matplotlib.pyplot as plt

# Given data
H = 0.02
mu = 0.001
rho = 1000.0
nu = mu / rho

N = 101
max_iter = 100

# Cases
dpdx_cases = [-100.0, -1000.0, -10000.0]
grid_ratios = [0.96, 0.94, 0.93]


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


def velocity_gradient(y, u):
    dudyy = np.zeros_like(u)

    # wall
    dudyy[0] = (u[1] - u[0]) / (y[1] - y[0])

    # interior points
    for i in range(1, N - 1):
        dudyy[i] = (u[i + 1] - u[i - 1]) / (y[i + 1] - y[i - 1])

    # centerline symmetry
    dudyy[-1] = 0.0

    return dudyy

# Thomas Algorithm to solve the tridiagonal system for velocity update
def solve_velocity(y, mu_e, Cp):
    E = np.zeros(N)
    F = np.zeros(N)
    u = np.zeros(N)

    # BC at i=1 : u1 = 0  --> E1 = 0, F1 = 0
    E[0] = 0.0
    F[0] = 0.0

    # forward sweep for i = 2 ... N-1
    for i in range(1, N - 1):
        Delta_i = y[i] - y[i - 1]
        Delta_ip1 = y[i + 1] - y[i]

        mu_e_iphalf = 0.5 * (mu_e[i] + mu_e[i + 1])
        mu_e_imhalf = 0.5 * (mu_e[i - 1] + mu_e[i])

        Ai = mu_e_iphalf / (Delta_ip1 * ((Delta_ip1 + Delta_i) / 2.0))
        Ci = mu_e_imhalf / (Delta_i * ((Delta_ip1 + Delta_i) / 2.0))
        Bi = Ai + Ci
        Di = -Cp

        den = Bi - Ci * E[i - 1]

        E[i] = Ai / den
        F[i] = (Di + Ci * F[i - 1]) / den

    # BC at i=N : du/dy = 0
    # (uN - uN-1)/Delta = 0  --> uN = uN-1
    Delta_top = y[-1] - y[-2]
    BCN = 0.0

    u[-1] = (F[-2] + Delta_top * BCN) / (1.0 - E[-2])

    # back substitution
    for i in range(N - 2, -1, -1):
        u[i] = E[i] * u[i + 1] + F[i]

    return u


results = {}

for Cp, r in zip(dpdx_cases, grid_ratios):
    y, delta = generate_grid(H, N, r)

    u = laminar_initial_velocity(y, H, mu, Cp)
    mu_t = np.zeros(N)

    errors = []

    for it in range(max_iter):
        dudy = velocity_gradient(y, u)

        tau_w = -Cp * H
        u_star = np.sqrt(tau_w / rho)
        y_plus = y * u_star / nu

        f_mu = 1.0 - np.exp(-y_plus / 26.0)
        eta = 1.0 - y / H
        lm = H * (0.14 - 0.08 * eta**2 - 0.06 * eta**4) * f_mu

        mu_t_new = rho * lm**2 * np.abs(dudy)
        mu_t = 0.5 * (mu_t + mu_t_new)
        mu_e = mu + mu_t

        u_new = solve_velocity(y, mu_e, Cp)

        error = np.sum(np.abs(u[1:] - u_new[1:])) / ((N - 1) * max(abs(u_new[-1]), 1e-12))
        errors.append(error)

        print(f'dp/dx = {Cp:8.1f} N/m^3   iteration = {it+1:3d}   error = {error:.6e}')

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

print("\nMaximum centerline velocities:")
for Cp in dpdx_cases:
    umax = results[Cp]["u"][-1]
    print(f"dp/dx = {Cp:8.1f} N/m^3   U_max = {umax:.6f} m/s")

plt.figure(1)
for Cp in dpdx_cases:
    y_plus = results[Cp]["y_plus"]
    u_plus = results[Cp]["u_plus"]
    mask = y_plus > 0
    plt.semilogx(y_plus[mask], u_plus[mask], label=f'dp/dx = {Cp:.0f} N/m^3')

# reference log-law line
ylog = np.linspace(30, 500, 300)
ulog = (1.0 / 0.41) * np.log(ylog) + 5.0

plt.semilogx(ylog, ulog, 'k--', label=r'$u^+ = \frac{1}{0.41}\ln(y^+) + 5.0$')

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