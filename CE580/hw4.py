import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# GIVEN DATA
# -----------------------------
Um = 2.0               # centerline velocity, m/s
R = 0.05               # pipe radius, m
nu = 1.0e-6            # molecular viscosity, m^2/s
rho = 1000.0           # density, kg/m^3

N = 31                 # total number of grid points
ratio = 0.82           # grid ratio
beta = 0.82            # constant grid ratio
dt = 3.50e-4           # time step, s
n_iter = 100000        # number of iterations


def generate_grid(R, N, ratio):
    delta_min = R * (1 - ratio) / (1 - ratio**(N - 1))

    delta_rev = np.zeros(N - 1)
    for i in range(N - 1):
        delta_rev[i] = delta_min * ratio**i

    delta = delta_rev[::-1]

    y = np.zeros(N)
    for i in range(1, N):
        y[i] = y[i - 1] + delta[i - 1]

    return y, delta

y, delta = generate_grid(R, N, ratio)

u = np.zeros(N)
for i in range(N):
    u[i] = Um * (y[i] / R)**(1/7)

dudy = np.zeros(N)

dudy[0] = (u[1] - u[0]) / (y[1] - y[0])

for i in range(1, N - 1):
    dudy[i] = (u[i + 1] - u[i - 1]) / (y[i + 1] - y[i - 1])

dudy[N - 1] = (u[N - 1] - u[N - 2]) / (y[N - 1] - y[N - 2])

tau_w = rho * nu * dudy[0]
u_star = np.sqrt(tau_w / rho)

A_plus = 26.0

y_plus = np.zeros(N)
f_mu = np.zeros(N)
lm = np.zeros(N)

for i in range(N):
    y_plus[i] = y[i] * u_star / nu
    f_mu[i] = 1.0 - np.exp(-y_plus[i] / A_plus)
    lm[i] = R * (0.14 - 0.08 * (1 - y[i] / R)**2 - 0.06 * (1 - y[i] / R)**4) * f_mu[i]

nu_t = np.zeros(N)
nu_e = np.zeros(N)

for i in range(N):
    nu_t[i] = lm[i]**2 * abs(dudy[i])
    nu_e[i] = nu + nu_t[i]

r_coord = np.zeros(N)
for i in range(N):
    r_coord[i] = R - y[i]

A = np.zeros(N)
C = np.zeros(N)

for i in range(1, N - 1):
    r_i = r_coord[i]
    r_iph = 0.5 * (r_coord[i] + r_coord[i + 1])
    r_imh = 0.5 * (r_coord[i] + r_coord[i - 1])

    Delta_i = delta[i - 1]
    Delta_ip1 = delta[i]

    A[i] = 2.0 * r_iph / (r_i * (Delta_ip1 + Delta_i) * Delta_ip1)
    C[i] = 2.0 * r_imh / (r_i * (Delta_ip1 + Delta_i) * Delta_i)