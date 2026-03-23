import numpy as np
import matplotlib.pyplot as plt

# Given data
Um = 2.0               # centerline velocity, m/s
R = 0.05               # pipe radius, m
nu = 1.0e-6            # molecular viscosity, m^2/s
rho = 1000.0           # density, kg/m^3

N = 31                 # total number of grid points
ratio = 0.82           # grid ratio
beta = 0.82            # constant grid ratio
dt = 3.50e-4           # time step, s
n_iter = 100000        # number of iterations
A_plus = 26.0          # damping constant

# Generate non-uniform grid
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

# Initial velocity profile (power-law)
u = np.zeros(N)
for i in range(N):
    u[i] = Um * (y[i] / R)**(1/7)

# compute r coordinate for cylindrical coordinates
r_coord = np.zeros(N)
for i in range(N):
    r_coord[i] = R - y[i]

# Storage 
dudy = np.zeros(N)
y_plus = np.zeros(N)
f_mu = np.zeros(N)
lm = np.zeros(N)
nu_t = np.zeros(N)
nu_e = np.zeros(N)
u_plus = np.zeros(N)

A = np.zeros(N)
C = np.zeros(N)
eps = np.zeros(N)
u_new = np.zeros(N)

residuals = []


# Iterative solver
for n in range(n_iter):

    # du/dy
    dudy[0] = (u[1] - u[0]) / (y[1] - y[0])

    for i in range(1, N - 1):
        dudy[i] = (u[i + 1] - u[i - 1]) / (y[i + 1] - y[i - 1])

    dudy[N - 1] = (u[N - 1] - u[N - 2]) / (y[N - 1] - y[N - 2])

    # wall shear and pressure-gradient constant
    tau_w = rho * nu * dudy[0]
    Cp = -2.0 * tau_w / R

    # friction velocity
    u_star = np.sqrt(tau_w / rho)

    # damping function, mixing length, turbulent viscosity
    for i in range(N):
        y_plus[i] = y[i] * u_star / nu
        u_plus[i] = u[i] / u_star
        f_mu[i] = 1.0 - np.exp(-y_plus[i] / A_plus)

        lm[i] = R * (
            0.14
            - 0.08 * (1.0 - y[i] / R)**2
            - 0.06 * (1.0 - y[i] / R)**4
        ) * f_mu[i]

        nu_t[i] = lm[i]**2 * abs(dudy[i])
        nu_e[i] = nu + nu_t[i]

    # explicit solution for interior nodes
    for i in range(1, N - 1):
        r_i = r_coord[i]
        r_iph = 0.5 * (r_coord[i] + r_coord[i + 1])
        r_imh = 0.5 * (r_coord[i] + r_coord[i - 1])

        Delta_i = delta[i - 1]
        Delta_ip1 = delta[i]

        A[i] = 2.0 * r_iph / (r_i * (Delta_ip1 + Delta_i) * Delta_ip1)
        C[i] = 2.0 * r_imh / (r_i * (Delta_ip1 + Delta_i) * Delta_i)

        nu_e_iph = 0.5 * (nu_e[i] + nu_e[i + 1])
        nu_e_imh = 0.5 * (nu_e[i] + nu_e[i - 1])

        eps[i] = dt * (
            -Cp / rho
            + A[i] * nu_e_iph * (u[i + 1] - u[i])
            - C[i] * nu_e_imh * (u[i] - u[i - 1])
        )

        u_new[i] = u[i] + eps[i]

    # boundary conditions
    u_new[0] = 0.0
    u_new[N - 1] = Um

    # residual error
    residual = 0.0
    for i in range(1, N - 1):
        residual += abs(eps[i])

    residual = residual / ((N - 2) * Um)
    residuals.append(residual)

    # update velocity for next iteration
    u[:] = u_new

d_f = np.zeros(N)

for i in range(1, N - 1):
    Delta_i = delta[i - 1]
    d_f[i] = nu_e[i] * dt / (Delta_i**2)

# Determine discharge, average velocity, Reynolds number, and friction factors
Q = 2.0 * np.pi * np.trapezoid(u * r_coord, y)
V_avg = Q / (np.pi * R**2)
D = 2.0 * R
Re = V_avg * D / nu

f_d = 8.0 * tau_w / (rho * V_avg**2)
f_m = 0.25 / (np.log10(5.74 / Re**0.9)**2)

print(f"Computed Darcy friction factor: {f_d:.6f}")
print(f"Swamee-Jain friction factor: {f_m:.6f}")

# Plotting
plt.figure(1)

plt.semilogx(y_plus[1:], u_plus[1:], '-', label='Numerical')

# log law: u+ = (1/kappa) ln(y+) + B
kappa = 0.41
B = 5.0
yplus_log = np.linspace(30, np.max(y_plus), 200)
uplus_log = (1 / kappa) * np.log(yplus_log) + B
plt.semilogx(yplus_log, uplus_log, '--', label='Log law')

plt.xlabel('log(y+)')
plt.ylabel('u+')
plt.title('Velocity Profile')
plt.grid(True, which='both')
plt.legend()

plt.figure(2)
plt.loglog(np.arange(1, n_iter + 1), residuals)
plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.title('Residual Error')
plt.grid(True)

plt.figure(3)
plt.semilogx(y_plus[1:N-1], d_f[1:N-1], '-')
plt.xlabel('log(y+)')
plt.ylabel('d_f')
plt.title('Diffusion Number')
plt.grid(True, which='both')

plt.figure(4)
plt.semilogy(y[1:] / R, nu_t[1:] / nu, '-')
plt.xlabel('y / R')
plt.ylabel('log(nu_t / nu)')
plt.title('Turbulent Viscosity Ratio')
plt.grid(True)
plt.show()