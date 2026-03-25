import numpy as np
import matplotlib.pyplot as plt

# Parameters
H = 0.1          # m
dpdx = -1.0      # Pa/m
mu = 0.001       # N·s/m²
Cp = dpdx        # pressure gradient term

# Analytical solution 
def analytical(y):
    return (Cp / mu) * (y**2 / 2 - H * y)

y = np.linspace(0, H, 100)
u_a = analytical(y)

Umax = analytical(H)
print(f"Umax = u(H) = {Umax:.4f} m/s")

# Solve using Dirichlet BCs at both boundaries (u=0 at wall, u=Umax at centerline)
def dirichlet(N):
    dy = H / (N - 1)
    y = np.linspace(0, H, N)

    A = np.ones(N)
    B = 2 * np.ones(N)
    C = np.ones(N)
    D = -(Cp / mu) * dy**2 * np.ones(N)

    E = np.zeros(N)
    F = np.zeros(N)
    u = np.zeros(N)

    # Wall BC: u[0]=0
    E[0] = 0.0
    F[0] = 0.0

    # Forward sweep of Thomas algorithm
    for i in range(1, N):
        denom = B[i] - C[i] * E[i-1]
        E[i] = A[i] / denom
        F[i] = (D[i] + C[i] * F[i-1]) / denom

    # Centerline BC: u=Umax at y=H
    Umax = analytical(H)
    u[-1] = Umax

    # Backward substitution
    for i in range(N-2, -1, -1):
        u[i] = E[i] * u[i+1] + F[i]
    
    return y, u

# Solve using Dirichlet at wall and Neumann (du/dy=0) at centerline
def neumann(N):
    dy = H / (N - 1)
    y = np.linspace(0, H, N)
    
    A = np.ones(N)
    B = 2 * np.ones(N)
    C = np.ones(N)
    D = -(Cp / mu) * dy**2 * np.ones(N)
    
    E = np.zeros(N)
    F = np.zeros(N)
    u = np.zeros(N)
    
    # Wall BC: u[0]=0
    E[0] = 0.0
    F[0] = 0.0
    
    # Forward sweep
    for i in range(1, N):
        denom = B[i] - C[i] * E[i-1]
        E[i] = A[i] / denom
        F[i] = (D[i] + C[i] * F[i-1]) / denom
    
    # Centerline BC: du/dy=0 (symmetry condition)
    BCN = 0.0
    u[-1] = (F[-2] + dy * BCN) / (1 - E[-2])
    
    # Backward substitution
    for i in range(N-2, -1, -1):
        u[i] = E[i] * u[i+1] + F[i]
    
    return y, u

# Compute normalized absolute errors for both boundary conditions
def compute_errors(N):
    y, u_b = dirichlet(N)
    y, u_c = neumann(N)
    u_a = analytical(y)
    Umax = analytical(H)
    
    # Calculate mean absolute error (normalized by Umax)
    E_dirichlet = (1 / (N-1)) * np.sum(np.abs(u_b[1:] - u_a[1:])) / Umax
    E_neumann   = (1 / (N-1)) * np.sum(np.abs(u_c[1:] - u_a[1:])) / Umax
    
    return E_dirichlet, E_neumann

# Calculate errors for varying grid sizes
N_values = np.arange(10, 5001, 10)
E_dir = []
E_neu = []

for N in N_values:
    ed, en = compute_errors(N)
    E_dir.append(ed)
    E_neu.append(en)

# Errors vs N
plt.figure(1)
plt.semilogy(N_values, E_dir, label='Dirichlet')
plt.semilogy(N_values, E_neu, label='Neumann')
plt.xlabel('N')
plt.ylabel('Error')
plt.title('Errors vs N')
plt.legend()
plt.grid(True)


# Centerline velocity u(H) vs N for Neumann case
u_centerline = []

for N in N_values:
    y, u_c = neumann(N)
    u_centerline.append(u_c[-1])  # u at y=H

plt.figure(2)
plt.plot(N_values, u_centerline)
plt.axhline(y=analytical(H), color='r', linestyle='--', label='Analytical')
plt.xlabel('N')
plt.ylabel('u(H)')
plt.title('Centerline Velocity vs N (Neumann)')
plt.legend()
plt.grid(True)


# Dirichlet solver with single precision (float32) for precision comparison
def dirichlet_single(N):
    dy = np.float32(H) / np.float32(N - 1)
    y = np.linspace(0, H, N, dtype=np.float32)
    
    A = np.ones(N, dtype=np.float32)
    B = np.float32(2) * np.ones(N, dtype=np.float32)
    C = np.ones(N, dtype=np.float32)
    D = np.float32(-(Cp / mu)) * dy**2 * np.ones(N, dtype=np.float32)
    
    E = np.zeros(N, dtype=np.float32)
    F = np.zeros(N, dtype=np.float32)
    u = np.zeros(N, dtype=np.float32)
    
    E[0] = np.float32(0.0)
    F[0] = np.float32(0.0)
    
    for i in range(1, N):
        denom = B[i] - C[i] * E[i-1]
        E[i] = A[i] / denom
        F[i] = (D[i] + C[i] * F[i-1]) / denom
    
    u[-1] = np.float32(analytical(H))
    
    for i in range(N-2, -1, -1):
        u[i] = E[i] * u[i+1] + F[i]
    
    return y, u

# Neumann solver with single precision (float32) for precision comparison
def neumann_single(N):
    dy = np.float32(H) / np.float32(N - 1)
    y = np.linspace(0, H, N, dtype=np.float32)
    
    A = np.ones(N, dtype=np.float32)
    B = np.float32(2) * np.ones(N, dtype=np.float32)
    C = np.ones(N, dtype=np.float32)
    D = np.float32(-(Cp / mu)) * dy**2 * np.ones(N, dtype=np.float32)
    
    E = np.zeros(N, dtype=np.float32)
    F = np.zeros(N, dtype=np.float32)
    u = np.zeros(N, dtype=np.float32)
    
    # Wall BC: u[0]=0
    E[0] = np.float32(0.0)
    F[0] = np.float32(0.0)
    
    # Forward sweep
    for i in range(1, N):
        denom = B[i] - C[i] * E[i-1]
        E[i] = A[i] / denom
        F[i] = (D[i] + C[i] * F[i-1]) / denom
    
    # Centerline BC: du/dy=0
    BCN = np.float32(0.0)
    u[-1] = (F[-2] + dy * BCN) / (1 - E[-2])
    
    # Backward substitution
    for i in range(N-2, -1, -1):
        u[i] = E[i] * u[i+1] + F[i]
    
    return y, u


N_vals = np.arange(10, 5001, 10)
diff_d = []
diff_n = []

for N in N_vals:
    y_d_d, u_d_double = dirichlet(N)
    y_d_s, u_d_single = dirichlet_single(N)
    diff_d.append(np.max(np.abs(u_d_double - u_d_single.astype(np.float64))))

    y_n_d, u_n_double = neumann(N)
    y_n_s, u_n_single = neumann_single(N)
    diff_n.append(np.max(np.abs(u_n_double - u_n_single.astype(np.float64))))

# Precision loss due to single precision arithmetic
plt.figure(3)
plt.semilogy(N_vals, diff_d, label='Dirichlet')
plt.semilogy(N_vals, diff_n, label='Neumann')
plt.xlabel('N')
plt.ylabel('Max difference')
plt.title('Double vs Single Precision Difference (Dirichlet & Neumann)')
plt.legend()
plt.grid(True)

plt.show()