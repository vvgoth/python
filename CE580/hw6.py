import numpy as np
import matplotlib.pyplot as plt

N = 21
x = np.linspace(0, 1, N)  # N points from 0 to 1
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)  # 2D coordinate arrays

U = X**2 - Y**2  # analytical solution at every grid point

S = np.zeros((N, N))  # start everything at zero

# set boundary values from U
S[0, :]  = U[0, :]   # bottom row  (y = 0)
S[-1, :] = U[-1, :]  # top row     (y = 1)
S[:, 0]  = U[:, 0]   # left column (x = 0)
S[:, -1] = U[:, -1]  # right column (x = 1)

omega = 1.5  # just a test value for now

for iteration in range(20):
    for i in range(1, N-1):
        for j in range(1, N-1):
            R = 0.25 * (S[i-1,j] + S[i+1,j] + S[i,j-1] + S[i,j+1]) - S[i,j]
            S[i,j] = S[i,j] + omega * R
            
error = np.sum(np.abs(U[1:N-1, 1:N-1] - S[1:N-1, 1:N-1])) / (N-2)**2
print(f"omega = {omega:.3f}, error = {error:.6e}")

N_values = [21, 31, 41, 51, 61, 81, 101]
omega_vals = np.arange(1.0, 2.0 + 1e-9, 0.002)

all_errors = {}
omega_opt = {}

for N in N_values:
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    U = X**2 - Y**2
    
    errors = np.zeros(len(omega_vals))
    
    for k, omega in enumerate(omega_vals):
        S = np.zeros((N, N))
        S[0, :]  = U[0, :]
        S[-1, :] = U[-1, :]
        S[:, 0]  = U[:, 0]
        S[:, -1] = U[:, -1]
        
        for iteration in range(100):
            for i in range(1, N-1):
                for j in range(1, N-1):
                    R = 0.25 * (S[i-1,j] + S[i+1,j] + S[i,j-1] + S[i,j+1]) - S[i,j]
                    S[i,j] = S[i,j] + omega * R
        
        errors[k] = np.sum(np.abs(U[1:N-1, 1:N-1] - S[1:N-1, 1:N-1])) / (N-2)**2
    
    all_errors[N] = errors
    omega_opt[N] = omega_vals[np.argmin(errors)]
    print(f"N={N}, omega_opt={omega_opt[N]:.3f}")

plt.figure(figsize=(10, 6))
for N in N_values:
    plt.plot(omega_vals, np.log10(all_errors[N]), label=f"N={N}")

plt.xlabel("omega")
plt.ylabel("log10(Error)")
plt.title("PSOR Error vs Omega after 20 iterations")
plt.legend()
plt.grid(True)

N = 51
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)
U = X**2 - Y**2

omega_vals_B = [1.6, 1.7, 1.8, 1.9, 1.99]

plt.figure(figsize=(10, 6))

for omega in omega_vals_B:
    S = np.zeros((N, N))
    S[0, :]  = U[0, :]
    S[-1, :] = U[-1, :]
    S[:, 0]  = U[:, 0]
    S[:, -1] = U[:, -1]
    
    iter_errors = []
    
    for iteration in range(1000):
        for i in range(1, N-1):
            for j in range(1, N-1):
                R = 0.25 * (S[i-1,j] + S[i+1,j] + S[i,j-1] + S[i,j+1]) - S[i,j]
                S[i,j] = S[i,j] + omega * R
        
        error = np.sum(np.abs(U[1:N-1, 1:N-1] - S[1:N-1, 1:N-1])) / (N-2)**2
        iter_errors.append(error)
    
    plt.plot(range(1, 1001), np.log10(iter_errors), label=f"omega={omega}")

plt.xlabel("Iteration")
plt.ylabel("log10(Error)")
plt.title("Part B - PSOR Convergence (N=51)")
plt.legend()
plt.grid(True)
plt.show()