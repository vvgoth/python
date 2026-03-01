import matplotlib.pyplot as plt
import numpy as np

# ── Parameters ──────────────────────────────────────────────────────────────
g = 9.81
H = 8
D = 0.3
ks = 0.0001
μ = 0.001
rho = 1000
delta_t = 0.02
T = 60
eps = 0.01
L_cases = [50, 100, 200, 500]
time_array = np.arange(0, T + delta_t, delta_t)

def Swamee_jain(Re, ks, D):
    f = 0.25 / (np.log10(ks/(3.7*D) + 5.74/Re**0.9))**2
    return f

def steady_state_velocity(H, D, L, ks, rho, μ, g=9.81):
    f = 0.014

    for i in range(100):
        Vs = np.sqrt(2*g*H / (1 + f*L/D))
        Re = rho * Vs * D / μ
        f_new = Swamee_jain(Re, ks, D)

        if abs(f_new - f) < 10**-6:
            break
        f = f_new
    return Vs

def velocity_time_series(H, D, L, ks, rho, μ, delta_t):
    f = 0.014
    Vs_i = 0
    velocity_series = [0]

    for t in time_array[1:]:
        Vs = Vs_i + delta_t * (H - (1 + f*L/D) * Vs_i**2 / (2*g)) * g / L
        velocity_series.append(Vs)
        Re = rho * Vs * D / μ
        f = Swamee_jain(Re, ks, D)
        Vs_i = Vs

    return velocity_series

for L in L_cases:
    Vs = steady_state_velocity(H, D, L, ks, rho, μ)
    velocity_series = velocity_time_series(H, D, L, ks, rho, μ, delta_t)

    for i, V in enumerate(velocity_series):
        if abs(V - Vs) / Vs < eps and i > 0:
            Ts = time_array[i]
            VTs = V
            print(f"L={L}m | Ts={Ts:.2f} s | V(Ts)={VTs:.4f} m/s | Vs={Vs:.4f} m/s")
            break

    plt.plot(time_array, velocity_series, label=f'L={L} m')
plt.axhline(y=Vs, color='r', linestyle='--', label='Steady-State Velocity')
plt.title('Velocity Time Series for Different Pipe Lengths')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid()
plt.show()

