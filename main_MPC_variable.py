import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# IMPORTAR MÓDULOS DEL PROYECTO
# =========================================================
from Modelo import (
    g,
    c_r,
    r_w,
    M_f,
    eta_t,
    F_b_max,
    T_m_max,
    aerodynamic_drag,
    traction_force,
    motor_speed,
    euler_step,
    torque_brake_from_u,
)

from Ruta9 import (
    theta,
    elevation,
    grade,
    get_stop_positions,
    get_stop_names,
)

from driving_profile_9 import (
    DT,
    build_full_cycle_profile,
    build_reference_functions,
    print_profile_summary,
)

from Modelo_variable import (
    generate_passenger_profile,
    print_passenger_profile,
    build_mass_profile_along_route,
    bus_dynamics_open_loop_variable_mass,
    mass_at_position,
)

from MPC_control_sim import MPCController


# =========================================================
# CONFIGURACIÓN DE PASAJEROS
# =========================================================
PASSENGER_SEED = None  # None: cambia en cada ejecución
PASSENGERS_MIN = 5
PASSENGERS_MAX = 70
INITIAL_PASSENGERS = None
MAX_CHANGE_PER_STOP = 18


# =========================================================
# CONFIGURACIÓN MPC NOMINAL
# =========================================================
# El MPC usa su modelo nominal interno.
# La planta simulada usa masa variable.
Ts_mpc = 0.15
Np_mpc = 25

MPC_PARAMS = dict(
    Ts=Ts_mpc,
    Np=Np_mpc,
    v_esc=20.0,
    u_esc=None,
    du_esc=7000.0,
    alpha_v=6.0,
    alpha_u=0.05,
    alpha_du=1.0,
    du_max=9000.0,
)

v_max_const = 35.0 / 3.6  # [m/s]


# =========================================================
# CONSTRUIR REFERENCIA DE CONDUCCIÓN
# =========================================================
print_profile_summary()

t_ref, s_ref, v_ref, a_ref = build_full_cycle_profile(dt=DT)
_, _, _, _, s_of_t, v_of_t, a_of_t = build_reference_functions(dt=DT)

N = len(t_ref)
dt = DT


# =========================================================
# GENERAR PERFIL DE PASAJEROS Y MASA VARIABLE
# =========================================================
passenger_counts = generate_passenger_profile(
    seed=PASSENGER_SEED,
    n_min=PASSENGERS_MIN,
    n_max=PASSENGERS_MAX,
    initial_passengers=INITIAL_PASSENGERS,
    max_change_per_stop=MAX_CHANGE_PER_STOP,
)

print_passenger_profile(passenger_counts)

s_mass_grid, passengers_grid, mass_grid = build_mass_profile_along_route(passenger_counts)


# =========================================================
# FUERZAS DE REFERENCIA CON MASA VARIABLE
# =========================================================
m_ref_var = np.array([
    mass_at_position(s_k, passenger_counts)
    for s_k in s_ref
])

Fd_ref = np.array([aerodynamic_drag(vk) for vk in v_ref])
Fr_ref = m_ref_var * g * (np.sin(theta(s_ref)) + c_r * np.cos(theta(s_ref)))

u_ref = m_ref_var * a_ref + Fd_ref + Fr_ref

Fm_ref = np.maximum(u_ref, 0.0)
Fb_ref = np.maximum(-u_ref, 0.0)
Fb_ref = np.clip(Fb_ref, 0.0, F_b_max)

Tm_ref = (r_w * eta_t / M_f) * Fm_ref
Tm_ref = np.clip(Tm_ref, 0.0, T_m_max)

Fm_ref = np.array([traction_force(Tm_k) for Tm_k in Tm_ref])
omega_m_ref = np.array([motor_speed(vk) for vk in v_ref])


# =========================================================
# SIMULACIÓN: MPC NOMINAL CON PLANTA DE MASA VARIABLE
# =========================================================
x_mpc = np.zeros((N, 2))
x_mpc[0, :] = [0.0, 0.0]

Fm_hist_mpc = np.zeros(N)
Fb_hist_mpc = np.zeros(N)
Fd_hist_mpc = np.zeros(N)
Fr_hist_mpc = np.zeros(N)
Tm_hist_mpc = np.zeros(N)
wm_hist_mpc = np.zeros(N)
u_hist_mpc = np.zeros(N)
grade_hist_mpc = np.zeros(N)
elev_hist_mpc = np.zeros(N)
mass_hist_mpc = np.zeros(N)
pass_hist_mpc = np.zeros(N)

cost_hist_mpc = np.zeros(N)
u_cmd_hist_mpc = np.zeros(N)

mpc = MPCController(**MPC_PARAMS)

u_current = 0.0
u_prev_mpc = 0.0
next_mpc_time = 0.0

for k in range(N - 1):
    tk = t_ref[k]
    xk = x_mpc[k, :]

    if tk >= next_mpc_time - 1e-9:
        future_times = tk + Ts_mpc * np.arange(Np_mpc)

        v_ref_horizon = v_of_t(future_times)
        v_ref_horizon = np.maximum(v_ref_horizon, 0.0)

        v_max_horizon = np.full(Np_mpc, v_max_const)

        u_current, pred = mpc.solve(
            x0_val=xk,
            u_prev_val=u_prev_mpc,
            v_ref_horizon=v_ref_horizon,
            v_max_horizon=v_max_horizon,
        )

        u_prev_mpc = u_current
        next_mpc_time += Ts_mpc
        cost_hist_mpc[k] = pred["cost"]
    else:
        cost_hist_mpc[k] = cost_hist_mpc[k - 1] if k > 0 else 0.0

    Fm_cmd, Fb_cmd, Tm_cmd = torque_brake_from_u(u_current)

    dx, Fm_k, Fd_k, Fr_k, m_k, n_pass_k = bus_dynamics_open_loop_variable_mass(
        xk,
        Tm_cmd,
        Fb_cmd,
        theta,
        passenger_counts,
    )

    x_mpc[k + 1, :] = euler_step(xk, dx, dt)

    Tm_hist_mpc[k] = Tm_cmd
    Fb_hist_mpc[k] = Fb_cmd
    Fm_hist_mpc[k] = Fm_k
    Fd_hist_mpc[k] = Fd_k
    Fr_hist_mpc[k] = Fr_k
    wm_hist_mpc[k] = motor_speed(xk[1])
    u_hist_mpc[k] = Fm_k - Fb_cmd
    u_cmd_hist_mpc[k] = u_current
    grade_hist_mpc[k] = grade(xk[0])
    elev_hist_mpc[k] = elevation(xk[0])
    mass_hist_mpc[k] = m_k
    pass_hist_mpc[k] = n_pass_k

Tm_hist_mpc[-1] = Tm_hist_mpc[-2]
Fb_hist_mpc[-1] = Fb_hist_mpc[-2]
u_cmd_hist_mpc[-1] = u_cmd_hist_mpc[-2]
cost_hist_mpc[-1] = cost_hist_mpc[-2]

dx, Fm_k, Fd_k, Fr_k, m_k, n_pass_k = bus_dynamics_open_loop_variable_mass(
    x_mpc[-1, :],
    Tm_hist_mpc[-1],
    Fb_hist_mpc[-1],
    theta,
    passenger_counts,
)

Fm_hist_mpc[-1] = Fm_k
Fd_hist_mpc[-1] = Fd_k
Fr_hist_mpc[-1] = Fr_k
wm_hist_mpc[-1] = motor_speed(x_mpc[-1, 1])
u_hist_mpc[-1] = Fm_hist_mpc[-1] - Fb_hist_mpc[-1]
grade_hist_mpc[-1] = grade(x_mpc[-1, 0])
elev_hist_mpc[-1] = elevation(x_mpc[-1, 0])
mass_hist_mpc[-1] = m_k
pass_hist_mpc[-1] = n_pass_k

s_mpc = x_mpc[:, 0]
v_mpc = x_mpc[:, 1]


# =========================================================
# ERRORES Y MÉTRICAS
# =========================================================
e_s_mpc = s_ref - s_mpc
e_v_mpc = v_ref - v_mpc

IAE_mpc = np.trapezoid(np.abs(e_v_mpc), t_ref)
ISE_mpc = np.trapezoid(e_v_mpc**2, t_ref)

print("\\n========================================")
print("MÉTRICAS DE SEGUIMIENTO EN VELOCIDAD")
print("MPC NOMINAL CON PLANTA DE MASA VARIABLE")
print("========================================")
print(f"Control MPC -> IAE = {IAE_mpc:.4f}, ISE = {ISE_mpc:.4f}")


# =========================================================
# GRÁFICAS
# =========================================================
plt.figure(figsize=(12, 5))
plt.step(s_mass_grid, passengers_grid, where="post", label="Pasajeros")
plt.xlabel("Posición s [m]")
plt.ylabel("Número de pasajeros")
plt.title("Perfil aleatorio de pasajeros por tramo")
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 5))
plt.step(s_mass_grid, mass_grid, where="post", label="Masa total")
plt.xlabel("Posición s [m]")
plt.ylabel("Masa [kg]")
plt.title("Masa variable del autobús")
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 5))
plt.plot(t_ref, v_ref * 3.6, "--", label="v_ref [km/h]")
plt.plot(t_ref, v_mpc * 3.6, label="v_MPC con masa variable [km/h]")
plt.xlabel("Tiempo [s]")
plt.ylabel("Velocidad [km/h]")
plt.title("Seguimiento de velocidad con MPC nominal y masa variable")
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 5))
plt.plot(t_ref, e_v_mpc * 3.6, label="e_v MPC [km/h]")
plt.xlabel("Tiempo [s]")
plt.ylabel("Error de velocidad [km/h]")
plt.title("Error de seguimiento con MPC nominal y masa variable")
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 5))
plt.plot(s_ref, v_ref * 3.6, "--", label="v_ref(s) [km/h]")
plt.plot(s_mpc, v_mpc * 3.6, label="v_MPC(s) [km/h]")
plt.xlabel("Distancia [m]")
plt.ylabel("Velocidad [km/h]")
plt.title("Velocidad vs distancia con MPC nominal y masa variable")
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 5))
plt.plot(t_ref, Tm_ref, "--", label="T_m ref variable mass [N·m]")
plt.plot(t_ref, Tm_hist_mpc, label="T_m MPC [N·m]")
plt.xlabel("Tiempo [s]")
plt.ylabel("Par [N·m]")
plt.title("Par del motor con MPC nominal y masa variable")
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 5))
plt.plot(t_ref, Fb_ref, "--", label="F_b ref variable mass [N]")
plt.plot(t_ref, Fb_hist_mpc, label="F_b MPC [N]")
plt.xlabel("Tiempo [s]")
plt.ylabel("Fuerza de frenado [N]")
plt.title("Frenado con MPC nominal y masa variable")
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 5))
plt.plot(t_ref, mass_hist_mpc, label="Masa total [kg]")
plt.xlabel("Tiempo [s]")
plt.ylabel("Masa [kg]")
plt.title("Masa variable durante la simulación MPC")
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 5))
plt.plot(t_ref, pass_hist_mpc, label="Pasajeros")
plt.xlabel("Tiempo [s]")
plt.ylabel("Número de pasajeros")
plt.title("Pasajeros durante la simulación MPC")
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 5))
plt.plot(t_ref, Fm_hist_mpc, label="F_m MPC [N]")
plt.plot(t_ref, Fb_hist_mpc, label="F_b MPC [N]")
plt.plot(t_ref, Fd_hist_mpc, label="F_d MPC [N]")
plt.plot(t_ref, Fr_hist_mpc, label="F_r MPC [N]")
plt.xlabel("Tiempo [s]")
plt.ylabel("Fuerza [N]")
plt.title("Fuerzas longitudinales con MPC nominal y masa variable")
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 5))
plt.plot(t_ref, u_cmd_hist_mpc, label="u_cmd MPC [N]")
plt.xlabel("Tiempo [s]")
plt.ylabel("Fuerza neta comandada [N]")
plt.title("Entrada longitudinal comandada por el MPC")
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 5))
plt.plot(t_ref, cost_hist_mpc, label="Costo MPC")
plt.xlabel("Tiempo [s]")
plt.ylabel("Costo")
plt.title("Costo óptimo del MPC durante la simulación")
plt.legend()
plt.grid(True)

stop_positions = get_stop_positions()
stop_names = get_stop_names()

plt.figure(figsize=(12, 5))
plt.plot(s_ref, v_ref * 3.6, "--", label="v_ref(s) [km/h]")
plt.plot(s_mpc, v_mpc * 3.6, label="v_MPC(s) [km/h]")

for i, stop_pos in enumerate(stop_positions[:-1]):
    plt.axvline(stop_pos, linestyle="--")
    plt.text(stop_pos, 2, stop_names[i], rotation=90, fontsize=8)

plt.xlabel("Distancia [m]")
plt.ylabel("Velocidad [km/h]")
plt.title("MPC nominal con masa variable sobre perfil de paradas")
plt.legend()
plt.grid(True)

plt.show()