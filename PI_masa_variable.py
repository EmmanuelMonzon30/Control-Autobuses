import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# IMPORTAR MÓDULOS DEL PROYECTO
# =========================================================
from Modelo import (
    traction_force,
    aerodynamic_drag,
    motor_speed,
    euler_step,
)

from Modelo_variable import (
    generate_passenger_profile,
    bus_dynamics_open_loop_variable_mass,
    build_mass_profile_along_route,
    print_passenger_profile,
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
    print_profile_summary,
)

from PI_control import (
    pi_control_step,
    print_pi_summary,
)


# =========================================================
# CONSTRUIR REFERENCIA DE CONDUCCIÓN
# =========================================================
print_profile_summary()
print_pi_summary()

t_ref, s_ref, v_ref, a_ref = build_full_cycle_profile(dt=DT)

N = len(t_ref)
dt = DT


# =========================================================
# GENERAR PERFIL ALEATORIO DE PASAJEROS / MASA
# =========================================================
passenger_counts = generate_passenger_profile(
    seed=None,              # Cambia a un número fijo si quieres repetir resultados
    n_min=5,
    n_max=80,
    initial_passengers=None,
    max_change_per_stop=18,
)

print_passenger_profile(passenger_counts)


# =========================================================
# SIMULACIÓN: PLANTA CON MASA VARIABLE + CONTROL PI
# =========================================================
x_pi = np.zeros((N, 2))
x_pi[0, :] = [0.0, 0.0]

z_int = 0.0

Fm_hist_pi = np.zeros(N)
Fb_hist_pi = np.zeros(N)
Fd_hist_pi = np.zeros(N)
Fr_hist_pi = np.zeros(N)
Tm_hist_pi = np.zeros(N)
wm_hist_pi = np.zeros(N)
u_hist_pi = np.zeros(N)
grade_hist_pi = np.zeros(N)
elev_hist_pi = np.zeros(N)

mass_hist_pi = np.zeros(N)
pass_hist_pi = np.zeros(N)

z_hist_pi = np.zeros(N)
e_v_hist_pi = np.zeros(N)
Fr_hat_hist_pi = np.zeros(N)
Fd_hat_hist_pi = np.zeros(N)
u_cmd_hist_pi = np.zeros(N)

for k in range(N - 1):

    # -----------------------------------------------------
    # Controlador PI
    # -----------------------------------------------------
    Tm_k, Fb_k, z_next, ctrl_info = pi_control_step(
        x=x_pi[k, :],
        z_int=z_int,
        v_ref_k=v_ref[k],
        dt=dt,
        theta_func=theta,
    )

    # -----------------------------------------------------
    # Planta real con masa variable
    # -----------------------------------------------------
    dx, Fm_k, Fd_k, Fr_k, m_k, n_pass_k = bus_dynamics_open_loop_variable_mass(
        x=x_pi[k, :],
        Tm=Tm_k,
        Fb=Fb_k,
        theta_func=theta,
        passenger_counts=passenger_counts,
    )

    # -----------------------------------------------------
    # Integración Euler
    # -----------------------------------------------------
    x_pi[k + 1, :] = euler_step(x_pi[k, :], dx, dt)

    # -----------------------------------------------------
    # Guardar variables
    # -----------------------------------------------------
    Tm_hist_pi[k] = Tm_k
    Fb_hist_pi[k] = Fb_k
    Fm_hist_pi[k] = Fm_k
    Fd_hist_pi[k] = Fd_k
    Fr_hist_pi[k] = Fr_k
    wm_hist_pi[k] = motor_speed(x_pi[k, 1])
    u_hist_pi[k] = Fm_k - Fb_k

    grade_hist_pi[k] = grade(x_pi[k, 0])
    elev_hist_pi[k] = elevation(x_pi[k, 0])

    mass_hist_pi[k] = m_k
    pass_hist_pi[k] = n_pass_k

    z_hist_pi[k] = z_int
    e_v_hist_pi[k] = ctrl_info["e_v"]
    Fr_hat_hist_pi[k] = ctrl_info["Fr_hat"]
    Fd_hat_hist_pi[k] = ctrl_info["Fd_hat"]
    u_cmd_hist_pi[k] = ctrl_info["u_cmd"]

    z_int = z_next


# =========================================================
# ÚLTIMO VALOR
# =========================================================
Tm_hist_pi[-1] = Tm_hist_pi[-2]
Fb_hist_pi[-1] = Fb_hist_pi[-2]
Fm_hist_pi[-1] = traction_force(Tm_hist_pi[-1])
Fd_hist_pi[-1] = aerodynamic_drag(x_pi[-1, 1])
Fr_hist_pi[-1] = Fr_hist_pi[-2]
wm_hist_pi[-1] = motor_speed(x_pi[-1, 1])
u_hist_pi[-1] = Fm_hist_pi[-1] - Fb_hist_pi[-1]

grade_hist_pi[-1] = grade(x_pi[-1, 0])
elev_hist_pi[-1] = elevation(x_pi[-1, 0])

mass_hist_pi[-1] = mass_hist_pi[-2]
pass_hist_pi[-1] = pass_hist_pi[-2]

z_hist_pi[-1] = z_int
e_v_hist_pi[-1] = v_ref[-1] - x_pi[-1, 1]
Fr_hat_hist_pi[-1] = Fr_hat_hist_pi[-2]
Fd_hat_hist_pi[-1] = Fd_hat_hist_pi[-2]
u_cmd_hist_pi[-1] = u_cmd_hist_pi[-2]


# =========================================================
# VARIABLES DE RESPUESTA
# =========================================================
s_pi = x_pi[:, 0]
v_pi = x_pi[:, 1]

e_s_pi = s_ref - s_pi
e_v_pi = v_ref - v_pi


# =========================================================
# MÉTRICAS
# =========================================================
IAE_pi = np.trapezoid(np.abs(e_v_pi), t_ref)
ISE_pi = np.trapezoid(e_v_pi**2, t_ref)
RMSE_pi = np.sqrt(np.mean(e_v_pi**2))
MAX_ERROR_pi = np.max(np.abs(e_v_pi))

print("\n========================================")
print("MÉTRICAS DE SEGUIMIENTO CON PI Y MASA VARIABLE")
print("========================================")
print(f"IAE  = {IAE_pi:.4f}")
print(f"ISE  = {ISE_pi:.4f}")
print(f"RMSE = {RMSE_pi:.4f} m/s")
print(f"Error máximo = {MAX_ERROR_pi:.4f} m/s")
print(f"Error máximo = {MAX_ERROR_pi*3.6:.4f} km/h")


# =========================================================
# PERFIL DE MASA PARA GRAFICAR EN DISTANCIA
# =========================================================
s_mass_grid, passengers_grid, mass_grid = build_mass_profile_along_route(
    passenger_counts,
    n_points=1000,
)


# =========================================================
# GRÁFICAS
# =========================================================

# 1) Velocidad de referencia vs PI
plt.figure(figsize=(12, 5))
plt.plot(t_ref, v_ref * 3.6, '--', label='v_ref [km/h]')
plt.plot(t_ref, v_pi * 3.6, label='v_PI masa variable [km/h]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Velocidad [km/h]")
plt.title("Seguimiento de velocidad con PI y masa variable")
plt.legend()
plt.grid(True)

# 2) Error de velocidad
plt.figure(figsize=(12, 5))
plt.plot(t_ref, e_v_pi * 3.6, label='e_v PI [km/h]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Error de velocidad [km/h]")
plt.title("Error de seguimiento con PI y masa variable")
plt.legend()
plt.grid(True)

# 3) Velocidad vs distancia
plt.figure(figsize=(12, 5))
plt.plot(s_ref, v_ref * 3.6, '--', label='v_ref(s) [km/h]')
plt.plot(s_pi, v_pi * 3.6, label='v_PI(s) masa variable [km/h]')
plt.xlabel("Distancia [m]")
plt.ylabel("Velocidad [km/h]")
plt.title("Velocidad vs distancia con PI y masa variable")
plt.legend()
plt.grid(True)

# 4) Masa en el tiempo
plt.figure(figsize=(12, 5))
plt.step(t_ref, mass_hist_pi, where='post', label='Masa total [kg]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Masa [kg]")
plt.title("Masa variable durante la simulación")
plt.legend()
plt.grid(True)

# 5) Pasajeros en el tiempo
plt.figure(figsize=(12, 5))
plt.step(t_ref, pass_hist_pi, where='post', label='Pasajeros')
plt.xlabel("Tiempo [s]")
plt.ylabel("Número de pasajeros")
plt.title("Pasajeros durante la simulación")
plt.legend()
plt.grid(True)

# 6) Masa vs distancia
plt.figure(figsize=(12, 5))
plt.step(s_mass_grid, mass_grid, where='post', label='Masa total [kg]')
plt.xlabel("Distancia [m]")
plt.ylabel("Masa [kg]")
plt.title("Perfil de masa a lo largo de la ruta")
plt.legend()
plt.grid(True)

# 7) Torque del motor
plt.figure(figsize=(12, 5))
plt.plot(t_ref, Tm_hist_pi, label='T_m PI [N·m]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Torque [N·m]")
plt.title("Torque del motor con PI y masa variable")
plt.legend()
plt.grid(True)

# 8) Fuerza de frenado
plt.figure(figsize=(12, 5))
plt.plot(t_ref, Fb_hist_pi, label='F_b PI [N]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Fuerza de frenado [N]")
plt.title("Frenado con PI y masa variable")
plt.legend()
plt.grid(True)

# 9) Fuerzas longitudinales
plt.figure(figsize=(12, 5))
plt.plot(t_ref, Fm_hist_pi, label='F_m [N]')
plt.plot(t_ref, Fb_hist_pi, label='F_b [N]')
plt.plot(t_ref, Fd_hist_pi, label='F_d [N]')
plt.plot(t_ref, Fr_hist_pi, label='F_r [N]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Fuerza [N]")
plt.title("Fuerzas longitudinales con PI y masa variable")
plt.legend()
plt.grid(True)

# 10) Estado integral
plt.figure(figsize=(12, 5))
plt.plot(t_ref, z_hist_pi, label='z_int')
plt.xlabel("Tiempo [s]")
plt.ylabel("Estado integral")
plt.title("Estado integral del controlador PI")
plt.legend()
plt.grid(True)

# 11) Señal de fuerza deseada del PI
plt.figure(figsize=(12, 5))
plt.plot(t_ref, u_cmd_hist_pi, label='u_cmd PI [N]')
plt.plot(t_ref, u_hist_pi, label='u_aplicada = F_m - F_b [N]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Fuerza longitudinal [N]")
plt.title("Fuerza deseada vs fuerza aplicada")
plt.legend()
plt.grid(True)

# 12) Paradas sobre velocidad vs distancia
stop_positions = get_stop_positions()
stop_names = get_stop_names()

plt.figure(figsize=(12, 5))
plt.plot(s_ref, v_ref * 3.6, '--', label='v_ref(s) [km/h]')
plt.plot(s_pi, v_pi * 3.6, label='v_PI(s) masa variable [km/h]')

for i, stop_pos in enumerate(stop_positions[:-1]):
    plt.axvline(stop_pos, linestyle='--')
    plt.text(stop_pos, 2, stop_names[i], rotation=90, fontsize=8)

plt.xlabel("Distancia [m]")
plt.ylabel("Velocidad [km/h]")
plt.title("Seguimiento de velocidad por paradas con masa variable")
plt.legend()
plt.grid(True)

plt.show()