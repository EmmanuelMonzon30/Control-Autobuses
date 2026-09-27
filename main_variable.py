import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# IMPORTAR MÓDULOS DEL PROYECTO
# =========================================================
from Modelo import (
    g,
    c_r,
    aerodynamic_drag,
    torque_brake_from_u,
    motor_speed,
    euler_step,
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

from Modelo_variable import (
    generate_passenger_profile,
    print_passenger_profile,
    build_mass_profile_along_route,
    bus_dynamics_open_loop_variable_mass,
    mass_at_position,
)


# =========================================================
# CONFIGURACIÓN DE PASAJEROS
# =========================================================
# seed=None hace que el perfil cambie en cada ejecución.
PASSENGER_SEED = None

PASSENGERS_MIN = 5
PASSENGERS_MAX = 80
INITIAL_PASSENGERS = None
MAX_CHANGE_PER_STOP = 40


# =========================================================
# CONSTRUIR REFERENCIA DE CONDUCCIÓN
# =========================================================
print_profile_summary()
print_pi_summary()

t_ref, s_ref, v_ref, a_ref = build_full_cycle_profile(dt=DT)

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

s_mass_grid, passengers_grid, mass_grid = (
    build_mass_profile_along_route(passenger_counts)
)


# =========================================================
# FUERZAS DE REFERENCIA CON MASA VARIABLE
# =========================================================
m_ref_var = np.array([
    mass_at_position(s_k, passenger_counts)
    for s_k in s_ref
])

Fd_ref = np.array([
    aerodynamic_drag(v_k)
    for v_k in v_ref
])

Fr_ref = (
    m_ref_var
    * g
    * (
        np.sin(theta(s_ref))
        + c_r * np.cos(theta(s_ref))
    )
)

# Fuerza longitudinal neta requerida para reproducir
# la aceleración de referencia:
#
# F_u,ref = m*a_ref + F_d + F_r
Fu_ref = m_ref_var * a_ref + Fd_ref + Fr_ref

# Distribuir la fuerza longitudinal entre los actuadores.
# torque_brake_from_u() considera:
# - límite máximo de par;
# - límite máximo de potencia;
# - límite máximo de frenado;
# - exclusión mutua entre tracción y frenado.
#
# Cada fila contiene:
# [Fuerza motriz, fuerza de frenado, par del motor]
actuator_ref = np.array([
    torque_brake_from_u(Fu_k, v_k)
    for Fu_k, v_k in zip(Fu_ref, v_ref)
], dtype=float)

Fm_ref = actuator_ref[:, 0]
Fb_ref = actuator_ref[:, 1]
Tm_ref = actuator_ref[:, 2]

# Velocidad angular del motor asociada con la referencia
omega_m_ref = np.array([
    motor_speed(v_k)
    for v_k in v_ref
])


# =========================================================
# SIMULACIÓN: CONTROL PI CON PLANTA DE MASA VARIABLE
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
Fu_hist_pi = np.zeros(N)

grade_hist_pi = np.zeros(N)
elev_hist_pi = np.zeros(N)
mass_hist_pi = np.zeros(N)
pass_hist_pi = np.zeros(N)

z_hist_pi = np.zeros(N)
e_v_hist_pi = np.zeros(N)
Fr_hat_hist_pi = np.zeros(N)
Fd_hat_hist_pi = np.zeros(N)


for k in range(N - 1):

    # -----------------------------------------------------
    # Controlador PI nominal
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
    dx, Fm_k, Fd_k, Fr_k, m_k, n_pass_k = (
        bus_dynamics_open_loop_variable_mass(
            x_pi[k, :],
            Tm_k,
            Fb_k,
            theta,
            passenger_counts,
        )
    )

    # -----------------------------------------------------
    # Integración mediante Euler explícito
    # -----------------------------------------------------
    x_pi[k + 1, :] = euler_step(
        x_pi[k, :],
        dx,
        dt,
    )

    # -----------------------------------------------------
    # Guardar variables
    # -----------------------------------------------------
    Tm_hist_pi[k] = Tm_k
    Fb_hist_pi[k] = Fb_k
    Fm_hist_pi[k] = Fm_k
    Fd_hist_pi[k] = Fd_k
    Fr_hist_pi[k] = Fr_k

    wm_hist_pi[k] = motor_speed(x_pi[k, 1])
    Fu_hist_pi[k] = Fm_k - Fb_k

    grade_hist_pi[k] = grade(x_pi[k, 0])
    elev_hist_pi[k] = elevation(x_pi[k, 0])

    mass_hist_pi[k] = m_k
    pass_hist_pi[k] = n_pass_k

    z_hist_pi[k] = z_int
    e_v_hist_pi[k] = ctrl_info["e_v"]
    Fr_hat_hist_pi[k] = ctrl_info["Fr_hat"]
    Fd_hat_hist_pi[k] = ctrl_info["Fd_hat"]

    z_int = z_next


# =========================================================
# ÚLTIMO VALOR
# =========================================================
Tm_hist_pi[-1] = Tm_hist_pi[-2]
Fb_hist_pi[-1] = Fb_hist_pi[-2]

dx, Fm_k, Fd_k, Fr_k, m_k, n_pass_k = (
    bus_dynamics_open_loop_variable_mass(
        x_pi[-1, :],
        Tm_hist_pi[-1],
        Fb_hist_pi[-1],
        theta,
        passenger_counts,
    )
)

Fm_hist_pi[-1] = Fm_k
Fd_hist_pi[-1] = Fd_k
Fr_hist_pi[-1] = Fr_k

wm_hist_pi[-1] = motor_speed(x_pi[-1, 1])
Fu_hist_pi[-1] = Fm_hist_pi[-1] - Fb_hist_pi[-1]

grade_hist_pi[-1] = grade(x_pi[-1, 0])
elev_hist_pi[-1] = elevation(x_pi[-1, 0])

mass_hist_pi[-1] = m_k
pass_hist_pi[-1] = n_pass_k

z_hist_pi[-1] = z_int
e_v_hist_pi[-1] = v_ref[-1] - x_pi[-1, 1]


# =========================================================
# VARIABLES DE RESPUESTA
# =========================================================
s_pi = x_pi[:, 0]
v_pi = x_pi[:, 1]


# =========================================================
# ERRORES Y MÉTRICAS
# =========================================================
e_s_pi = s_ref - s_pi
e_v_pi = v_ref - v_pi

IAE_pi = np.trapezoid(
    np.abs(e_v_pi),
    t_ref,
)

ISE_pi = np.trapezoid(
    e_v_pi**2,
    t_ref,
)

RMSE_pi = np.sqrt(
    np.mean(e_v_pi**2)
)

MAX_ERROR_pi = np.max(
    np.abs(e_v_pi)
)

print("\n========================================")
print("MÉTRICAS DE SEGUIMIENTO EN VELOCIDAD")
print("CONTROL PI CON MASA VARIABLE")
print("========================================")
print(f"IAE = {IAE_pi:.4f} m")
print(f"ISE = {ISE_pi:.4f} m²/s")
print(f"RMSE = {RMSE_pi:.4f} m/s")
print(f"Error máximo = {MAX_ERROR_pi:.4f} m/s")
print(f"Error máximo = {MAX_ERROR_pi * 3.6:.4f} km/h")


# =========================================================
# GRÁFICAS
# =========================================================

# 1) Pasajeros en función de la posición
plt.figure(figsize=(12, 5))
plt.step(
    s_mass_grid,
    passengers_grid,
    where="post",
    label="Pasajeros",
)
plt.xlabel("Posición s [m]")
plt.ylabel("Número de pasajeros")
plt.title("Perfil aleatorio de pasajeros por tramo")
plt.legend()
plt.grid(True)


# 2) Masa en función de la posición
plt.figure(figsize=(12, 5))
plt.step(
    s_mass_grid,
    mass_grid,
    where="post",
    label="Masa total",
)
plt.xlabel("Posición s [m]")
plt.ylabel("Masa [kg]")
plt.title("Masa variable del autobús")
plt.legend()
plt.grid(True)


# 3) Seguimiento de velocidad
plt.figure(figsize=(12, 5))
plt.plot(
    t_ref,
    v_ref * 3.6,
    "--",
    label="v_ref [km/h]",
)
plt.plot(
    t_ref,
    v_pi * 3.6,
    label="v_PI con masa variable [km/h]",
)
plt.xlabel("Tiempo [s]")
plt.ylabel("Velocidad [km/h]")
plt.title("Seguimiento de velocidad con PI y masa variable")
plt.legend()
plt.grid(True)


# 4) Error de velocidad
plt.figure(figsize=(12, 5))
plt.plot(
    t_ref,
    e_v_pi * 3.6,
    label="e_v PI [km/h]",
)
plt.xlabel("Tiempo [s]")
plt.ylabel("Error de velocidad [km/h]")
plt.title("Error de seguimiento con PI y masa variable")
plt.legend()
plt.grid(True)


# 5) Velocidad respecto de la distancia
plt.figure(figsize=(12, 5))
plt.plot(
    s_ref,
    v_ref * 3.6,
    "--",
    label="v_ref(s) [km/h]",
)
plt.plot(
    s_pi,
    v_pi * 3.6,
    label="v_PI(s) [km/h]",
)
plt.xlabel("Distancia [m]")
plt.ylabel("Velocidad [km/h]")
plt.title("Velocidad vs distancia con PI y masa variable")
plt.legend()
plt.grid(True)


# 6) Par del motor
plt.figure(figsize=(12, 5))
plt.plot(
    t_ref,
    Tm_ref,
    "--",
    label="T_m de referencia [N·m]",
)
plt.plot(
    t_ref,
    Tm_hist_pi,
    label="T_m PI [N·m]",
)
plt.xlabel("Tiempo [s]")
plt.ylabel("Torque [N·m]")
plt.title("Torque del motor con PI y masa variable")
plt.legend()
plt.grid(True)


# 7) Fuerza de frenado
plt.figure(figsize=(12, 5))
plt.plot(
    t_ref,
    Fb_ref,
    "--",
    label="F_b de referencia [N]",
)
plt.plot(
    t_ref,
    Fb_hist_pi,
    label="F_b PI [N]",
)
plt.xlabel("Tiempo [s]")
plt.ylabel("Fuerza de frenado [N]")
plt.title("Frenado con PI y masa variable")
plt.legend()
plt.grid(True)


# 8) Masa en función del tiempo
plt.figure(figsize=(12, 5))
plt.plot(
    t_ref,
    mass_hist_pi,
    label="Masa total [kg]",
)
plt.xlabel("Tiempo [s]")
plt.ylabel("Masa [kg]")
plt.title("Masa variable durante la simulación PI")
plt.legend()
plt.grid(True)


# 9) Pasajeros en función del tiempo
plt.figure(figsize=(12, 5))
plt.plot(
    t_ref,
    pass_hist_pi,
    label="Pasajeros",
)
plt.xlabel("Tiempo [s]")
plt.ylabel("Número de pasajeros")
plt.title("Pasajeros durante la simulación PI")
plt.legend()
plt.grid(True)


# 10) Fuerzas longitudinales
plt.figure(figsize=(12, 5))
plt.plot(
    t_ref,
    Fm_hist_pi,
    label="F_m PI [N]",
)
plt.plot(
    t_ref,
    Fb_hist_pi,
    label="F_b PI [N]",
)
plt.plot(
    t_ref,
    Fd_hist_pi,
    label="F_d PI [N]",
)
plt.plot(
    t_ref,
    Fr_hist_pi,
    label="F_r PI [N]",
)
plt.xlabel("Tiempo [s]")
plt.ylabel("Fuerza [N]")
plt.title("Fuerzas longitudinales con PI y masa variable")
plt.legend()
plt.grid(True)


# 11) Estado integral
plt.figure(figsize=(12, 5))
plt.plot(
    t_ref,
    z_hist_pi,
    label="z_int PI",
)
plt.xlabel("Tiempo [s]")
plt.ylabel("Estado integral")
plt.title("Estado integral del controlador PI con masa variable")
plt.legend()
plt.grid(True)


# 12) Paradas sobre el perfil espacial
stop_positions = get_stop_positions()
stop_names = get_stop_names()

plt.figure(figsize=(12, 5))

plt.plot(
    s_ref,
    v_ref * 3.6,
    "--",
    label="v_ref(s) [km/h]",
)

plt.plot(
    s_pi,
    v_pi * 3.6,
    label="v_PI(s) [km/h]",
)

for i, stop_pos in enumerate(stop_positions[:-1]):
    plt.axvline(
        stop_pos,
        linestyle="--",
    )

    plt.text(
        stop_pos,
        2,
        stop_names[i],
        rotation=90,
        fontsize=8,
    )

plt.xlabel("Distancia [m]")
plt.ylabel("Velocidad [km/h]")
plt.title("PI con masa variable sobre perfil de paradas")
plt.legend()
plt.grid(True)

plt.show()