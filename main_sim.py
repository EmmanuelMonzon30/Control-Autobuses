import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# IMPORTAR MÓDULOS DEL PROYECTO
# =========================================================
from Modelo import (
    m,
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
    bus_dynamics_open_loop,
    euler_step,
)

from Ruta9 import (
    theta,
    elevation,
    grade,
    route_length,
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

# =========================================================
# FUERZAS RESISTIVAS SOBRE LA REFERENCIA
# =========================================================
Fd_ref = np.array([aerodynamic_drag(vk) for vk in v_ref])
Fr_ref = m * g * (np.sin(theta(s_ref)) + c_r * np.cos(theta(s_ref)))

# Fuerza longitudinal neta requerida:
# u = m*a_ref + Fd + Fr
u_ref = m * a_ref + Fd_ref + Fr_ref

# =========================================================
# SEPARAR EN TRACCIÓN Y FRENADO
# =========================================================
Fm_ref = np.maximum(u_ref, 0.0)
Fb_ref = np.maximum(-u_ref, 0.0)
Fb_ref = np.clip(Fb_ref, 0.0, F_b_max)

# Reconstrucción de torque:
# Tm = (r_w * eta_t / M_f) * Fm
Tm_ref = (r_w * eta_t / M_f) * Fm_ref
Tm_ref = np.clip(Tm_ref, 0.0, T_m_max)

# Recalcular Fm por si el torque quedó saturado
Fm_ref = np.array([traction_force(Tm_k) for Tm_k in Tm_ref])

# Velocidad angular del motor asociada a la referencia
omega_m_ref = np.array([motor_speed(vk) for vk in v_ref])

# =========================================================
# SIMULACIÓN 1: PLANTA REAL EN LAZO ABIERTO (ORIGINAL)
# =========================================================
N = len(t_ref)
dt = DT

x_ol = np.zeros((N, 2))
x_ol[0, :] = [0.0, 0.0]   # s(0)=0, v(0)=0

Fm_hist_ol = np.zeros(N)
Fb_hist_ol = np.zeros(N)
Fd_hist_ol = np.zeros(N)
Fr_hist_ol = np.zeros(N)
Tm_hist_ol = np.zeros(N)
wm_hist_ol = np.zeros(N)
u_hist_ol = np.zeros(N)
grade_hist_ol = np.zeros(N)
elev_hist_ol = np.zeros(N)

for k in range(N - 1):
    Tm_k = Tm_ref[k]
    Fb_k = Fb_ref[k]

    dx, Fm_k, Fd_k, Fr_k = bus_dynamics_open_loop(x_ol[k, :], Tm_k, Fb_k, theta)

    # Integración Euler
    x_ol[k + 1, :] = euler_step(x_ol[k, :], dx, dt)

    # Guardar variables
    Tm_hist_ol[k] = Tm_k
    Fb_hist_ol[k] = Fb_k
    Fm_hist_ol[k] = Fm_k
    Fd_hist_ol[k] = Fd_k
    Fr_hist_ol[k] = Fr_k
    wm_hist_ol[k] = motor_speed(x_ol[k, 1])
    u_hist_ol[k] = Fm_k - Fb_k
    grade_hist_ol[k] = grade(x_ol[k, 0])
    elev_hist_ol[k] = elevation(x_ol[k, 0])

# Último valor
Tm_hist_ol[-1] = Tm_ref[-1]
Fb_hist_ol[-1] = Fb_ref[-1]
Fm_hist_ol[-1] = traction_force(Tm_hist_ol[-1])
Fd_hist_ol[-1] = aerodynamic_drag(x_ol[-1, 1])
Fr_hist_ol[-1] = m * g * (np.sin(theta(x_ol[-1, 0])) + c_r * np.cos(theta(x_ol[-1, 0])))
wm_hist_ol[-1] = motor_speed(x_ol[-1, 1])
u_hist_ol[-1] = Fm_hist_ol[-1] - Fb_hist_ol[-1]
grade_hist_ol[-1] = grade(x_ol[-1, 0])
elev_hist_ol[-1] = elevation(x_ol[-1, 0])

s_ol = x_ol[:, 0]
v_ol = x_ol[:, 1]

# =========================================================
# SIMULACIÓN 2: PLANTA REAL CON CONTROL PI
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

# Variables extra del PI
z_hist_pi = np.zeros(N)
e_v_hist_pi = np.zeros(N)
Fr_hat_hist_pi = np.zeros(N)
Fd_hat_hist_pi = np.zeros(N)

for k in range(N - 1):
    # Controlador PI
    Tm_k, Fb_k, z_next, ctrl_info = pi_control_step(
        x=x_pi[k, :],
        z_int=z_int,
        v_ref_k=v_ref[k],
        dt=dt,
        theta_func=theta,
    )

    dx, Fm_k, Fd_k, Fr_k = bus_dynamics_open_loop(x_pi[k, :], Tm_k, Fb_k, theta)

    # Integración Euler
    x_pi[k + 1, :] = euler_step(x_pi[k, :], dx, dt)

    # Guardar variables
    Tm_hist_pi[k] = Tm_k
    Fb_hist_pi[k] = Fb_k
    Fm_hist_pi[k] = Fm_k
    Fd_hist_pi[k] = Fd_k
    Fr_hist_pi[k] = Fr_k
    wm_hist_pi[k] = motor_speed(x_pi[k, 1])
    u_hist_pi[k] = Fm_k - Fb_k
    grade_hist_pi[k] = grade(x_pi[k, 0])
    elev_hist_pi[k] = elevation(x_pi[k, 0])

    z_hist_pi[k] = z_int
    e_v_hist_pi[k] = ctrl_info["e_v"]
    Fr_hat_hist_pi[k] = ctrl_info["Fr_hat"]
    Fd_hat_hist_pi[k] = ctrl_info["Fd_hat"]

    z_int = z_next

# Último valor
Tm_hist_pi[-1] = Tm_hist_pi[-2]
Fb_hist_pi[-1] = Fb_hist_pi[-2]
Fm_hist_pi[-1] = traction_force(Tm_hist_pi[-1])
Fd_hist_pi[-1] = aerodynamic_drag(x_pi[-1, 1])
Fr_hist_pi[-1] = m * g * (np.sin(theta(x_pi[-1, 0])) + c_r * np.cos(theta(x_pi[-1, 0])))
wm_hist_pi[-1] = motor_speed(x_pi[-1, 1])
u_hist_pi[-1] = Fm_hist_pi[-1] - Fb_hist_pi[-1]
grade_hist_pi[-1] = grade(x_pi[-1, 0])
elev_hist_pi[-1] = elevation(x_pi[-1, 0])
z_hist_pi[-1] = z_int
e_v_hist_pi[-1] = v_ref[-1] - x_pi[-1, 1]

s_pi = x_pi[:, 0]
v_pi = x_pi[:, 1]

# =========================================================
# ERRORES DE SEGUIMIENTO
# =========================================================
e_s_ol = s_ref - s_ol
e_v_ol = v_ref - v_ol

e_s_pi = s_ref - s_pi
e_v_pi = v_ref - v_pi

# =========================================================
# MÉTRICAS
# =========================================================
IAE_ol = np.trapezoid(np.abs(e_v_ol), t_ref)
ISE_ol = np.trapezoid(e_v_ol**2, t_ref)

IAE_pi = np.trapezoid(np.abs(e_v_pi), t_ref)
ISE_pi = np.trapezoid(e_v_pi**2, t_ref)

print("\n========================================")
print("MÉTRICAS DE SEGUIMIENTO EN VELOCIDAD")
print("========================================")
print(f"Lazo abierto -> IAE = {IAE_ol:.4f}, ISE = {ISE_ol:.4f}")
print(f"Control PI   -> IAE = {IAE_pi:.4f}, ISE = {ISE_pi:.4f}")

# =========================================================
# GRÁFICAS ORIGINALES (LAZO ABIERTO)
# =========================================================

# 1) Posición: referencia vs real (lazo abierto)
plt.figure(figsize=(12, 5))
plt.plot(t_ref, s_ref, '--', label='s_ref [m]')
plt.plot(t_ref, s_ol, label='s_open_loop [m]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Posición [m]")
plt.title("Posición: referencia vs respuesta (lazo abierto)")
plt.legend()
plt.grid(True)

# 2) Velocidad: referencia vs real (lazo abierto)
plt.figure(figsize=(12, 5))
plt.plot(t_ref, v_ref * 3.6, '--', label='v_ref [km/h]')
plt.plot(t_ref, v_ol * 3.6, label='v_open_loop [km/h]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Velocidad [km/h]")
plt.title("Velocidad: referencia vs respuesta (lazo abierto)")
plt.legend()
plt.grid(True)

# 3) Error de velocidad (lazo abierto)
plt.figure(figsize=(12, 5))
plt.plot(t_ref, e_v_ol * 3.6, label='e_v open_loop [km/h]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Error de velocidad [km/h]")
plt.title("Error de seguimiento en velocidad (lazo abierto)")
plt.legend()
plt.grid(True)

# =========================================================
# NUEVAS GRÁFICAS DE COMPARACIÓN
# =========================================================

# 4) Comparación de velocidad
plt.figure(figsize=(12, 5))
plt.plot(t_ref, v_ref * 3.6, '--', label='v_ref [km/h]')
plt.plot(t_ref, v_ol * 3.6, label='v_open_loop [km/h]')
plt.plot(t_ref, v_pi * 3.6, label='v_PI [km/h]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Velocidad [km/h]")
plt.title("Comparación del seguimiento de velocidad")
plt.legend()
plt.grid(True)

# 5) Comparación del error de velocidad
plt.figure(figsize=(12, 5))
plt.plot(t_ref, e_v_ol * 3.6, label='e_v open_loop [km/h]')
plt.plot(t_ref, e_v_pi * 3.6, label='e_v PI [km/h]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Error de velocidad [km/h]")
plt.title("Comparación del error de seguimiento")
plt.legend()
plt.grid(True)

# 6) Velocidad vs distancia
plt.figure(figsize=(12, 5))
plt.plot(s_ref, v_ref * 3.6, '--', label='v_ref(s) [km/h]')
plt.plot(s_ol, v_ol * 3.6, label='v_open_loop(s) [km/h]')
plt.plot(s_pi, v_pi * 3.6, label='v_PI(s) [km/h]')
plt.xlabel("Distancia [m]")
plt.ylabel("Velocidad [km/h]")
plt.title("Comparación de velocidad vs distancia")
plt.legend()
plt.grid(True)

# 7) Comparación de torque
plt.figure(figsize=(12, 5))
plt.plot(t_ref, Tm_ref, '--', label='T_m ref [N·m]')
plt.plot(t_ref, Tm_hist_ol, label='T_m open_loop [N·m]')
plt.plot(t_ref, Tm_hist_pi, label='T_m PI [N·m]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Torque [N·m]")
plt.title("Comparación del torque del motor")
plt.legend()
plt.grid(True)

# 8) Comparación de frenado
plt.figure(figsize=(12, 5))
plt.plot(t_ref, Fb_ref, '--', label='F_b ref [N]')
plt.plot(t_ref, Fb_hist_ol, label='F_b open_loop [N]')
plt.plot(t_ref, Fb_hist_pi, label='F_b PI [N]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Fuerza de frenado [N]")
plt.title("Comparación de frenado")
plt.legend()
plt.grid(True)

# 9) Comparación de posición
plt.figure(figsize=(12, 5))
plt.plot(t_ref, s_ref, '--', label='s_ref [m]')
plt.plot(t_ref, s_ol, label='s_open_loop [m]')
plt.plot(t_ref, s_pi, label='s_PI [m]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Posición [m]")
plt.title("Comparación de posición")
plt.legend()
plt.grid(True)

# 10) Fuerzas longitudinales con PI
plt.figure(figsize=(12, 5))
plt.plot(t_ref, Fm_hist_pi, label='F_m PI [N]')
plt.plot(t_ref, Fb_hist_pi, label='F_b PI [N]')
plt.plot(t_ref, Fd_hist_pi, label='F_d PI [N]')
plt.plot(t_ref, Fr_hist_pi, label='F_r PI [N]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Fuerza [N]")
plt.title("Fuerzas longitudinales con controlador PI")
plt.legend()
plt.grid(True)

# 11) Estado integral del PI
plt.figure(figsize=(12, 5))
plt.plot(t_ref, z_hist_pi, label='z_int')
plt.xlabel("Tiempo [s]")
plt.ylabel("Estado integral")
plt.title("Estado integral del controlador PI")
plt.legend()
plt.grid(True)

# 12) Paradas sobre la distancia
stop_positions = get_stop_positions()
stop_names = get_stop_names()

plt.figure(figsize=(12, 5))
plt.plot(s_ref, v_ref * 3.6, '--', label='v_ref(s) [km/h]')
plt.plot(s_ol, v_ol * 3.6, label='v_open_loop(s) [km/h]')
plt.plot(s_pi, v_pi * 3.6, label='v_PI(s) [km/h]')
for i, stop_pos in enumerate(stop_positions[:-1]):
    plt.axvline(stop_pos, linestyle='--')
    plt.text(stop_pos, 2, stop_names[i], rotation=90, fontsize=8)
plt.xlabel("Distancia [m]")
plt.ylabel("Velocidad [km/h]")
plt.title("Comparación sobre el perfil de paradas")
plt.legend()
plt.grid(True)

plt.show()