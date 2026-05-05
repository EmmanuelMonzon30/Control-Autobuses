import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# PARÁMETROS DEL AUTOBÚS
# =========================================================
m = 15000.0          # masa [kg]
rho = 1.225          # densidad del aire [kg/m^3]
A_bus = 8.0          # área frontal [m^2]
c_a = 0.7            # coeficiente aerodinámico [-]
c_r = 0.008          # coeficiente de rodadura [-]
g = 9.81             # gravedad [m/s^2]
r_w = 0.5            # radio de la rueda [m]
M_f = 6.0            # relación final de engrane [-]
eta_t = 0.95         # eficiencia de transmisión [-]

# Cotas físicas
T_m_max = 2500.0     # torque máximo [N·m]
T_m_min = 0.0        # por ahora solo tracción
F_b_max = 30000.0    # frenado máximo [N]
F_b_min = 0.0


# =========================================================
# TIEMPO DE SIMULACIÓN
# =========================================================
dt = 0.1
t0 = 0.0
tf = 60.0
t = np.arange(t0, tf + dt, dt)


# =========================================================
# PERFIL DE RUTA
# =========================================================
def theta(s):
    """
    Ángulo de pendiente de la ruta [rad].
    Primera versión: ruta plana.
    """
    return 0.0


# =========================================================
# PERFIL SUAVE DE VELOCIDAD DESEADA v_d(t)
# =========================================================
def smooth_step(t, t_start, t_end, y0, y1):
    """
    Transición suave tipo coseno entre y0 y y1.
    """
    if t <= t_start:
        return y0
    elif t >= t_end:
        return y1
    else:
        tau = (t - t_start) / (t_end - t_start)
        return y0 + (y1 - y0) * 0.5 * (1 - np.cos(np.pi * tau))


def desired_speed(t):
    """
    Perfil suave de velocidad deseada [m/s].

    Fases:
    1) 0-15 s: arranque suave de 0 a 12 m/s
    2) 15-35 s: crucero a 12 m/s
    3) 35-55 s: frenado suave de 12 a 0 m/s
    4) 55-60 s: reposo
    """
    if t <= 15.0:
        return smooth_step(t, 0.0, 15.0, 0.0, 12.0)
    elif t <= 35.0:
        return 12.0
    elif t <= 55.0:
        return smooth_step(t, 35.0, 55.0, 12.0, 0.0)
    else:
        return 0.0


# =========================================================
# CONSTRUCCIÓN DE TRAYECTORIAS DESEADAS
# =========================================================
v_d = np.array([desired_speed(tk) for tk in t])

# Derivada numérica para obtener a_d(t)
a_d = np.gradient(v_d, dt)

# Integración numérica para obtener s_d(t)
s_d = np.zeros_like(t)
for k in range(len(t) - 1):
    s_d[k + 1] = s_d[k] + v_d[k] * dt


# =========================================================
# FUERZAS RESISTIVAS EQUIVALENTES SOBRE LA TRAYECTORIA DESEADA
# =========================================================
def aerodynamic_drag(v):
    """
    F_d = 0.5 * rho * A_bus * c_a * v^2
    """
    return 0.5 * rho * A_bus * c_a * v**2


def rolling_gravity_force(s):
    """
    F_r = m*g*(sin(theta(s)) + c_r*cos(theta(s)))
    """
    th = theta(s)
    return m * g * (np.sin(th) + c_r * np.cos(th))


Fd_d = np.array([aerodynamic_drag(vk) for vk in v_d])
Fr_d = np.array([rolling_gravity_force(sk) for sk in s_d])


# =========================================================
# ENTRADA LONGITUDINAL NETA REQUERIDA EN LAZO ABIERTO
# =========================================================
# u = m*a_d + F_d + F_r
u = m * a_d + Fd_d + Fr_d

# Separación física: nunca actúan al mismo tiempo
Fm_ref = np.maximum(u, 0.0)
Fb_ref = np.maximum(-u, 0.0)

# Aplicar saturaciones físicas
Fb_ref = np.clip(Fb_ref, F_b_min, F_b_max)

# Torque reconstruido a partir de F_m
# F_m = (M_f / (r_w * eta_t)) * T_m
# => T_m = (r_w * eta_t / M_f) * F_m
Tm_ref = (r_w * eta_t / M_f) * Fm_ref
Tm_ref = np.clip(Tm_ref, T_m_min, T_m_max)

# Recalcular F_m por si el torque quedó saturado
Fm_ref = (M_f / (r_w * eta_t)) * Tm_ref

# Velocidad angular del motor
omega_m_ref = (M_f / r_w) * v_d


# =========================================================
# DINÁMICA REAL DEL AUTOBÚS
# =========================================================
def bus_dynamics_open_loop(x, Tm, Fb):
    """
    Dinámica en lazo abierto con entradas ya prescritas.

    x = [s, v]
    """
    s, v = x

    Fm = (M_f / (r_w * eta_t)) * Tm
    Fd = aerodynamic_drag(v)
    Fr = rolling_gravity_force(s)

    dsdt = v
    dvdt = (Fm - Fb - Fd - Fr) / m

    return np.array([dsdt, dvdt]), Fm, Fd, Fr


# =========================================================
# SIMULACIÓN DE LA PLANTA REAL
# =========================================================
x = np.zeros((len(t), 2))
x[0, :] = [0.0, 0.0]   # s(0)=0, v(0)=0

Fm_hist = np.zeros(len(t))
Fb_hist = np.zeros(len(t))
Fd_hist = np.zeros(len(t))
Fr_hist = np.zeros(len(t))
Tm_hist = np.zeros(len(t))
wm_hist = np.zeros(len(t))
u_hist = np.zeros(len(t))

for k in range(len(t) - 1):
    Tm_k = Tm_ref[k]
    Fb_k = Fb_ref[k]

    dx, Fm_k, Fd_k, Fr_k = bus_dynamics_open_loop(x[k, :], Tm_k, Fb_k)

    # Euler explícito
    x[k + 1, :] = x[k, :] + dt * dx

    # Evitar velocidad negativa numérica
    if x[k + 1, 1] < 0:
        x[k + 1, 1] = 0.0

    Tm_hist[k] = Tm_k
    Fb_hist[k] = Fb_k
    Fm_hist[k] = Fm_k
    Fd_hist[k] = Fd_k
    Fr_hist[k] = Fr_k
    wm_hist[k] = (M_f / r_w) * x[k, 1]
    u_hist[k] = Fm_k - Fb_k

# Último valor
Tm_hist[-1] = Tm_ref[-1]
Fb_hist[-1] = Fb_ref[-1]
Fd_hist[-1] = aerodynamic_drag(x[-1, 1])
Fr_hist[-1] = rolling_gravity_force(x[-1, 0])
Fm_hist[-1] = (M_f / (r_w * eta_t)) * Tm_hist[-1]
wm_hist[-1] = (M_f / r_w) * x[-1, 1]
u_hist[-1] = Fm_hist[-1] - Fb_hist[-1]

s = x[:, 0]
v = x[:, 1]


# =========================================================
# GRÁFICAS
# =========================================================

# 1) Posición: deseada vs real
plt.figure(figsize=(10, 5))
plt.plot(t, s_d, '--', label='s_d [m]')
plt.plot(t, s, label='s [m]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Posición [m]")
plt.title("Posición del autobús")
plt.legend()
plt.grid(True)

# 2) Velocidad: deseada vs real
plt.figure(figsize=(10, 5))
plt.plot(t, v_d, '--', label='v_d [m/s]')
plt.plot(t, v, label='v [m/s]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Velocidad [m/s]")
plt.title("Velocidad del autobús")
plt.legend()
plt.grid(True)

# 3) Aceleración deseada
plt.figure(figsize=(10, 5))
plt.plot(t, a_d, label='a_d [m/s²]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Aceleración [m/s²]")
plt.title("Aceleración deseada")
plt.legend()
plt.grid(True)

# 4) Entrada longitudinal neta
plt.figure(figsize=(10, 5))
plt.plot(t, u, label='u requerida [N]')
plt.plot(t, u_hist, '--', label='u aplicada [N]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Fuerza [N]")
plt.title("Entrada longitudinal neta")
plt.legend()
plt.grid(True)

# 5) Torque reconstruido
plt.figure(figsize=(10, 5))
plt.plot(t, Tm_ref, label='T_m requerida [N·m]')
plt.plot(t, Tm_hist, '--', label='T_m aplicada [N·m]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Torque [N·m]")
plt.title("Torque del motor")
plt.legend()
plt.grid(True)

# 6) Frenado reconstruido
plt.figure(figsize=(10, 5))
plt.plot(t, Fb_ref, label='F_b requerida [N]')
plt.plot(t, Fb_hist, '--', label='F_b aplicada [N]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Fuerza de frenado [N]")
plt.title("Fuerza de frenado")
plt.legend()
plt.grid(True)

# 7) Fuerzas longitudinales reales
plt.figure(figsize=(10, 5))
plt.plot(t, Fm_hist, label='F_m [N]')
plt.plot(t, Fb_hist, label='F_b [N]')
plt.plot(t, Fd_hist, label='F_d [N]')
plt.plot(t, Fr_hist, label='F_r [N]')
plt.xlabel("Tiempo [s]")
plt.ylabel("Fuerzas [N]")
plt.title("Fuerzas longitudinales")
plt.legend()
plt.grid(True)

# 8) Velocidad angular del motor
plt.figure(figsize=(10, 5))
plt.plot(t, omega_m_ref, '--', label=r'$\omega_{m,d}$ [rad/s]')
plt.plot(t, wm_hist, label=r'$\omega_m$ [rad/s]')
plt.xlabel("Tiempo [s]")
plt.ylabel(r"$\omega_m$ [rad/s]")
plt.title("Velocidad angular del motor")
plt.legend()
plt.grid(True)

plt.show()