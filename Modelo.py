import numpy as np

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

# =========================================================
# COTAS FÍSICAS
# =========================================================
T_m_max = 2500.0     # torque máximo [N·m]
T_m_min = 0.0        # por ahora solo tracción
F_b_max = 30000.0    # frenado máximo [N]
F_b_min = 0.0


# =========================================================
# SATURACIONES
# =========================================================
def saturate_torque(Tm: float) -> float:
    """
    Satura el torque del motor dentro de sus límites físicos.
    """
    return float(np.clip(Tm, T_m_min, T_m_max))


def saturate_brake(Fb: float) -> float:
    """
    Satura la fuerza de frenado dentro de sus límites físicos.
    """
    return float(np.clip(Fb, F_b_min, F_b_max))


# =========================================================
# VARIABLES AUXILIARES DEL TREN MOTRIZ
# =========================================================
def motor_speed(v: float) -> float:
    """
    Velocidad angular del motor [rad/s].

    omega_m = (M_f / r_w) * v
    """
    return (M_f / r_w) * v


def traction_force(Tm: float) -> float:
    """
    Fuerza motriz equivalente [N] producida por el torque del motor.

    F_m = (M_f / (r_w * eta_t)) * T_m
    """
    Tm_sat = saturate_torque(Tm)
    return (M_f / (r_w * eta_t)) * Tm_sat


def torque_from_force(Fm: float) -> float:
    """
    Reconstruye el torque [N·m] a partir de una fuerza motriz [N].

    T_m = (r_w * eta_t / M_f) * F_m
    """
    Tm = (r_w * eta_t / M_f) * Fm
    return saturate_torque(Tm)


# =========================================================
# FUERZAS RESISTIVAS
# =========================================================
def aerodynamic_drag(v: float) -> float:
    """
    Fuerza de arrastre aerodinámico [N].

    F_d = 0.5 * rho * A_bus * c_a * v^2
    """
    return 0.5 * rho * A_bus * c_a * v**2


def rolling_gravity_force(s: float, theta_func) -> float:
    """
    Fuerza que agrupa rodadura y efecto gravitacional [N].

    F_r = m*g*(sin(theta(s)) + c_r*cos(theta(s)))

    Parámetros
    ----------
    s : float
        Posición longitudinal [m]
    theta_func : callable
        Función que recibe s y devuelve theta(s) en radianes
    """
    th = float(theta_func(s))
    return m * g * (np.sin(th) + c_r * np.cos(th))


# =========================================================
# ENTRADA LONGITUDINAL NETA
# =========================================================
def split_longitudinal_force(u: float) -> tuple[float, float]:
    """
    Separa una fuerza longitudinal neta u en:
    - fuerza motriz F_m
    - fuerza de frenado F_b

    Regla:
    - si u >= 0: hay tracción, no frenado
    - si u < 0 : hay frenado, no tracción
    """
    if u >= 0.0:
        Fm = float(u)
        Fb = 0.0
    else:
        Fm = 0.0
        Fb = float(-u)

    Fb = saturate_brake(Fb)
    return Fm, Fb


def torque_brake_from_u(u: float) -> tuple[float, float, float]:
    """
    A partir de una fuerza longitudinal neta u [N], calcula:
    - F_m [N]
    - F_b [N]
    - T_m [N·m]

    Útil para reconstruir actuadores a partir de una referencia longitudinal.
    """
    Fm, Fb = split_longitudinal_force(u)
    Tm = torque_from_force(Fm)
    Fm = traction_force(Tm)  # recalcular por si hubo saturación
    return Fm, Fb, Tm


# =========================================================
# DINÁMICA DEL AUTOBÚS
# =========================================================
def bus_dynamics_open_loop(x: np.ndarray, Tm: float, Fb: float, theta_func):
    """
    Dinámica longitudinal del autobús en lazo abierto.

    Estados
    -------
    x = [s, v]
        s : posición [m]
        v : velocidad [m/s]

    Entradas
    --------
    Tm : float
        Torque del motor [N·m]
    Fb : float
        Fuerza de frenado [N]
    theta_func : callable
        Función theta(s) [rad]

    Modelo
    ------
    s_dot = v

    m*v_dot = F_m - F_b - F_d - F_r
    """
    s, v = x

    Tm_sat = saturate_torque(Tm)
    Fb_sat = saturate_brake(Fb)

    Fm = traction_force(Tm_sat)
    Fd = aerodynamic_drag(v)
    Fr = rolling_gravity_force(s, theta_func)

    dsdt = v
    dvdt = (Fm - Fb_sat - Fd - Fr) / m

    return np.array([dsdt, dvdt], dtype=float), Fm, Fd, Fr


# =========================================================
# INTEGRADORES NUMÉRICOS
# =========================================================
def euler_step(x: np.ndarray, dx: np.ndarray, dt: float) -> np.ndarray:
    """
    Un paso de integración Euler explícito.
    """
    x_next = x + dt * dx

    # Evitar velocidad negativa numérica
    if x_next[1] < 0.0:
        x_next[1] = 0.0

    return x_next


def rk4_step(x: np.ndarray, dt: float, dynamics_func, *args):
    """
    Un paso de integración Runge-Kutta de orden 4.
    dynamics_func debe retornar:
        dx, ...
    y solo se toma dx para integrar.
    """
    k1 = dynamics_func(x, *args)[0]
    k2 = dynamics_func(x + 0.5 * dt * k1, *args)[0]
    k3 = dynamics_func(x + 0.5 * dt * k2, *args)[0]
    k4 = dynamics_func(x + dt * k3, *args)[0]

    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Evitar velocidad negativa numérica
    if x_next[1] < 0.0:
        x_next[1] = 0.0

    return x_next