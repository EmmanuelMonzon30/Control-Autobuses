import numpy as np


# =========================================================
# PARÁMETROS DEL AUTOBÚS
# =========================================================
m = 15000.0          # masa nominal [kg]
rho = 1.225          # densidad del aire [kg/m^3]
A_bus = 8.0          # área frontal [m^2]
c_a = 0.7            # coeficiente aerodinámico [-]
c_r = 0.008          # coeficiente de rodadura [-]
g = 9.81             # aceleración gravitacional [m/s^2]

r_w = 0.5            # radio de la rueda [m]
M_f = 6.0            # relación final de transmisión [-]
eta_t = 0.95         # eficiencia de transmisión [-]


# =========================================================
# COTAS FÍSICAS
# =========================================================
T_m_max = 2500.0     # par máximo del motor [N·m]
T_m_min = 0.0        # únicamente tracción positiva

P_m_max = 250000.0   # potencia mecánica máxima [W]

F_b_max = 30000.0    # fuerza máxima de frenado [N]
F_b_min = 0.0


# =========================================================
# VARIABLES AUXILIARES DEL TREN MOTRIZ
# =========================================================
def motor_speed(v: float) -> float:
    """
    Velocidad angular del motor [rad/s].

        omega_m = (M_f / r_w) * v
    """
    return (M_f / r_w) * float(v)


def available_motor_torque(v: float) -> float:
    """
    Par máximo disponible para la velocidad actual.

    En la región de par constante:

        T_lim = T_m_max

    En la región de potencia constante:

        T_lim = P_m_max / omega_m
    """
    omega_m = abs(motor_speed(v))

    # Evitar una división entre cero cuando el vehículo
    # se encuentra detenido.
    if omega_m <= 1e-9:
        return T_m_max

    torque_power_limit = P_m_max / omega_m

    return float(
        min(T_m_max, torque_power_limit)
    )


# =========================================================
# SATURACIONES
# =========================================================
def saturate_torque(Tm: float, v: float) -> float:
    """
    Satura el par del motor entre cero y el límite
    disponible para la velocidad actual.
    """
    Tm_limit = available_motor_torque(v)

    return float(
        np.clip(Tm, T_m_min, Tm_limit)
    )


def saturate_brake(Fb: float) -> float:
    """
    Satura la fuerza de frenado dentro de sus
    límites físicos.
    """
    return float(
        np.clip(Fb, F_b_min, F_b_max)
    )


# =========================================================
# FUERZA MOTRIZ Y PAR DEL MOTOR
# =========================================================
def traction_force(Tm: float, v: float) -> float:
    """
    Fuerza motriz transmitida a las ruedas [N].

        F_m = (M_f * eta_t / r_w) * T_m

    El par se satura considerando tanto su límite
    máximo como la potencia disponible.
    """
    Tm_sat = saturate_torque(Tm, v)

    return (
        M_f
        * eta_t
        / r_w
        * Tm_sat
    )


def torque_from_force(Fm: float, v: float) -> float:
    """
    Par requerido para generar una fuerza motriz [N·m].

        T_m = (r_w / (M_f * eta_t)) * F_m
    """
    Tm_requested = (
        r_w
        / (M_f * eta_t)
        * float(Fm)
    )

    return saturate_torque(
        Tm_requested,
        v,
    )


# =========================================================
# FUERZAS RESISTIVAS
# =========================================================
def aerodynamic_drag(v: float) -> float:
    """
    Fuerza de arrastre aerodinámico [N].

        F_d = 0.5 * rho * A_bus * c_a * v^2
    """
    return (
        0.5
        * rho
        * A_bus
        * c_a
        * float(v)**2
    )


def rolling_gravity_force(
    s: float,
    theta_func,
) -> float:
    """
    Fuerza nominal que agrupa el efecto gravitacional
    y la resistencia a la rodadura [N].

        F_r = m*g*(sin(theta(s)) + c_r*cos(theta(s)))

    Se utiliza la masa nominal m=15000 kg.
    """
    th = float(theta_func(s))

    return (
        m
        * g
        * (
            np.sin(th)
            + c_r * np.cos(th)
        )
    )


# =========================================================
# DISTRIBUCIÓN DE LA FUERZA LONGITUDINAL
# =========================================================
def split_longitudinal_force(
    Fu_requested: float,
) -> tuple[float, float]:
    """
    Separa una fuerza longitudinal deseada en:

    - fuerza motriz solicitada F_m;
    - fuerza de frenado F_b.

    Si Fu_requested >= 0:
        se aplica tracción y no se aplica frenado.

    Si Fu_requested < 0:
        se aplica frenado y no se aplica tracción.
    """
    if Fu_requested >= 0.0:
        Fm_requested = float(Fu_requested)
        Fb = 0.0
    else:
        Fm_requested = 0.0
        Fb = float(-Fu_requested)

    Fb = saturate_brake(Fb)

    return Fm_requested, Fb


def torque_brake_from_u(
    u: float,
    v: float,
) -> tuple[float, float, float]:
    """
    Convierte la fuerza longitudinal neta deseada
    en las acciones físicas de los actuadores.

    Parámetros
    ----------
    u : float
        Fuerza longitudinal neta deseada [N].
    v : float
        Velocidad actual del autobús [m/s].

    Retorna
    -------
    Fm : float
        Fuerza motriz realmente disponible [N].
    Fb : float
        Fuerza de frenado aplicada [N].
    Tm : float
        Par del motor realmente disponible [N·m].
    """
    Fm_requested, Fb = split_longitudinal_force(u)

    # Convertir la fuerza solicitada en par y aplicar
    # los límites de par y potencia.
    Tm = torque_from_force(
        Fm_requested,
        v,
    )

    # Reconstruir la fuerza realmente disponible después
    # de aplicar las saturaciones.
    Fm = traction_force(
        Tm,
        v,
    )

    return Fm, Fb, Tm


# =========================================================
# DINÁMICA DEL AUTOBÚS CON MASA NOMINAL
# =========================================================
def bus_dynamics_open_loop(
    x: np.ndarray,
    Tm: float,
    Fb: float,
    theta_func,
):
    """
    Dinámica longitudinal del autobús con masa nominal.

    Estados
    -------
    x = [s, v]

    Modelo
    ------
        ds/dt = v

        m*dv/dt = F_m - F_b - F_d - F_r
    """
    s = float(x[0])
    v = float(x[1])

    Fb_sat = saturate_brake(Fb)

    # traction_force aplica las restricciones de par
    # y potencia utilizando la velocidad actual.
    Fm = traction_force(
        Tm,
        v,
    )

    Fd = aerodynamic_drag(v)

    Fr = rolling_gravity_force(
        s,
        theta_func,
    )

    dsdt = v

    dvdt = (
        Fm
        - Fb_sat
        - Fd
        - Fr
    ) / m

    dx = np.array(
        [dsdt, dvdt],
        dtype=float,
    )

    return dx, Fm, Fd, Fr


# =========================================================
# INTEGRADORES NUMÉRICOS
# =========================================================
def euler_step(
    x: np.ndarray,
    dx: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Realiza un paso de integración mediante
    Euler explícito.
    """
    x_next = x + dt * dx

    # Evitar velocidades negativas causadas por
    # la integración numérica.
    if x_next[1] < 0.0:
        x_next[1] = 0.0

    return x_next


def rk4_step(
    x: np.ndarray,
    dt: float,
    dynamics_func,
    *args,
) -> np.ndarray:
    """
    Realiza un paso de integración mediante el
    método de Runge-Kutta de cuarto orden.

    dynamics_func debe retornar primero la derivada
    del estado.
    """
    k1 = dynamics_func(
        x,
        *args,
    )[0]

    k2 = dynamics_func(
        x + 0.5 * dt * k1,
        *args,
    )[0]

    k3 = dynamics_func(
        x + 0.5 * dt * k2,
        *args,
    )[0]

    k4 = dynamics_func(
        x + dt * k3,
        *args,
    )[0]

    x_next = x + (
        dt / 6.0
    ) * (
        k1
        + 2.0 * k2
        + 2.0 * k3
        + k4
    )

    # Evitar velocidades negativas causadas por
    # la integración numérica.
    if x_next[1] < 0.0:
        x_next[1] = 0.0

    return x_next