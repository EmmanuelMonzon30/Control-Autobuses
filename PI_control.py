import numpy as np

from Modelo import (
    rolling_gravity_force,
    aerodynamic_drag,
    torque_brake_from_u,
    F_b_max,
    T_m_max,
)

# =========================================================
# PARÁMETROS DEL CONTROLADOR PI
# =========================================================
Kp = 5000
Ki = 100

# Estado integral z = \int (v_ref - v) dt
Z_MIN = -100.0
Z_MAX = 100.0

# Opciones de compensación
USE_SLOPE_COMPENSATION = True
USE_DRAG_COMPENSATION = False   # Puedes ponerlo en True más adelante

# Anti-windup
USE_ANTI_WINDUP = True


# =========================================================
# UTILIDADES
# =========================================================
def clamp(value: float, vmin: float, vmax: float) -> float:
    return float(np.clip(value, vmin, vmax))


def compute_speed_error(v_ref: float, v: float) -> float:
    """
    Error de velocidad:
        e = v_ref - v
    """
    return float(v_ref - v)


def slope_compensation_force(s: float, theta_func) -> float:
    """
    Compensación feedforward por pendiente y rodadura:

        u_ff = F_r_hat(s)

    usando el mismo modelo resistivo de Modelo.py.
    """
    return float(rolling_gravity_force(s, theta_func))


def drag_compensation_force(v: float) -> float:
    """
    Compensación opcional del arrastre aerodinámico.
    """
    return float(aerodynamic_drag(v))


def pi_feedback_force(e_v: float, z_int: float) -> float:
    """
    Término PI:
        u_PI = Kp*e + Ki*z
    """
    return float(Kp * e_v + Ki * z_int)


# =========================================================
# LEY DE CONTROL
# =========================================================
def compute_pi_control_force(
    s: float,
    v: float,
    v_ref: float,
    z_int: float,
    theta_func,
):
    """
    Calcula la fuerza longitudinal neta deseada:

        u = u_ff + u_PI

    donde:
        u_ff = compensación de pendiente (y opcionalmente arrastre)
        u_PI = Kp*e + Ki*z

    Retorna:
        u_cmd, e_v, Fr_hat, Fd_hat
    """
    e_v = compute_speed_error(v_ref, v)

    Fr_hat = slope_compensation_force(s, theta_func) if USE_SLOPE_COMPENSATION else 0.0
    Fd_hat = drag_compensation_force(v) if USE_DRAG_COMPENSATION else 0.0

    u_ff = Fr_hat + Fd_hat
    u_pi = pi_feedback_force(e_v, z_int)

    u_cmd = u_ff + u_pi

    return float(u_cmd), float(e_v), float(Fr_hat), float(Fd_hat)


def compute_pi_actuators(
    s: float,
    v: float,
    v_ref: float,
    z_int: float,
    theta_func,
):
    """
    A partir del estado actual y la referencia, calcula:

    - u_cmd : fuerza neta longitudinal deseada
    - Fm_cmd: fuerza motriz aplicada
    - Fb_cmd: fuerza de frenado aplicada
    - Tm_cmd: torque del motor aplicado
    - e_v   : error de velocidad
    - Fr_hat: compensación por pendiente
    - Fd_hat: compensación por arrastre
    """
    u_cmd, e_v, Fr_hat, Fd_hat = compute_pi_control_force(
        s=s,
        v=v,
        v_ref=v_ref,
        z_int=z_int,
        theta_func=theta_func,
    )

    Fm_cmd, Fb_cmd, Tm_cmd = torque_brake_from_u(u_cmd)

    return {
        "u_cmd": float(u_cmd),
        "Fm_cmd": float(Fm_cmd),
        "Fb_cmd": float(Fb_cmd),
        "Tm_cmd": float(Tm_cmd),
        "e_v": float(e_v),
        "Fr_hat": float(Fr_hat),
        "Fd_hat": float(Fd_hat),
    }


# =========================================================
# ACTUALIZACIÓN DEL ESTADO INTEGRAL
# =========================================================
def update_integral_state(
    z_int: float,
    e_v: float,
    dt: float,
    u_cmd: float,
    Fm_cmd: float,
    Fb_cmd: float,
):
    """
    Actualiza el estado integral con anti-windup simple.

    Idea:
    - si la fuerza deseada excede la capacidad de tracción y el error pide
      todavía más tracción, se congela la integración
    - si la fuerza deseada excede la capacidad de frenado y el error pide
      todavía más frenado, se congela la integración
    - en otro caso, integra normalmente

    Nota:
    Como torque_brake_from_u() ya reconstruye Fm y Fb saturados,
    usamos la comparación entre u_cmd y la fuerza neta realmente aplicable.
    """
    if not USE_ANTI_WINDUP:
        return clamp(z_int + dt * e_v, Z_MIN, Z_MAX)

    u_applied = Fm_cmd - Fb_cmd

    saturated_high = (u_cmd > u_applied + 1e-9)
    saturated_low = (u_cmd < u_applied - 1e-9)

    if (saturated_high and e_v > 0.0) or (saturated_low and e_v < 0.0):
        z_next = z_int
    else:
        z_next = z_int + dt * e_v

    return clamp(z_next, Z_MIN, Z_MAX)


# =========================================================
# INTERFAZ PRINCIPAL PARA main_sim.py
# =========================================================
def pi_control_step(
    x: np.ndarray,
    z_int: float,
    v_ref_k: float,
    dt: float,
    theta_func,
):
    """
    Paso de control PI.

    Parámetros
    ----------
    x : np.ndarray
        Estado actual [s, v]
    z_int : float
        Estado integral actual
    v_ref_k : float
        Velocidad de referencia en el instante k [m/s]
    dt : float
        Paso temporal [s]
    theta_func : callable
        Función theta(s) de la ruta

    Retorna
    -------
    Tm_cmd : float
        Torque del motor [N·m]
    Fb_cmd : float
        Fuerza de frenado [N]
    z_next : float
        Nuevo estado integral
    ctrl_info : dict
        Información auxiliar del controlador
    """
    s = float(x[0])
    v = float(x[1])

    ctrl = compute_pi_actuators(
        s=s,
        v=v,
        v_ref=v_ref_k,
        z_int=z_int,
        theta_func=theta_func,
    )

    z_next = update_integral_state(
        z_int=z_int,
        e_v=ctrl["e_v"],
        dt=dt,
        u_cmd=ctrl["u_cmd"],
        Fm_cmd=ctrl["Fm_cmd"],
        Fb_cmd=ctrl["Fb_cmd"],
    )

    return ctrl["Tm_cmd"], ctrl["Fb_cmd"], z_next, ctrl


# =========================================================
# FUNCIÓN OPCIONAL DE RESUMEN
# =========================================================
def print_pi_summary():
    print("========================================")
    print("Controlador PI sobre velocidad")
    print("========================================")
    print(f"Kp = {Kp:.3f}")
    print(f"Ki = {Ki:.3f}")
    print(f"Anti-windup = {USE_ANTI_WINDUP}")
    print(f"Compensación de pendiente = {USE_SLOPE_COMPENSATION}")
    print(f"Compensación de arrastre = {USE_DRAG_COMPENSATION}")
    print(f"z_int en [{Z_MIN:.3f}, {Z_MAX:.3f}]")