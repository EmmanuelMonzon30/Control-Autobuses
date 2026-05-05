import numpy as np

# =========================================================
# MODELO DE MASA VARIABLE DEL AUTOBÚS
# =========================================================
# Este archivo NO reemplaza Modelo.py.
# Su objetivo es complementar la simulación cuando se desea que
# la planta tenga masa variable por ascenso/descenso de pasajeros.

from Modelo import (
    m as M_BUS_EMPTY_DEFAULT,
    rho,
    A_bus,
    c_a,
    c_r,
    g,
    saturate_torque,
    saturate_brake,
    traction_force,
    aerodynamic_drag,
)

from Ruta9 import (
    STOPS,
    STOP_POSITIONS,
    TOTAL_ROUTE_LENGTH,
    wrap_position,
)


# =========================================================
# PARÁMETROS DE CAPACIDAD Y MASA DE PASAJEROS
# =========================================================
# Supuesto para un autobús urbano estándar de 12 m:
# - aproximadamente 30 pasajeros sentados
# - aproximadamente 50 pasajeros parados
# - capacidad total usada en simulación: 80 pasajeros
#
# Puedes modificar estos valores si después tienes la ficha técnica
# exacta del autobús de estudio.

N_SEATED = 30
N_STANDING = 50
N_PASSENGERS_MAX_PHYSICAL = N_SEATED + N_STANDING

# Masa promedio por pasajero [kg]
M_PASSENGER = 70.0

# Masa base del autobús sin pasajeros [kg]
M_BUS_EMPTY = float(M_BUS_EMPTY_DEFAULT)


# =========================================================
# GENERACIÓN DE PASAJEROS POR PARADA
# =========================================================
def generate_passenger_profile(
    seed=None,
    n_min=5,
    n_max=80, # 70
    initial_passengers=None,
    max_change_per_stop=40, # 18
):
    """
    Genera un perfil aleatorio de pasajeros para una vuelta completa.

    La salida representa el número de pasajeros dentro del autobús
    después de salir de cada parada.
    """
    rng = np.random.default_rng(seed)
    n_stops = len(STOPS)

    n_min = int(max(0, n_min))
    n_max = int(min(n_max, N_PASSENGERS_MAX_PHYSICAL))

    if n_min > n_max:
        raise ValueError("n_min no puede ser mayor que n_max.")

    if initial_passengers is None:
        current = int(rng.integers(n_min, n_max + 1))
    else:
        current = int(np.clip(initial_passengers, n_min, n_max))

    passenger_counts = np.zeros(n_stops, dtype=int)

    for i in range(n_stops):
        if i == 0:
            passenger_counts[i] = current
        else:
            delta = int(rng.integers(-max_change_per_stop, max_change_per_stop + 1))
            current = int(np.clip(current + delta, n_min, n_max))
            passenger_counts[i] = current

    return passenger_counts


def generate_independent_passenger_profile(
    seed=None,
    n_min=5,
    n_max=80,
):
    """
    Genera pasajeros aleatorios independientes en cada parada.

    Esta opción cambia más bruscamente que generate_passenger_profile().
    Puede ser útil para pruebas extremas.
    """
    rng = np.random.default_rng(seed)

    n_min = int(max(0, n_min))
    n_max = int(min(n_max, N_PASSENGERS_MAX_PHYSICAL))

    if n_min > n_max:
        raise ValueError("n_min no puede ser mayor que n_max.")

    return rng.integers(n_min, n_max + 1, size=len(STOPS)).astype(int)


# =========================================================
# PASAJEROS Y MASA COMO FUNCIÓN DE LA POSICIÓN
# =========================================================
def segment_index_from_position(s):
    """
    Devuelve el índice del tramo actual a partir de la posición s.
    """
    s_wrapped = float(wrap_position(s))

    idx = int(np.searchsorted(STOP_POSITIONS, s_wrapped, side="right") - 1)
    idx = int(np.clip(idx, 0, len(STOPS) - 1))

    return idx


def passengers_at_position(s, passenger_counts):
    """
    Número de pasajeros dentro del autobús en la posición s.
    """
    idx = segment_index_from_position(s)
    return int(passenger_counts[idx])


def mass_from_passengers(
    n_passengers,
    m_bus_empty=M_BUS_EMPTY,
    m_passenger=M_PASSENGER,
):
    """
    Masa total del autobús con pasajeros.
    """
    return float(m_bus_empty + n_passengers * m_passenger)


def mass_at_position(
    s,
    passenger_counts,
    m_bus_empty=M_BUS_EMPTY,
    m_passenger=M_PASSENGER,
):
    """
    Masa total del autobús como función de la posición.
    """
    n_pass = passengers_at_position(s, passenger_counts)
    return mass_from_passengers(n_pass, m_bus_empty, m_passenger)


# =========================================================
# FUERZA DE RODADURA + GRAVEDAD CON MASA VARIABLE
# =========================================================
def rolling_gravity_force_variable_mass(
    s,
    theta_func,
    m_current,
):
    """
    Fuerza de pendiente + rodadura considerando masa variable.

    F_r = m_current*g*(sin(theta(s)) + c_r*cos(theta(s)))
    """
    th = float(theta_func(s))
    return float(m_current * g * (np.sin(th) + c_r * np.cos(th)))


# =========================================================
# DINÁMICA DE LA PLANTA CON MASA VARIABLE
# =========================================================
def bus_dynamics_open_loop_variable_mass(
    x,
    Tm,
    Fb,
    theta_func,
    passenger_counts,
    m_bus_empty=M_BUS_EMPTY,
    m_passenger=M_PASSENGER,
):
    """
    Dinámica longitudinal del autobús con masa variable.

    Estados
    -------
    x = [s, v]
        s : posición [m]
        v : velocidad [m/s]

    Entradas
    --------
    Tm : float
        Torque del motor [N*m]

    Fb : float
        Fuerza de frenado [N]

    theta_func : callable
        Función theta(s) [rad]

    passenger_counts : np.ndarray
        Número de pasajeros por tramo/parada.

    Modelo
    ------
    s_dot = v

    m(s)*v_dot = F_m - F_b - F_d - F_r(s, m(s))
    """
    s, v = x

    Tm_sat = saturate_torque(Tm)
    Fb_sat = saturate_brake(Fb)

    n_pass = passengers_at_position(s, passenger_counts)
    m_current = mass_from_passengers(n_pass, m_bus_empty, m_passenger)

    Fm = traction_force(Tm_sat)
    Fd = aerodynamic_drag(v)
    Fr = rolling_gravity_force_variable_mass(s, theta_func, m_current)

    dsdt = v
    dvdt = (Fm - Fb_sat - Fd - Fr) / m_current

    dx = np.array([dsdt, dvdt], dtype=float)

    return dx, Fm, Fd, Fr, m_current, n_pass


# =========================================================
# UTILIDADES PARA SIMULACIÓN Y GRÁFICAS
# =========================================================
def build_mass_profile_along_route(
    passenger_counts,
    n_points=1000,
    m_bus_empty=M_BUS_EMPTY,
    m_passenger=M_PASSENGER,
):
    """
    Construye perfiles s, pasajeros(s), masa(s) para graficar.
    """
    s_grid = np.linspace(0.0, TOTAL_ROUTE_LENGTH, n_points)

    passengers_grid = np.array(
        [passengers_at_position(s, passenger_counts) for s in s_grid],
        dtype=int,
    )

    mass_grid = np.array(
        [mass_from_passengers(n, m_bus_empty, m_passenger) for n in passengers_grid],
        dtype=float,
    )

    return s_grid, passengers_grid, mass_grid


def print_passenger_profile(passenger_counts):
    """
    Imprime en consola el número de pasajeros por tramo.
    """
    print("========================================")
    print("Perfil de pasajeros por tramo")
    print("========================================")
    print(f"Capacidad sentados : {N_SEATED}")
    print(f"Capacidad parados  : {N_STANDING}")
    print(f"Capacidad total    : {N_PASSENGERS_MAX_PHYSICAL}")
    print(f"Masa pasajero      : {M_PASSENGER:.1f} kg")
    print(f"Masa bus vacío     : {M_BUS_EMPTY:.1f} kg")
    print("----------------------------------------")

    for i, stop in enumerate(STOPS):
        n_pass = int(passenger_counts[i])
        m_total = mass_from_passengers(n_pass)
        next_stop = STOPS[(i + 1) % len(STOPS)]
        print(
            f"{i+1:02d}. {stop} -> {next_stop}: "
            f"{n_pass:2d} pasajeros, masa = {m_total:.1f} kg"
        )


# =========================================================
# PRUEBA DEL ARCHIVO
# =========================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    passenger_counts = generate_passenger_profile(
        seed=None,
        n_min=5,
        n_max=80,
        initial_passengers=None,
        max_change_per_stop=40,
    )

    print_passenger_profile(passenger_counts)

    s_grid, n_grid, m_grid = build_mass_profile_along_route(passenger_counts)

    plt.figure(figsize=(12, 5))
    plt.step(s_grid, n_grid, where="post", label="Pasajeros")
    plt.xlabel("Posición s [m]")
    plt.ylabel("Número de pasajeros")
    plt.title("Perfil aleatorio de pasajeros por tramo")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(12, 5))
    plt.step(s_grid, m_grid, where="post", label="Masa total")
    plt.xlabel("Posición s [m]")
    plt.ylabel("Masa [kg]")
    plt.title("Masa variable del autobús")
    plt.grid(True)
    plt.legend()

    plt.show()
