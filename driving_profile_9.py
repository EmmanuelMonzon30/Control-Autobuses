import numpy as np
from Ruta9 import STOPS, DIST_BETWEEN_STOPS

# =========================================================
# PARÁMETROS DEL PERFIL DE CONDUCCIÓN
# =========================================================
KMH_TO_MS = 1.0 / 3.6

V_SLOW = 15.0 * KMH_TO_MS
V_MEDIUM = 25.0 * KMH_TO_MS
V_FAST = 30.0 * KMH_TO_MS

A_ACC = 0.8      # [m/s^2]
A_BRAKE = 1.2    # [m/s^2]

DWELL_DEFAULT = 10.0   # [s]
DWELL_BASE = 30.0      # [s]

DT = 0.1  # paso temporal [s]

# =========================================================
# CATEGORÍA DE CADA TRAMO
# =========================================================
SEGMENT_SPEED_TYPES = [
    "fast",    # Base Metrobús CU -> Estadio de Prácticas
    "fast",    # Estadio de Prácticas -> MUCA
    "medium",  # MUCA -> Rectoría
    "slow",    # Rectoría -> Psicología
    "slow",    # Psicología -> Facultad de Filosofía
    "slow",    # Facultad de Filosofía -> Facultad de Derecho
    "slow",    # Facultad de Derecho -> Facultad de Economía
    "slow",    # Facultad de Economía -> Facultad de Odontología
    "slow",    # Facultad de Odontología -> Facultad de Medicina
    "fast",    # Facultad de Medicina -> Invernadero
    "medium",  # Invernadero -> Anexo de Ingeniería
    "slow",    # Anexo de Ingeniería -> Camino Verde
    "medium",  # Camino Verde -> Facultad de Contaduría y Administración
    "medium",  # Facultad de Contaduría y Administración -> Escuela de Trabajo Social
    "medium",  # Escuela de Trabajo Social -> Base Metrobús CU
]

def speed_from_type(speed_type: str) -> float:
    if speed_type == "slow":
        return V_SLOW
    elif speed_type == "medium":
        return V_MEDIUM
    elif speed_type == "fast":
        return V_FAST
    else:
        raise ValueError(f"Tipo de velocidad no reconocido: {speed_type}")

def dwell_time_for_stop(stop_name: str) -> float:
    if stop_name == "Base Metrobús CU":
        return DWELL_BASE
    return DWELL_DEFAULT

def triangular_peak_speed(distance: float, a_acc: float = A_ACC, a_brake: float = A_BRAKE) -> float:
    """
    Velocidad pico alcanzable cuando no hay suficiente distancia
    para llegar a la velocidad objetivo y luego frenar a cero.
    """
    return np.sqrt(2.0 * distance * a_acc * a_brake / (a_acc + a_brake))

def build_segment_profile(distance: float, v_target: float, dt: float = DT):
    """
    Construye un perfil por tramo:
    0 -> acelera -> crucero (si cabe) -> frena -> 0

    Devuelve:
        t_seg, s_seg, v_seg, a_seg
    """
    d_acc = v_target**2 / (2.0 * A_ACC)
    d_brake = v_target**2 / (2.0 * A_BRAKE)
    d_min = d_acc + d_brake

    if distance >= d_min:
        # Perfil trapezoidal
        t_acc = v_target / A_ACC
        t_brake = v_target / A_BRAKE
        d_cruise = distance - d_min
        t_cruise = d_cruise / v_target

        t1 = np.arange(0.0, t_acc, dt)
        v1 = A_ACC * t1
        s1 = 0.5 * A_ACC * t1**2
        a1 = np.full_like(t1, A_ACC)

        t2 = np.arange(dt, t_cruise + dt, dt)
        v2 = np.full_like(t2, v_target)
        s2_start = s1[-1] if len(t1) > 0 else 0.0
        s2 = s2_start + v_target * t2
        a2 = np.zeros_like(t2)

        t3 = np.arange(dt, t_brake + dt, dt)
        v3 = np.maximum(v_target - A_BRAKE * t3, 0.0)
        s3_start = s2[-1] if len(t2) > 0 else s2_start
        s3 = s3_start + v_target * t3 - 0.5 * A_BRAKE * t3**2
        a3 = np.full_like(t3, -A_BRAKE)

        t_seg = np.concatenate([t1, t_acc + t2, t_acc + t_cruise + t3])
        s_seg = np.concatenate([s1, s2, s3])
        v_seg = np.concatenate([v1, v2, v3])
        a_seg = np.concatenate([a1, a2, a3])

    else:
        # Perfil triangular
        v_peak = triangular_peak_speed(distance)
        t_acc = v_peak / A_ACC
        t_brake = v_peak / A_BRAKE

        t1 = np.arange(0.0, t_acc, dt)
        v1 = A_ACC * t1
        s1 = 0.5 * A_ACC * t1**2
        a1 = np.full_like(t1, A_ACC)

        t2 = np.arange(dt, t_brake + dt, dt)
        v2 = np.maximum(v_peak - A_BRAKE * t2, 0.0)
        s2_start = s1[-1] if len(t1) > 0 else 0.0
        s2 = s2_start + v_peak * t2 - 0.5 * A_BRAKE * t2**2
        a2 = np.full_like(t2, -A_BRAKE)

        t_seg = np.concatenate([t1, t_acc + t2])
        s_seg = np.concatenate([s1, s2])
        v_seg = np.concatenate([v1, v2])
        a_seg = np.concatenate([a1, a2])

    # Asegurar inicio exacto en cero
    if len(s_seg) > 0:
        s_seg[0] = 0.0
        s_seg[-1] = distance
        v_seg[-1] = 0.0

    return t_seg, s_seg, v_seg, a_seg

def build_full_cycle_profile(dt: float = DT):
    """
    Construye una vuelta completa de la ruta con parada en todas las estaciones.

    Devuelve:
        t_ref, s_ref, v_ref, a_ref
    """
    t_all = [0.0]
    s_all = [0.0]
    v_all = [0.0]
    a_all = [0.0]

    t_offset = 0.0
    s_offset = 0.0

    for i, distance in enumerate(DIST_BETWEEN_STOPS):
        speed_type = SEGMENT_SPEED_TYPES[i]
        v_target = speed_from_type(speed_type)

        t_seg, s_seg_local, v_seg, a_seg = build_segment_profile(distance, v_target, dt=dt)

        if len(t_seg) > 0:
            t_seg_global = t_offset + t_seg
            s_seg_global = s_offset + s_seg_local

            t_all.extend(t_seg_global.tolist())
            s_all.extend(s_seg_global.tolist())
            v_all.extend(v_seg.tolist())
            a_all.extend(a_seg.tolist())

            t_offset = t_seg_global[-1]
            s_offset = s_seg_global[-1]

        next_stop_name = STOPS[(i + 1) % len(STOPS)]
        dwell = dwell_time_for_stop(next_stop_name)

        t_dwell = np.arange(dt, dwell + dt, dt)
        if len(t_dwell) > 0:
            t_all.extend((t_offset + t_dwell).tolist())
            s_all.extend(np.full_like(t_dwell, s_offset, dtype=float).tolist())
            v_all.extend(np.zeros_like(t_dwell, dtype=float).tolist())
            a_all.extend(np.zeros_like(t_dwell, dtype=float).tolist())
            t_offset = t_offset + t_dwell[-1]

    return (
        np.array(t_all, dtype=float),
        np.array(s_all, dtype=float),
        np.array(v_all, dtype=float),
        np.array(a_all, dtype=float),
    )

def build_reference_functions(dt: float = DT):
    """
    Construye funciones interpoladas de referencia:
        s_ref(t), v_ref(t), a_ref(t)
    """
    t_ref, s_ref, v_ref, a_ref = build_full_cycle_profile(dt=dt)

    def s_of_t(t):
        return np.interp(np.asarray(t), t_ref, s_ref)

    def v_of_t(t):
        return np.interp(np.asarray(t), t_ref, v_ref)

    def a_of_t(t):
        return np.interp(np.asarray(t), t_ref, a_ref)

    return t_ref, s_ref, v_ref, a_ref, s_of_t, v_of_t, a_of_t

def print_profile_summary():
    print("========================================")
    print("Perfil de conducción - 1 vuelta")
    print("========================================")
    print(f"Velocidad lenta : {V_SLOW*3.6:.1f} km/h")
    print(f"Velocidad media : {V_MEDIUM*3.6:.1f} km/h")
    print(f"Velocidad rápida: {V_FAST*3.6:.1f} km/h")
    print(f"Aceleración     : {A_ACC:.2f} m/s^2")
    print(f"Desaceleración  : {A_BRAKE:.2f} m/s^2")
    print(f"Parada general  : {DWELL_DEFAULT:.1f} s")
    print(f"Parada base     : {DWELL_BASE:.1f} s")

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print_profile_summary()

    t_ref, s_ref, v_ref, a_ref = build_full_cycle_profile(dt=DT)

    plt.figure(figsize=(12, 5))
    plt.plot(t_ref, v_ref * 3.6, label="v_ref")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Velocidad [km/h]")
    plt.title("Perfil de conducción: velocidad de referencia")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(12, 5))
    plt.plot(t_ref, s_ref, label="s_ref")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Distancia recorrida [m]")
    plt.title("Perfil de conducción: distancia recorrida")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(12, 5))
    plt.plot(t_ref, a_ref, label="a_ref")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Aceleración [m/s²]")
    plt.title("Perfil de conducción: aceleración de referencia")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(12, 5))
    plt.plot(s_ref, v_ref * 3.6, label="v_ref(s)")
    plt.xlabel("Distancia recorrida [m]")
    plt.ylabel("Velocidad [km/h]")
    plt.title("Perfil de conducción: velocidad de referencia vs distancia")
    plt.grid(True)
    plt.legend()

    plt.show()