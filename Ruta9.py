import numpy as np
from scipy.ndimage import gaussian_filter1d

# =========================================================
# PARADAS DE LA RUTA
# =========================================================
STOPS = [
    "Base Metrobús CU",
    "Estadio de Prácticas",
    "MUCA",
    "Rectoría",
    "Psicología",
    "Facultad de Filosofía",
    "Facultad de Derecho",
    "Facultad de Economía",
    "Facultad de Odontología",
    "Facultad de Medicina",
    "Invernadero",
    "Anexo de Ingeniería",
    "Camino Verde",
    "Facultad de Contaduría y Administración",
    "Escuela de Trabajo Social"
]

# =========================================================
# DISTANCIAS ENTRE PARADAS [m]
# =========================================================
DIST_BETWEEN_STOPS = [
    331,  # Base Metrobús CU -> Estadio de Prácticas
    490,  # Estadio de Prácticas -> MUCA
    186,  # MUCA -> Rectoría
    217,  # Rectoría -> Psicología
    206,  # Psicología -> Facultad de Filosofía
    525,  # Facultad de Filosofía -> Facultad de Derecho
    199,  # Facultad de Derecho -> Facultad de Economía
    217,  # Facultad de Economía -> Facultad de Odontología
    367,  # Facultad de Odontología -> Facultad de Medicina
    478,  # Facultad de Medicina -> Invernadero
    245,  # Invernadero -> Anexo de Ingeniería
    195,  # Anexo de Ingeniería -> Camino Verde
    355,  # Camino Verde -> Facultad de Contaduría y Administración
    223,  # Facultad de Contaduría y Administración -> Escuela de Trabajo Social
    150   # Escuela de Trabajo Social -> Base Metrobús CU
]

# =========================================================
# DISTANCIAS ACUMULADAS DE LAS PARADAS
# =========================================================
def cumulative_distances(distances):
    cum = [0.0]
    total = 0.0
    for d in distances:
        total += d
        cum.append(total)
    return np.array(cum, dtype=float)

STOP_POSITIONS = cumulative_distances(DIST_BETWEEN_STOPS)
TOTAL_ROUTE_LENGTH = float(STOP_POSITIONS[-1])

# =========================================================
# DATOS DETALLADOS DE CADA TRAMO
# =========================================================
ROUTE_SEGMENTS = [
    {
        "name": "Base Metrobús CU - Estadio de Prácticas",
        "d_local": np.array([0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 331], dtype=float),
        "elevation": np.array([2286, 2286, 2285, 2284, 2285, 2286, 2286, 2287, 2287, 2287, 2286, 2288, 2288, 2289], dtype=float),
    },
    {
        "name": "Estadio de Prácticas - MUCA",
        "d_local": np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 490], dtype=float),
        "elevation": np.array([2287, 2287, 2289, 2292, 2292, 2293, 2292, 2291, 2288, 2287, 2286], dtype=float),
    },
    {
        "name": "MUCA - Rectoría",
        "d_local": np.array([0, 25, 50, 75, 100, 125, 150, 175, 186], dtype=float),
        "elevation": np.array([2286, 2286, 2285, 2286, 2286, 2286, 2286, 2286, 2286], dtype=float),
    },
    {
        "name": "Rectoría - Psicología",
        "d_local": np.array([0, 25, 50, 75, 100, 125, 150, 175, 200, 217], dtype=float),
        "elevation": np.array([2286, 2285, 2287, 2288, 2288, 2287, 2288, 2288, 2287, 2287], dtype=float),
    },
    {
        "name": "Psicología - Facultad de Filosofía",
        "d_local": np.array([0, 25, 50, 75, 100, 125, 150, 175, 206], dtype=float),
        "elevation": np.array([2287, 2284, 2283, 2282, 2281, 2280, 2280, 2279, 2279], dtype=float),
    },
    {
        "name": "Facultad de Filosofía - Facultad de Derecho",
        "d_local": np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 525], dtype=float),
        "elevation": np.array([2279, 2278, 2277, 2277, 2276, 2274, 2274, 2276, 2277, 2276, 2274], dtype=float),
    },
    {
        "name": "Facultad de Derecho - Facultad de Economía",
        "d_local": np.array([0, 25, 50, 75, 100, 125, 150, 175, 199], dtype=float),
        "elevation": np.array([2274, 2273, 2272, 2273, 2273, 2272, 2271, 2271, 2270], dtype=float),
    },
    {
        "name": "Facultad de Economía - Facultad de Odontología",
        "d_local": np.array([0, 25, 50, 75, 100, 125, 150, 175, 200, 217], dtype=float),
        "elevation": np.array([2270, 2269, 2269, 2269, 2270, 2268, 2268, 2268, 2267, 2267], dtype=float),
    },
    {
        "name": "Facultad de Odontología - Facultad de Medicina",
        "d_local": np.array([0, 50, 100, 150, 200, 250, 300, 367], dtype=float),
        "elevation": np.array([2267, 2266, 2266, 2264, 2263, 2262, 2262, 2264], dtype=float),
    },
    {
        "name": "Facultad de Medicina - Invernadero",
        "d_local": np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 478], dtype=float),
        "elevation": np.array([2264, 2265, 2269, 2271, 2273, 2275, 2274, 2275, 2274, 2274, 2274], dtype=float),
    },
    {
        "name": "Invernadero - Anexo de Ingeniería",
        "d_local": np.array([0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 245], dtype=float),
        "elevation": np.array([2274, 2273, 2272, 2272, 2272, 2272, 2273, 2273, 2274, 2275, 2276], dtype=float),
    },
    {
        "name": "Anexo de Ingeniería - Camino Verde",
        "d_local": np.array([0, 25, 50, 75, 100, 125, 150, 175, 195], dtype=float),
        "elevation": np.array([2276, 2277, 2279, 2280, 2280, 2281, 2282, 2282, 2282], dtype=float),
    },
    {
        "name": "Camino Verde - Facultad de Contaduría y Administración",
        "d_local": np.array([0, 50, 100, 150, 200, 250, 300, 355], dtype=float),
        "elevation": np.array([2282, 2284, 2284, 2286, 2287, 2289, 2289, 2290], dtype=float),
    },
    {
        "name": "Facultad de Contaduría y Administración - Escuela de Trabajo Social",
        "d_local": np.array([0, 25, 50, 75, 100, 125, 150, 175, 200, 223], dtype=float),
        "elevation": np.array([2290, 2291, 2292, 2292, 2292, 2292, 2292, 2291, 2291, 2291], dtype=float),
    },
    {
        "name": "Escuela de Trabajo Social - Base Metrobús CU",
        "d_local": np.array([0, 25, 50, 75, 100, 125, 150], dtype=float),
        "elevation": np.array([2291, 2290, 2290, 2289, 2290, 2289, 2288], dtype=float),
    },
]

# =========================================================
# CONSTRUCCIÓN GLOBAL DEL PERFIL DE RUTA
# =========================================================
def build_route_profile():
    s_global = []
    h_global = []
    s_offset = 0.0

    for i, seg in enumerate(ROUTE_SEGMENTS):
        d_local = seg["d_local"]
        h_local = seg["elevation"]

        if len(d_local) != len(h_local):
            raise ValueError(f"Longitudes incompatibles en el tramo: {seg['name']}")

        s_seg = s_offset + d_local

        if i == 0:
            s_global.extend(s_seg.tolist())
            h_global.extend(h_local.tolist())
        else:
            s_global.extend(s_seg[1:].tolist())
            h_global.extend(h_local[1:].tolist())

        s_offset = s_seg[-1]

    return np.array(s_global, dtype=float), np.array(h_global, dtype=float)

S_ROUTE, H_ROUTE = build_route_profile()

# =========================================================
# SUAVIZADO DEL PERFIL
# =========================================================
SMOOTH_SIGMA = 1.0
H_ROUTE_SMOOTH = gaussian_filter1d(H_ROUTE, sigma=SMOOTH_SIGMA)
GRADE_ROUTE = np.gradient(H_ROUTE_SMOOTH, S_ROUTE)
THETA_ROUTE = np.arctan(GRADE_ROUTE)

# =========================================================
# FUNCIONES AUXILIARES DE LA RUTA
# =========================================================
def route_length():
    return TOTAL_ROUTE_LENGTH

def wrap_position(s):
    return np.mod(s, TOTAL_ROUTE_LENGTH)

def elevation(s):
    s_wrapped = wrap_position(np.asarray(s))
    return np.interp(s_wrapped, S_ROUTE, H_ROUTE_SMOOTH)

def grade(s):
    s_wrapped = wrap_position(np.asarray(s))
    return np.interp(s_wrapped, S_ROUTE, GRADE_ROUTE)

def theta(s):
    s_wrapped = wrap_position(np.asarray(s))
    return np.interp(s_wrapped, S_ROUTE, THETA_ROUTE)

def get_stop_positions():
    return STOP_POSITIONS.copy()

def get_stop_names():
    return STOPS.copy()

def get_route_profile():
    return S_ROUTE.copy(), H_ROUTE.copy()

def get_smoothed_route_profile():
    return S_ROUTE.copy(), H_ROUTE_SMOOTH.copy()

def print_route_summary():
    print("========================================")
    print("Resumen de la ruta Pumabús CU")
    print("========================================")
    print(f"Número de paradas: {len(STOPS)}")
    print(f"Longitud total del circuito: {TOTAL_ROUTE_LENGTH:.2f} m")
    print("\nParadas y posiciones acumuladas:")
    for i, pos in enumerate(STOP_POSITIONS[:-1]):
        print(f"{i+1:02d}. {STOPS[i]} -> {pos:.2f} m")
    print(f"\nCierre del circuito -> {STOP_POSITIONS[-1]:.2f} m")

# =========================================================
# PRUEBA
# =========================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print_route_summary()

    s_route, h_route = get_route_profile()
    _, h_route_smooth = get_smoothed_route_profile()
    stop_positions = get_stop_positions()
    stop_names = get_stop_names()

    plt.figure(figsize=(12, 5))
    plt.plot(s_route, h_route, label="Elevación original", alpha=0.5)
    plt.plot(s_route, h_route_smooth, label="Elevación suavizada", linewidth=2)
    plt.scatter(stop_positions[:-1], elevation(stop_positions[:-1]), marker='o', label="Paradas")
    plt.xlabel("Distancia recorrida s [m]")
    plt.ylabel("Elevación [m]")
    plt.title("Perfil de elevación de la ruta")
    plt.grid(True)
    plt.legend()

    for i, name in enumerate(stop_names):
        plt.text(stop_positions[i], elevation(stop_positions[i]) + 0.5, name, rotation=45, fontsize=8)

    plt.figure(figsize=(12, 5))
    plt.plot(S_ROUTE, 100 * GRADE_ROUTE, label="Pendiente [%]")
    plt.scatter(stop_positions[:-1], 100 * grade(stop_positions[:-1]), marker='o', label="Paradas")
    plt.xlabel("Distancia recorrida s [m]")
    plt.ylabel("Pendiente [%]")
    plt.title("Pendiente de la ruta")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(12, 5))
    plt.plot(S_ROUTE, THETA_ROUTE, label=r"Inclinación $\theta(s)$ [rad]")
    plt.scatter(stop_positions[:-1], theta(stop_positions[:-1]), marker='o', label="Paradas")
    plt.xlabel("Distancia recorrida s [m]")
    plt.ylabel(r"$\theta(s)$ [rad]")
    plt.title("Ángulo de inclinación de la ruta")
    plt.grid(True)
    plt.legend()

    plt.show()