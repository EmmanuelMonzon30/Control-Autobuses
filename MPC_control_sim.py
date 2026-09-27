import numpy as np
import matplotlib.pyplot as plt

try:
    import casadi as ca
except ImportError as exc:
    raise ImportError(
        "CasADi no está instalado. Instálalo con:\n\n"
        "    pip install casadi\n"
    ) from exc

# =========================================================
# IMPORTAR MODELO, RUTA Y PERFIL
# =========================================================
from Modelo import (
    m,
    rho,
    A_bus,
    c_a,
    c_r,
    g,
    T_m_max,
    F_b_max,
    traction_force,
    torque_brake_from_u,
    bus_dynamics_open_loop,
    rk4_step,
)

from Ruta9 import (
    theta,
    route_length,
    S_ROUTE,
    THETA_ROUTE,
)

from driving_profile_9 import (
    build_reference_functions,
    DT as DT_REF,
)


# =========================================================
# CLASE MPC
# =========================================================
class MPCController:
    """
    MPC nominal no lineal para seguimiento de velocidad.

    Estado:
        x = [s, v]

    Entrada optimizada:
        u = fuerza longitudinal neta [N]

    Convención:
        u > 0  -> tracción
        u < 0  -> frenado
    """

    def __init__(
        self,
        Ts=0.2,
        Np=50,
        v_esc=20.0,
        u_esc=None,
        du_esc=3000.0,
        alpha_v=5.0,
        alpha_u=0.10,
        alpha_du=1.0,
        du_max=6000.0,
    ):
        self.Ts = Ts
        self.Np = Np

        # Fuerza máxima de tracción a partir del torque máximo del modelo
        self.u_max = float(traction_force(T_m_max))

        # Entrada negativa para representar frenado
        self.u_min = -float(F_b_max)

        # Restricción de variación de control
        self.du_max = float(du_max)
        self.du_min = -float(du_max)

        # Escalas de normalización
        self.v_esc = float(v_esc)
        self.u_esc = float(u_esc) if u_esc is not None else self.u_max
        self.du_esc = float(du_esc)

        # Pesos adimensionales
        self.alpha_v = float(alpha_v)
        self.alpha_u = float(alpha_u)
        self.alpha_du = float(alpha_du)

        self._build_theta_interpolant()
        self._build_optimizer()

    def _build_theta_interpolant(self):
        """
        Crea interpolante CasADi para theta(s).

        Como la ruta es cíclica, dentro del optimizador usamos una
        aproximación directa sobre el intervalo [0, L].
        En simulación, s normalmente avanza de 0 a L en una vuelta.
        """
        s_grid = np.asarray(S_ROUTE, dtype=float)
        theta_grid = np.asarray(THETA_ROUTE, dtype=float)

        # CasADi interpolant 1D
        self.theta_ca = ca.interpolant(
            "theta_ca",
            "linear",
            [s_grid],
            theta_grid
        )

    def _dynamics(self, x, u):
        """
        Modelo discreto usando Forward Euler.
        """
        s = x[0]
        v = x[1]

        # Mantener s dentro del rango físico aproximado del circuito.
        # Para una vuelta, s se mantiene dentro de [0, L].
        L = route_length()
        s_clip = ca.fmin(ca.fmax(s, 0.0), L)

        th = self.theta_ca(s_clip)

        Fd = 0.5 * rho * A_bus * c_a * v**2
        Fr = m * g * (ca.sin(th) + c_r * ca.cos(th))

        s_next = s + self.Ts * v
        v_next = v + (self.Ts / m) * (u - Fd - Fr)

        return ca.vertcat(s_next, v_next)

    def _build_optimizer(self):
        Np = self.Np

        # Variables de optimización
        X = ca.SX.sym("X", 2, Np + 1)
        U = ca.SX.sym("U", 1, Np)

        # Parámetros
        x0 = ca.SX.sym("x0", 2)
        u_prev = ca.SX.sym("u_prev", 1)
        v_ref = ca.SX.sym("v_ref", Np)
        v_max = ca.SX.sym("v_max", Np)

        cost = 0.0
        constraints = []
        lbg = []
        ubg = []

        # Condición inicial
        constraints.append(X[:, 0] - x0)
        lbg += [0.0, 0.0]
        ubg += [0.0, 0.0]

        for k in range(Np):
            xk = X[:, k]
            uk = U[0, k]
            vk = X[1, k]

            # Delta u
            if k == 0:
                duk = uk - u_prev
            else:
                duk = uk - U[0, k - 1]

            # Costo normalizado
            e_v = (vk - v_ref[k]) / self.v_esc
            e_u = uk / self.u_esc
            e_du = duk / self.du_esc

            cost += self.alpha_v * e_v**2
            cost += self.alpha_u * e_u**2
            cost += self.alpha_du * e_du**2

            # Dinámica
            x_next = self._dynamics(xk, uk)
            constraints.append(X[:, k + 1] - x_next)
            lbg += [0.0, 0.0]
            ubg += [0.0, 0.0]

            # Restricción de velocidad: 0 <= v <= v_max
            constraints.append(vk)
            lbg.append(0.0)
            ubg.append(np.inf)

            constraints.append(vk - v_max[k])
            lbg.append(-np.inf)
            ubg.append(0.0)

            # Restricción de entrada
            constraints.append(uk)
            lbg.append(self.u_min)
            ubg.append(self.u_max)

            # Restricción de variación de entrada
            constraints.append(duk)
            lbg.append(self.du_min)
            ubg.append(self.du_max)

        # =====================================================
        # COSTO TERMINAL
        # =====================================================
        v_terminal = X[1, Np]
        e_v_terminal = (v_terminal - v_ref[Np - 1]) / self.v_esc

        # Penaliza que al final del horizonte la velocidad no se acerque
        # a la última referencia futura. Ayuda a anticipar frenados.
        cost += 5.0 * e_v_terminal**2

        # Velocidad terminal no negativa
        constraints.append(v_terminal)
        lbg.append(0.0)
        ubg.append(np.inf)

        opt_vars = ca.vertcat(
            ca.reshape(X, -1, 1),
            ca.reshape(U, -1, 1)
        )

        params = ca.vertcat(
            x0,
            u_prev,
            v_ref,
            v_max,
        )

        nlp = {
            "x": opt_vars,
            "f": cost,
            "g": ca.vertcat(*constraints),
            "p": params,
        }

        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 100,
        }

        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        self.nX = 2 * (Np + 1)
        self.nU = Np

        self.lbg = np.array(lbg, dtype=float)
        self.ubg = np.array(ubg, dtype=float)

        self.last_solution = None

    def solve(self, x0_val, u_prev_val, v_ref_horizon, v_max_horizon):
        Np = self.Np

        x0_val = np.asarray(x0_val, dtype=float).reshape(2,)
        v_ref_horizon = np.asarray(v_ref_horizon, dtype=float).reshape(Np,)
        v_max_horizon = np.asarray(v_max_horizon, dtype=float).reshape(Np,)

        # Evita valores negativos numéricos en la referencia.
        # Esto ayuda especialmente en zonas cercanas a paradas.
        v_ref_horizon = np.maximum(v_ref_horizon, 0.0)

        params = np.concatenate([
            x0_val,
            np.array([u_prev_val], dtype=float),
            v_ref_horizon,
            v_max_horizon,
        ])

        # Warm start simple
        if self.last_solution is None:
            X_guess = np.tile(x0_val.reshape(2, 1), (1, Np + 1))
            U_guess = np.zeros((1, Np))
            z0 = np.concatenate([
                X_guess.reshape(-1, order="F"),
                U_guess.reshape(-1, order="F")
            ])
        else:
            z0 = self.last_solution.copy()

        sol = self.solver(
            x0=z0,
            p=params,
            lbg=self.lbg,
            ubg=self.ubg,
        )

        z_opt = np.array(sol["x"]).flatten()
        self.last_solution = z_opt.copy()

        X_opt = z_opt[:self.nX].reshape((2, Np + 1), order="F")
        U_opt = z_opt[self.nX:self.nX + self.nU].reshape((1, Np), order="F")

        u_cmd = float(U_opt[0, 0])

        pred = {
            "X_opt": X_opt,
            "U_opt": U_opt,
            "cost": float(sol["f"]),
        }

        return u_cmd, pred


# =========================================================
# SIMULACIÓN INDEPENDIENTE DEL MPC
# =========================================================
def run_mpc_simulation():
    # Horizonte MPC
    Ts_mpc = 0.1
    Np = 50

    # Simulación
    dt_sim = 0.1

    # Referencias del perfil existente
    t_ref, s_ref, v_ref, a_ref, s_of_t, v_of_t, a_of_t = build_reference_functions(dt=DT_REF)
    t_final = float(t_ref[-1])

    # Tiempo de simulación
    t = np.arange(0.0, t_final, dt_sim)
    n_steps = len(t)

    # Límite de velocidad. Usamos un margen sobre la referencia para esta prueba.
    # Más adelante podemos construir v_max(s) por tramos.
    v_max_const = 35.0 / 3.6  # [m/s]

    mpc = MPCController(
        Ts=Ts_mpc,
        Np=Np,
        v_esc=20.0,
        u_esc=None,
        du_esc=6000.0,
        alpha_v=3.0,
        alpha_u=0.10,
        alpha_du=1,
        du_max=6000.0,
    )
    
    # Estado inicial
    x = np.array([0.0, 0.0], dtype=float)
    u_prev = 0.0

    # Historiales
    X_hist = np.zeros((n_steps, 2))
    U_hist = np.zeros(n_steps)
    Tm_hist = np.zeros(n_steps)
    Fb_hist = np.zeros(n_steps)
    Fm_hist = np.zeros(n_steps)
    Vref_hist = np.zeros(n_steps)
    Aref_hist = np.zeros(n_steps)

    next_mpc_time = 0.0
    u_current = 0.0

    for j, tj in enumerate(t):
        # Resolver MPC cada Ts_mpc segundos
        if tj >= next_mpc_time - 1e-9:
            future_times = tj + Ts_mpc * np.arange(Np)

            v_ref_horizon = v_of_t(future_times)
            v_max_horizon = np.full(Np, v_max_const)

            u_current, pred = mpc.solve(
                x0_val=x,
                u_prev_val=u_prev,
                v_ref_horizon=v_ref_horizon,
                v_max_horizon=v_max_horizon,
            )

            u_prev = u_current
            next_mpc_time += Ts_mpc

        # Reconstruir actuadores desde u
        Fm_cmd, Fb_cmd, Tm_cmd = torque_brake_from_u(u_current)

        # Simular planta con RK4 y el modelo original
        x = rk4_step(
            x,
            dt_sim,
            bus_dynamics_open_loop,
            Tm_cmd,
            Fb_cmd,
            theta,
        )

        # Guardar datos
        X_hist[j, :] = x
        U_hist[j] = u_current
        Tm_hist[j] = Tm_cmd
        Fb_hist[j] = Fb_cmd
        Fm_hist[j] = Fm_cmd
        Vref_hist[j] = v_of_t(tj)
        Aref_hist[j] = a_of_t(tj)

    results = {
        "t": t,
        "X": X_hist,
        "u": U_hist,
        "Tm": Tm_hist,
        "Fb": Fb_hist,
        "Fm": Fm_hist,
        "v_ref": Vref_hist,
        "a_ref": Aref_hist,
        "t_ref": t_ref,
        "s_ref": s_ref,
        "v_ref_profile": v_ref,
        "a_ref_profile": a_ref,
    }

    return results


# =========================================================
# GRÁFICAS
# =========================================================
def plot_results(results):
    t = results["t"]
    X = results["X"]
    u = results["u"]
    Tm = results["Tm"]
    Fb = results["Fb"]
    Fm = results["Fm"]
    v_ref = results["v_ref"]

    s = X[:, 0]
    v = X[:, 1]

    err_v = v_ref - v

    plt.figure(figsize=(12, 5))
    plt.plot(t, v_ref * 3.6, label="Referencia")
    plt.plot(t, v * 3.6, label="MPC")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Velocidad [km/h]")
    plt.title("Seguimiento de velocidad con MPC")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(12, 5))
    plt.plot(t, err_v * 3.6, label="Error v_ref - v")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Error de velocidad [km/h]")
    plt.title("Error de seguimiento")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(12, 5))
    plt.plot(t, s, label="s(t)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Posición [m]")
    plt.title("Posición recorrida")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(12, 5))
    plt.plot(t, u, label="u neta")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Fuerza neta [N]")
    plt.title("Entrada longitudinal neta del MPC")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(12, 5))
    plt.plot(t, Tm, label="Torque motor")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Torque [N·m]")
    plt.title("Torque reconstruido desde la fuerza neta")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(12, 5))
    plt.plot(t, Fb, label="Frenado")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Fuerza de frenado [N]")
    plt.title("Fuerza de frenado reconstruida")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(12, 5))
    plt.plot(s, v * 3.6, label="MPC")
    plt.xlabel("Posición [m]")
    plt.ylabel("Velocidad [km/h]")
    plt.title("Velocidad vs posición")
    plt.grid(True)
    plt.legend()

    plt.show()


# =========================================================
# EJECUCIÓN
# =========================================================
if __name__ == "__main__":
    results = run_mpc_simulation()
    plot_results(results)
