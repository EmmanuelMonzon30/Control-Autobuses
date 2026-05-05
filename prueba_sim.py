import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Ejemplo: sistema masa-resorte-amortiguador
# x1 = posición, x2 = velocidad
m = 1.0
c = 0.5
k = 4.0

def f(t, x):
    x1, x2 = x
    u = 0.0  # entrada (puedes cambiarla)
    dx1 = x2
    dx2 = (u - c*x2 - k*x1)/m
    return [dx1, dx2]

t_span = (0, 10)
x0 = [1.0, 0.0]
t_eval = np.linspace(t_span[0], t_span[1], 2000)

sol = solve_ivp(f, t_span, x0, t_eval=t_eval)

plt.plot(sol.t, sol.y[0], label="posición x1")
plt.plot(sol.t, sol.y[1], label="velocidad x2")
plt.grid(True)
plt.xlabel("t [s]")
plt.legend()
plt.show()