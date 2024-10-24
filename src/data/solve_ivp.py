import numpy as np
from scipy.integrate import odeint


def ODE_nonautonomous(y, t, u, m, c, k):
    y1 = y[0]
    y2 = y[1]
    dxdt1 = y2
    dxdt2 = (-k * y1 - c * y2 + u) / m
    dxdt = [dxdt1, dxdt2]
    return dxdt


def solve_nonautonomous_RK4(y0, tspan, u, m, c, k):
    y = np.zeros((len(tspan), 2))

    for i in range(1, len(tspan)):
        t = [tspan[i - 1], tspan[i]]
        yn = odeint(ODE_nonautonomous, y0, t, args=(u[i], m, c, k))[1]
        y[i] = yn
        y0 = yn

    return y
