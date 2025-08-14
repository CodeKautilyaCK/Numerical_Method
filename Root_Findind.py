import numpy as np

def bisection(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) >= 0:
        raise ValueError("Function must have opposite signs at endpoints.")
    for _ in range(max_iter):
        c = (a + b) / 2
        if abs(func(c)) < tol:
            return c
        elif func(a) * func(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

def newton(func, dfunc, x0, tol=1e-6, max_iter=100):
    x = x0
    for _ in range(max_iter):
        fx = func(x)
        dfx = dfunc(x)
        if dfx == 0:
            raise ValueError("Derivative is zero.")
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

def secant(func, x0, x1, tol=1e-6, max_iter=100):
    for _ in range(max_iter):
        f0, f1 = func(x0), func(x1)
        if f1 - f0 == 0:
            raise ValueError("Zero division in secant method.")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x2 - x1) < tol:
            return x2
        x0, x1 = x1, x2
    return x1
