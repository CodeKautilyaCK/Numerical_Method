def gaussian_elimination(A, b):
    """Solve Ax = b using Gaussian elimination."""
    import copy
    n = len(A)
    M = copy.deepcopy(A)
    b = copy.deepcopy(b)

    for k in range(n):
        # Partial pivoting
        max_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if max_row != k:
            M[k], M[max_row] = M[max_row], M[k]
            b[k], b[max_row] = b[max_row], b[k]

        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n):
                M[i][j] -= factor * M[k][j]
            b[i] -= factor * b[k]

    # Back substitution
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (b[i] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def lu_decomposition(A, b):
    """LU decomposition without pivoting."""
    import copy
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = copy.deepcopy(A)

    for i in range(n):
        L[i][i] = 1.0
        for j in range(i + 1, n):
            factor = U[j][i] / U[i][i]
            L[j][i] = factor
            for k in range(i, n):
                U[j][k] -= factor * U[i][k]

    # Forward substitution
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))

    # Back substitution
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]

    return L, U, y, x


def jacobi(A, b, tol=1e-10, max_iterations=100):
    """Solve Ax = b using Jacobi iteration."""
    n = len(A)
    x = [0.0] * n

    for _ in range(max_iterations):
        x_new = [0.0] * n
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        if all(abs(x_new[i] - x[i]) < tol for i in range(n)):
            return x_new
        x = x_new

    return x
