import numpy as np

def hp_filter(series, lamb=1600):
    y = np.array(series, dtype=float)
    T = len(y)

    D = np.zeros((T - 2, T))
    for i in range(T - 2):
        D[i, i]   =  1
        D[i, i+1] = -2
        D[i, i+2] =  1

    trend = np.linalg.solve(np.eye(T) + lamb * D.T @ D, y)
    cycle = y - trend

    return trend, cycle
    