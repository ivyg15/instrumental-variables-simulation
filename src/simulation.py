import numpy as np

def generate_data(data_size, beta, pi, gamma, delta=0):
    rng = np.random.default_rng()
    U = rng.normal(size=data_size)
    Z = rng.normal(size=data_size)
    e_x = rng.normal(size=data_size)
    e_y = rng.normal(size=data_size)

    X = pi * Z + gamma * U + e_x
    Y = beta * X + delta * Z + U + e_y

    return X, Y, Z