import numpy as np
from src.simulation import generate_data

def get_OLS_estimates(N, **data_kwargs):
    beta_OLS_estimates = np.zeros(N)
    for i in range(N):
        X, Y, _ = generate_data(**data_kwargs)
        beta_OLS = 1/(X.T @ X) * (X.T @ Y)
        beta_OLS_estimates[i] = beta_OLS
    return beta_OLS_estimates

def get_IV_estimates(N, **data_kwargs):
    beta_IV_estimates = np.zeros(N)
    for i in range(N):
        X, Y, Z = generate_data(**data_kwargs)
        pi_hat = 1/(Z.T @ Z) * (Z.T @ X)
        X_hat = pi_hat * Z
        beta_IV = 1/(X_hat.T @ X_hat) * (X_hat.T @ Y)
        beta_IV_estimates[i] = beta_IV
    return beta_IV_estimates