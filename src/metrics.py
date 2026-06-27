import numpy as np

def get_metrics(estimates, beta):
    return {
        "var": np.var(estimates),
        "bias": np.mean(estimates) - beta,
        "mse": np.mean((estimates - beta) ** 2)
    }
