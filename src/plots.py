import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

def plot_histogram(estimates_list, titles, beta, suptitle=None):
    num_plots = len(estimates_list)
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    if suptitle:
        fig.suptitle(suptitle)

    for i in range(num_plots):
        estimates = estimates_list[i]
        axes[i].hist(estimates, bins=20)
        axes[i].axvline(x=beta, color="orange", label="true value")
        axes[i].axvline(x=sum(estimates)/len(estimates), color="red", label="mean of estimates")
        axes[i].legend()
        axes[i].set_title(titles[i])
    plt.show()
    return None

def plot_metrics(data_sizes, OLS_trends, IV_trends):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Metrics Comparison")

    axes[0].plot(data_sizes, IV_trends["bias"], color="blue", label="IV")
    axes[0].plot(data_sizes, OLS_trends["bias"], color="red", label="OLS")
    axes[0].set_title("Bias Comparison")
    axes[0].set_xlabel("Sample Size")
    axes[0].set_ylabel("Bias")
    axes[0].legend()

    axes[1].plot(data_sizes, np.log(IV_trends["var"]), color="blue", label="IV")
    axes[1].plot(data_sizes, np.log(OLS_trends["var"]), color="red", label="OLS")
    axes[1].set_title("Variance Comparison")
    axes[1].set_xlabel("Sample Size")
    axes[1].set_ylabel("ln(Variance)")
    axes[1].legend()

    axes[2].plot(data_sizes, np.log(IV_trends["mse"]), color="blue", label="IV")
    axes[2].plot(data_sizes, np.log(OLS_trends["mse"]), color="red", label="OLS")
    axes[2].set_title("MSE Comparison")
    axes[2].set_xlabel("Sample Size")
    axes[2].set_ylabel("ln(MSE)")
    axes[2].legend()
    plt.show()
    return None

def plot_heatmap(results):
    norm = SymLogNorm(
        linthresh=0.01,
        linscale=1.0,
        vmin=results.min(),
        vmax=results.max(),
        base=10
    )

    fig, ax = plt.subplots()
    im = ax.imshow(results, origin="lower", norm=norm, cmap="coolwarm")
    ax.set_ylabel("Instrument Strength (pi)")
    ax.set_yticks(np.arange(0, 25, 5), np.round(np.arange(0.01, 1.5, 0.298), decimals=2))
    ax.set_xlabel("Endogeneity Strength (gamma)")
    ax.set_xticks(np.arange(0, 25, 5), np.round(np.arange(0.05, 2.0, 0.39), decimals=2))
    ax.set_title("MSE Difference")
    plt.colorbar(im, label="MSE(IV) - MSE(OLS)")
    plt.show()
    return None