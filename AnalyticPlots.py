## Analytical Plots

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable

def ess(chain, max_lag=None):
    """ESS = N/ (1 + 2 * sum_{k=1}^K acf(k))"""
    
    N = len(chain)
    x = chain - np.mean(chain) # centre of chain
    
    if max_lag is None:
        max_lag = N // 2
    
    
    acf = []

    var = np.var(x)
    if var < 1e-14:
        return float(N) # ESS = N for very small variance

    for lag in range(max_lag+1):
        acf_lag = (np.sum(x[:N-lag]*x[lag:]) / (N - lag)) / var # covariance at lag / var
        acf.append(acf_lag)

    s = 0.0
    for lag in range(1, max_lag+1):
        if acf[lag] <= 0.0:
            break # lag becomes strictly negative
        s += acf[lag]
    
    ess = N / (1 + 2*s)
    return ess

def summary(theta, param_names):
    """PrettyTable of each param's mean, std, ess"""
    table = PrettyTable()
    table.field_names = ["Parameter", "Mean", "Standard Deviation", "Effective Sample Size (ESS)"]
    
    for i, name in enumerate(param_names):
        param = theta[:, i]
        mean = np.mean(param)
        std = np.std(param)
        ess_ = ess(param)
        
        table.add_row([name, f"{mean:.4f}", f"{std:.4f}", f"{ess_:.6f}"])

    print(table)
    
def plots(theta, param_names):
    """Trace, ACF, Histogram, and KDE plots"""
    
    num_params = len(param_names)
    colours = ['darkgreen', 'steelblue', 'darkorange', 'purple', 'brown', 'teal']
    
    fig, axs = plt.subplots(num_params, 4, figsize=(10, 4 * num_params))

    for i, name in enumerate(param_names):
        param = theta[:, i]
        colour = colours[i % len(colours)]
        
        #Trace 
        axs[i, 0].plot(param, color=colour)
        axs[i, 0].set_title("Trace of sigma")
        axs[i, 0].set_xlabel("Iteration")
        axs[i, 0].set_ylabel("sigma")
    
        # ACF
        axs[i, 1].acorr(param - np.mean(param), maxlags=100,
                usevlines=True, normed=True, color=colour)
        axs[i, 1].set_title("ACF of sigma")
    
        # Histogram
        axs[i, 2].hist(param, bins=40, density=True, color=colour, alpha=0.6)
        axs[i, 2].set_title("Histogram of sigma")
        axs[i, 2].set_xlabel("sigma")
    
        # KDE 
        sns.kdeplot(param, ax=axs[i, 3], fill=True, color=colour)
        axs[i, 3].set_title("KDE of sigma")
    
        plt.tight_layout()
        plt.show()

def post_pred(y_obs, y_preds):
    """Creates posterior predictive intervals for a time-series forecast"""
    T = len(y_obs)
    steps = y_preds.shape[1]
    time_hist = np.arange(T)
    time_future = np.arange(T, T + steps)

    percentiles = [2.5, 25, 50, 75, 97.5]
    inf_data = np.percentile(y_preds, percentiles, axis=0) 

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_hist, y_obs, color='black', label="Historical data")
    median_forecast = inf_data[2]  
    plt.plot(time_future, median_forecast, color='blue', label="Median forecast")

    plt.fill_between(
        time_future, 
        inf_data[1],  # 25th
        inf_data[3],  # 75th
        color='blue',
        alpha=0.2,
        label="50% credible interval"
    )
    plt.fill_between(
        time_future,
        inf_data[0],  # 2.5th
        inf_data[-1], # 97.5th
        color='blue',
        alpha=0.1,
        label="95% credible interval"
    )

    plt.xlabel("Time Index")
    plt.ylabel("Y")
    plt.title("Posterior Predictive")
    plt.legend(loc='upper left')
    plt.show()