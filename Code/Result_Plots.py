import numpy as np
import scipy.stats as st
from scipy.optimize import brentq
from scipy.integrate import quad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


from Closed_Form_Sols import (bs_call_price, implied_vol_call, OptionPricingModels)

plt.style.use('ggplot')

plt.rcParams.update({
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "figure.titlesize": 20,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,

    "figure.constrained_layout.use": True,

    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "grid.alpha": 0.35,})

POINT_KW = dict(marker="o", facecolors="none", edgecolors="black",
                linewidths=1.4, s=60, zorder=5)
LINE_KW  = dict(linewidth=2.0, zorder=3)
BAND_KW  = dict(alpha=0.25, linewidth=0)

def _call_model(model_fn, *, K, T, extra_args, params):
    """Call model_fn for either single-asset (Asian) or exchange (Margrabe)."""
    # margrabe
    if "S1" in extra_args and "S2" in extra_args:
        S1, S2 = extra_args["S1"], extra_args["S2"]
        r   = extra_args.get("r", 0.0)
        return model_fn(S1, S2, K, T, r, *tuple(params))
    # Single-asset
    S0 = extra_args.get("S0", 1.0)
    r  = extra_args.get("r", 0.0)
    return model_fn(S0, K, T, r, *tuple(params))


def _band_from_samples(model_fn, Ks, T, extra_args, param_samples, qlo=2.5, qhi=97.5):
    """Credible band and mean across samples at each strike (fixed T)."""
    means, los, his = [], [], []
    for K in Ks:
        vals = [_call_model(model_fn, K=K, T=T, extra_args=extra_args, params=ps)
                for ps in param_samples]
        means.append(np.mean(vals))
        q = np.percentile(vals, [qlo, qhi])
        los.append(q[0]); his.append(q[1])
    return np.array(means), (np.array(los), np.array(his))



# Price vs Strike (single panel, with obs + bands)
def plot_price_vs_strike_obs(model_fn, params_mean, strikes, T, extra_args, obs_prices,
                             param_samples=None, qlo=2.5, qhi=97.5, ax=None,
                             label_mean="Posterior mean"):
    """
    Single-panel price-vs-strike with optional credible band and observed points.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
    else:
        fig = ax.figure

    line = [_call_model(model_fn, K=k, T=T, extra_args=extra_args, params=params_mean)
            for k in strikes]
    ax.plot(strikes, line, label=label_mean, **LINE_KW)

    if param_samples is not None and len(param_samples) > 1:
        _, (lo, hi) = _band_from_samples(model_fn, strikes, T, extra_args, param_samples, qlo=qlo, qhi=qhi)
        ax.fill_between(strikes, lo, hi, **BAND_KW, label="95% credible band")

    ax.scatter(strikes, obs_prices, **POINT_KW, label="Observed")

    ax.set_xlabel("Strike"); ax.set_ylabel("Price")
    ax.legend(frameon=False)
    ax.margins(x=0.02)
    return fig, ax


# 2) Price vs Strike — multiple maturities
def plot_price_vs_strike_obs_multiT(model_fn, params_mean, df, extra_args,
                                    param_samples=None, group_by="expiry",
                                    qlo=2.5, qhi=97.5, ncols=1,
                                    figure_title=None):
    """
    Multi-panel price vs strike. 
    Note: df must have columns: ['expiry','strike','T','obs_price'].
    """
    groups = list(df[group_by].unique())
    n = len(groups)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8.0, 4.0*nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, g in zip(axes, groups):
        sub = df[df[group_by] == g].sort_values("strike")
        strikes = sub["strike"].values
        T = float(sub["T"].iloc[0])
        obs = sub["obs_price"].values

        line = [_call_model(model_fn, K=k, T=T, extra_args=extra_args, params=params_mean) for k in strikes]
        ax.plot(strikes, line, label="Posterior mean", **LINE_KW)

        if param_samples is not None and len(param_samples) > 1:
            _, (lo, hi) = _band_from_samples(model_fn, strikes, T, extra_args, param_samples, qlo=qlo, qhi=qhi)
            ax.fill_between(strikes, lo, hi, **BAND_KW, label="95% credible band")

        ax.scatter(strikes, obs, **POINT_KW, label="Observed")

        ax.set_title(f"expiry={g}")
        ax.set_xlabel("Strike"); ax.set_ylabel("Price")
        if ax is axes[0]:
            ax.legend(frameon=False)

    for ax in axes[len(groups):]:
        ax.axis("off")

    if figure_title:
        fig.suptitle(figure_title)
    return fig, axes[:n]


# Residuals vs Strike
def plot_residuals_vs_strike(model_fn, params_mean, strikes, T, extra_args, obs_prices, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
    else:
        fig = ax.figure

    model = np.array([_call_model(model_fn, K=k, T=T, extra_args=extra_args, params=params_mean) for k in strikes])
    residuals = obs_prices - model

    ax.axhline(0.0, lw=1.0, color="firebrick")
    ax.scatter(strikes, residuals, **POINT_KW)
    ax.set_xlabel("Strike"); ax.set_ylabel("Residual (obs − model)")
    return fig, ax



# Posterior Predictive 
def plot_posterior_predictive(model_fn, param_samples, K, T, extra_args=None):
    """Histogram of prices generated from the parameter samples at (K,T)."""
    extra_args = extra_args or {}
    prices = [_call_model(model_fn, K=K, T=T, extra_args=extra_args, params=ps) for ps in param_samples]
    ci = np.percentile(prices, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.hist(prices, bins="auto", density=False, alpha=0.7)
    ax.axvline(ci[0], linestyle="--", color="firebrick", linewidth=1.5)
    ax.axvline(ci[1], linestyle="--", color="firebrick", linewidth=1.5)
    ax.set_xlabel("Price"); ax.set_ylabel("Frequency"); ax.set_title("Posterior Predictive")
    return fig


# Greeks 
def plot_greeks(model_fn, params, K_range, T, extra_args=None):
    """
    Greeks vs strike.
    """
    extra_args = extra_args or {}
    Ks = np.linspace(K_range[0], K_range[1], 200)
    dK = 1e-3 * np.clip(np.mean(K_range), 1e-6, None)

    def price_at(k, T_=T, ea=extra_args, p=params):
        return _call_model(model_fn, K=k, T=T_, extra_args=ea, params=p)

    # Delta, Gamma
    base = np.array([price_at(k) for k in Ks])
    plus = np.array([price_at(k + dK) for k in Ks])
    minus= np.array([price_at(k - dK) for k in Ks])
    delta = (plus - minus) / (2.0*dK)
    gamma = (plus - 2.0*base + minus) / (dK**2)

    # Theta
    dT = max(1e-4, 0.01*T)
    theta = np.array([(price_at(k, T_=T + dT) - price_at(k, T_=T - dT)) / (2.0*dT) for k in Ks])

    # Vega 
    eps = 1e-4
    def _bump(p, i, h):
        arr = np.array(p, dtype=float)
        if 0 <= i < arr.size:
            arr[i] = max(arr[i] + h, 1e-8)
        return arr

    if len(params) == 4:
        vega = np.array([(price_at(k, ea=extra_args, p=_bump(params, 0, +eps)) -
                          price_at(k, ea=extra_args, p=_bump(params, 0, -eps))) / (2*eps) for k in Ks])
    elif len(params) == 3:
        vega = np.array([(price_at(k, ea=extra_args, p=_bump(_bump(params, 0, +eps), 1, +eps)) -
                          price_at(k, ea=extra_args, p=_bump(_bump(params, 0, -eps), 1, -eps))) / (2*eps)
                         for k in Ks])
    else:
        vega = np.zeros_like(Ks)

    # Rho
    if "r" in extra_args:
        ea_hi = extra_args.copy(); ea_hi["r"] = extra_args["r"] + 1e-5
        ea_lo = extra_args.copy(); ea_lo["r"] = extra_args["r"] - 1e-5
        rho = np.array([(price_at(k, ea=ea_hi) - price_at(k, ea=ea_lo)) / (2e-5) for k in Ks])
    else:
        rho = np.zeros_like(Ks)

    fig, axs = plt.subplots(3, 2, figsize=(10.0, 10.0))
    axs = axs.ravel()

    axs[0].plot(Ks, delta, **LINE_KW); axs[0].set_title("Delta vs Strike")
    axs[1].plot(Ks, gamma, **LINE_KW); axs[1].set_title("Gamma vs Strike")
    axs[2].plot(Ks, vega,  **LINE_KW); axs[2].set_title("Vega vs Strike")
    axs[3].plot(Ks, theta, **LINE_KW); axs[3].set_title("Theta vs Strike")
    axs[4].plot(Ks, rho,   **LINE_KW); axs[4].set_title("Rho vs Strike")
    axs[5].axis("off")

    for ax in axs[:5]:
        ax.set_xlabel("Strike"); ax.set_ylabel(ax.get_title().split()[0])

    return fig, axs

# 3-D Price surface: Calibrated vs Standard + Obs pts
def plot_price_surface_3d(model_fn, params_cal, params_std, strikes, maturities,
                          extra_args=None, ax=None, market_points=None, title=None):
    """
    3-D surface plot for (Calibrated vs Standard) and optional observed market points.
    Note: market_points may be either a tuple (K_pts, T_pts, P_pts), or a DataFrame with 
    columns ['strike','T','obs_price' or 'price'].
    """
    extra_args = extra_args or {}

    K = np.asarray(strikes)
    T = np.asarray(maturities)
    Kg, Tg = np.meshgrid(K, T, indexing="xy")

    def _surface(params):
        Z = np.zeros_like(Kg, dtype=float)
        for i in range(Kg.shape[0]):
            for j in range(Kg.shape[1]):
                Z[i, j] = _call_model(model_fn, K=Kg[i, j], T=Tg[i, j], extra_args=extra_args, params=params)
        return Z

    Z_cal = _surface(params_cal)
    Z_std = _surface(params_std)

    if ax is None:
        fig = plt.figure(figsize=(6.3, 5.7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    s1 = ax.plot_surface(Kg, Tg, Z_cal, alpha=0.60, color="tab:red")
    s2 = ax.plot_surface(Kg, Tg, Z_std, alpha=0.55, color="tab:blue")

    # Observed market points
    if market_points is not None:
        if hasattr(market_points, "columns"):
            Ks = market_points["strike"].values
            Ts = market_points["T"].values if "T" in market_points.columns else np.full_like(Ks, Tg.min())
            price_col = "obs_price" if "obs_price" in market_points.columns else market_points.columns[-1]
            Ps = market_points[price_col].values
        else:
            Ks, Ts, Ps = market_points
        ax.scatter(Ks, Ts, Ps, s=55, facecolors="none", edgecolors="black",
                   linewidths=1.6, depthshade=False, zorder=8)

    handles = [Patch(facecolor="tab:red", alpha=0.60, label="Calibrated"),
               Patch(facecolor="tab:blue", alpha=0.55, label="Standard")]
    if market_points is not None:
        handles.append(Line2D([0], [0], **POINT_KW, label="Observed"))
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.98))

    ax.set_xlabel("Strike"); ax.set_ylabel("Maturity"); ax.set_zlabel("Price")
    if title:
        ax.set_title(title)
    return fig


# Model comparison bars (BIC/DIC)
def plot_bic_dic(names, bic, dic, ax=None):
    x = np.arange(len(names))
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.6, 5.3))
    else:
        fig = ax.figure
    w = 0.35
    ax.bar(x - w/2, bic, width=w, label="BIC")
    ax.bar(x + w/2, dic, width=w, label="DIC")
    ax.set_xticks(x, names); ax.set_ylabel("Score"); ax.set_title("Model Comparison")
    ax.legend(frameon=False)
    return fig, ax


# Simple surfaces difference
def plot_price_surface_diff(model_fn, params_cal, params_std, strikes, maturities, extra_args=None, ax=None):
    """single surface showing Calibrated - Standard differences."""
    extra_args = extra_args or {}
    K = np.asarray(strikes)
    T = np.asarray(maturities)
    Kg, Tg = np.meshgrid(K, T, indexing="xy")
    Z = np.zeros_like(Kg, dtype=float)
    for i in range(Kg.shape[0]):
        for j in range(Kg.shape[1]):
            Z[i, j] = (_call_model(model_fn, K=Kg[i, j], T=Tg[i, j], extra_args=extra_args, params=params_cal) -
                       _call_model(model_fn, K=Kg[i, j], T=Tg[i, j], extra_args=extra_args, params=params_std))
    if ax is None:
        fig = plt.figure(figsize=(6.3, 5.7)); ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
    ax.plot_surface(Kg, Tg, Z, cmap="viridis")
    ax.set_xlabel("Strike"); ax.set_ylabel("Maturity"); ax.set_zlabel("Cal - Std")
    ax.set_title("Price Surface Difference")
    return fig


# Simple parameter evolution
def plot_parameter_evolution(param_series, dates, ax=None):
    ax = ax or plt.gca()
    for name, series in param_series.items():
        ax.plot(dates, series, **LINE_KW, label=name)
    ax.legend(frameon=False)
    ax.set_title("Parameter Evolution")
    return ax


# Path simulation helpers 
def simulate_paths(model_sde, params, S0, T, steps, n_paths):
    dt = T / steps
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = S0
    for i in range(steps):
        dW = np.random.normal(0.0, np.sqrt(dt), size=n_paths)
        paths[:, i+1] = model_sde(paths[:, i], dW, dt, params)
    return paths


def plot_path_simulation(model_sde, params_cal, params_std, S0, T, steps, n_paths):
    pc = simulate_paths(model_sde, params_cal, S0, T, steps, n_paths)
    ps = simulate_paths(model_sde, params_std, S0, T, steps, n_paths)
    t = np.linspace(0.0, T, steps + 1)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for i in range(min(n_paths, 10)):
        ax.plot(t, pc[i], color="tab:blue", alpha=0.35)
        ax.plot(t, ps[i], color="tab:red",  alpha=0.35)
    ax.set_title("Sample Paths (blue=cal, red=std)")
    ax.set_xlabel("Time"); ax.set_ylabel("Price")
    return fig