# main.py

'''

We are limited on result plot data since we have no access to real data on margrabe and asian option data for 
any asset (all behind a pay wall). I made up some data points but with no real value behind the option contract

The result data only compares the results of the MCMC params to random params I made up below. As a result,
we can't technically (unless you have any idea) track whether the MCMC was more accurate or not.

Let me know how you want to move forward on this in terms of the result section of the paper.



To Do List:
    --- Run HMC code

'''

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import matplotlib.pyplot as plt

from Closed_Form_Sols import OptionPricingModels
from data_generator import (
    gen_asian_quotes, 
    gen_margrabe_quotes, 
    asian_posterior_fit, 
    margrabe_posterior_fit, 
    scores)
from Result_Plots import (
    plot_price_surface_3d,
    plot_price_surface_diff,
    plot_greeks,
    plot_posterior_predictive,
    plot_bic_dic,
    plot_parameter_evolution,
    plot_path_simulation,)

from Black_Scholes_3_0 import adaptive_rwm, transform ## Black Scholes Multi-Asset
from temp_BDMCMC import birth_death_mcmc, transform as mjd_transform ## Merton Single-Asset model (CHANGE TO HMC)#

## include multi asset Merton HMC
## include single asset Heston HMC
## include multi asset Heston HMC 

def burnin(chain, burn_frac=0.2):
    n = chain.shape[0]
    start = int(burn_frac * n)
    return chain[start:].copy()

tickers = ["CORN", "SOYB"]
data = yf.download(tickers, period='3y', auto_adjust=False)["Adj Close"]
corndata = data[tickers[0]]
soybdata = data[tickers[1]]
r = 0.01

def get_chain_df(ticker):
    tk = yf.Ticker(ticker)
    rows = []
    for expiry in tk.options:
        chain = tk.option_chain(expiry).calls
        chain = chain.assign(
            expiry=expiry,
            ticker=ticker,
            mid=0.5*(chain.bid + chain.ask)
        )
        rows.append(chain[["ticker","expiry","strike","bid","ask","mid"]])
    return pd.concat(rows, ignore_index=True)

df_calls = get_chain_df("CORN")
df_puts  = get_chain_df("SOYB").rename(columns={"bid":"bid_put","ask":"ask_put","mid":"mid_put"})

# Synthetic Margrabe
df_margrabe = (
    df_calls.merge(df_puts, on=["expiry","strike"], how="inner")
            [["expiry","strike"]]
            .drop_duplicates()
            .assign(T=lambda d: (pd.to_datetime(d["expiry"]) - pd.Timestamp(date.today())).dt.days/365)
            .sort_values(["expiry","strike"])
            .reset_index(drop=True))


# Asian Options
expiry_asian = df_calls.expiry.unique()[-1]
df_asian = df_calls.query("expiry == @expiry_asian").copy()


######## Add MCMC Code Here ##########

## AMwG <- Margrabe

print("Running Multi Asset MCMC code...")
xA = np.log(corndata.values)
xB = np.log(soybdata.values)
theta0_bs = np.array([np.log(0.2), np.log(0.2), np.arctanh(0.0)])  
chain_bs = adaptive_rwm(
    N=20000,
    xproc=xA, yproc=xB,
    theta0=theta0_bs,
    r=r,
    initial_cov=np.diag([0.01,0.01,0.01]),
    adaptation_start=500,
    adaptation_cooling=0.995)
chain_params_bs = np.array([transform(th) for th in chain_bs])  

# MJD‐Asian MCMC (Birth–Death but might switch to HMC)
print("Running Single-Asset MCMC code...")


theta0_mjd = np.array([0.2, 1.0, 0.0, 0.1]) 
N0 = np.random.poisson(theta0_mjd[1], size=(len(xA)+1,))
chain_mjd, chainN_mjd = birth_death_mcmc(
    sims=20000,
    xproc=xA,
    theta0=theta0_mjd,
    N0=N0,
    r=r)
chain_params_mjd = np.array([mjd_transform(th) for th in chain_mjd]) 

def mjd_daily_to_annual(params):
    sigma_d, lam_d, nu, delta_d = params
    return np.array([sigma_d * np.sqrt(252),
                     lam_d * 252,nu,delta_d * np.sqrt(252)])


chain_params_mjd_annual = np.apply_along_axis(mjd_daily_to_annual, 1, chain_params_mjd)
cal_par_mjd_annual     = chain_params_mjd_annual.mean(axis=0)


chain_params_bs  = burnin(chain_params_bs,  burn_frac=0.2)
chain_params_mjd = burnin(chain_params_mjd, burn_frac=0.2)

##########################
# closed‐form model prices
##########################
# Using the MCMC posterior means:

print("Running Closed-Form Solutions")
cal_par_bs  = np.mean(chain_params_bs, axis=0) # [sigma1, sigma2, rho]
cal_par_mjd = np.mean(chain_params_mjd, axis=0) # [sigma, lam, nu, delta]

T_asian    = (pd.to_datetime(expiry_asian)    - pd.Timestamp(date.today())).days / 365
T_margrabe = (pd.to_datetime(df_margrabe.expiry.iloc[0]) - pd.Timestamp(date.today())).days / 365



#########
## DATA GENERATOR
#########

dgp_asian   = "heston" # "bs" | "mjd" | "heston"
dgp_marg    = "bs" # "bs" | "mjd" | "heston"

theta_star_asian = (0.04, 1.5, 0.04, 0.5, -0.4) # (v0, kappa, theta, xi, rho) for Heston
#theta_star_asian = (0.2, 0.1, -0.05, 0.3)
theta_star_marg  = tuple(cal_par_bs) # (sigma1, sigma2, rho) for BS

# Asian observed quotes (single expiry) 
df_asian = df_asian.sort_values("strike").copy()
asian_obs = gen_asian_quotes(
    dgp=dgp_asian, theta_star=theta_star_asian,
    S0=corndata.iloc[-1], r=r, strikes=df_asian["strike"].values, T=T_asian,
    noise_frac=0.02)

#assert asian_obs[0] > 0.0, "ITM Asian price should be > 0"
#assert np.all(np.diff(asian_obs) <= 1e-10), "Asian price must be non-increasing in strike"

df_asian["obs_price"] = asian_obs

# Margrabe observed quotes (multiple expiries) 
S1_0 = float(corndata.iloc[-1])
S2_0 = float(soybdata.iloc[-1])
K_ratios = np.linspace(0.5, 1.2, 10)
expiries_mar = pd.to_datetime(df_calls.expiry.unique())
t_maturities_mar = ((expiries_mar - pd.Timestamp(date.today())).days / 365).values
t_maturities_mar = np.sort(t_maturities_mar[t_maturities_mar > 0])[:4]

df_margrabe = (pd.DataFrame([(T, K) for T in t_maturities_mar for K in K_ratios],
        columns=["T","K_ratio"]))
df_margrabe["strike"] = df_margrabe["K_ratio"]                      
df_margrabe["expiry"] = (
    pd.Timestamp(date.today()) + pd.to_timedelta((df_margrabe["T"]*365).round().astype(int), unit="D")
).dt.strftime("%Y-%m-%d")

df_margrabe["model_bs_margrabe_cal"] = df_margrabe.apply(
    lambda row: OptionPricingModels.bs_margrabe_call(S1=S1_0, S2=S2_0,
        K=float(row["K_ratio"]), T=float(row["T"]),
        r=r, sigma1=cal_par_bs[0], sigma2=cal_par_bs[1], rho=cal_par_bs[2]
    ), axis=1)

mar_obs = gen_margrabe_quotes(
    dgp=dgp_marg, theta_star=theta_star_marg, S1=S1_0, S2=S2_0,
    r=r, strikes=df_margrabe["strike"].values,
    T=df_margrabe["T"].values, noise_frac=0.02)
df_margrabe["obs_price"] = mar_obs

###########  


### Benchmark parameters ###
std_par_M = (0.3, 0.25, 0.0) # BS Margrabe: sigma1, sigma2, rho
std_par_A = (0.2, 1.0, 0.0, 0.1) # MJD Asian: sigma, lam, nu, delta


# Fair price values
df_asian['model_mjd_asian'] = df_asian.strike.apply(
    lambda K: OptionPricingModels.merton_asian_geometric_call(
        S0=corndata.iloc[-1], K=K, T=T_asian, r=r,
        sigma=cal_par_mjd_annual[0], lam=cal_par_mjd_annual[1],
        nu=cal_par_mjd_annual[2], delta=cal_par_mjd_annual[3]))

asian_mean, asian_band = asian_posterior_fit(
    S0=corndata.iloc[-1], r=r,
    strikes=df_asian["strike"].values, T=T_asian,
    param_samples=chain_params_mjd_annual,  
    model="mjd")
asian_scores = scores(df_asian["obs_price"].values, asian_mean, asian_band)
print("Asian — MJD fit vs", dgp_asian, "market:", asian_scores)

mar_mean, mar_band = margrabe_posterior_fit(
    S1=corndata.iloc[-1], S2=soybdata.iloc[-1], r=r,
    strikes=df_margrabe["strike"].values, T=df_margrabe["T"].values,
    param_samples=chain_params_bs,
    model="bs")
mar_scores = scores(df_margrabe["obs_price"].values, mar_mean, mar_band)
print("Margrabe — BS fit vs", dgp_marg, "market:", mar_scores)

#####################
# Result Plots
#####################

print("Running Plots...")

expiries_mar = pd.to_datetime(df_margrabe.expiry.unique())
t_maturities_mar = ((expiries_mar - pd.Timestamp(date.today())).days / 365).values
T_asian = (pd.to_datetime(expiry_asian) - pd.Timestamp(date.today())).days / 365
t_maturities_asian = np.linspace(T_asian*0.2, T_asian, 10)


# 3D surface plot 
fig_asian = plot_price_surface_3d(
    model_fn=OptionPricingModels.merton_asian_geometric_call,
    params_cal=cal_par_mjd_annual,
    params_std=std_par_A,
    strikes=np.sort(df_asian.strike.unique()),
    maturities=t_maturities_asian,
    extra_args={"S0": corndata.iloc[-1], "r":r})
fig_asian.suptitle("Asian (CORN) Price Surface - MJD vs Benchmark")

fig_mar = plot_price_surface_3d(
    model_fn=OptionPricingModels.bs_margrabe_call,
    params_cal=cal_par_bs,
    params_std=std_par_M,
    strikes=np.sort(df_margrabe.strike.unique()),
    maturities=t_maturities_mar,
    extra_args={"S1": corndata.iloc[-1], "S2": soybdata.iloc[-1], "r":r})

# Greeks
fig_g_asian, axes_asian = plot_greeks(
    model_fn=OptionPricingModels.merton_asian_geometric_call,
    params=cal_par_mjd_annual,
    K_range=(df_asian.strike.min(), df_asian.strike.max()),
    T=T_asian,
    extra_args={"S0": corndata.iloc[-1], "r": r,
        "sigma": cal_par_mjd_annual[0],
        "lam":  cal_par_mjd_annual[1],
        "nu":   cal_par_mjd_annual[2],
        "delta":cal_par_mjd_annual[3],})
fig_g_asian.suptitle("Asian Option Greeks — MJD Calibrated")

fig_g_mar, axes_mar = plot_greeks(
    model_fn=OptionPricingModels.bs_margrabe_call,
    params=cal_par_bs,
    K_range=(df_margrabe.strike.min(), df_margrabe.strike.max()),
    T=t_maturities_mar[0],
    extra_args={"S1": corndata.iloc[-1], "S2": soybdata.iloc[-1],
        "r": r, "sigma1": cal_par_bs[0],
        "sigma2": cal_par_bs[1], "rho": cal_par_bs[2],})
fig_g_mar.suptitle("Margrabe Option Greeks — BS Calibrated")

# Posterior‑predictive.
fig_pp = plot_posterior_predictive(
    model_fn=OptionPricingModels.merton_asian_geometric_call,
    param_samples=chain_params_mjd_annual,
    K=df_asian.strike.iloc[len(df_asian)//2],
    T=T_asian,
    extra_args={"S0": corndata.iloc[-1], "r": r})

from temp_BDMCMC import llike as mjd_llike
from Black_Scholes_3_0 import llike as bs_llike

def _loglikes_mjd(param_samples, nproc_samples, xproc, r):
    return np.array([mjd_llike(th, xproc, N, r) for th, N in zip(param_samples, nproc_samples)])

def _loglikes_bs(param_samples, xproc, yproc, r):
    return np.array([bs_llike(th, xproc, yproc, r) for th in param_samples])

def _dic_bic_from_loglikes(loglikes, loglike_bar, k, n, shift_c=None):
    """Compute DIC/BIC from per-draw log-likelihood. shift_c preserves rankings but pushes values positive."""
    if shift_c is None:
        # conventional
        D = -2.0 * loglikes
        Dbar = D.mean()
        D_at_bar = -2.0 * loglike_bar
        pD = Dbar - D_at_bar
        DIC = Dbar + pD
        BIC = -2.0 * loglikes.max() + k * np.log(n)
        return DIC, BIC
    else:
        # shifted
        D_shift = -2.0 * (loglikes - shift_c)
        Dbar = D_shift.mean()
        D_at_bar_shift = -2.0 * (loglike_bar - shift_c)
        pD = Dbar - D_at_bar_shift
        DIC = Dbar + pD

        BIC = -2.0 * (loglikes.max() - shift_c) + k * np.log(n)
        return DIC, BIC

loglikes_mjd = _loglikes_mjd(chain_params_mjd, chainN_mjd, xA, r)
loglikes_bs  = _loglikes_bs(chain_params_bs, xA, xB, r)

theta_bar_mjd = chain_params_mjd.mean(axis=0)
N_bar_mjd = np.rint(chainN_mjd.mean(axis=0)).astype(int)
loglike_bar_mjd = mjd_llike(theta_bar_mjd, xA, N_bar_mjd, r)

theta_bar_bs = chain_params_bs.mean(axis=0)
loglike_bar_bs = bs_llike(theta_bar_bs, xA, xB, r)

global_shift_c = max(loglikes_mjd.max(), loglikes_bs.max(), loglike_bar_mjd, loglike_bar_bs)

k_mjd = chain_params_mjd.shape[1]
k_bs  = chain_params_bs.shape[1] 
n_mjd = len(xA) - 1
n_bs  = len(xA) - 1

# conventional values:
dic_asian_raw, bic_asian_raw = _dic_bic_from_loglikes(loglikes_mjd, loglike_bar_mjd, k_mjd, n_mjd, shift_c=None)
dic_marg_raw,  bic_marg_raw  = _dic_bic_from_loglikes(loglikes_bs,  loglike_bar_bs,  k_bs,  n_bs,  shift_c=None)

# shifted-positive versions: This is so that it is easier to read in the paper
dic_asian_pos, bic_asian_pos = _dic_bic_from_loglikes(loglikes_mjd, loglike_bar_mjd, k_mjd, n_mjd, shift_c=global_shift_c)
dic_marg_pos,  bic_marg_pos  = _dic_bic_from_loglikes(loglikes_bs,  loglike_bar_bs,  k_bs,  n_bs,  shift_c=global_shift_c)

fig_mc, ax_mc = plt.subplots()
plot_bic_dic(
    names=["MJD-Asian", "BS-Margrabe"],
    bic=[bic_asian_pos, bic_marg_pos],
    dic=[dic_asian_pos, dic_marg_pos],
    ax=ax_mc
)


from Result_Plots import plot_price_vs_strike_obs, plot_residuals_vs_strike, plot_price_vs_strike_obs_multiT

# Asian: observed vs model with band
fig_asian_obs, ax_asian_obs = plt.subplots()
plot_price_vs_strike_obs(
    model_fn=OptionPricingModels.merton_asian_geometric_call,
    params_mean=np.mean(chain_params_mjd_annual, axis=0),
    strikes=df_asian["strike"].values, T=T_asian,
    extra_args={"S0": corndata.iloc[-1], "r": r},
    obs_prices=df_asian["obs_price"].values,
    param_samples=chain_params_mjd_annual)
ax_asian_obs.set_title(f"Asian — Fitted MJD vs {dgp_asian.upper()} market")

# Asian residuals
fig_asian_res, ax_asian_res = plt.subplots()
plot_residuals_vs_strike(
    model_fn=OptionPricingModels.merton_asian_geometric_call,
    params_mean=np.mean(chain_params_mjd_annual, axis=0),
    strikes=df_asian["strike"].values, T=T_asian,
    extra_args={"S0": corndata.iloc[-1], "r": r},
    obs_prices=df_asian["obs_price"].values,
    ax=ax_asian_res)

# Margrabe residuals 
one_exp = df_margrabe['expiry'].unique()[0]
sub = df_margrabe[df_margrabe['expiry'] == one_exp].sort_values('strike')
fig_marg_res, ax_marg_res = plt.subplots()
plot_residuals_vs_strike(
    model_fn=OptionPricingModels.bs_margrabe_call,
    params_mean=np.mean(chain_params_bs, axis=0),
    strikes=sub['strike'].values,
    T=float(sub['T'].iloc[0]),
    extra_args={"S1": S1_0, "S2": S2_0, "r": r},
    obs_prices=sub['obs_price'].values,
    ax=ax_marg_res)

# Margrabe (multi-expiry) observed vs posterior mean + band
fig_marg_obs, axes_marg_obs = plot_price_vs_strike_obs_multiT(
    model_fn=OptionPricingModels.bs_margrabe_call,
    params_mean=np.mean(chain_params_bs, axis=0),
    df=df_margrabe[['expiry','strike','T','obs_price']],
    extra_args={"S1": corndata.iloc[-1], "S2": soybdata.iloc[-1], "r": r},
    param_samples=chain_params_bs,
    group_by='expiry')


# Margrabe posterior-predictive (same expiry + mid-strike)
K_pp = sub['strike'].iloc[len(sub)//2]
T_pp = float(sub['T'].iloc[0])
fig_pp_marg = plot_posterior_predictive(
    model_fn=OptionPricingModels.bs_margrabe_call,
    param_samples=chain_params_bs,
    K=K_pp, T=T_pp,
    extra_args={"S1": S1_0, "S2": S2_0, "r": r})
fig_pp_marg.suptitle("Margrabe — Posterior Predictive")


plt.show()