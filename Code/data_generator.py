## Data Generator

'''
This is to generate option prices for the Asian and Margrabe options since there are no free APIs for exotic options 

How this works:
    1) generate option prices using one of the pricing models (we have three: BS, MJD, Heston)
    2) After picking one we calculate fair prices using the other two pricing models
    3) Simple and hopefully a quick fix for the paper
'''

import numpy as np
from typing import Tuple, Dict, Literal, Optional
from Closed_Form_Sols import OptionPricingModels

DGP = Literal["bs", "mjd", "heston"]

### Pricing Kernels

def _asian_pricer(dgp: DGP):
    if dgp == "bs":
        return OptionPricingModels.bs_asian_geometric_call
    if dgp == "mjd":
        return OptionPricingModels.merton_asian_geometric_call
    if dgp == "heston":
        #return OptionPricingModels.heston_asian_geometric_call
        return OptionPricingModels.heston_asian_geometric_call_mm
    raise ValueError(f"Unknown dgp={dgp}")

def _margrabe_pricer(dgp: DGP):
    if dgp == "bs":
        return OptionPricingModels.bs_margrabe_call
    if dgp == "mjd":
        return OptionPricingModels.merton_margrabe_call  # supports many params
    if dgp == "heston":
        return OptionPricingModels.heston_margrabe_call
    raise ValueError(f"Unknown dgp={dgp}")
    
### Synthetic Data

def gen_asian_quotes(dgp: DGP, theta_star: Tuple[float, ...], *, S0: float, r: float,
                     strikes: np.ndarray, T: float, noise_frac: float = 0.02,
                     rng: Optional[np.random.Generator] = None,) -> np.ndarray:
    """
    semi-synthetic observed prices for geometric Asian call using the given data-generating model (dgp).
    Note: theta_star must match the dgp:
      - bs: (sigma,)
      - mjd: (sigma, lam, nu, delta)
      - heston: (v0, kappa, theta, xi, rho)
    """
    rng = rng or np.random.default_rng()
    pricer = _asian_pricer(dgp)
    prices = np.array([
        pricer(S0, float(K), float(T), r, *theta_star)
        for K in strikes], dtype=float)
    
    prices = np.asarray(prices, float)
    prices = np.where(prices < 0, np.maximum(prices, -1e-10), prices)
    noise = rng.normal(0.0, noise_frac * np.maximum(prices, 1e-12))
    return np.maximum(prices + noise, 0.0)


def gen_margrabe_quotes(dgp: DGP, theta_star: Tuple[float, ...], *,
                        S1: float, S2: float, r: float,
                        strikes: np.ndarray, T: np.ndarray, noise_frac: float = 0.02,
                        rng: Optional[np.random.Generator] = None,) -> np.ndarray:
    """
    semi-synthetic observed prices for Margrabe exchange option using the given data-generating model.

    Note: theta_star must match the dgp:
      - bs: (sigma1, sigma2, rho)
      - mjd: (sigma1, sigma2, lam1, lam2, nu1, nu2, delta1, delta2, rho)
      - heston: (v0, kappa, theta, xi, rho)
    """
    rng = rng or np.random.default_rng()
    pricer = _margrabe_pricer(dgp)
    strikes = np.asarray(strikes, dtype=float)
    T = np.asarray(T, dtype=float)
    if T.size == 1:
        T = np.full_like(strikes, T.item(), dtype=float)

    prices = np.array([
        pricer(S1, S2, float(K), float(t), r, *theta_star)
        for K, t in zip(strikes, T)], dtype=float)
    noise = rng.normal(0.0, noise_frac * np.maximum(prices, 1e-12))
    return prices + noise

### Posterior curve

def asian_posterior_fit(*, S0: float, r: float, strikes: np.ndarray, T: float,
                        param_samples: np.ndarray, model: Literal["bs", "mjd", "heston"] = "mjd",
                        ci: Tuple[float, float] = (2.5, 97.5),):
    if model == "bs":
        pricer = OptionPricingModels.bs_asian_geometric_call
    elif model == "mjd":
        pricer = OptionPricingModels.merton_asian_geometric_call
    elif model == "heston":
        pricer = OptionPricingModels.heston_asian_geometric_call
    else:
        raise ValueError("model must be either 'bs', 'mjd', or 'heston'")

    strikes = np.asarray(strikes, dtype=float)
    
    mean_curve = np.array([
        np.mean([pricer(S0, float(K), T, r, *tuple(ps)) for ps in param_samples]) for K in strikes])
    
    bands = np.array([
        np.percentile([pricer(S0, float(K), T, r, *tuple(ps)) for ps in param_samples], ci) for K in strikes])
    return mean_curve, bands


def margrabe_posterior_fit(*, S1: float, S2: float, r: float, 
                           strikes: np.ndarray, T: np.ndarray, param_samples: np.ndarray,   
                           model: Literal["bs", "mjd", "heston"] = "bs", ci: Tuple[float, float] = (2.5, 97.5),
                           ):
    if model == "bs":
        pricer = OptionPricingModels.bs_margrabe_call
    elif model == "mjd":
        pricer = OptionPricingModels.merton_margrabe_call
    elif model == "heston":
        pricer = OptionPricingModels.heston_margrabe_call
    else:
        raise ValueError("model must be either 'bs', 'mjd', or 'heston'")

    strikes = np.asarray(strikes, dtype=float)
    T = np.asarray(T, dtype=float)
    if T.size == 1:
        T = np.full_like(strikes, T.item(), dtype=float)

    mean_curve = np.array([
        np.mean([pricer(S1, S2, float(K), float(t), r, *tuple(ps)) for ps in param_samples])
        for K, t in zip(strikes, T)])
    bands = np.array([
        np.percentile([pricer(S1, S2, float(K), float(t), r, *tuple(ps)) for ps in param_samples], ci)
        for K, t in zip(strikes, T)])
    return mean_curve, bands

## RMSE MAE Coverage
def scores(observed: np.ndarray, fitted_mean: np.ndarray, bands: Optional[np.ndarray] = None) -> Dict[str, float]:
    obs = np.asarray(observed, dtype=float)
    fit = np.asarray(fitted_mean, dtype=float)
    rmse = float(np.sqrt(np.mean((obs - fit)**2)))
    mae  = float(np.mean(np.abs(obs - fit)))
    out = {"rmse": rmse, "mae": mae}
    if bands is not None:
        cover = float(np.mean((obs >= bands[:, 0]) & (obs <= bands[:, 1])))
        out["coverage"] = cover
    return out