## Merton Model

## Using Metropolis-Within-Gibbs Sampler

## Work in ptogress...

import numpy as np 
import yfinance as yf
from scipy import special

from SingleAssetPricingModels import merton_model
from AnalyticPlots import summary, plots, post_pred

# Note theta = (sigma, lambda, nu, delta)
# take the prior 
def lprior(theta):
    """log prior"""
    prior_mu = np.array([0.2, 1.0, 0.0, 0.1]) # Hyperparams 
    prior_std = np.array([0.1, 0.5, 0.2, 0.1]) # Can be changed
    
    return np.sum(np.log(np.exp(-((theta - prior_mu) ** 2) / (2* prior_std ** 2)) / 
                         np.sqrt(2 * np.pi * prior_std)))

def llike(theta, xproc, nproc, r):
    """log likelihood. Latent state Nt (nproc)"""
    sigma, lambd, nu, delta = theta
    
    k = np.exp(nu + delta**2/2) - 1
    logprobX = (-0.5 * np.log(2 * np.pi * (sigma**2 + nproc[1:-1] * delta**2)) 
               -0.5 * ((xproc[1:] -xproc[:-1] - ((r - lambd * k) + nproc[1:-1]*nu))**2 /
                       (sigma**2 + nproc[1:-1] * delta**2)
                   ))
    
    logprobN = nproc[1:-1] * np.log(lambd) - lambd - special.gammaln(nproc[1:-1] + 1)
    
    return np.sum(logprobX + logprobN)

def lfcd_Nt(t, nproc, xproc, theta, r):
    """Single step update"""
    sigma, lambd, nu, delta = theta
    
    T = len(xproc)
    val = 0.0
    k = np.exp(nu + delta**2 / 2) - 1

    # p(N_t| dX_t)
    if t>=0 and t<T:
        val += nproc[t] * np.log(lambd) - lambd - special.gammaln(nproc[0] + 1)
        
        val += (-0.5 * np.log(2 * np.pi * (sigma**2 + nproc[t] * delta**2))
                - 0.5 * ((xproc[t] - xproc[t-1] - ((r - lambd * k) + nproc[t]*nu))**2 /
                         (sigma**2 * nproc[t] * delta**2)
                ))
    
    return val

def gibb_mh(sims, data, theta0, N0, theta_proposal_cov=None, r=0.1, burnin=None):
    """Metropolis within Gibbs Algorithm"""
    
    if theta_proposal_cov is None:
        theta_proposal_cov = np.diag([0.01, 0.01, 0.01, 0.01])
    
    # Maybe add here transform function to tranform the parameters into an unconstraint space
    
    T = len(data)
    num_params = 4 # sigma, lambda, nu, delta
    mat = np.zeros((sims, num_params)) # store theta samples
    matN = np.zeros((sims, T + 1)) # store latent state samples
    
    theta_curr = theta0.copy()
    N_curr = N0.copy()
    
    mat[0, :] = theta_curr
    matN[0, :] = N_curr
    
    sigma_curr, lambd_curr, nu_curr, delta_curr = theta_curr
    count_N, count_theta = 0, 0
    
    # 2 block Gibbs deterministic scan
    for i in range(1, sims):
        
        # Update N by single-site metropolis
        for t in range(T+1):
            N_old = N_curr[t]
            log_curr = lfcd_Nt(t, N_curr, data, theta_curr, r)
            N_can = max(0, N_old + np.random.choice([-1, 1])) # removed the trivial case 0
            
            N_curr[t] = N_can
            log_can = lfcd_Nt(t, N_curr, data, theta_curr, r)
            
            laprob = log_can - log_curr # alpha(N* | N)
            if np.log(np.random.uniform(0, 1)) >= laprob:
                N_curr[t] = N_old # reject sample
            else:
                count_N += 1 # accept
            
        # Update theta by Random-Walk Metropolis
        log_posterior_old = lprior(theta_curr) + llike(theta_curr, data, N_curr, r)
        
        theta_can = np.random.multivariate_normal(theta_curr, theta_proposal_cov)
        log_posterior_can = lprior(theta_can) + llike(theta_can, data, N_curr, r)
        
        laprob = log_posterior_can - log_posterior_old
        if np.log(np.random.uniform(0, 1)) < laprob:
            theta_curr = theta_can # accept sample
            count_theta += 1 
        
        mat[i, :] = theta_curr
        matN[i, :] = N_curr
    
    print(f"Acceptance rates: theta = {count_theta / (sims - 1):.6f} | N = {count_N / (T*(sims - 1)):.6f}")
    return mat, matN

if __name__ == '__main__':
    
    ticker = ['AAPL']
    data = yf.download(ticker, period='1y')['Adj Close']
    data = data.dropna()
    r = 0.1

    xproc = np.log(data.values)
    
    np.random.seed(3421)
    sims =  10_000
    T = len(data)
    param_names = ["sigma", "lambda", "nu", "delta"]
    theta0 = np.array([0.01, 1.0, 0.0, 0.01])
    N0 = np.random.poisson(theta0[1], size=(T + 1))
    
    theta_proposal_cov = np.diag([0.01, 0.01, 0.01, 0.01])
    
    out_theta, out_N = gibb_mh(sims, xproc, theta0, N0, theta_proposal_cov=theta_proposal_cov)
    
    y_preds = ""
    
    summary(out_theta, param_names)
    plots(out_theta, param_names)
    #post_pred(data, y_preds.T)
    
    
    
