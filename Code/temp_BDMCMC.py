import numpy as np
import yfinance as yf
from AnalyticPlots import summary, plots
from scipy import special

"""
Merton Jump‑Diffusion model using a Birth–Death MCMC for the latent jumps
"""

def lprior(theta):
    """Log‐prior on (sigma, lambda, nu, delta) — simple Gaussian on each."""
    prior_mu = np.array([0.2, 1.0, 0.0, 0.1])
    prior_std = np.array([0.1, 0.5, 0.2, 0.1])
    return np.sum(-0.5 * ((theta - prior_mu)**2) / (prior_std**2)
                  - np.log(np.sqrt(2*np.pi)*prior_std))

def llike(theta, xproc, nproc, r):
    """Log‐likelihood under Merton jump‑diffusion."""
    sigma, lambd, nu, delta = theta
    k = np.exp(nu + 0.5*delta**2) - 1
    dt = 1.0
    # Continuous + jumps
    denom = sigma**2 * dt + nproc[1:-1]*delta**2
    mu = (r - lambd*k)*dt + nproc[1:-1]*nu
    X = xproc[1:] - xproc[:-1]
    ll_cont = -0.5*np.log(2*np.pi*denom) \
              - 0.5*( (X - mu)**2 / denom )
    # Poisson counts
    ll_jump = nproc[1:-1]*np.log(lambd*dt) \
              - lambd*dt \
              - special.gammaln(nproc[1:-1]+1)
    return np.sum(ll_cont + ll_jump)

def lfcd_N(t, nproc, xproc, theta, r):
    """Single‑site log‑posterior for N_t."""
    sigma, lambd, nu, delta = theta
    k = np.exp(nu + 0.5*delta**2) - 1
    dt = 1.0
    val = 0.0
    # prior + likelihood at site t
    val += nproc[t]*np.log(lambd*dt) - lambd*dt \
           - special.gammaln(nproc[t]+1)
    denom = sigma**2*dt + nproc[t]*delta**2
    mu = (r - lambd*k)*dt + nproc[t]*nu
    inc = xproc[t] - xproc[t-1]
    val += -0.5*np.log(2*np.pi*denom) \
           - 0.5*((inc - mu)**2 / denom)
    return val

def birth_death_mcmc(sims, xproc, theta0, N0,
                    theta_proposal_cov=None, r=0.1):
    """
    Metropolis‑within‑Gibbs with Birth–Death MCMC for N_t
    theta0 = [sigma, lambda, nu, delta]
    N0     = initial Poisson counts array of length T+1
    """
    if theta_proposal_cov is None:
        theta_proposal_cov = np.diag([0.01]*4)
    T = len(xproc)
    mat   = np.zeros((sims, 4))
    matN  = np.zeros((sims, T+1), dtype=int)
    theta_curr = theta0.copy()
    N_curr     = N0.copy()
    mat[0]  = theta_curr
    matN[0] = N_curr
    count_N = 0
    count_theta = 0

    for i in range(1, sims):
        # Birth–Death update for each interior N_t 
        for t in range(1, T):
            old = N_curr[t]
            # propose
            if old == 0:
                cand = 1
                q_old_can = 1.0
                q_can_old = 0.5
            else:
                if np.random.rand() < 0.5:
                    cand = old + 1
                    q_old_can = 0.5
                    q_can_old = 0.5
                else:
                    cand = old - 1
                    q_old_can = 0.5
                    q_can_old = 1.0 if cand == 0 else 0.5
            log_old = lfcd_N(t, N_curr, xproc, theta_curr, r)
            N_curr[t] = cand
            log_cand = lfcd_N(t, N_curr, xproc, theta_curr, r)
            laprob_N = log_cand - log_old \
                       + np.log(q_can_old) - np.log(q_old_can)
            if np.log(np.random.rand()) < laprob_N:
                count_N += 1
            else:
                N_curr[t] = old

        # Metropolis update for theta
        log_post_old = lprior(theta_curr) + llike(theta_curr, xproc, N_curr, r)
        theta_can = np.random.multivariate_normal(theta_curr, theta_proposal_cov)
        log_post_can = lprior(theta_can) + llike(theta_can, xproc, N_curr, r)
        laprob = log_post_can - log_post_old
        if np.log(np.random.rand()) < laprob:
            theta_curr = theta_can
            count_theta += 1

        mat[i]  = theta_curr
        matN[i] = N_curr

    print(f"Acceptance rate N: {count_N/((T-1)*(sims-1)):.4f}, " +
          f"theta: {count_theta/(sims-1):.4f}")
    return mat, matN

def transform(theta):
    sigma = np.abs(theta[0])
    lam   = np.abs(theta[1])
    nu    = theta[2]
    delta = np.abs(theta[3])
    return np.array([sigma, lam, nu, delta])

if __name__ == "__main__":
    # download data
    ticker = ['AAPL']
    data = yf.download(ticker, period='1y', auto_adjust=False)['Adj Close'].dropna()
    xproc = np.log(data.values)
    r = 0.1

    # settings
    np.random.seed(3421)
    sims = 10_000
    theta0 = np.array([0.2, 1.0, 0.0, 0.1])
    N0 = np.random.poisson(theta0[1], size=(len(xproc)+1,))
    # run
    out_theta, out_N = birth_death_mcmc(sims, xproc, theta0, N0, r=r)
    # diagnostics
    param_names = ['sigma','lambda','nu','delta']
    summary(out_theta, param_names)
    plots(out_theta, param_names, 'BirthDeath_Merton.pdf')