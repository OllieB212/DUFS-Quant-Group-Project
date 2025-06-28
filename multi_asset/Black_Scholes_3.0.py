import numpy as np
import yfinance as yf
from AnalyticPlots import summary
"""
Black Scholes model for 2 assets using an Adaptive MCMC process
"""

def transform(theta):
    """theta = (sigma, nu, rho)"""
    sigma = np.exp(theta[0])
    nu    = np.exp(theta[1])
    rho   = np.tanh(theta[2])
    return sigma, nu, rho

def log_jacobian(theta):
    """log of the determinant of the Jacobian"""
    return theta[0] + theta[1] + np.log(1 - np.tanh(theta[2])**2)

def lprior(theta_unconstrained, prior_mu=None, prior_std=None):
    """Unconstrained theta's prior. Commented out is the normal prior."""
    # N(0,1)
    return -0.5 * np.sum(theta_unconstrained**2)
    
    # a custom multivariate normal prior 
    #if prior_mu is None:
    #    # prior_mu = np.array([-1.2, -2.3, -7.6])  
    #    prior_mu = np.array([0.1, 0.1, 0.5])
    #if prior_std is None:
    #    prior_std = np.array([0.5, 0.5, 1.0]) 
    #
    #return np.sum(np.log(np.exp(-((theta_unconstrained - prior_mu) ** 2) / (2* prior_std ** 2)) / 
    #                     np.sqrt(2 * np.pi * prior_std)))


def llike(can, xproc, yproc, r):
    """ 
    can = (sigma, nu, rho), all in their rightful domain now.
    incorporate dt for daily increments. 
    """
    sigma, nu, rho = can
    dt = 1/252.0
    
    mu_x = (r - 0.5 * sigma**2) * dt
    mu_y = (r - 0.5 * nu**2)   * dt
    var_x = sigma**2 * dt
    var_y = nu**2   * dt
    cov = rho * sigma * nu * dt
    det = var_x*var_y - cov**2
    X = xproc[1:] - xproc[:-1]
    Y = yproc[1:] - yproc[:-1]
    
    inv = (1/det) * np.array([[ var_y,    -cov],
                              [  -cov,   var_x]])
    
    n = len(X) 
    log_like = -0.5*n*np.log(2*np.pi) - 0.5*n*np.log(np.abs(det))
    log_like += -0.5 * np.sum(np.einsum('ij,ij->i', 
                                       np.stack([X - mu_x, Y - mu_y], axis=1) @ inv, 
                                       np.stack([X - mu_x, Y - mu_y], axis=1)))
    return log_like


def log_posterior(theta_unconstrained, xproc, yproc, r):
    """log posterior = log prior + log likelihood + log Jacobian"""
    can = transform(theta_unconstrained)
    return (lprior(theta_unconstrained)
            + llike(can, xproc, yproc, r)
            + log_jacobian(theta_unconstrained))

def adaptive_rwm(N, xproc, yproc, theta0, # (unconstrained)
    r = 0.1, initial_cov=None,
    adaptation_start=100, 
    adaptation_cooling=0.99):  

    """
    Haario-style adaptive Metropolis. 
    """
    d = len(theta0)
    mat = np.zeros((N, d))
    count = 0
    
    if initial_cov is None:
        initial_cov = np.diag([0.01, 0.01, 0.01])
    C = initial_cov.copy()  # current guess of proposal covariance

    theta_curr = theta0
    lp_curr = log_posterior(theta_curr, xproc, yproc, r)
    M = np.zeros(d) # the running mean 
    S = np.zeros((d, d)) # the running covariance
    mat[0] = theta_curr
    M[:] = theta_curr

    for i in range(1, N):

        can_unc = np.random.multivariate_normal(theta_curr, C)
        lp_can = log_posterior(can_unc, xproc, yproc, r)
        laprob = lp_can - lp_curr
        if np.log(np.random.rand()) < laprob:
            # accept 
            theta_curr = can_unc
            lp_curr = lp_can
            count += 1

        mat[i] = theta_curr

        if i >= adaptation_start: # diminishing adaptation
            # reference: Haario et al. 2001
            old_M = M.copy()
            M += (theta_curr - old_M)/(i+1)
            S += np.outer(theta_curr - old_M, theta_curr - M)

            emp_cov = S/i # empirical covariance
            scale = (2.38**2)/d # scaling factor (regularised)
            
            newC = scale * emp_cov
            newC += 1e-5 * np.eye(d) 

            # covariance is updated using cooling method (partial interpolation)
            C = adaptation_cooling * C + (1 - adaptation_cooling)*newC

    print("Final acceptance rate:", count / N)
    return mat

if __name__=="__main__":
    tickers = ['AAPL', 'MSFT']
    data = yf.download(tickers, period='1y', auto_adjust=False)['Adj Close'].dropna()
    s1proc = np.array(data[tickers[0]])
    s2proc = np.array(data[tickers[1]])
    xproc  = np.log(s1proc)
    yproc  = np.log(s2proc)

    np.random.seed(3421)
    N = 10_000
    r = 0.1

    theta0 = np.array([np.log(0.2), np.log(0.2), np.arctanh(0.4)]) # simga = 0.2, nu = 0.2, rho = 0.4
    init_cov = np.diag([0.01, 0.01, 0.01])

    out_theta = adaptive_rwm(N, xproc, yproc, theta0, r, initial_cov=init_cov,
                             adaptation_start=100, adaptation_cooling=0.99)

    summary(out_theta, ["sigma", 'nu', 'rho'])
