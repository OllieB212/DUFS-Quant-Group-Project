'''
--------------
Pricing Models
--------------
'''
import numpy as np


def heston_model(S0, v0, rho, kappa, theta, sigma, r, T, steps):
    """
    Heston Model under risk-neutral measure.

    Parameters
    ----------
    S0 : float
        Initial Price.
    v0 : float
        Initial variance.
    rho : float
        Correlation between asset returns and variance.
    kappa : float
        Rate of mean reversion in variance process.
    theta : float
        Long-term mean of variance process.
    sigma : float
        Volatility of the variance process.
    T : int
        Time to maturity.
    steps : int
        Number of time steps.
    sims : int
        Number of simulations.

    Returns
    -------
    S : ndarray
         Asset prices over time.
    v : ndarray
         Variance over time.
    """
    
    dt = T / steps
    
    # Parameters for generating Brownian Motion Correlation
    mu = np.array([0, 0])
    cov = np.array([[1, rho],
                    [rho, 1]])
    
    # Brownian Motions of both the price and variance process
    Bt = np.random.multivariate_normal(mu, cov, steps)
    # Brownian Motion of the price process
    BtS = np.squeeze(Bt[:, 0])
    # Brownian Motion of the variance process
    Btv = np.squeeze(Bt[:, 1])
    
    S = np.full(shape=(steps+1), fill_value=S0)
    v = np.full(shape=(steps+1), fill_value=v0)
    
    for i in range(1, steps+1):
        
        v[i] = np.maximum(v[i-1] + kappa*(theta - v[i-1])*dt + 
                          sigma*np.sqrt(v[i-1]*dt)*Btv[i-1, :], 0)
        
        S[i] = (S[i-1] + r*S[i-1]*dt + 
                np.sqrt(v[i] * dt) * S[i-1] * BtS[i-1, :])
        
         
    return S, v

def merton_model(S0, sigma, lambd, mu_J, sigma_J, r, T, steps):
    """
    Merton Model under risk-neutral measure.

    Parameters
    ----------
    S0 : float
        Inital Price.
    sigma : float
        Volatility of the diffusion process.
    lambd : float
        Average number of jumps per year.
    mu_J : float
        Mean jump size (logarithmic).
    sigma_J : float
        Standard deviation of jump (logarithmic).
    r : float
        Risk free rate of return.
    T : float
        Time to maturity.
    steps : int
        Number of time steps.

    Returns
    -------
    S : ndarray
        Asset Prices over time
    """
    
    dt = T / steps
    
    Bt = np.random.normal(0, 1, steps) # Brownian motion increments
    Nt = np.random.poisson(lambd * dt, steps) # Poisson jumps per step
    Jt = np.random.lognormal(mu_J, sigma_J, steps) # Jump size
    
    S = np.full(shape=(steps+1), fill_value=S0)
    
    for i in range(1, steps+1):
        
        jump_mult = Jt[i-1] ** Nt[i-1] if Nt[i-1] > 0 else 1    
            
        S[i] = S[i-1] * np.exp((r - 0.5 * sigma**2 - lambd * 
                                (np.exp(mu_J + 0.5 * sigma_J**2) - 1)) *dt + 
                               sigma * np.sqrt(dt) * Bt[i-1]) * jump_mult
        
    return S

def heston_jump_diffusion_model(S0, v0, rho, kappa, theta, sigma, lambd, mu_J_S, sigma_J_S, mu_J_v, sigma_J_v, r, T, steps):
    """
    Heston model with jump diffusions in the price and volatility processes under risk-neutral measure.

    Parameters
    ----------
    S0 : float
        Initial Price.
    v0 : float
        Initial variance.
    rho : float
        Correlation between asset returns and variance.
    kappa : float
        Rate of mean reversion in variance process.
    theta : float
        Long-term mean of variance process.
    sigma : float
        Volatility of the variance process
    lambd : float
        Average number of jumps per year.
    mu_J : float
        Mean jump size (logarithmic).
    sigma_J : float
        Standard deviation of jump (logarithmic).
    r : float
        Risk free rate of return.
    T : float
        Time to maturity.
    steps : int
        Number of time steps.

    Returns
    -------
    S : ndarray
        Asset prices over time.
    v : ndarray
        Volatility over time.

    """
    
    dt = T / steps
    
    # Parameters for generating Brownian Motion Correlation
    mu = np.array([0, 0])
    cov = np.array([[1, rho],
                    [rho, 1]])
    
    # Brownian Motions of both the price and variance process
    Bt = np.random.multivariate_normal(mu, cov, steps)
    # Brownian Motion of the price process
    BtS = np.squeeze(Bt[:, 0])
    # Brownian Motion of the variance process
    Btv = np.squeeze(Bt[:, 1])
    
    Nt_S = np.random.poisson(lambd * dt, steps) # Poisson jumps per step for price 
    Jt_S = np.random.lognormal(mu_J_S, sigma_J_S, steps) # Jump size for price
    
    Nt_v = np.random.poisson(lambd * dt, steps) # Poisson jumps per step for variance
    Jt_v = np.random.lognormal(mu_J_S, sigma_J_S, steps) # Jump size for variance
    
    # Note: we are assuming independent jumps between the stock and variance to reduce complexity in calibration stage
    
    S = np.full(shape=(steps+1), fill_value=S0)
    v = np.full(shape=(steps+1), fill_value=v0)
    
    for i in range(1, steps + 1):
        
        jump_mult_v = Jt_v[i-1] ** Nt_v[i-1] if Nt_v[i-1] > 0 else 1 
        
        jump_mult_S = Jt_S[i-1] ** Nt_S[i-1] if Nt_S[i-1] > 0 else 1 
        
        v[i] = np.maximum(v[i-1] + kappa * (theta - v[i-1]) * dt +
                          sigma * np.sqrt(v[i-1] * dt) * Btv[i-1] +
                          jump_mult_v, 0)
        
        S[i] = S[i-1] * np.exp((r - 0.5 * v[i] - lambd * (
            np.exp(mu_J_S + 0.5 * sigma_J_S ** 2) - 1)) * dt +
            np.sqrt(v[i] * dt) * BtS[i-1]) * jump_mult_S
        
    
    return S, v

def SABR_model(F0, v0, beta, rho, nu, r, T, steps):
    '''
    SABR model under risk-neutral measure

    Parameters
    ----------
    F0 : float
        Initial forward price.
    v0 : float
        Initial volatility.
    beta : float
        Elasticity parameter beta \in [0, 1].
    rho : float
        Correlation between forward price and variance.
    nu : float
        Volatility of the variance process.
    r : float
        Risk free rate of return.
    T : float
        Time to maturity.
    steps : int
        Number of time steps.

    Returns
    -------
    F : ndarray
        Forward prices over time.
    v : ndarray
        Volatility over time.
    '''
    
    dt = T / steps
    
    # Parameters for generating Brownian Motion Correlation
    mu = np.array([0, 0])
    cov = np.array([[1, rho],
                    [rho, 1]])
    
    # Brownian Motions of both the forward price and variance process
    Bt = np.random.multivariate_normal(mu, cov, steps)
    # Brownian Motion of the forward price process
    BtF = np.squeeze(Bt[:, 0])
    # Brownian Motion of the variance process
    Btv = np.squeeze(Bt[:, 1])
    
    F = np.full(steps + 1, F0)
    v = np.full(steps + 1, v0)
    
    for i in range(1, steps + 1):
        
        v[i] = v[i-1] * np.exp(-0.5 * nu ** 2 * dt + nu * np.sqrt(dt) * Btv[i-1])
        
        F[i] = F[i-1] * np.exp((r - 0.5 * (v[i] ** 2) * (F[i-1] ** (2 * beta - 2))) * dt +
                               v[i] * (F[i-1] **  (beta - 1)) * np.sqrt(dt) * BtF[i-1])
        
    return F, v

def Kou_model(S0, sigma, lambd, p, eta1, eta2, r, T, steps):
    """
    

    Parameters
    ----------
    S0 : float
        Initial price.
    sigma : float
        Volatility of the diffusion process.
    lambd : float
        Jump intensity.
    p : float
        Probability of upward jumps.
    eta1 : float
        Rate of upward jumps.
    eta2 : float
        Rate of downward jumps.
    r : float
        Risk free rate of return.
    T : float
        Time to maturity.
    steps : int
        Number of time steps.

    Returns
    -------
    S : ndarray
        Asset Prices over time.

    """
    
    dt = T / steps
    
    Bt = np.random.normal(0, 1, steps) # Brownian motion increments
    Nt = np.random.poisson(lambd * dt, steps) # Poisson jumps per step
    Jt = np.zeros(steps) # Jump size
    for i in range(steps):

        if Nt[i]>0:
            jumps = np.random.choice([
                np.random.exponential(1 / eta1) - 1, 
                - (np.random.exponential(1 / eta2) - 1)],
                size=Nt[i],
                p=[p, 1 - p])
            
            Jt[i] = np.sum(jumps)
    
    S = np.full(steps+1, S0)
    
    for i in range(1, steps):
        
        S[i] = S[i -1] * np.exp((r - 0.5 * sigma ** 2 - lambd * (p / eta1 - (1 - p) / eta2)) * dt +
                                sigma * np.sqrt(dt) * Bt[i-1] + Jt[i-1])
    
    return S

def Exponential_Volatility_model(S0, v0, rho, kappa, theta, sigma, r, T, steps):
    """
    Exponential Volatility model under risk-neutral measure.

    Parameters
    ----------
    S0 : float
        Initial Price.
    v0 : float
        Initial variance.
    rho : float
        Correlation between asset returns and variance.
    kappa : float
        Rate of mean reversion in variance process.
    theta : float
        Long-term mean of variance process.
    sigma : float
        Volatility of the variance process.
    T : int
        Time to maturity.
    steps : int
        Number of time steps.
    sims : int
        Number of simulations.

    Returns
    -------
    S : ndarray
         Asset prices over time.
    v : ndarray
         Variance over time.
    """
    
    dt = T / steps
    
    # Parameters for generating Brownian Motion Correlation
    mu = np.array([0, 0])
    cov = np.array([[1, rho],
                    [rho, 1]])
    
    # Brownian Motions of both the price and variance process
    Bt = np.random.multivariate_normal(mu, cov, steps)
    # Brownian Motion of the price process
    BtS = np.squeeze(Bt[:, 0])
    # Brownian Motion of the variance process
    Btv = np.squeeze(Bt[:, 1])
    
    S = np.full(shape=(steps+1), fill_value=S0)
    v = np.full(shape=(steps+1), fill_value=v0)
    
    for i in range(1, steps+1):
        
        v[i] = (v[i-1] + kappa * (theta - v[i-1]) * dt + 
                sigma * np.sqrt(dt) * Btv[i-1])
        
        S[i] = S[i-1] * np.exp((r -0.5 * np.exp(v[i])) * dt + 
                np.exp(0.5 * v[i]) * np.sqrt(dt) * BtS[i-1])
         
    return S, v