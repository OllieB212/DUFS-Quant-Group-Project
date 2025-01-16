### Heston Model (NO JUMPS) (STANDARD MODEL)

## MwG

## As of right now code gives low ESS for all parameters 

import numpy as np 
import yfinance as yf
from scipy import special

from SingleAssetPricingModels import heston_model
from AnalyticPlots import summary, plots, post_pred
# from tuning import historic_std

# Note: Params = (sigma, kappa, theta = xi, rho) <- xi to remove confusion with MwG theta

# prior for params
def lprior(theta, prior_mu = None, prior_std = None):
    """log prior. 
    Note: we using untransformed parameters here so we don't need to use a Jacobian Matrix"""
    if prior_mu is None:
        prior_mu = np.array([-1.2, -2.3, -7.6, -0.87])  
    if prior_std is None:
        prior_std = np.array([0.5, 0.5, 1.0, 0.3]) 
    
    return np.sum(np.log(np.exp(-((theta - prior_mu) ** 2) / (2* prior_std ** 2)) / 
                         np.sqrt(2 * np.pi * prior_std)))


def llike(theta, xproc, vproc, r):
    """log likelihood"""
    sigma, kappa, xi, rho = theta
    
    # individual stuff
    mu_x = (r - 0.5 * vproc[1:-1])
    mu_v = kappa * (xi - vproc[1:-1])
    var_x = vproc[1:-1]
    var_v = sigma ** 2 * vproc[1:-1]
    cov = rho * sigma * vproc[1:-1]
    
    # Bivariate distribution mean and variance
    # Mu = np.array([mu_x, mu_v])
    # Var = np.array([[var_x**2, cov], [cov, var_v**2]])
    
    # useful stuff
    det = (var_x * var_v) - cov**2

    X = xproc[1:] - xproc[:-1]
    v = vproc[1:-1] - vproc[:len(vproc)-2]
    
    return np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * np.log(det) 
                  - 0.5 * (X - mu_x - (cov / var_v) * (v - mu_v))**2 /
                            (det / (var_v))
                  )

def lfcd_vt(t, vproc, xproc, theta, r, alpha=2.0, beta=0.02):
    """log single-site metropolis"""
    sigma, kappa, xi, rho = theta
    
    T = len(xproc)
    val = 0.0
    
    #vproc = np.maximum(1e-10, vproc)
    
    # prior
    # Gamma prior <- strictly positive
    lprior_vt = - np.log(beta ** alpha * special.gamma(alpha)) + (alpha - 1) * np.log(vproc[t]) - vproc[t] / beta
    
    val += lprior_vt
    
    # individual stuff
    mu_x = (r - 0.5 * vproc)
    mu_v = kappa * (xi - vproc)
    var_x = vproc
    var_v = sigma ** 2 * vproc
    cov = rho * sigma * vproc
    
    if t > 0 and t<T-1:
        
        # Bivariate distribution mean and variance
        
        # useful stuff
        det = (var_x[t-1] * var_v[t-1]) - cov[t-1]**2
        
        
        llike_prev = (-0.5 * np.log(2 * np.pi) 
                      - 0.5 * np.log(det) 
                  - 0.5 * (xproc[t] - xproc[t-1] - mu_x[t-1] - (cov[t-1] /var_v[t-1]) * (vproc[t] - vproc[t-1] - mu_v[t-1]))**2 /
                           (det / (var_v[t-1]))
                 )
                  
            
        val += llike_prev

        
    if t< T-1: 
        
        det = (var_x[t] * var_v[t]) - cov[t]**2
        
        llike_next = (-0.5 * np.log(2 * np.pi)
                      -0.5 * np.log(det) 
                  - 0.5 * (xproc[t+1] - xproc[t] - mu_x[t] - (cov[t] / var_v[t]) * (vproc[t+1] - vproc[t] - mu_v[t]))**2 /
                           (det / (var_v[t]))
                  )
            
        val += llike_next
        
        
        
    return val

def gibbs_mh(sims, data, theta0, v0, 
             prior_mu=None, prior_std=None, alpha = 2.0, beta = 0.0005,
             theta_proposal_cov=None, v_proposal=0.1, r=0.1, burnin=None):
    """Metropolis within Gibbs Algorithm"""
    
    if theta_proposal_cov is None:
        proposal_cov_theta = np.diag([0.02, 0.02, 0.02, 0.02])
        
    if prior_mu is None:
        prior_mu = np.array([-1.2, -2.3, -7.6, -0.87])
        
    if prior_std is None:
        prior_std = np.array([0.5, 0.5, 1.0, 0.3])
    
    def transform(theta):
        """transform params to the unconstraint space"""
        sigma = np.exp(theta[0])
        kappa = np.exp(theta[1])
        xi = np.exp(theta[2])
        rho = np.tanh(theta[3])
        
        return sigma, kappa, xi, rho
    
    T = len(data)
    num_params = 4 # sigma, lambda, nu, delta
    mat = np.zeros((sims, num_params)) # store theta samples
    matv = np.zeros((sims, T + 1)) # store latent state samples
    
    theta_curr_unconstraint = theta0.copy()
    v_curr = v0.copy()
    
    sigma_curr, lambd_curr, nu_curr, delta_curr = transform(theta_curr_unconstraint)
    theta_curr = np.array([sigma_curr, lambd_curr, nu_curr, delta_curr])
    
    mat[0, :] = theta_curr
    matv[0, :] = v_curr
    
    count_v, count_theta = 0, 0
    
    # Gibbs two block update 
    for i in range(1, sims):
        
        # Update v by Single-site Metropolis
        for t in range(T+1):
            v_old = v_curr[t]
            log_curr = lfcd_vt(t, v_curr, data, theta_curr, r, alpha, beta)
            
            v_can = v_curr[t] + np.random.normal(0, v_proposal)
            v_curr[t] = v_can
            log_can = lfcd_vt(t, v_curr, data, theta_curr, r, alpha, beta)
            
            laprob = log_can - log_curr
            
            if np.log(np.random.uniform(0, 1)) >= laprob:
                v_curr[t] = v_old # reject sample
            else:
                count_v += 1
            
        # Update theta
        log_post_curr = lprior(theta_curr_unconstraint, prior_mu=prior_mu, prior_std=prior_std) + llike(theta_curr, data, v_curr, r)
        
        theta_can_unconstraint = np.random.multivariate_normal(theta_curr_unconstraint, 
                                                               theta_proposal_cov)

        theta_can = np.array(transform(theta_can_unconstraint))
        
        log_post_can = lprior(theta_can_unconstraint) + llike(theta_can, data, v_curr, r)
        
        laprob_theta = log_post_can - log_post_curr
        
        if np.log(np.random.uniform(0, 1)) < laprob_theta:
            theta_curr_unconstraint = theta_can_unconstraint # accept sample
            theta_curr = theta_can
            count_theta += 1
        
        mat[i, :] = theta_curr
        matv[i, :] = v_curr
                
    print(f"Acceptance rates: theta = {count_theta / (sims - 1):.6f} | v = {count_v / (T*(sims - 1)):.6f}")
    
    return mat, matv


if __name__ == '__main__':
    
    ticker = ['AAPL']
    data = yf.download(ticker, period='1y')['Adj Close']
    data = data.dropna()
    r = 0.1
    # hist_std = historic_std(data)
    
    # ALPHA = 2.0
    # beta = hist_std**2 / ALPHA

    xproc = np.log(data.values)
    
    np.random.seed(3421)
    sims =  10_000
    T = len(data)
    
    param_names = ["sigma", "kappa", "xi", "rho"]
    theta0 = np.array([-1.2, -2.3, -7.6, -0.87])
    v0 = np.random.lognormal(theta0[1], size=(T + 1))
    
    
    prior_mu = np.array([-1.2, -2.3, -7.6, -0.87])
    prior_std = np.array([0.5, 0.5, 1.0, 0.3])
    theta_proposal_cov = np.diag([0.04, 0.04, 0.04, 0.04])
    
    out_theta, out_v = gibbs_mh(sims, xproc, theta0, v0, 
                                prior_mu=prior_mu, prior_std=prior_std, 
                                theta_proposal_cov=theta_proposal_cov)
    
    y_preds = ""
    
    summary(out_theta, param_names)
