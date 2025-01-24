"""
Black Scholes Multi (2 Asset) Asset Model

1) constant correlation
"""
import numpy as np
import yfinance as yf

def transform(theta):
    """theta = (sigma, nu, rho)"""
    theta[2] = np.tanh(theta[2])
    return theta

def lprior(theta, prior_mu = None, prior_std = None):
    """log prior. 
    Note: we using untransformed parameters here so we don't need to use a Jacobian Matrix"""
    if prior_mu is None:
        # prior_mu = np.array([-1.2, -2.3, -7.6])  
        prior_mu = np.array([0.1, 0.1, 0.5])
    if prior_std is None:
        prior_std = np.array([0.5, 0.5, 1.0]) 
    
    return np.sum(np.log(np.exp(-((theta - prior_mu) ** 2) / (2* prior_std ** 2)) / 
                         np.sqrt(2 * np.pi * prior_std)))


def llike(theta, xproc, yproc, r):
    """log likelihood"""
    sigma, nu, rho = theta
    
    # individual stuff
    mu_x = (r - 0.5 * xproc[1:])
    mu_y = (r - 0.5 * yproc[1:])
    var_x = sigma ** 2
    var_y = nu ** 2
    cov = rho * sigma * nu
    
    # Bivariate distribution mean and variance
    Mu = np.array([mu_x, mu_y])
    #Var = np.array([[var_x**2, cov], [cov, var_y**2]])
    
    # useful stuff
    det = (var_x * var_y) - cov**2
    inv = 1/det * np.array([[-var_x, cov], [cov, -var_y]])

    X = xproc[1:] - xproc[:-1]
    y = yproc[1:] - yproc[:-1]
    
    return np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * np.log(det) 
                  - 0.5 *  np.dot(np.matmul(np.array([X - mu_x, y - mu_y]).transpose(), 
                                    inv), np.array([X - mu_x, y - mu_y])))
                  

def rwm(N, xproc, yproc, theta0, proposal_cov=None, r=0.1):
    """random walk metropolis"""
    if proposal_cov is None:
        proposal_cov = np.diag([0.01, 0.01, 0.01])
        
    T = len(xproc)
    
    if T != len(yproc):
        yproc = yproc[-T:]
        
    mat = np.zeros(shape=(N, len(theta0)))
    theta_unconstraint = theta0
    theta_curr = transform(theta_unconstraint)
    mat[0, :] = theta_curr
    count = 0
    
    for i in range(N):
        
        can_unconstraint = np.random.multivariate_normal(theta_unconstraint, proposal_cov)
        can = transform(can_unconstraint)
        
        laprob = (lprior(can_unconstraint) + llike(can, xproc, yproc, r) 
                  - lprior(theta_unconstraint) - llike(theta_curr, xproc, yproc, r))
        print(laprob)
        
        if np.log(np.random.uniform(0, 1)) < laprob:
            theta_curr = can
            count += 1
            
        mat[i, :] = theta_curr
    
    print(f'Acceptance Rate: {count / (N -1)}')
    return mat

if __name__=="__main__":
    
    tickers = ['AAPL', 'MSFT']
    data = yf.download(tickers, period='1y')['Adj Close']
    data = data.dropna()
    s1proc, s2proc = np.array(data[f'{tickers[0]}']), np.array(data[f'{tickers[1]}'])
    xproc, yproc = np.log(s1proc), np.log(s2proc)
    
    np.random.seed(3421)
    sims =  10_000
    theta0 = np.array([0.1, 0.1, -0.7])
    r = 0.1
    proposal_cov = np.diag([0.1, 0.1, 0.1])

    out_theta = rwm(sims, xproc, yproc, theta0=theta0, proposal_cov=proposal_cov, r=r)
    #y_preds = predict(data, out_sigma, horizon=100)
    #summary(out_sigma.reshape(-1, 1), ["sigma"])
    #post_pred(data, y_preds.T)
    
    
    
