## Black Scholes

## Using Random Walk Metropolis 

import numpy as np 
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns

from AnalyticPlots import summary, plots, post_pred

def lprior(sigma):
    """log prior"""
    return np.sum(np.log(np.exp(-((sigma - 0.2) ** 2) / (2*0.1)) / 
                         np.sqrt(2 * np.pi * 0.1)))


def llike(sigma, xproc, r):
    """log likelihood"""
    return -np.sum(0.5 * np.log(2 * np.pi * sigma ** 2) + 
                   (xproc[1:] - xproc[:-1] - (r - 0.5 * sigma**2))**2 / (2 * sigma **2))


def rwm(N, data, sigma0, V, r):
    """random walk metropolis"""
    T = len(data)
    mat = np.zeros((N, ))
    sigma_curr = sigma0
    mat[0] = sigma_curr
    count = 0
    
    for i in range(N):
        
        can = sigma_curr + np.random.normal(0, V)
        
        laprob = (lprior(can) + llike(can, xproc, r) 
                  - lprior(sigma_curr) - llike(sigma_curr, xproc, r))
        
        if np.log(np.random.uniform(0, 1)) < laprob:
            sigma_curr = can
            count += 1
            
        mat[i] = sigma_curr
    
    print(f'Acceptance Rate: {count / (N -1)}')
    return mat

def predict(data, sigma, horizon=1):
    """predict values up to horizon time steps"""
    
    dt = 1/252 # one time step 
    sims = len(sigma)
    S0 = data.iloc[-1]
    # Brownian Motion of the price process
    Bt = np.random.normal(0, 1, (horizon, sims))
    
    S = np.full(shape=(horizon, sims), fill_value=S0)
    for i in range(1, horizon):
        S[i] = S[i-1] * np.exp((r - (sigma**2/2))*dt + sigma * np.sqrt(dt)*Bt[i-1, :])
        
    return S


ticker = ['AAPL']
data = yf.download(ticker, period="1y")
data = data['Close']
data = data.dropna()

xproc = np.log(data.values)

np.random.seed(3421)
sims =  10_000
sigma0 = 0.1
r = 0.1
V = 0.01

out_sigma = rwm(sims, xproc, sigma0, V, r)
y_preds = predict(data, out_sigma, horizon=100)
summary(out_sigma.reshape(-1, 1), ["sigma"])
post_pred(data, y_preds.T)