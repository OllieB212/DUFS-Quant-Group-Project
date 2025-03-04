import math
import numpy as np
import SingleAssetPricingModels
import loc_vol
from scipy.stats import norm
N = norm.cdf

#Black-Scholes for european call option
def BlackScholes(S, K, T, r, sigma):
#    if (K == 0):
#        return S #just a Forward. Price = discount factor * expected value of S
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)

#derivative of BlackScholes() wrt T assuming sigma constant
def BlackScholesDerivT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S*(np.exp(-(d1**2)/2)/np.sqrt(2*math.pi))*sigma/(2*np.sqrt(T)) \
        +r*K*np.exp(-r*T)*N(d2)

#1st derivative of BlackScholes() wrt K assuming sigma constant
def BlackScholesDerivK1(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return -N(d2)*np.exp(-r*T)

#2nd derivative of BlackScholes() wrt K
def BlackScholesDerivK2(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    return S*(np.exp(-(d1**2)/2)/np.sqrt(2*math.pi))/ \
        ((K ** 2)*sigma*np.sqrt(T))

#utility function: calculate math.exp(-r * T) * np.mean(np.fmax(paths[-1]-K, 0))
#but handles K being array like or not
def computeDiscountedMeanOverMax(r, T, paths, K):
    if isinstance(K, int) or isinstance(K, float):
        pricesBeforeDF = np.mean(np.fmax(paths-K, 0))
    else:
        pricesBeforeDF =np.empty(len(K))
        for i in range(0, len(K)):
            pricesBeforeDF[i] = np.mean(np.fmax(paths-K[i], 0))
            
    price = math.exp(-r * T) * pricesBeforeDF
    return price
    

#MCLN for european call option
def MCLogNormal(S, K, T, r, sigma, numIter, rng):
    #get the simulated paths. Can just use one time point
    paths = SingleAssetPricingModels.black_scholes_model(
        S, r, sigma, T, 1, numIter, rng)
    return computeDiscountedMeanOverMax(r, T, paths[-1], K)

#MC using local vol. Supports array of strikes
def MCLV(S, K, T, r, volSurf, numIter, rng, loStrike, hiStrike,
         timePtsPerDay = 3, numStrikes=200):
    numTimePoints = max(int(T*365*timePtsPerDay),3)
    maturities = np.linspace(T/numTimePoints, T, numTimePoints)
    strikes = np.linspace(loStrike, hiStrike, numStrikes)
    #calculate grid of implied vols
    vols = volSurf.calcSmoothVols(maturities, strikes)
    #calculate option prices on this grid
    print("About to calculate option prices", flush=True)
    option_prices = loc_vol.option_price_grid(strikes, maturities, vols, S, r)
    #create splines for interpolation
    print("About to create splines", flush=True)
    splines = loc_vol.create_splines(strikes, maturities, option_prices)
    print("About to start MC", flush=True)
    paths = loc_vol.MCLV(
        S, r, T, numTimePoints, strikes, maturities, option_prices,
        vols, splines, numIter, 0.0, rng)
    return computeDiscountedMeanOverMax(r, T, paths[-1], K)
