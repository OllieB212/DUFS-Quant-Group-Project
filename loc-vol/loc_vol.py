#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import scipy
import EuropeanCall
import VolSurface

#create grid of european option prices given a vol surface and spot etc info
def option_price_grid(strikes, maturities, vols, S, r):
    prices = np.zeros((len(maturities), len(strikes)))
    for i in range(0, len(maturities)): 
        prices[i] = BS_CALL(S, strikes, maturities[i], r, vols[i])
    return prices

#create cubic spline for each specific time point
def create_splines(strikes, maturities, option_prices):
    splines = [None]*len(maturities)
    for iMat in range(0, len(maturities)):
        #how to extrapolate outside range of strikes?
        #Could handle manually but would need to search paths to see if
        #any outside of strike bounds
        splines[iMat] = scipy.interpolate.CubicSpline(strikes,
                                                      option_prices[iMat],
                                                      bc_type = 'clamped',
                                                      extrapolate = False)
    return splines

#calculates partial derivative of option_prices at t=maturities[iMat].
#assumes maturities are equally spaced
def calc_time_deriv(maturities, splines, paths, iMat):
    if iMat == 0:
        #print("[",2,"]: option_prices: ", splines[2](paths))
        #print("[",1,"]: option_prices: ", splines[1](paths))
        #print("[",0,"]: option_prices: ", splines[0](paths))
        time_deriv = (4*splines[1](paths) - splines[2](paths) -
                      3*splines[0](paths))/ \
            (maturities[2] - maturities[0])
    elif iMat == len(maturities) - 1:
        #print("[",iMat,"]: option_prices: ", splines[-1](paths))
        #print("[",iMat-1,"]: option_prices: ", splines[-2](paths))
        #print("[",iMat-2,"]: option_prices: ", splines[-3](paths))
        time_deriv = -(4*splines[-2](paths) - splines[-3](paths) -
                       3*splines[-1](paths))/ \
                       (maturities[-1] - maturities[-3])
    else:
        #print("[",iMat+1,"]: option_prices: ", splines[iMat+1](paths))
        #print("[",iMat,"]: option_prices: ", splines[iMat](paths))
        #print("[",iMat-1,"]: option_prices: ", splines[iMat-1](paths))
        time_deriv = (splines[iMat+1](paths) - splines[iMat-1](paths))/ \
            (maturities[iMat+1] - maturities[iMat-1])

    #print("time_deriv", time_deriv);
    return time_deriv

    
def compute_local_volatility(strikes, maturities, option_prices,
                             vols, splines,
                             paths, iMat, r, q = 0.0):
    #compute time derivative
    #actualTimeDeriv = EuropeanCall.BlackScholesDerivT(100.0, paths,
    #                                                  maturities[iMat],0.05,0.3)
    #print("Actual time deriv=", actualTimeDeriv)
    time_deriv = calc_time_deriv(maturities, splines, paths, iMat)
    #time_deriv = actualTimeDeriv
    #evaluate spline plus 1st and 2nd derivs
    option_prices0 = splines[iMat](paths, 0)
    #actual_prices0 = EuropeanCall.BlackScholes(100.0, paths,
    #                                           maturities[iMat],r,0.3)
    #print("actual_prices0", actual_prices0)
    #print("option_prices0", option_prices0);
    option_prices1 = splines[iMat](paths, 1)
    #actual_prices1 = EuropeanCall.BlackScholesDerivK1(100.0, paths,
    #                                                  maturities[iMat],r,0.3)
    #print("actual_prices1", actual_prices1)
    #print("option_prices1", option_prices1);
    option_prices2 = splines[iMat](paths, 2)
    #actual_prices2 = EuropeanCall.BlackScholesDerivK2(100.0, paths,
    #                                                  maturities[iMat],r,0.3)
    #np.set_printoptions(suppress=False)
    #print("actual_prices2", actual_prices2)
    #print("option_prices2", option_prices2);
    local_vol_Sq_num = 2 * \
        (time_deriv + q * option_prices0 + (r-q)*paths*option_prices1)
    #print("local_vol_Sq_num:", local_vol_Sq_num)
    local_vol_Sq_den = (paths**2) * option_prices2
    #print("local_vol_Sq_den:", local_vol_Sq_den)
    #np.set_printoptions(suppress=True)
    local_vol_Sq = local_vol_Sq_num/local_vol_Sq_den
    #print("local vol^2:", local_vol_Sq)
    #handle numerical noise causing issues (can happen for deeply in the money
    #or out of the money options) when 2nd deriv is close to zero or we
    #end up with a negative number for local vol squared
    local_vol_Sq = \
        np.where(np.logical_or(local_vol_Sq < 0.0,
                               option_prices2 < 1e-10),
                 np.interp(paths, strikes, vols[iMat])**2, local_vol_Sq)
    
    #print("adj local vol^2:", local_vol_Sq)
    local_vol = np.sqrt(local_vol_Sq)
    #actual_local_vol = np.sqrt(2 * \
    #    (actualTimeDeriv + q * actual_prices0 + (r-q)*paths*actual_prices1)/\
    #    ((paths**2) * actual_prices2))
    #print("actual local vol:", actual_local_vol)
    #print("computed local vol before test:", local_vol)
    #handle cases where paths are outside of range of strikes in vol surface
    #In vol world we extrapolate flat so just replace local vol with
    #implied vol (ignores time variation but hopefully a decent approximation)
    local_vol = np.where(paths < strikes[0], vols[iMat][0], local_vol)
    local_vol = np.where(paths > strikes[-1], vols[iMat][-1], local_vol)

    return local_vol
    
def quadratic_interpolation(S_prev, S_curr, S_next, C_prev, C_curr, C_next):

    # matrix for the system of equations
    A = np.array([
        [S_prev**2, S_prev, 1],
        [S_curr**2, S_curr, 1],
        [S_next**2, S_next, 1]
    ])
    
    B = np.array([C_prev, C_curr, C_next])

    coeffs = np.linalg.solve(A, B)
    
    a, b, c = coeffs
    return a, b, c

def compute_local_volatility2(option_prices, strikes, maturities, r, q=0.0):

    local_vol = np.zeros_like(option_prices, dtype=np.float64)
    
    # Calculate the time step differences (dT)
    dT = np.gradient(maturities)
    
    # Loop through maturities and strikes
    for i in range(1, len(maturities) - 1): 
        for j in range(1, len(strikes) - 1):  
            
            S_prev, S_curr, S_next = strikes[j-1], strikes[j], strikes[j+1]
            C_prev, C_curr, C_next = option_prices[i, j-1], option_prices[i, j], option_prices[i, j+1]
            
            #quadratic interpolation
            a, b, c = quadratic_interpolation(S_prev, S_curr, S_next, C_prev, C_curr, C_next)
            
            # Compute first and second derivatives of C(S)
            dC_dS = 2 * a * S_curr + b  # First derivative
            d2C_dS2 = 2 * a  # Second derivative
            
            # Compute the time derivative
            dC_dT = (option_prices[i+1, j] - option_prices[i-1, j]) / (2 * dT[i])
            
            #local volatility using Dupire's formula
            numerator = dC_dT + r * S_curr * dC_dS + q * S_curr * (- dC_dS)+q*C_curr
            denominator = 0.5 * S_curr**2 * d2C_dS2
            
            # Calculate local volatility
            if denominator > 1e-10:
                local_vol[i, j] = np.sqrt(numerator / denominator)
            else:
                local_vol[i, j] = np.nan  # If the denominator is too small, set it to NaN (or handle appropriately)
    
    return local_vol

def MCLV(S0, r, T, steps,
         strikes, maturities, option_prices, vols, splines,
         sims, q = 0.0,
         rng = np.random.default_rng()):
    
    dt = T / steps
    
    # Brownian Motion of the price process
    Bt = rng.normal(0, 1, (steps, sims))
    
    S = np.full(shape=(steps+1, sims), fill_value=S0)
    
    for i in range(1, steps+1):
        local_vol = compute_local_volatility(strikes, maturities, option_prices,
                                             vols, splines, S[i-1],
                                             i-1, r, q)
        #print(i-1,": ", local_vol)
        S[i] = S[i-1] * np.exp((r - (local_vol**2/2))*dt +
                               local_vol * np.sqrt(dt)*Bt[i-1, :])
        print('.', end='', flush=True)
        
    print(flush=True)
    return S


from scipy.stats import norm
N = norm.cdf

#Black-Scholes for european call option
def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)

#Black-Scholes for european put option
def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return K*np.exp(-r*T)*N(-d2) - S*N(-d1)


def test():
    #create flat vol surface with 30% for testing
    #strikes = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    #strikes = [50, 100, 150]
    strikesVS = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0]
    maturitiesVS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    #maturities = [0.5, 1.0]
    volsVS = np.full((len(maturitiesVS), len(strikesVS)), 0.3)
    volSurf = VolSurface.VolSurface(maturitiesVS, strikesVS, volsVS)
    S=100
    r = 0.05
    maturities = np.linspace(0.001, 10*0.001, 20)
    strikes = np.linspace(90.0, 110, 21)
    vols = volSurf.calcVols(maturities, strikes)
    option_prices = option_price_grid(strikes, maturities, vols, S, r)
    np.set_printoptions(suppress=True)
    #print("option prices:\n", option_prices)
    #paths = np.array([90.0, 94.0, 98.0, 102.0, 106.0, 110.0])
    paths = np.array([91.0])
    #create splines
    splines = create_splines(strikes, maturities, option_prices)
    #print("local vol:")
    for iMat in range(0, len(maturities)):
        local_vol = compute_local_volatility(strikes, maturities, option_prices,
                                             vols, 
                                             splines, paths, iMat, r, 0.0)
        print("[",iMat,"]: local vol:", local_vol)

if __name__ == "__main__":
    test()    


