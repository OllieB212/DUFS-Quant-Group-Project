import numpy as np
import scipy.stats as st
from scipy.optimize import brentq
from scipy.integrate import quad
import matplotlib.pyplot as plt

class OptionPricingModels:
    '''
    Closed form solutions for the exotic options under Black Scholes, Merton Jump Diffusion, and Heston model.
    '''
    @staticmethod
    def bs_asian_geometric_call(S0, K, T, r, q, sigma):
        '''
        Closed-form price for a geometric Asian call under Black-Scholes.
        '''
        sigma_hat = sigma * np.sqrt((2 * T + 1) / 6)
        mu_hat = 0.5 * sigma_hat**2 + (r - q - 0.5 * sigma**2) * (T + 1) / (2 * T)
        d1 = (np.log(S0 / K) + (mu_hat + 0.5 * sigma_hat**2) * T) / (sigma_hat * np.sqrt(T))
        d2 = d1 - sigma_hat * np.sqrt(T)
        return np.exp(-r * T) * (S0 * np.exp(mu_hat * T) * st.norm.cdf(d1) - K * st.norm.cdf(d2))

    @staticmethod
    def bs_margrabe_call(S1, S2, K, T, r, q1, q2, sigma1, sigma2, rho):
        '''
        Margrabe formula for exchange option under Black-Scholes.
        '''
        d1 = (np.log(S1 / S2) + (q2 - q1 + 0.5 * np.sqrt(sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2)**2) * T
              ) / (np.sqrt(sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2) * np.sqrt(T))
        d2 = d1 - np.sqrt(sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2) * np.sqrt(T)
        return np.exp(-q1 * T) * S1 * st.norm.cdf(d1) - np.exp(-q2 * T) * S2 * st.norm.cdf(d2)

    @staticmethod
    def merton_asian_geometric_call(S0, K, T, r, q, sigma, lam, nu, delta):
        '''
        Geometric Asian call under Merton jump diffusion.
        '''
        
        k = np.exp(nu + 0.5 * delta**2) - 1
        
        mu_Y = np.log(S0) + (r - lam * k - 0.5 * sigma**2) * T / 2 + lam * nu * T / 2
        var_Y = (sigma**2 * T / 3 + lam * T * (nu**2 + delta**2) / 3)
        sigma_Y = np.sqrt(var_Y)
        
        d1 = (mu_Y + var_Y - np.log(K)) / sigma_Y
        d2 = d1 - sigma_Y
        return np.exp(-r * T) * (np.exp(mu_Y + 0.5 * var_Y) * st.norm.cdf(d1) - K * st.norm.cdf(d2))

    @staticmethod
    def merton_margrabe_call(S1, S2, K, T, r, q1, q2,
                             sigma1, sigma2, lam1, lam2, nu1, nu2, delta1, delta2, rho,
                             n_max=20):
        '''
        Margrabe exchange option under Merton jump diffusion.
        '''
        
        X0 = np.log(S1 / S2)
        k1 = np.exp(nu1 + 0.5 * delta1**2) - 1
        k2 = np.exp(nu2 + 0.5 * delta2**2) - 1
        sigma_eff = np.sqrt(sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2)
        price = 0.0
        for n1 in range(n_max):
            p1 = np.exp(-lam1 * T) * (lam1 * T)**n1 / np.math.factorial(n1)
            for n2 in range(n_max):
                p2 = np.exp(-lam2 * T) * (lam2 * T)**n2 / np.math.factorial(n2)
                # conditional mean and variance of log ratio
                mu0 = -0.5 * (sigma1**2 - sigma2**2) - (lam1 * k1 - lam2 * k2)
                mean = X0 + mu0 * T + n1 * nu1 - n2 * nu2
                var = sigma_eff**2 * T + n1 * delta1**2 + n2 * delta2**2
                std = np.sqrt(var)
                # d1, d2 for strike=1
                d1 = (mean + var) / std
                d2 = d1 - std
                price += p1 * p2 * S2 * (st.norm.cdf(d1) - st.norm.cdf(d2))
        return price


    @staticmethod
    def heston_asian_geometric_call(S0, K, T, r, q, v0, kappa, theta, xi, rho):
        '''
        Geometric Asian call under Heston.
        '''
        
        u = 1.0

        B_T = -u * T / T

        def gamma(s):
            return 0.5 * (u**2 / T**2) * s**2 + (u / (2 * T)) * s
        def p(s):
            return rho * xi * (-u * s / T) - kappa

        def K(s):
            return np.exp((rho * xi * u / (2 * T)) * s**2 + kappa * s)

        integrand_C = lambda s: K(s) * gamma(s)
        integral_C, _ = quad(integrand_C, 0, T)
        C_T = np.exp(- (rho * xi * u / (2 * T)) * T**2 - kappa * T) * integral_C
        
        def C_s(s):
            int_Cs, _ = quad(integrand_C, 0, s)
            return np.exp(- (rho * xi * u / (2 * T)) * s**2 - kappa * s) * int_Cs
        
        integral_A, _ = quad(lambda s: C_s(s), 0, T)
        A_T = -r * u * T / 2 + kappa * theta * integral_A
        X0 = np.log(S0)
        return np.exp(-r * T) * (np.exp(A_T + B_T * X0 + C_T * v0) - K)

    @staticmethod
    def heston_margrabe_call(S1, S2, K, T, r, q1, q2, v0, kappa, theta, xi, rho):
        X0 = np.log(S1 / S2)

        def phi(z):
            d = np.sqrt(kappa**2 + 2j * xi**2 * z)
            g = (kappa - d) / (kappa + d)
            exp_neg_dT = np.exp(-d * T)
            B_z = (kappa - d) / xi**2 * (1 - exp_neg_dT) / (1 - g * exp_neg_dT)
            A_z = kappa * theta / xi**2 * ((kappa - d) * T - 2 * np.log((1 - g * exp_neg_dT) / (1 - g)))
            return np.exp(A_z + B_z * v0)
        
        def g_hat(z): # payoff transform
            return (np.exp(0.5j * z) - 1) / (1j * z)

        integrand = lambda z: np.real(phi(z) * g_hat(-z) * np.exp(-1j * z * X0))
        integral, _ = quad(integrand, 0, np.inf, limit=200)
        return S2 * (1 - (1 / np.pi) * integral)
    
    
    
# ------------------------------
# Implied volatility inversion
# ------------------------------
def bs_call_price(S0, K, T, r, q, sigma):
    if T <= 0:
        return max(S0 - K, 0.0)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-q * T) * S0 * st.norm.cdf(d1) - np.exp(-r * T) * K * st.norm.cdf(d2)

def implied_vol_call(target_price, S0, K, T, r, q, tol=1e-6, maxiter=100):
    f = lambda vol: bs_call_price(S0, K, T, r, q, vol) - target_price
    try:
        return brentq(f, 1e-6, 5.0, xtol=tol, maxiter=maxiter)
    except ValueError:
        return np.nan
