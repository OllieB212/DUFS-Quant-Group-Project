import numpy as np
import scipy.stats as st
from scipy.optimize import brentq
from scipy.integrate import quad
import matplotlib.pyplot as plt
import cmath

class OptionPricingModels:
    '''
    Closed form solutions for the exotic options under Black Scholes, Merton Jump Diffusion, and Heston model.
    '''
    @staticmethod
    def bs_asian_geometric_call(S0, K, T, r, sigma):
        '''
        Closed-form price for a geometric Asian call under Black-Scholes.
        '''
        if T <= 0:
            return max(S0 - K, 0.0)
        
        mu_Y  = np.log(S0) + 0.5 * (r - 0.5 * sigma**2) * T
        sY    = np.sqrt((sigma**2) * T / 3.0)
        
        d2 = (mu_Y - np.log(K)) / sY
        d1 = d2 + sY
        return np.exp(-r * T) * (np.exp(mu_Y + 0.5 * ((sigma**2) * T / 3.0)) * st.norm.cdf(d1) - K * st.norm.cdf(d2))


    @staticmethod
    def bs_margrabe_call(S1, S2, K, T, r, sigma1, sigma2, rho):
        """
        Margrabe exchange with general strike K: payoff = (S1 - K*S2)^+ under Black-Scholes.
        """
        sigma_eff = np.sqrt(sigma1**2 + sigma2**2 - 2.0 * rho * sigma1 * sigma2)
        if T <= 0 or sigma_eff <= 0:
            return max(S1 - K * S2, 0.0)

        d1 = (np.log(S1 / (K * S2)) + 0.5 * sigma_eff**2 * T) / (sigma_eff * np.sqrt(T))
        d2 = d1 - sigma_eff * np.sqrt(T)
        return S1 * st.norm.cdf(d1) - K * S2 * st.norm.cdf(d2)

    @staticmethod
    def merton_asian_geometric_call(S0, K, T, r, sigma, lam, nu, delta):
        '''
        Geometric Asian call under Merton jump diffusion.
        '''
        
        k = np.exp(nu + 0.5 * delta**2) - 1.0

        mu_Y  = np.log(S0) + (r - lam * k - 0.5 * sigma**2) * T / 2.0 + lam * nu * T / 2.0
        var_Y = (sigma**2) * T / 3.0 + lam * T * (nu**2 + delta**2) / 3.0
        sY    = np.sqrt(var_Y)

        d1 = (mu_Y + var_Y - np.log(K)) / sY
        d2 = d1 - sY
        return np.exp(-r * T) * (np.exp(mu_Y + 0.5 * var_Y) * st.norm.cdf(d1) - K * st.norm.cdf(d2))


    @staticmethod
    def merton_margrabe_call(S1, S2, K, T, r, 
                             sigma1, sigma2, lam1, lam2, nu1, nu2, delta1, delta2, rho,
                             n_max=20):
        '''
        Margrabe exchange option under Merton jump diffusion.
        '''
        
        X0 = np.log(S1 / (K * S2))
        sigma_eff = np.sqrt(sigma1**2 + sigma2**2 - 2.0 * rho * sigma1 * sigma2)

        k1 = np.exp(nu1 + 0.5 * delta1**2) - 1.0
        k2 = np.exp(nu2 + 0.5 * delta2**2) - 1.0

        price = 0.0
        for n1 in range(n_max):
            p1 = np.exp(-lam1 * T) * (lam1 * T)**n1 / np.math.factorial(n1)
            for n2 in range(n_max):
                p2 = np.exp(-lam2 * T) * (lam2 * T)**n2 / np.math.factorial(n2)

                mu0  = -0.5 * (sigma1**2 - sigma2**2) - (lam1 * k1 - lam2 * k2)
                mean = X0 + mu0 * T + n1 * nu1 - n2 * nu2
                var  = sigma_eff**2 * T + n1 * delta1**2 + n2 * delta2**2

                d1   = (mean + var) / (np.sqrt(var))
                d2   = d1 - np.sqrt(var)
                price += p1 * p2 * S2 * (st.norm.cdf(d1) - st.norm.cdf(d2))

        return price


    @staticmethod
    def heston_asian_geometric_call(S0, K, T, r, v0, kappa, theta, xi, rho):
        """
        Geometric Asian call under Heston via affine MGF of Y := (1/T)\int_0^T X_s ds,
        then a 1-D Fourier inversion for the payoff (e^Y - K)^+. More information
        in the paper section 3.5.1
        """
    
        X0    = np.log(S0)
        k     = np.log(K)
        alpha = 1.25
        Umax  = 120.0

        # affine ODEs for M(u) = E[e^{u Y}] (u complex)
        def _ABC(u):
            N  = max(500, int(30 * abs(u) + 150 * T))
            dt = T / N
            C = 0.0 + 0.0j
            A = 0.0 + 0.0j
            for n in range(N):
                tau = n * dt
                B   = (u / T) * tau

                dC = 0.5 * (B**2) + rho * xi * B * C + 0.5 * (xi**2) * (C**2) - 0.5 * B - kappa * C
                dA = r * B + kappa * theta * C

                # RK4
                k1C, k1A = dC, dA
                B2 = (u / T) * (tau + 0.5 * dt)
                C2 = C + 0.5 * dt * k1C
                dC2 = 0.5 * (B2**2) + rho * xi * B2 * C2 + 0.5 * (xi**2) * (C2**2) - 0.5 * B2 - kappa * C2
                dA2 = r * B2 + kappa * theta * C2

                C3 = C + 0.5 * dt * dC2
                dC3 = 0.5 * (B2**2) + rho * xi * B2 * C3 + 0.5 * (xi**2) * (C3**2) - 0.5 * B2 - kappa * C3
                dA3 = r * B2 + kappa * theta * C3

                B4 = (u / T) * (tau + dt)
                C4 = C + dt * dC3
                dC4 = 0.5 * (B4**2) + rho * xi * B4 * C4 + 0.5 * (xi**2) * (C4**2) - 0.5 * B4 - kappa * C4
                dA4 = r * B4 + kappa * theta * C4

                C += (dt / 6.0) * (k1C + 2 * dC2 + 2 * dC3 + dC4)
                A += (dt / 6.0) * (k1A + 2 * dA2 + 2 * dA3 + dA4)

            B_T = u
            return A, B_T, C

        def M(u):
            A, B, C = _ABC(u)
            z = A + B * X0 + C * v0
            if np.real(z) > 700:
                z = 700 + 1j * np.imag(z)
            return np.exp(z)

        # Might need to cite this... (Carr–Madan integral for (e^Y - K)^+)
        def kernel(u):
            u = float(u)
            denom = (alpha * (alpha + 1.0) - u * u) + 1j * ((2.0 * alpha + 1.0) * u)
            num   = np.exp(-1j * u * k) * M((alpha + 1.0) + 1j * u)
            return np.real(num / denom)

        val, _ = quad(kernel, 0.0, Umax, limit=600, epsabs=1e-8, epsrel=1e-8)
        price_fourier = np.exp(-r * T) * np.exp(-alpha * k) * (val / np.pi)

        # fallback option
        if not np.isfinite(price_fourier) or price_fourier <= 1e-10:
            m1 = np.real(M(1.0))  # E[e^Y]
            m2 = np.real(M(2.0))  # E[e^{2Y}]
            m1 = max(m1, 1e-16)
            m2 = max(m2, m1 * m1 * (1 + 1e-12))
            var_ln = np.log(m2 / (m1 * m1))
            s_ln   = np.sqrt(var_ln)
            mu_ln  = np.log(m1) - 0.5 * var_ln
            d2     = (mu_ln - np.log(K)) / s_ln
            d1     = d2 + s_ln
            price_mm = np.exp(-r * T) * (np.exp(mu_ln + 0.5 * var_ln) * st.norm.cdf(d1) - K * st.norm.cdf(d2))
            return float(max(price_mm, 0.0))

        return float(max(price_fourier, 0.0))
    
    @staticmethod
    def heston_asian_geometric_call_mm(S0, K, T, r, v0, kappa, theta, xi, rho):
        """
        Robust generator for Geometric Asian call under Heston via moment-matched lognormal.
        Uses M(1)=E[e^Y] and M(2)=E[e^{2Y}] from the affine MGF of Y=(1/T)\int log S_s ds.
        """
        X0 = np.log(S0)

        def M(u):
            N  = max(500, int(150 * T) + 200)
            dt = T / N
            C, A = 0.0, 0.0
            for n in range(N):
                tau = n * dt
                B   = (u / T) * tau

                dC  = 0.5 * (B**2) + rho * xi * B * C + 0.5 * (xi**2) * (C**2) - 0.5 * B - kappa * C
                dA  = r * B + kappa * theta * C

                # Heun (RK2)
                C_pred = C + dt * dC
                A_pred = A + dt * dA
                B2     = (u / T) * (tau + dt)
                dC2    = 0.5 * (B2**2) + rho * xi * B2 * C_pred + 0.5 * (xi**2) * (C_pred**2) - 0.5 * B2 - kappa * C_pred
                dA2    = r * B2 + kappa * theta * C_pred

                C += 0.5 * dt * (dC + dC2)
                A += 0.5 * dt * (dA + dA2)

            z = A + (u) * X0 + C * v0
            z = min(np.real(z), 700.0)
            return np.exp(z)

        m1 = float(M(1.0))
        m2 = float(M(2.0))
        m1 = max(m1, 1e-12)
        m2 = max(m2, m1 * m1 * (1 + 1e-12))

        var_ln = np.log(m2 / (m1 * m1))
        s_ln   = np.sqrt(var_ln)
        mu_ln  = np.log(m1) - 0.5 * var_ln

        d2 = (mu_ln - np.log(K)) / s_ln
        d1 = d2 + s_ln
        return np.exp(-r * T) * (np.exp(mu_ln + 0.5 * var_ln) * st.norm.cdf(d1) - K * st.norm.cdf(d2))
        
    
    @staticmethod
    def heston_margrabe_call(S1, S2, K, T, r, v0, kappa, theta, xi, rho_S):
        """
        Margrabe under single-factor Heston.
        Use \phi_I(z) from the CIR integrated-variance transform, then 1-D Fourier.
        More information in the paper at section 3.5.2
        """
        X0 = np.log(S1 / S2)
        c  = 2.0 * (1.0 - rho_S)

        def phi_I(z):
            zc = z * c
            d  = np.sqrt(kappa**2 - 2.0j * xi**2 * zc)
            g  = (kappa - d) / (kappa + d)
            e  = np.exp(-d * T)
            B  = (kappa - d) / (xi**2) * (1 - e) / (1 - g * e)
            A  = (kappa * theta / (xi**2)) * ((kappa - d) * T - 2.0 * np.log((1 - g * e) / (1 - g)))
            return np.exp(A + B * v0)

        def g_hat(z):
            return (np.exp(0.5j * z) - 1.0) / (1j * z)

        def integrand(z):
            return np.real(phi_I(z) * g_hat(-z) * np.exp(-1j * z * X0))

        integral, _ = quad(integrand, 0.0, np.inf, limit=200)
        return S2 * (1.0 - (1.0 / np.pi) * integral)
    
    
    

# Implied volatility inversion (Old stuff not needed anymore)
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
