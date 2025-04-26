import yfinance as yf
import numpy as np
import stan
import arviz as az
import matplotlib.pyplot as plt
ticker = ['AAPL']
data = yf.download(ticker, period="1y")
data = data['Close']
returns = data.pct_change().values.flatten()
returns = returns[1:]
T = len(returns)
dt = 1/T  # 1/ Trading days in a year

stan_data = {
    'T': T,
    'y': returns,
    'dt': dt
}
with open("heston.stan", "r+") as heston_model:
    model = heston_model.read()
posterior = stan.build(model, data=stan_data)
fit = posterior.sample(num_chains=4, num_samples=2000, num_warmup=500)
posterior = az.from_pystan(fit)

az.plot_trace(posterior, var_names=['mu', 'kappa', 'theta', 'sigma', 'rho'])
plt.tight_layout()
plt.savefig('AAPL_parameter_traces.png', dpi=300)
plt.show()

az.plot_posterior(posterior, var_names=['mu', 'kappa', 'theta', 'sigma', 'rho'])
plt.tight_layout()
plt.show()

summary = az.summary(posterior, var_names=['mu', 'kappa', 'theta', 'sigma', 'rho'])
print(summary)

mu_mean = summary.loc['mu', 'mean']
kappa_mean = summary.loc['kappa', 'mean']
theta_mean = summary.loc['theta', 'mean']
sigma_mean = summary.loc['sigma', 'mean']
rho_mean = summary.loc['rho', 'mean']

print(f"Mean parameter values:")
print(f"mu: {mu_mean:.4f}")
print(f"kappa: {kappa_mean:.4f}")
print(f"theta: {theta_mean:.4f}")
print(f"sigma: {sigma_mean:.4f}")
print(f"rho: {rho_mean:.4f}")