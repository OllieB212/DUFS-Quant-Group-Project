import stan
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
#Note can only be run in linux
model_code = """
data {
    int<lower=1> T; // Time series length
    vector[T] y; //  log prices
    real<lower=0> init_vol; // Initial volatility 
}

parameters {
    real<lower=0> sigma_raw;     // unscaled vol of vol
    real<lower=0> kappa_raw;     // unscaled mean reversion 
    real<lower=0, upper=1> theta;    // Long-run var 
    real<lower=-1, upper=1> rho;     // corr
    vector<lower=1e-6>[T] vl;        // var latent
}

transformed parameters {
    //Transforming stuff
    real<lower=0> sigma = sigma_raw * 0.2;  
    real<lower=0> kappa = kappa_raw * 5;    
    vector[T-1] log_means;  

    for (t in 1:(T-1)) {
        real drift = vl[t] - kappa * (vl[t] - theta);
        real variance_temp = fmax(drift, 1e-6);  //  lower bound
        log_means[t] = log(variance_temp) - 0.5 * sigma * sigma;
    }
}

model {
    // More realistic priors for financial data
    sigma_raw ~ gamma(2, 10);        // Using unscaled parameters with _raw
    kappa_raw ~ gamma(2, 0.5);       
    theta ~ beta(5, 15);             
    rho ~ normal(-0.7, 0.2);        

    
    vl[1] ~ normal(init_vol, 0.05);  

    // Variance process
    for (t in 2:T) {
        vl[t] ~ lognormal(log_means[t-1], sigma);

        
        real vol = sqrt(vl[t]);
        y[t] ~ normal(y[t-1] + rho * vol, vol);
    }
}

generated quantities {
    real avg_volatility = mean(sqrt(vl));
    real annualized_vol = avg_volatility * sqrt(252); // Annualize for comparison with real world data
    real effective_theta = theta;
    real effective_kappa = kappa;
    real effective_sigma = sigma;
    real effective_rho = rho;

    // Log likelihood for model comparison
    vector[T-1] log_likelihood;
    for (t in 2:T) {
        real vol = sqrt(vl[t]);
        log_likelihood[t-1] = normal_lpdf(y[t] | y[t-1] + rho * vol, vol);
    }
}
"""

# Fetch stock data
ticker = "AAPL"
data = yf.download(ticker, period="2y")["Close"]
data = data.dropna()
log_prices = np.log(data.values)
log_prices = log_prices.flatten()
returns = np.diff(log_prices)
init_vol_estimate = np.var(returns)

# Prep data
stan_data = {
    "T": len(log_prices),
    "y": log_prices,
    "init_vol": init_vol_estimate
}

posterior = stan.build(model_code, data=stan_data)

# Needs fixing, slow as but works better than default, yet still not accurate enough.
fit = posterior.sample(
    num_chains=4,
    num_samples=2000,
    num_warmup=2000
)

#Get params
sigma = fit["effective_sigma"]
kappa = fit["effective_kappa"]
theta = fit["effective_theta"]
rho = fit["effective_rho"]
avg_vol = fit["avg_volatility"]
annualized_vol = fit["annualized_vol"]

print("\nParameter Summaries:")
print(f"Volatility of volatility (sigma): {np.mean(sigma):.4f} ± {np.std(sigma):.4f}")
print(f"Mean reversion speed (kappa): {np.mean(kappa):.4f} ± {np.std(kappa):.4f}")
print(f"Long-run variance (theta): {np.mean(theta):.4f} ± {np.std(theta):.4f}")
print(f"Price-volatility correlation (rho): {np.mean(rho):.4f} ± {np.std(rho):.4f}")
print(f"Average volatility: {np.mean(avg_vol):.4f} ± {np.std(avg_vol):.4f}")
print(f"Annualized volatility: {np.mean(annualized_vol):.4f} ± {np.std(annualized_vol):.4f}")

# CI for more info
print("\nParameter 95% Credible Intervals:")
for param_name, param_values in {
    "sigma": sigma,
    "kappa": kappa,
    "theta": theta,
    "rho": rho,
    "annualized_vol": annualized_vol
}.items():
    q_low, q_median, q_high = np.percentile(param_values, [2.5, 50, 97.5])
    print(f"{param_name}: {q_low:.4f} | {q_median:.4f} | {q_high:.4f}")


