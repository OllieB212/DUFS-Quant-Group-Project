data {
  int<lower=1> T;
  vector[T] y;
  real<lower=0> dt;
}

parameters {
  real mu;
  real<lower=0> kappa;
  real<lower=0> theta;
  real<lower=0> sigma;
  real<lower=-1, upper=1> rho;
  vector<lower=0>[T] v;
}

transformed parameters {
  matrix[2,2] L;
  {
    matrix[2, 2] Sigma;
    Sigma[1, 1] = 1;
    Sigma[1, 2] = rho;
    Sigma[2, 1] = rho;
    Sigma[2, 2] = 1;
    L = cholesky_decompose(Sigma);
  }
}

model {
  mu ~ normal(0, 1);
  kappa ~ normal(1, 1);
  theta ~ normal(0.2, 0.1);
  sigma ~ normal(0.5, 0.2);
  rho ~ uniform(-1, 1);

  v[1] ~ normal(theta, sigma);

  for (t in 2:T) {
    real v_mean = v[t-1] + kappa * (theta - v[t-1]) * dt;
    real v_sd = sigma * sqrt(v[t-1] * dt);

    v[t] ~ normal(v_mean, v_sd);
    y[t] ~ normal((mu - 0.5 * v[t-1]) * dt, sqrt(v[t-1] * dt));
  }
}

generated quantities {
  vector[T] v_sim;
  vector[T] y_sim;

  v_sim[1] = v[1];
  y_sim[1] = y[1];

  for (t in 2:T) {
    vector[2] eps;
    real v_mean;
    real v_sd;

    eps = L * to_vector(normal_rng(rep_vector(0, 2), rep_vector(1, 2)));

    v_mean = v_sim[t-1] + kappa * (theta - v_sim[t-1]) * dt;
    v_sd = sigma * sqrt(v_sim[t-1] * dt);
    v_sim[t] = v_mean + v_sd * eps[2];

    y_sim[t] = (mu - 0.5 * v_sim[t-1]) * dt + sqrt(v_sim[t-1] * dt) * eps[1];
  }
}