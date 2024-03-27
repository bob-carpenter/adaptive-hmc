data {
  int<lower=0> I;
  int<lower=0> J;
  array[I, J] int<lower=0, upper=1> y;
}
parameters {
  vector[J] theta;
  real<lower=0> sigma_log_a;
  vector<multiplier=sigma_log_a>[I] log_a;
  real mu_b;
  real<lower=0> sigma_b;
  vector<offset=mu_b, multiplier=sigma_b>[I] b;
}
model {
  sigma_log_a ~ lognormal(0, 2);
  log_a ~ normal(0, sigma_log_a);
  vector[I] a = exp(log_a);

  theta ~ normal(0, 1);

  mu_b ~ normal(0, 5);
  sigma_b ~ lognormal(0, 2);
  b ~ normal(mu_b, sigma_b);

  for (i in 1:I) {
    y[i] ~ bernoulli_logit(a[i] * (theta - b[i]));
  }
}


