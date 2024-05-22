data {
  int<lower=0> J;
  array[J] real y;
  array[J] real<lower=0> sigma;
}
parameters {
  real<lower=0> tau;
  real mu;
  vector<offset=mu, multiplier=tau>[J] theta;
}
model {
  theta ~ normal(mu, tau);
  y ~ normal(theta, sigma);
  mu ~ normal(0, 5);
  tau ~ normal(0, 10); // cauchy(0, 5);
}

