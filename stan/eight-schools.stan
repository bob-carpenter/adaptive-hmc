data {
  int<lower=0> J;
  array[J] real y;
  array[J] real<lower=0> sigma;
}
parameters {
  real<lower=0> tau;
  real mu;
  vector[J] theta_std;
}
model {
  vector[J] theta = mu + tau * theta_std;
  theta_std ~ normal(0, 1);
  y ~ normal(theta, sigma);
  mu ~ normal(0, 5);
  tau ~ normal(0, 10); // cauchy(0, 5);
}

