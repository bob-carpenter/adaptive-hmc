data {
  int<lower=0> J; // number of schools
  array[J] real y; // estimated treatment
  array[J] real<lower=0> sigma; // std of estimated effect
}
parameters {
  real<lower=1e-10> tau; // hyper-parameter of sd
  real<multiplier=10> mu; // hyper-parameter of mean
  vector<offset = mu, multiplier=tau>[J] theta; // transformation of theta
}
model {
  theta ~ normal(mu, tau);
  y ~ normal(theta, sigma);
  mu ~ normal(0, 5); // a non-informative prior
  tau ~ cauchy(0, 5);
}

