data {
  int<lower=0> n; // Number of years
  array[n] int<lower=0> C; // Counts
  vector[n] year; // Year
}
transformed data {
  vector[n] year_squared;
  vector[n] year_cubed;
  
  year_squared = year .* year;
  year_cubed = year .* year .* year;
}
parameters {
  real<lower=-20, upper=20> alpha;
  real<lower=-10, upper=10> beta1;
  real<lower=-10, upper=20> beta2;
  real<lower=-10, upper=10> beta3;
  real<lower=0, upper=5> sigma;
  vector<multiplier=sigma>[n] eps; // Year effects
}
model {
  vector[n] log_lambda
    = alpha + beta1 * year + beta2 * year_squared + beta3 * year_cubed + eps;
  C ~ poisson_log(log_lambda);
  eps ~ normal(0, sigma);
}
