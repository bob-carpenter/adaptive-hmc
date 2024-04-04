functions {
  vector std(vector x) {
    return (x - mean(x)) / sd(x);
  }
}
data {
  int<lower=0> n; // Number of years
  array[n] int<lower=0> C; // Counts
  vector[n] year; // Year
}
transformed data {
  vector[n] year_squared = year .* year;
  vector[n] year_cubed = year_squared .* year;
  matrix[n, 3] x = append_col(year,
                              append_col(year_squared,
                                         year_cubed));
}
parameters {
  real alpha;
  vector[3] beta;
  real<lower=0> sigma;
  vector<multiplier=sigma>[n] eps; // Year effects
}
model {
  alpha ~ normal(0, 5);
  beta ~ normal(0, 5);
  sigma ~ lognormal(0, 1);
  eps ~ normal(0, sigma);
  C ~ poisson_log(alpha + x * beta + eps);
}
