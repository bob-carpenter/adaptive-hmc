data {
  int<lower=0> K;
  int<lower=0> T;
  vector[T] y;
}
parameters {
  real alpha;
  row_vector[K] beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  sigma ~ normal(0, 1);
  for (t in (K + 1):T) {
    y[t] ~ normal(alpha + beta * y[t - K:t - 1], sigma);
  }
}


