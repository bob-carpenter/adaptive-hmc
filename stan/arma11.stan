// ARMA(1, 1)

data {
  int<lower=1> T;
  vector[T] y;
}
parameters {
  real mu;
  real phi;
  real theta;
  real<lower=0> sigma;
}
model {
  mu ~ normal(0, 10);
  phi ~ normal(0, 2);
  theta ~ normal(0, 2);
  sigma ~ normal(0, 2.5);

  vector[T] nu;
  vector[T] err;
  nu[1] = mu + phi * mu;  // assumes err[0] = 0; could impute
  err[1] = y[1] - nu[1];
  for (t in 2:T) {
    nu[t] = mu + phi * y[t - 1] + theta * err[t - 1];
    err[t] = y[t] - nu[t];
  }
  
  err ~ normal(0, sigma);
}


