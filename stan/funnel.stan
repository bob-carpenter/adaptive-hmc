data {
  int<lower=0> D;
}
parameters {
  real double_log_sigma;
  vector[D] alpha;
}
model {
  double_log_sigma ~ normal(0, 3);
  alpha ~ normal(0, exp(0.5 * double_log_sigma));
}  
