data {
  int<lower=0> D;
}
parameters {
  vector[D] y;
}
model {
  y ~ normal(0, 1);
}
