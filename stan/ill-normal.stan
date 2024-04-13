data {
  int<lower=0> D;
}
transformed data {
  vector[D] s = linspaced_vector(D, 1, D) / sqrt(D);
}
parameters {
  vector[D] y;
}
model {
  y ~ normal(0, s);
}
