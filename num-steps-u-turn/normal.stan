data {
  int<lower=0> D;
}
parameters {
  vector[D] y;
}
model {
  y ~ normal(0, 1);
}
generated quantities {
  vector<lower=0>[D] y_sq = square(y);
}
