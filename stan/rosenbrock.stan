data {
  int<lower=0> D;
}
parameters {
  vector[D] v; 
  vector[D] theta;
}

model {
  v ~ normal(1, 1);
  theta ~ normal(v^2, 0.1);
}
