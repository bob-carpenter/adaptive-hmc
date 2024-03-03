data {
  int<lower=0> D;
}

parameters {
  real v; 
  vector[D] theta;
}

model {
v ~ normal(0, 3);
theta ~ normal(0, exp(v/2));
}
