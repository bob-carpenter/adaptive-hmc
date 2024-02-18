
parameters {
  real v; 
  vector[10] theta;
}

model {
v ~ normal(0, 3);
theta ~ normal(0, exp(v/2));
}
