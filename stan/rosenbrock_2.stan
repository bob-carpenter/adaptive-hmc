parameters {
  vector[1] v; 
  vector[1] theta;
}

model {
v ~ normal(1, 1);
theta ~ normal(v^2, 0.1^0.5);
}
