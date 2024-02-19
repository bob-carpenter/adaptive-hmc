parameters {
  vector[10] v; 
  vector[10] theta;
}

model {
v ~ normal(1, 1);
theta ~ normal(v^2, 0.1^0.5);
}
