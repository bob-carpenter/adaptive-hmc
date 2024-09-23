transformed data {
  vector[2] m = [0, 0]';
  cov_matrix[2] S = [[1, 0.9], [0.9, 1]];
}
parameters {
  vector[2] alpha;
}
model {
  alpha ~ multi_normal(m, S);
}
