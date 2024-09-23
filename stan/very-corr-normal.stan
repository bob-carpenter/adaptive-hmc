data {
  int<lower=0> N;
  real<lower=-1, upper=1> rho;
}
transformed data {
  matrix[N, N] Sigma = rep_matrix(rho, N, N);
  for (n in 1:N) {
    Sigma[n, n] = 1;
  }
  matrix[N, N] L_Sigma = cholesky_decompose(Sigma);
  vector[N] zeros = rep_vector(0, N);
}
parameters {
  vector[N] y;
}
model {
  y ~ multi_normal_cholesky(zeros, L_Sigma);
}
