// simple hmm example (1 output; 2 states)
data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  vector<lower=0, upper=1>[2] theta;
  positive_ordered[2] mu;
}
model {
  mu ~ normal([3, 10], 1);
  matrix[2, 2] log_theta = log([[theta[1], theta[2]],
                                [1 - theta[1], 1-theta[2]]]);
  array[N, 2] real gamma;
  for (k in 1 : 2) {
    gamma[1, k] = normal_lpdf(y[1] | mu[k], 1);
  }
  array[2] real acc;
  for (t in 2 : N) {
    for (k in 1 : 2) {
      real norm_lpdf = normal_lpdf(y[t] | mu[k], 1);
      for (j in 1 : 2) {
        acc[j] = gamma[t - 1, j] + log_theta[j, k] + norm_lpdf;
      }
      gamma[t, k] = log_sum_exp(acc);
    }
  }
  target += log_sum_exp(gamma[N]);
}
