// simple hmm example (1 output; 2 states)
data {
  int<lower=0> N;
  int<lower=0> K;
  array[N] real y;
}
parameters {
  simplex[K] theta1;
  simplex[K] theta2;
  positive_ordered[K] mu;
}
model {
  array[2] simplex[K] theta = {theta1, theta2};

  target += normal_lpdf(mu[1] | 3, 1);
  target += normal_lpdf(mu[2] | 10, 1);
  array[K] real acc;
  array[N, K] real gamma;
  for (k in 1 : K) {
    gamma[1, k] = normal_lpdf(y[1] | mu[k], 1);
  }
  for (t in 2 : N) {
    for (k in 1 : K) {
      for (j in 1 : K) {
        acc[j] = gamma[t - 1, j] + log(theta[j, k])
            + normal_lpdf(y[t] | mu[k], 1);
      }
      gamma[t, k] = log_sum_exp(acc);
    }
  }
  target += log_sum_exp(gamma[N]);
}
