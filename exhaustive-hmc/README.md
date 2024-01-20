Exhaustive HMC was originally developed by Michael Betancourt in *Identifying
the Optimal Integration Time in Hamiltonian Monte Carlo*[1].  Honestly, I don't
understand much of the math.  But the paper contains an idea for a stopping
criterion that can be made to satisfiy Section 5.1's Algorithm 3 and Lemma 4 of
*A general perspective on the Metropolis-Hastings kernel*[2] by Christophe
Andrieu, Anthony Lee, and Sam Livingstone.  Lemma 4 in particular can guarantee
100% acceptance rates and reduced computational cost, relative to the code like
that developed in our folder num-steps-u-turn.  The code in our folder
num-steps-u-turn also satisfies Algorithm 3, but not Lemma 4, of the paper by
Andrieu et al.

Exhaustive HMC as developed by Betancourt used a balanced binary tree (similar
to Stan) to acheive detailed balance.  It turns out this strategy is more than
necessary for detailed balance to hold.  Betancourt's code is in the
stan-dev/Stan repository under the branch named exhaustions: two files of note
are
[base_exhaustive.hpp](https://github.com/stan-dev/stan/blob/exhaustions/src/stan/mcmc/hmc/exhaustive/base_exhaustive.hpp)
and
[diag_e_metric.hpp](https://github.com/stan-dev/stan/blob/exhaustions/src/stan/mcmc/hmc/hamiltonians/diag_e_metric.hpp).

The code in exhaustive.py uses Andrieu et al's Algorithm 3 and Lemma 4 to rework
the exahustion based stopping criterion into an HMC algorithm that acheives
detailed balance, 100% acceptance rate, and uses roughly 1.5x the computational
cost of reaching the stopping criterion.  In comparison, the algorithm in
num-steps-u-turn/nsut.py uses on average about 2.5x the computational cost
beyond the cost of reaching a u-turn.

The downside is significantly reduced effective sample size, relative to what
Stan or nsut.py achieve.  This fault comes from the design choice of the
stopping criterion, since exhastive.py does not use the u-turn criterion.
exhaustive.py uses an exhaustion as defined in Definition 1 of the Betancourt
paper.

Depsite the flaws of the algorithm in exhaustive.py, my hope is that this is a
starting point for a new way to think about designing MCMC algorithms.  Start
from a base algorithm of high acceptance rates and little computational cost.
Can we design into this base algorithm satisfactory effective sample sizes?

* [1] <https://arxiv.org/abs/1601.00225>
* [2] <https://arxiv.org/abs/2012.14881>
