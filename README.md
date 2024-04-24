# GIST: Gibbs self-tuning for locally adaptive Hamiltonian Monte Carlo

This repository is being used for our ongoing research on the Gibbs self-tuning HMC.

## The basic idea

The paper defining the basic method is here.

* Nawaf Bou-Rabee, Bob Carpenter, Milo Marsden.  2024. [GIST: Gibbs self-tuning for locally adaptive Hamiltonian Monte Carlo](https://arxiv.org/abs/2404.15253). *arXiv* 2404.15253.

### Replicating or reproducing the paper's results

To replicate the results and plots in the paper, all you need is a single Python script.

```
> cd adaptive-hmc/python
> python3 experiments.py
```

To exactly reproduce the state of the repository at the point the paper was submitted check out tag `arxiv-v1`.  Exactly reproducing the paper's results will also require the same C++ toolchain and settings and the same operating system and version (otherwise small details of floating point behavior or library function behavior may vary).  The results should replicate on different platforms using different choices of random seeds.

You'll need the following packages:

* [`cmdstanpy`](https://cmdstanpy.readthedocs.io/en/v1.2.0/)
* [`bridgestan`](https://roualdes.github.io/bridgestan/latest/)

as well as `numpy` and `scipy` for math, `pandas` for data frame manipulatiion, and `plotnine` for plotting.  The CmdStanPy and BridgeStan documentation (to which the above links point), have detailed installation instructions, which require a relatively up to date C++ toolchain in order to compile the code Stan generates.


## Code

### Warning! Research code

We have tried to write our research code as cleanly as possible, but this is not a product with thorough unit tests and there are no guarantees of stability from day to day.  To ensure compatiblity, you can work against a specific hash of the code here.

### Top-level experiment code

The top-level code to run the experiments is in `experiments.py`.  Comment out the tests a the end of the file that you wish to run.  The code itself is organized into a series of functions that hopefully make the intention clear.

### Stan code

The Stan code is in the directory `stan/` and includes files suffixed `.stan` for the Stan programs and `.json` for any necessary data.  These Stan programs were taken from [`posteriordb`](https://github.com/stan-dev/posteriordb) and modified to Stan's best practices.  The simple normal models are new.

### Python code

The Python code is in the directory `python/` and includes both the samplers and the experimental code.  The file `python/experiments.py` contains the current set of experiments. The code is organized into functions to avoid code duplication to the extent possible.  The graphics are generated using `plotnine`, which is a Python clone of R's `ggplot2`.

The base sampler is `HmcSamplerBase` in the file `hmc.py`.  It is set up to act as an iterator through the `draw()` method.  The concrete `GistSampler` in `gist_sampler.py` extends the base class to define the uniform sampler.  The member variable `frac` determines what fraction of the final steps are used for uniform generation.  The `GistBinomialSampler` in `gist_binomial_sampler` is set up the same way, but `frac` determines the success probability in the binomial generation.

BridgeStan acts like Stan in that it defines a density on unconstrained parameters and provides a constraining function to convert unconstrained parameter values to their natural constrained form as written in the Stan program.  For example, a Stan variable declared to be positive will be log transformed (with appropriate change of variables correction).  BridgeStan provides the constraining and unconstraining transformed, which are applied where necessary to convert between the constrained and unconstrained representations.








