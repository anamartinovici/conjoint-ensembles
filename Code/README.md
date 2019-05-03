Code
================

## R

  - `conj_hmnl.R`: A random-walk Metropolis-Hastings algorithm for an
    HMNL with a conjuctive screen.
  - `conj_hmnl.cpp`: Supporting C++ functions for `conj_hmnl.R`.
  - `simulation-experiment.R`: Data simulation in R and parameter
    recovery using a Stan model.
  - `hit_prob.R`: Function for computing a model’s hit probability.
  - `hit_rate.R`: Function for computing a model’s hit rate.

## Python

  - `conjoint.py`: Functions for setting up data and running an HMNL,
    creating an ensemble, and stacking.
  - `demo.py`: Demo that calls on `conjoint.py`.
  - `test_utils.py`: Functions for testing.
  - `utils.py`: Utility functions.
  - `visualize.py`: Visualization functions.

## Stan

  - `hmnl.stan`: Hierarchical multinomial logit model.
  - `meta_mnl.stan`: Aggregate multinomial logit meta-learning model.
  - `mnl.stan`: Aggregate multinomial logit model.
  - `mnl_vanilla.stan`: An original hierarchical multinomial logit
    model?