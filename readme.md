# IEVD (Incremental Eigenvalue Decomposition) Project
## Project Overview
IEVD is a Python implementation for efficiently computing rank-1 modification eigenvalue decomposition. The project provides solutions for two types of rank-1 updates: standard rank-1 update and incremental rank-1 update, with support for multiple stopping criteria to enhance computational stability.

## Core Features
- Two Rank-1 Update Types :
  - Standard rank-1 update (standard_rank1)
  - Incremental rank-1 update (incremental_rank1)
# - standard_rank1: Standard rank-1 problem
# - incremental_rank1: Incremental rank-1 problem 

- Two Stopping Criteria :
  - Classical bisection method (std)
  - Gu-Eisenstat method (gu)
# - std: Classical bisection (based on interval length |b - a| ≤ 2εmax(1, |a|, |b|))
# - gu: Gu-Eisenstat method (based on function value |f(μ)| ≤ ε(1 + sum_terms))


## Matrix Generation Modes
The project supports multiple matrix generation modes (specified via the mode parameter):

- mode 0-6 : Different types of matrix generation methods, including random matrices, Laplacian matrices, etc.
## Performance Evaluation
The project evaluates the following metrics:

- Computation Time : Execution time of IEVD and standard EIG methods
- Residual : ||B - QB * EB * QB^T|| / (||B|| * N)
- Orthogonality : ||QB * QB^T - I|| / N
- Eigenvalue Error : ||EB - EB_|| / N , where EB_ is the result from the dsyevd function

## Dependencies
- NumPy
- SciPy
- pandas
- PyYAML
