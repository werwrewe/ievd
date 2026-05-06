# IEVD — Incremental Eigenvalue Decomposition

Efficient computation of eigenvalues and eigenvectors after rank-1 modifications, without full re-decomposition.

## Background

Given a symmetric matrix **A** with known eigendecomposition `A = Q_A Σ Q_A^T`, we want the eigendecomposition of a rank-1 modified matrix **B** without computing it from scratch. Two problem types are supported:

### Standard rank-1 update
```
B = A + ρ · u · u^T
```
Rotated to diagonal form: `D = Σ + ρ · z · z^T`, where `z = Q_A^T u`. Solved via the **secular equation**:
```
f(λ) = 1 + ρ Σ z_i² / (λ_i - λ) = 0
```

### Incremental rank-1 update
```
    ┌         ┐
B = │ A     α │
    │ α^T   w │
    └         ┘
```
Equivalent to an **arrowhead matrix** `C`: `C = [[Σ, β], [β^T, w]]` where `β = Q_A^T α`. Solved via the secular equation:
```
f(μ) = w - μ - Σ β_i² / (λ_i - μ) = 0
```

## Algorithm Pipeline

1. **Rotation to diagonal basis** — `z = Q_A^T u` (standard) or `β = Q_A^T α` (incremental)
2. **Deflation (shrinkage)** — Remove zero components and repeated eigenvalues via Givens rotations, reducing the secular equation size
3. **Secular equation solving** — Bisection with Gu-Eisenstat or classical stopping criterion on each interval
4. **Beta / z recomputation** — Recompute for numerical stability (Gu-Eisenstat method)
5. **Eigenvector construction** — From secular equation roots, then transform back via `Q_A` and deflation matrix `V_defl`/`U_defl`

## Project Structure

| File | Purpose |
|------|---------|
| `main.py` | Entry point — runs batch tests and exports results to `results.xlsx` |
| `run.py` | Test harness — `batchNtest`/`batchtest`/`evdtest` functions |
| `evd.py` | Core solvers — `evd()` (incremental) and `evd_standard()` (standard rank-1) |
| `deflation.py` | Deflation (shrinkage) — zero-component and repeated-diagonal deflation with Givens rotations |
| `bisection.py` | Secular equation root-finders — Gu-Eisenstat and classical bisection |
| `func.py` | Secular equation functions — `cpfunc` (incremental), `cpfunc_standard` (standard) |
| `tools.py` | Utilities — eigenvalue sorting, beta/z recomputation, eigenvector construction |
| `matrix.py` | Test matrix generators — 6 modes (see below) |
| `evaluation.py` | Error metrics — residual, orthogonality |
| `config.yaml` | Configuration — `target_type`, `stop_criterion` |

## Deflation (large time complexity)

Implements the paper's complete deflation method with two phases:

- **Phase 1 — Zero-component deflation**: If `|β_i| ≤ tol` (or `|z_i| ≤ tol`), the component contributes nothing to the secular equation. The corresponding eigenvalue of **A** is also an eigenvalue of **B**, with the same eigenvector.

- **Phase 2 — Repeated diagonal deflation**: If `|Σ_i − Σ_{i+1}| ≤ tol`, apply a Givens rotation `G` to zero out `β_{i+1}`. The rotated eigenvalue at position `i+1` becomes a deflated eigenvalue. This repeats until no more deflatable pairs remain.

Tolerance (paper's formula): `tol = 2 n² ε · max(|λ₁|, |λₙ|)`

Cumulative transformation matrices `V_defl` (incremental, size `(n+1)×(n+1)`) and `U_defl` (standard, size `n×n`) track all Givens rotations for eigenvector recovery.

## Matrix Generation Modes

| Mode | Type | Description |
|------|------|-------------|
| 0 | Random Symmetric | Uniform distribution [-1, 1], symmetrized |
| 1 | Weak Diagonally Dominant | Tridiagonal [1, μ_i, 1] with μ_i = i·10⁻⁶ |
| 2 | Geometric Distribution | Eigenvalues in geometric progression [1, 1/cond] |
| 3 | Arithmetic Distribution | Eigenvalues uniformly spaced in [1/cond, 1] |
| 4 | Log-Uniform Distribution | log(λ) uniformly distributed in [log(1/cond), 0] |
| 5 | Random Laplacian | Laplacian matrix from random adjacency graph |

## Stopping Criteria

| Criterion | Condition | Config value |
|-----------|-----------|--------------|
| Classical (std) | `|b − a| ≤ 2ε · max(1, |a|, |b|)` | `std` |
| Gu-Eisenstat (gu) | `|f(μ)| ≤ ε · N · (|μ| + |w − origin| + Σ|terms|)` | `gu` |

## Usage

```bash
python main.py
```

Results are written to `results.xlsx`. Configure via `config.yaml`:
- `target_type`: `incremental_rank1` or `standard_rank1`
- `stop_criterion`: `gu` or `std`

## Error Metrics

- **Residual**: `‖B − Q_B E_B Q_B^T‖ / (‖B‖ · N)`
- **Orthogonality**: `max_i ‖Q_B^T q_i − e_i‖₂ / N`
- **Eigenvalue Error**: `‖E_B − E_B^ref‖ / N` (reference via LAPACK `dsyevd`)

## Dependencies

- NumPy, SciPy, pandas, PyYAML
