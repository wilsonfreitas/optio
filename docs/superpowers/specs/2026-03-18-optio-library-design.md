# Optio — Python Option Pricing Library Design

## Overview

**optio** is a comprehensive Python option pricing library implementing all pricing functions from Haug's "The Complete Guide to Option Pricing Formulas" (2nd ed., 2006). It follows functional programming patterns and is designed for future API exposure.

This spec covers the **initial release**: distributions + Chapters 1–2 (Generalized Black-Scholes, Black 76, Garman-Kohlhagen).

## Package Structure

```
optio/
  __init__.py             # re-exports all public functions
  py.typed                # PEP 561 marker
  _core/
    __init__.py
    arrays.py             # scalar → ndarray coercion, broadcast helpers
    distributions.py      # ND, CND, CNDEV, CBND
    constants.py          # default epsilons, common constants
    differentials.py      # generic finite_difference utility
  settings.py             # library-wide config (differentiation method)
  vanilla.py              # bs_*, black_*, gk_* (Chapters 1-2)
tests/
  test_distributions.py
  test_vanilla.py
pyproject.toml
LICENSE
```

## Dependencies

- **Required:** `numpy`
- **Test:** `pytest`, `scipy` (for cross-validation of distributions)

`scipy` is NOT a runtime dependency. Distributions are implemented from scratch in numpy to match Haug's exact algorithms.

## Vectorization & Coercion (`_core/arrays.py`)

Every public function coerces inputs at entry via `broadcast_args()`:

1. Scalars (`int`, `float`) → 0-d `np.ndarray`
2. Lists/tuples → `np.asarray`
3. Pandas Series → duck-typed via `hasattr(x, 'values')` to avoid pandas import
4. `np.broadcast_shapes` validates compatibility
5. Returns ndarrays ready for element-wise operations

Output is always `np.ndarray`. Scalar inputs yield 0-d arrays (behave like scalars, `float(result)` works).

The `flag` parameter (`"call"` / `"put"`) also supports arrays for vectorized mixed-flag computation.

## Distributions (`_core/distributions.py`)

Ported from Haug's VBA `Distiributions.vb`:

- **`nd(x)`** — Standard normal PDF: `1/sqrt(2π) * exp(-x²/2)`
- **`cnd(x)`** — Cumulative normal CDF (Haug's implementation, Hart rational approximation — Haug pp. 505–506)
- **`cndev(x)`** — Inverse CND (Abramowitz & Stegun approximation)
- **`cbnd(x, y, rho)`** — Cumulative bivariate normal (Genz/Drezner-Wesolowsky method)

All accept and return ndarrays. Implemented from scratch in numpy to match Haug's exact algorithms for reproducibility.

## Settings (`settings.py`)

Module-level configuration for differentiation strategy:

```python
from optio import settings

settings.diff_method = "analytical"   # default — closed-form Greeks
settings.diff_method = "finite_diff"  # numerical finite differences
settings.diff_method = "ad"           # future: automatic differentiation (raises NotImplementedError)
```

Context manager for temporary overrides:

```python
with settings.override(diff_method="finite_diff"):
    d = bs_delta("call", S=100, K=105, T=0.5, r=0.05, b=0.05, v=0.2)
```

Greek functions check `settings.diff_method` to dispatch:
- `"analytical"` → explicit formula
- `"finite_diff"` → generic `finite_difference()` from `_core/differentials.py`
- `"ad"` → `NotImplementedError` (placeholder)

## Finite Difference Utility (`_core/differentials.py`)

A generic function that computes numerical Greeks for any pricing function:

```python
def finite_difference(fn, args: tuple, param_index: int, epsilon: float, order: int = 1):
    """
    fn: pricing function (e.g., bs_price)
    args: tuple of all arguments including flag (e.g., ("call", S, K, T, r, b, v))
    param_index: index into args of the parameter to bump (e.g., 1 for S)
    epsilon: bump size
    order: 1 for first derivative (delta, vega), 2 for second (gamma)
    """
    ...
```

Works with any pricing function — serves as the universal fallback for models without analytical Greeks.

## Vanilla Pricing (`vanilla.py`) — Chapters 1 & 2

### Call/Put Convention

String-based: `"call"` or `"put"`. Supports vectorized arrays of flags.

### Strike Parameter Naming

Uses `K` (standard in modern literature) rather than Haug's `X`. Applied consistently across public API and internals.

### Edge Cases

- `T=0` (at expiration): return intrinsic value (`max(S-K, 0)` for calls, `max(K-S, 0)` for puts)
- `v=0` (zero vol): return discounted intrinsic value
- Division by `v * sqrt(T)` in `d1`/`d2` is guarded — when either `v=0` or `T=0`, bypass the formula and return the limit value

### Cost-of-Carry Parameter `b`

Kept explicit per Haug's formulation:
- `b = r` → Black-Scholes 1973 (stock)
- `b = r - q` → Merton (stock with dividend yield)
- `b = 0` → Black 1976 (futures)
- `b = r - rf` → Garman-Kohlhagen (FX)

### Internal Shared Computation

- `_gbs_d1_d2(S, K, T, b, v)` → returns `(d1, d2)`, computed once and reused

### Public API — Generalized Black-Scholes (`bs_*`)

- `bs_price(flag, S, K, T, r, b, v)`
- `bs_delta(flag, S, K, T, r, b, v)`
- `bs_gamma(flag, S, K, T, r, b, v)`
- `bs_vega(flag, S, K, T, r, b, v)`
- `bs_theta(flag, S, K, T, r, b, v)`
- `bs_rho(flag, S, K, T, r, b, v)`
- `bs_carry(flag, S, K, T, r, b, v)` — sensitivity to `b`

### Public API — Black 76 (`black_*`)

Thin wrappers that set `b=0`, `S=F`:
- `black_price(flag, F, K, T, r, v)`
- `black_delta(...)`, `black_gamma(...)`, `black_vega(...)`, `black_theta(...)`, `black_rho(...)`

No `black_carry` — `b` is fixed at 0, carry sensitivity is not meaningful.

### Public API — Garman-Kohlhagen (`gk_*`)

Thin wrappers that set `b = r - rf`:
- `gk_price(flag, S, K, T, r, rf, v)`
- `gk_delta(...)`, `gk_gamma(...)`, `gk_vega(...)`, `gk_theta(...)`, `gk_rho(...)`, `gk_phi(...)`

`gk_rho` returns domestic rate sensitivity (∂V/∂r). `gk_phi` returns foreign rate sensitivity (∂V/∂rf), which maps to `bs_carry` under the hood since `b = r - rf`.

### Higher-Order Greeks

Vomma, charm, speed, zomma, etc. — deferred to a follow-up iteration. Not in initial release scope.

## Testing

**Framework:** `pytest`

**Two validation layers:**
1. **Book reference values** — worked examples from Haug markdown, hardcoded as test cases
2. **VBA cross-validation** — known inputs run through VBA spreadsheets, captured as reference values

**Test files:**
- `test_distributions.py` — `nd`, `cnd`, `cndev`, `cbnd` validated against `scipy.stats` and known tabulated values
- `test_vanilla.py` — `bs_price`, `bs_delta`, etc. validated against book examples and VBA outputs
- Both scalar and vectorized inputs tested (single float, 1000-element arrays, mixed broadcast shapes)

**Tolerances:** `np.testing.assert_allclose` with `rtol=1e-6` for prices and analytical Greeks, `rtol=1e-4` for finite difference Greeks only.

## Future Scope (Not in Initial Release)

- Chapters 3–14 (American approximations, exotics, trees, Monte Carlo, etc.)
- Higher-order Greeks (vomma, charm, speed, zomma, etc.)
- Implied volatility solver
- Automatic differentiation (`"ad"` setting)
- Rust backend for hot paths (via pyo3/maturin)
- API layer
