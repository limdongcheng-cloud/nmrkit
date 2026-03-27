"""Baseline correction for frequency-domain NMR spectra.

Provides multiple algorithms for estimating and removing baseline distortion:
- AsLS: Asymmetric Least Squares (Eilers & Boelens 2005) — best general-purpose
- airPLS: Adaptive Iteratively Reweighted PLS (Zhang et al. 2010) — auto-tuning
- Polynomial: Iterative polynomial fitting — simple, numpy-only fallback
"""

import numpy as np
from typing import Optional

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from nmrkit.core import NMRData
from nmrkit.utils import validate_dimension


def _asls_1d(
    spectrum: np.ndarray,
    lambda_: float = 1e5,
    p: float = 0.01,
    max_iter: int = 10,
    tol: float = 1e-6,
) -> np.ndarray:
    """Estimate baseline using Asymmetric Least Squares smoothing.

    Iteratively solves a penalized least squares problem with asymmetric
    weights: points above the baseline (peaks) get low weight, points
    below get high weight.

    Reference: Eilers, P.H.C. & Boelens, H.F.M. (2005).

    Args:
        spectrum: 1D real-valued array (frequency-domain spectrum).
        lambda_: Smoothness parameter. Larger values produce smoother baselines.
            Typical range: 1e3 (flexible) to 1e7 (very smooth).
        p: Asymmetry parameter. Smaller values keep baseline below peaks.
            Typical range: 0.001 to 0.1.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance on relative weight change.

    Returns:
        Estimated baseline array (same length as spectrum).
    """
    n = len(spectrum)
    if n < 4:
        return np.zeros(n)

    # Second-order difference matrix D (n-2 x n), stored as CSC for spsolve
    e = np.ones(n)
    D = diags([e[:-2], -2 * e[:-1], e], [0, 1, 2], shape=(n - 2, n), format="csc")
    H = lambda_ * D.T.dot(D)

    w = np.ones(n)
    baseline = np.copy(spectrum)

    for _ in range(max_iter):
        W = diags(w, format="csc")
        baseline = spsolve(W + H, w * spectrum)

        # Asymmetric weight update
        w_new = np.where(spectrum > baseline, p, 1.0 - p)

        # Convergence check
        if np.linalg.norm(w_new - w) / (np.linalg.norm(w) + 1e-15) < tol:
            break
        w = w_new

    return baseline


def _airpls_1d(
    spectrum: np.ndarray,
    lambda_: float = 1e5,
    max_iter: int = 15,
    tol: float = 1e-6,
) -> np.ndarray:
    """Estimate baseline using Adaptive Iteratively Reweighted PLS (airPLS).

    A variant of AsLS with adaptive weight updates that requires only
    one hyperparameter (lambda_). Weights are updated based on the
    exponential of negative residuals.

    Reference: Zhang, Z.-M. et al., Analyst 135, 1138-1146 (2010).

    Args:
        spectrum: 1D real-valued array.
        lambda_: Smoothness parameter (only hyperparameter needed).
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance on negative residual sum.

    Returns:
        Estimated baseline array.
    """
    n = len(spectrum)
    if n < 4:
        return np.zeros(n)

    # Second-order difference matrix, stored as CSC for spsolve
    e = np.ones(n)
    D = diags([e[:-2], -2 * e[:-1], e], [0, 1, 2], shape=(n - 2, n), format="csc")
    H = lambda_ * D.T.dot(D)

    w = np.ones(n)
    spectrum_abs_sum = np.abs(spectrum).sum()
    if spectrum_abs_sum == 0:
        return np.zeros(n)

    for i in range(1, max_iter + 1):
        W = diags(w, format="csc")
        baseline = spsolve(W + H, w * spectrum)

        residual = spectrum - baseline

        # Sum of absolute negative residuals
        neg_mask = residual < 0
        d_neg_sum = np.abs(residual[neg_mask]).sum()

        # Convergence: negative residuals are negligible
        if d_neg_sum < tol * spectrum_abs_sum:
            break

        # Adaptive weight update
        # Positive residuals (peaks above baseline): weight → 0
        # Negative residuals: exponential weight based on magnitude
        w = np.zeros(n)
        if d_neg_sum > 0:
            w[neg_mask] = np.exp(i * np.abs(residual[neg_mask]) / d_neg_sum)

        # Boundary weights to prevent edge effects
        w[0] = np.exp(i * 0.5)
        w[-1] = np.exp(i * 0.5)

    return baseline


def _polynomial_1d(
    spectrum: np.ndarray,
    order: int = 5,
    max_iter: int = 100,
    threshold_factor: float = 1.5,
) -> np.ndarray:
    """Estimate baseline using iterative polynomial fitting.

    Fits a polynomial to the spectrum, then iteratively removes points
    above the fit (peaks) and refits until convergence. Only requires numpy.

    Args:
        spectrum: 1D real-valued array.
        order: Polynomial degree. Typical range: 2-8.
        max_iter: Maximum iterations.
        threshold_factor: Points with residuals > threshold_factor * std
            are classified as peaks and excluded from the next fit.

    Returns:
        Estimated baseline array.
    """
    n = len(spectrum)
    if n < 4:
        return np.zeros(n)

    # Clamp polynomial order to avoid overfitting
    order = min(order, n - 1)

    x = np.arange(n, dtype=float)
    mask = np.ones(n, dtype=bool)

    baseline = np.zeros(n)

    for _ in range(max_iter):
        if mask.sum() < order + 1:
            # Not enough points to fit — use last baseline
            break

        coeffs = np.polyfit(x[mask], spectrum[mask], order)
        baseline = np.polyval(coeffs, x)

        residuals = spectrum - baseline
        std = np.std(residuals[mask])
        if std == 0:
            break

        new_mask = residuals < threshold_factor * std

        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    return baseline


def _center_baseline(corrected: np.ndarray, local: bool = False) -> np.ndarray:
    """Remove residual offset so noise fluctuates symmetrically around zero.

    After AsLS/airPLS/polynomial baseline subtraction, a small positive bias
    often remains. This function identifies noise-only points (excluding peaks
    via the IQR outlier rule) and subtracts their median.

    Args:
        corrected: 1D complex or real array after baseline subtraction.
        local: If True, estimate a slowly-varying local offset (better for
            wide spectra with regional drift, ~0.1s extra). If False
            (default), subtract a single global offset (~0.001s).

    Returns:
        Offset-corrected array.
    """
    real = corrected.real.copy()

    # Identify peaks (outliers above P75 + 1.5*IQR)
    q25 = np.percentile(real, 25)
    q75 = np.percentile(real, 75)
    iqr = q75 - q25
    is_noise = real < (q75 + 1.5 * iqr)

    if is_noise.sum() <= 10:
        return corrected

    if not local:
        # Global: subtract single noise median
        return corrected - np.median(real[is_noise])

    # Local: build a smooth offset curve from noise-interpolated data
    nn = len(real)
    noise_vals = real.copy()
    noise_idx = np.where(is_noise)[0]
    peak_idx = np.where(~is_noise)[0]
    if len(peak_idx) > 0 and len(noise_idx) > 1:
        noise_vals[peak_idx] = np.interp(peak_idx, noise_idx, real[noise_idx])

    # Two-pass boxcar smoothing (window ~2% of spectrum)
    win = max(32, nn // 50)
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win) / win
    offset = np.convolve(noise_vals, kernel, mode="same")
    offset = np.convolve(offset, kernel, mode="same")

    return corrected - offset


def baseline_correct(
    data: NMRData,
    dim: int = 0,
    method: str = "asls",
    refine: bool = True,
    **kwargs,
) -> NMRData:
    """Apply baseline correction to frequency-domain NMR data.

    Estimates the baseline from the real part of the spectrum and subtracts
    it. Operates independently on each slice along the specified dimension.

    Args:
        data: NMRData object (should be in frequency domain).
        dim: Dimension to correct (default: 0, the direct dimension).
        method: Baseline estimation algorithm. Options:
            - "asls": Asymmetric Least Squares (default). Good general-purpose.
              Params: lambda_ (float, 1e5), p (float, 0.01), max_iter, tol.
            - "airpls": Adaptive Iteratively Reweighted PLS. Auto-tuning,
              only 1 hyperparameter. Params: lambda_ (float, 1e5), max_iter, tol.
            - "polynomial" or "poly": Iterative polynomial fitting (TopSpin-
              style). Simple, numpy-only.
              Params: order (int, 5), max_iter, threshold_factor.
        refine: If True (default), use local (sliding-window) offset
            correction to remove regional baseline drift. Costs ~0.1s extra
            for 256K points but reduces noise-region bias from ~10 to ~0.5
            intensity units. Set to False for global-only centering.
        **kwargs: Method-specific parameters (see algorithm descriptions).

    Returns:
        New NMRData with baseline subtracted.

    Notes:
        - The baseline is estimated from the real part only. Subtracting a
          real-valued baseline from complex data only affects the real part,
          which is the standard NMR convention.
        - Should be applied AFTER phase correction.
        - For multi-dimensional data, each 1D slice along `dim` is corrected
          independently.
    """
    validate_dimension(data, dim)

    # Method dispatch
    methods = {
        "asls": _asls_1d,
        "airpls": _airpls_1d,
        "polynomial": _polynomial_1d,
        "poly": _polynomial_1d,
    }

    method_lower = method.lower()
    if method_lower not in methods:
        raise ValueError(
            f"Unknown baseline correction method '{method}'. "
            f"Available methods: {list(methods.keys())}"
        )

    algo = methods[method_lower]

    # Filter kwargs to only pass recognized parameters for each method
    asls_params = {"lambda_", "p", "max_iter", "tol"}
    airpls_params = {"lambda_", "max_iter", "tol"}
    poly_params = {"order", "max_iter", "threshold_factor"}

    param_sets = {
        "asls": asls_params,
        "airpls": airpls_params,
        "polynomial": poly_params,
        "poly": poly_params,
    }

    valid_params = param_sets[method_lower]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    result = data.copy()
    dim_size = result.dimensions[dim].size

    # Move target dimension to axis 0 for uniform iteration
    work = np.moveaxis(result.data, dim, 0)
    shape_rest = work.shape[1:]
    work_2d = work.reshape(dim_size, -1)

    for s in range(work_2d.shape[1]):
        spectrum = work_2d[:, s]
        real_part = spectrum.real.copy()

        # Estimate and subtract baseline
        baseline = algo(real_part, **filtered_kwargs)
        corrected = spectrum - baseline

        # Center noise around zero
        corrected = _center_baseline(corrected, local=refine)

        work_2d[:, s] = corrected

    # Restore original dimension order
    work = work_2d.reshape(dim_size, *shape_rest)
    result.data = np.moveaxis(work, 0, dim)

    # Record parameters in metadata
    meta = {"method": method_lower, "refine": refine}
    meta.update(filtered_kwargs)
    result.dimensions[dim].domain_metadata["baseline_correction"] = meta

    return result
