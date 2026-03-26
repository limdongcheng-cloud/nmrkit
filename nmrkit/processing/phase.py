import numpy as np
from typing import Dict, Optional
from scipy.optimize import minimize
from nmrkit.core import NMRData
from nmrkit.utils import (
    validate_dimension,
    create_dimension_shape,
    update_dimension_info,
    validate_param_value,
)


def _calculate_phase_factor(
    dim_size: int,
    ph0: float,
    ph1: float,
    pivot: int,
    ndim: int,
    dim: int,
) -> np.ndarray:
    """Calculate phase correction factor for a specific dimension.

    Args:
        dim_size: Size of the dimension to apply phase correction to
        ph0: Zero-order phase correction in degrees
        ph1: First-order phase correction in degrees
        pivot: Pivot point for first-order correction
        ndim: Total number of dimensions in the data
        dim: Dimension index to apply phase correction to

    Returns:
        np.ndarray: Phase correction factor array with appropriate shape
    """
    # Convert degrees to radians
    ph0_rad = np.deg2rad(ph0)
    ph1_rad = np.deg2rad(ph1)

    # Calculate phase correction for each point in the dimension
    indices = np.arange(dim_size)
    phase = ph0_rad + ph1_rad * \
        (indices - pivot) / (dim_size - 1)

    # Reshape phase array to match data dimensions
    phase_shape = create_dimension_shape(ndim, dim, dim_size)
    phase = phase.reshape(phase_shape)

    # Create phase correction factor (complex exponential)
    return np.exp(1j * phase)


def phase_correct(
    data: NMRData,
    dim: int = 0,
    ph0: float = 0.0,
    ph1: float = 0.0,
    pivot: Optional[int] = None,
) -> NMRData:
    """Apply phase correction to a specific dimension of NMR data.

    Args:
        data: NMRData object to process
        dim: Dimension index to apply phase correction to (default: 0)
        ph0: Zero-order phase correction in degrees (default: 0.0)
            This parameter adjusts the overall phase of the spectrum.
        ph1: First-order phase correction in degrees (default: 0.0)
            This parameter adjusts the phase linearly across the spectrum,
            which is useful for correcting phase distortions that vary with frequency.
        pivot: Pivot point for first-order correction (default: center of spectrum)
            The point around which the first-order phase correction is applied.

    Returns:
        NMRData: New NMRData object with phase correction applied
    """
    # Validate dimension
    validate_dimension(data, dim)

    # Create a copy to avoid modifying original data
    result = data.copy()

    # Get dimension size
    dim_size = result.dimensions[dim].size

    # Determine pivot point if not provided
    if pivot is None:
        pivot = dim_size // 2

    # Validate pivot value
    validate_param_value("pivot", pivot, min_value=0, max_value=dim_size - 1)

    # Calculate phase correction factor
    phase_factor = _calculate_phase_factor(
        dim_size, ph0, ph1, pivot, result.ndim, dim
    )

    # Apply phase correction to data
    result.data = result.data * phase_factor

    # Update domain metadata
    if "phase_correction" not in result.dimensions[dim].domain_metadata:
        result.dimensions[dim].domain_metadata["phase_correction"] = []

    result.dimensions[dim].domain_metadata["phase_correction"].append(
        {"ph0": ph0, "ph1": ph1, "pivot": pivot}
    )

    return result


def correct_digital_filter_phase(
    data: NMRData, dim: int = 0, group_delay: Optional[float] = None
) -> NMRData:
    """Apply phase correction to account for digital filter phase distortion.

    This function calculates and applies the phase correction needed to compensate
    for linear phase distortion caused by digital filters. The correction is based
    on the group delay parameter, which quantifies the filter's phase response.

    Args:
        data: NMRData object to process
        dim: Dimension index to apply phase correction to (default: 0)
        group_delay: Group delay in points (default: None)
            If None, the function will attempt to automatically extract the group delay
            from the data metadata (e.g., GRPDLY parameter for Bruker TopSpin data).
            If provided, this value will override any automatically extracted value.

    Returns:
        NMRData: New NMRData object with digital filter phase correction applied

    Notes:
        Digital filters introduce a linear phase shift with frequency, which is
        characterized by the group delay parameter. The correction is calculated as:
        phase_factor = exp(2j * π * group_delay * n / dim_size)
        where n is the frequency point index and dim_size is the size of the dimension.

        This implementation follows the approach used in nmrglue's rm_dig_filter function
        with post_proc=True, which is suitable for correcting already Fourier-transformed data.

        For Bruker TopSpin data, the group delay is automatically extracted from the
        GRPDLY parameter in the acquisition parameters.

        For future extension, this function can be enhanced to:
        1. Calculate group delay from data for formats that don't provide it directly
        2. Support different correction algorithms for complex cases
        3. Handle frequency-dependent group delay for more accurate correction
        4. Add support for other NMR data formats' specific parameters
    """
    # Validate dimension
    validate_dimension(data, dim)

    # Get dimension size
    dim_size = data.dimensions[dim].size

    # Try to extract group delay from metadata if not provided
    if group_delay is None:
        group_delay = 0.0

        # Check if this is Bruker TopSpin data with parameters
        if data.source_format == "topspin" and "parameters" in data.metadata:
            # For direct dimension (F2), check GRPDLY parameter
            if "direct" in data.metadata["parameters"]:
                group_delay = data.metadata["parameters"]["direct"].get(
                    "GRPDLY", 0.0)
            # For indirect dimensions, check if they have GRPDLY parameter
            # This is less common but possible for some experiments
            elif dim > 0:
                indirect_key = f"indirect{dim}"
                if indirect_key in data.metadata["parameters"]:
                    group_delay = data.metadata["parameters"][indirect_key].get(
                        "GRPDLY", 0.0)

        # Check if this is JEOL Delta data with computed group delay
        elif data.source_format == "delta":
            group_delay = data.metadata.get(
                "digital_filter_group_delay", 0.0
            )

    # Create copy to avoid modifying original data
    result = data.copy()

    # Calculate phase correction factor using the nmrglue approach (post_proc=True)
    # Formula: phase_factor = exp(2j * π * group_delay * n / dim_size)
    n = np.arange(dim_size)
    phase_factor = np.exp(2j * np.pi * group_delay * n / dim_size)

    # Reshape phase factor to match data dimensions
    # This handles multi-dimensional data correctly
    phase_shape = [1] * result.ndim
    phase_shape[dim] = dim_size
    phase_factor = phase_factor.reshape(phase_shape)

    # Apply phase correction to data
    result.data = result.data * phase_factor

    # Update domain metadata
    if "phase_correction" not in result.dimensions[dim].domain_metadata:
        result.dimensions[dim].domain_metadata["phase_correction"] = []

    result.dimensions[dim].domain_metadata["phase_correction"].append(
        {"type": "digital_filter", "group_delay": group_delay}
    )

    return result


def remove_digital_filter(
    data: NMRData, dim: int = 0, group_delay: Optional[float] = None
) -> NMRData:
    """Remove digital filter artifact from time-domain FID data by circular shift.

    This function removes the pre-echo artifact caused by the digital filter by
    circular-shifting the FID left by the group delay amount and zeroing the
    wrapped points. Must be called BEFORE Fourier transform.

    Args:
        data: NMRData object (must be time-domain)
        dim: Dimension index (default: 0)
        group_delay: Group delay in points. If None, extracted from metadata.

    Returns:
        NMRData: Copy with digital filter artifact removed
    """
    validate_dimension(data, dim)

    if group_delay is None:
        group_delay = 0.0
        if data.source_format == "topspin" and "parameters" in data.metadata:
            if "direct" in data.metadata["parameters"]:
                group_delay = data.metadata["parameters"]["direct"].get(
                    "GRPDLY", 0.0
                )
        elif data.source_format == "delta":
            group_delay = data.metadata.get(
                "digital_filter_group_delay", 0.0
            )

    shift = int(round(group_delay))
    if shift <= 0:
        return data.copy()

    result = data.copy()

    # Circular shift left by 'shift' points along the target dimension
    result.data = np.roll(result.data, -shift, axis=dim)

    # Zero the last 'shift' points (the wrapped pre-echo)
    slices = [slice(None)] * result.ndim
    slices[dim] = slice(-shift, None)
    result.data[tuple(slices)] = 0

    # Update metadata
    if "phase_correction" not in result.dimensions[dim].domain_metadata:
        result.dimensions[dim].domain_metadata["phase_correction"] = []
    result.dimensions[dim].domain_metadata["phase_correction"].append(
        {"type": "digital_filter_removal", "group_delay": group_delay, "shift": shift}
    )

    return result


def _apply_phase(spectrum: np.ndarray, ph0: float, ph1: float) -> np.ndarray:
    """Apply phase correction to a 1D complex spectrum (internal helper).

    Args:
        spectrum: 1D complex array
        ph0: Zero-order phase in radians
        ph1: First-order phase in radians

    Returns:
        Phase-corrected 1D complex array
    """
    n = len(spectrum)
    indices = np.arange(n) / max(n - 1, 1)  # normalized 0..1
    phase = ph0 + ph1 * indices
    return spectrum * np.exp(1j * phase)


def _detect_signal_region(spectrum: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Detect indices where signal (not noise) is present.

    Uses a moving-average envelope of the absolute spectrum and returns
    a boolean mask where the envelope exceeds threshold × median level.

    This is critical for wide spectra (e.g., 13C with 200+ ppm) where
    most points are pure noise. Computing entropy over noise misleads
    the optimizer.

    Args:
        spectrum: 1D complex frequency-domain array
        threshold: Signal/noise multiplier (default: 3.0)

    Returns:
        Boolean mask array (True = signal region)
    """
    absspec = np.abs(spectrum)
    n = len(absspec)

    # Smoothing window ~0.5% of spectrum width, at least 16 points
    win = max(16, n // 200)
    if win % 2 == 0:
        win += 1

    # Moving average via convolution
    kernel = np.ones(win) / win
    envelope = np.convolve(absspec, kernel, mode="same")

    # Noise level from median (robust against signal peaks)
    noise = np.median(envelope)
    if noise == 0:
        noise = np.mean(envelope) + 1e-15

    mask = envelope > threshold * noise

    # Dilate the mask to include peak tails
    dilation = win * 2
    indices = np.where(mask)[0]
    if len(indices) == 0:
        # No signal detected — fall back to full spectrum
        return np.ones(n, dtype=bool)

    dilated = np.zeros(n, dtype=bool)
    for idx in indices:
        lo = max(0, idx - dilation)
        hi = min(n, idx + dilation + 1)
        dilated[lo:hi] = True

    return dilated


def _acme_objective(params: np.ndarray, spectrum: np.ndarray,
                    signal_mask: np.ndarray = None) -> float:
    """ACME entropy minimization objective function.

    Computes the Shannon entropy of the derivative of the real part of the
    phase-corrected spectrum, plus a penalty for negative signal area.
    When signal_mask is provided, only the signal region contributes to
    the objective, avoiding noise domination in wide spectra.

    Reference: Chen et al., J. Magn. Reson. 158, 164-168 (2002)

    Args:
        params: [ph0, ph1] in radians
        spectrum: 1D complex frequency-domain array
        signal_mask: Optional boolean mask for signal regions

    Returns:
        Entropy value (lower is better)
    """
    ph0, ph1 = params
    corrected = _apply_phase(spectrum, ph0, ph1)
    real = corrected.real

    # If signal mask provided, focus on those regions
    if signal_mask is not None:
        real_sig = real[signal_mask]
    else:
        real_sig = real

    if len(real_sig) < 2:
        return 1e10

    # First derivative of the real spectrum
    deriv = np.diff(real_sig)

    # Normalized absolute derivative → probability distribution
    abs_deriv = np.abs(deriv)
    total = abs_deriv.sum()
    if total == 0:
        return 1e10

    p = abs_deriv / total

    # Shannon entropy (avoid log(0))
    mask = p > 1e-15
    entropy = -np.sum(p[mask] * np.log(p[mask]))

    # Penalty for negative area (encourages absorption-mode peaks)
    # Weight penalty more heavily for wide spectra where entropy alone
    # is insufficient to discriminate correct phasing
    neg_vals = real_sig[real_sig < 0]
    penalty = np.sum(neg_vals ** 2) / max(len(real_sig), 1) / max(total, 1e-15)

    return entropy + 1000.0 * penalty


def _peak_minima_objective(params: np.ndarray, spectrum: np.ndarray,
                           signal_mask: np.ndarray = None) -> float:
    """Peak-minima objective: minimize negative peak area.

    Simpler alternative to ACME. Minimizes the sum of squared negative values
    in the real part of the phase-corrected spectrum.

    Args:
        params: [ph0, ph1] in radians
        spectrum: 1D complex frequency-domain array
        signal_mask: Optional boolean mask for signal regions

    Returns:
        Negative-area score (lower is better)
    """
    ph0, ph1 = params
    corrected = _apply_phase(spectrum, ph0, ph1)
    real = corrected.real

    if signal_mask is not None:
        real = real[signal_mask]

    # Sum of squared negative values, normalized by spectrum length
    neg_area = np.sum(real[real < 0] ** 2)
    return neg_area / max(len(real), 1)


def _estimate_ph0_from_max_peak(spectrum: np.ndarray) -> float:
    """Estimate zero-order phase from the largest peak.

    Finds the point with maximum absolute value and returns the angle
    needed to rotate it to be purely real and positive.

    Args:
        spectrum: 1D complex frequency-domain array

    Returns:
        Estimated ph0 in radians
    """
    idx = np.argmax(np.abs(spectrum))
    return -np.angle(spectrum[idx])


def autophase(
    data: NMRData,
    dim: int = 0,
    method: str = "acme",
    **kwargs,
) -> NMRData:
    """Automatic phase correction of frequency-domain NMR data.

    Optimizes zero-order (ph0) and first-order (ph1) phase parameters
    to produce a properly phased absorption-mode spectrum.

    Args:
        data: NMRData object (must be frequency-domain complex data)
        dim: Dimension index to apply phase correction to (default: 0)
        method: Algorithm to use (default: "acme")
            - "acme": Entropy minimization (Chen et al., JMR 2002). Most
              robust for general use.
            - "peak_minima": Minimize negative peak area. Faster, works
              well for clean spectra with well-resolved peaks.
        **kwargs: Additional parameters:
            - ph0_init (float): Initial guess for ph0 in degrees (default: auto)
            - ph1_init (float): Initial guess for ph1 in degrees (default: 0)

    Returns:
        NMRData: New NMRData object with optimized phase correction applied
    """
    validate_dimension(data, dim)

    result = data.copy()
    dim_size = result.dimensions[dim].size

    # Select objective function
    if method == "acme":
        objective = _acme_objective
    elif method == "peak_minima":
        objective = _peak_minima_objective
    else:
        raise ValueError(f"Unknown autophase method: {method!r}. "
                         f"Use 'acme' or 'peak_minima'.")

    # --- Optimize per-slice along the target dimension ---
    # For 1D data this is a single optimization.
    # For nD data we iterate over all other dimensions (e.g., each row of a 2D).

    # Move target dim to axis 0 for uniform iteration
    work = np.moveaxis(result.data, dim, 0)
    shape_rest = work.shape[1:]
    work_2d = work.reshape(dim_size, -1)  # (dim_size, n_slices)

    n_slices = work_2d.shape[1]

    for s in range(n_slices):
        spectrum = work_2d[:, s]

        # Detect signal regions to focus the objective function
        signal_mask = _detect_signal_region(spectrum)

        # Initial guess
        ph0_init_deg = kwargs.get("ph0_init", None)
        ph1_init_deg = kwargs.get("ph1_init", None)

        if ph0_init_deg is None:
            ph0_init = _estimate_ph0_from_max_peak(spectrum)
        else:
            ph0_init = np.deg2rad(ph0_init_deg)

        # Multi-start search over ph1 initial values for robustness.
        # Wide spectra (13C, 19F) can have very large ph1 values that
        # a single start at 0 would never reach.
        if ph1_init_deg is not None:
            ph1_starts = [np.deg2rad(ph1_init_deg)]
        else:
            ph1_starts = [np.deg2rad(v) for v in [0, -90, 90, -180, 180]]

        best_obj = np.inf
        best_ph0, best_ph1 = ph0_init, 0.0

        for ph1_init in ph1_starts:
            # Pass 1 — optimize ph0 only (fix ph1) for a good starting point
            res0 = minimize(
                lambda p, _ph1=ph1_init: objective(
                    np.array([p[0], _ph1]), spectrum, signal_mask),
                x0=[ph0_init],
                method="Nelder-Mead",
                options={"xatol": 1e-4, "fatol": 1e-10, "maxiter": 500},
            )
            ph0_warm = res0.x[0]

            # Pass 2 — jointly optimize (ph0, ph1)
            res = minimize(
                objective,
                x0=[ph0_warm, ph1_init],
                args=(spectrum, signal_mask),
                method="Nelder-Mead",
                options={"xatol": 1e-4, "fatol": 1e-10, "maxiter": 3000},
            )

            if res.fun < best_obj:
                best_obj = res.fun
                best_ph0, best_ph1 = res.x

        # Apply correction to this slice
        work_2d[:, s] = _apply_phase(spectrum, best_ph0, best_ph1)

    # Reshape back and restore dimension order
    work = work_2d.reshape(dim_size, *shape_rest)
    result.data = np.moveaxis(work, 0, dim)

    # Convert final parameters to degrees for metadata (use last slice's values
    # for 1D, which is the only slice)
    ph0_deg = np.rad2deg(best_ph0) % 360
    ph1_deg = np.rad2deg(best_ph1)

    # Update metadata
    if "phase_correction" not in result.dimensions[dim].domain_metadata:
        result.dimensions[dim].domain_metadata["phase_correction"] = []

    result.dimensions[dim].domain_metadata["phase_correction"].append(
        {"type": "autophase", "method": method,
         "ph0": float(ph0_deg), "ph1": float(ph1_deg)}
    )

    return result
