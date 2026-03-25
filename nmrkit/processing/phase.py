import numpy as np
from typing import Dict, Optional
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


def autophase(data: NMRData, dim: int = 0, **kwargs) -> NMRData:
    """Automatic phase correction (to be implemented).

    Args:
        data: NMRData object to process
        dim: Dimension index to apply automatic phase correction to (default: 0)
        **kwargs: Additional parameters for automatic phase correction

    Returns:
        NMRData: New NMRData object with automatic phase correction applied
    """
    # Placeholder for future implementation
    # For now, just return a copy of the data
    return data.copy()
