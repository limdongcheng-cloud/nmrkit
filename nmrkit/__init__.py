"""nmrkit: A comprehensive library for NMR data processing and analysis."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("nmrkit")
except PackageNotFoundError:
    __version__ = "unknown"

# Core data structures
from .processing.workflows.basic_2d import process as basic_2d_process
from .processing.workflows.basic_1d import process as basic_1d_process
from .core import DimensionInfo, NMRData

# Input/Output functions
from .io import FormatReader, FormatWriter, read, write, DeltaReader

# Processing functions - main names
from .processing import (
    first_point_scaling,
    exponential,
    sine,
    cosine,
    trapezoidal,
    zero_fill,
    extract_region,
    fourier_transform,
    ft_shift,
    ft_unshift,
    phase_correct,
    correct_digital_filter_phase,
    remove_digital_filter,
    autophase,
    complexify_indirect_dim,
    baseline_correct,
)

# Complex conversion functions
from .utils.complex import complexify, decomplexify

# Visualization functions
from .visualization import plot

# Function aliases for convenience (numpy-style short names)
# Window functions
fps = first_point_scaling
fp_scaling = first_point_scaling
em = exponential
trap = trapezoidal

# Data manipulation
zf = zero_fill
extract = extract_region

# Fourier transform
ft = fourier_transform
ftshift = ft_shift
ftunshift = ft_unshift

# Phase processing
phase = phase_correct

# Baseline correction
bc = baseline_correct

# Complex processing
complexify_indirect = complexify_indirect_dim


# Auto processing workflow

# Workflow aliases
process_1d = basic_1d_process
process_2d = basic_2d_process


def auto_process(data, **kwargs):
    """Automatic processing of NMR data based on its type.

    Parameters
    ----------
    data : nmrkit.NMRData
        Input NMR data.
    **kwargs
        Additional processing parameters passed to the workflow.

    Returns
    -------
    data : nmrkit.NMRData
        Processed NMR data.
    """
    if data.ndim == 1:
        return basic_1d_process(data, **kwargs)
    elif data.ndim == 2:
        return basic_2d_process(data, **kwargs)
    else:
        raise ValueError(
            f"Auto processing not implemented for {
                data.ndim}D data")


# Define __all__ to control what gets imported with 'from nmrkit import *'
__all__ = [
    # Core data structures
    "DimensionInfo",
    "NMRData",
    # IO functions
    "FormatReader",
    "FormatWriter",
    "read",
    "write",
    "DeltaReader",
    # Window functions - main names
    "first_point_scaling",
    "exponential",
    "sine",
    "cosine",
    "trapezoidal",
    # Data manipulation - main names
    "zero_fill",
    "extract_region",
    # Fourier transform - main names
    "fourier_transform",
    "ft_shift",
    "ft_unshift",
    # Phase processing - main names
    "phase_correct",
    "correct_digital_filter_phase",
    "remove_digital_filter",
    "autophase",
    # Complex processing - main names
    "complexify_indirect_dim",
    # Complex conversion functions
    "complexify",
    "decomplexify",
    # Visualization
    "plot",
    # Aliases - numpy-style short names
    "fps",
    "fp_scaling",
    "em",
    "exp",
    "trap",
    "zf",
    "extract",
    "ft",
    "ftshift",
    "ftunshift",
    "phase",
    "autophase",
    "baseline_correct",
    "bc",
    "complexify_indirect",
    # Workflow functions
    "process_1d",
    "process_2d",
    "basic_1d_process",
    "basic_2d_process",
    # Auto processing
    "auto_process",
]
