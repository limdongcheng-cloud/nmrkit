import numpy as np
import pytest
from nmrkit.core.data import NMRData, DimensionInfo, LinearGenerator
from nmrkit.processing.phase import (
    phase_correct,
    correct_digital_filter_phase,
    remove_digital_filter,
    autophase,
)


# Helper function to create test data
def create_test_1d_data(
        size=128,
        complex=True,
        domain_type="frequency",
        can_ft=True):
    if complex:
        data_array = np.random.rand(size) + 1j * np.random.rand(size)
    else:
        data_array = np.random.rand(size)

    dims = [
        DimensionInfo(
            size=size,
            is_complex=complex,
            domain_type=domain_type,
            can_ft=can_ft,
            axis_generator=LinearGenerator(start=0.0, step=0.1),
        )
    ]

    return NMRData(data=data_array, dimensions=dims)


# Test phase_correct function
def test_phase_correct_basic():
    # Test basic phase correction functionality
    nmr_data = create_test_1d_data(
        size=128, complex=True, domain_type="frequency")
    original_data = nmr_data.data.copy()

    # Apply zero-order phase correction (90 degrees)
    result = phase_correct(nmr_data, dim=0, ph0=90.0, ph1=0.0)

    # Verify data has been modified
    assert not np.array_equal(result.data, original_data)

    # Check that metadata has been updated
    assert "phase_correction" in result.dimensions[0].domain_metadata
    correction = result.dimensions[0].domain_metadata["phase_correction"][-1]
    assert correction["ph0"] == 90.0
    assert correction["ph1"] == 0.0
    assert correction["pivot"] == 64  # Default pivot at center


def test_phase_correct_zero_order():
    # Test zero-order phase correction
    nmr_data = create_test_1d_data(
        size=128, complex=True, domain_type="frequency")

    # Create test data with known phase
    data_array = np.ones(128, dtype=np.complex128)
    nmr_data.data = data_array

    # Apply 0 degrees phase correction (should not change data)
    result_0 = phase_correct(nmr_data, dim=0, ph0=0.0)
    np.testing.assert_array_equal(result_0.data, data_array)

    # Apply 180 degrees phase correction (should invert sign)
    result_180 = phase_correct(nmr_data, dim=0, ph0=180.0)
    expected_180 = np.ones(128, dtype=np.complex128) * np.exp(1j * np.pi)
    np.testing.assert_array_almost_equal(result_180.data, expected_180)


def test_phase_correct_first_order():
    # Test first-order phase correction
    nmr_data = create_test_1d_data(
        size=128, complex=True, domain_type="frequency")

    # Create test data
    data_array = np.ones(128, dtype=np.complex128)
    nmr_data.data = data_array

    # Apply first-order phase correction
    result = phase_correct(nmr_data, dim=0, ph0=0.0, ph1=360.0, pivot=64)

    # Check that phase changes linearly across the spectrum
    angles = np.angle(result.data)

    # Calculate expected angles using the same formula as in the code
    indices = np.arange(128)
    expected_angles = np.deg2rad(360.0) * (indices - 64) / (128 - 1)

    # Normalize both actual and expected angles to the same range
    angles_normalized = np.angle(np.exp(1j * angles))
    expected_normalized = np.angle(np.exp(1j * expected_angles))

    np.testing.assert_array_almost_equal(
        angles_normalized, expected_normalized, decimal=4
    )


def test_phase_correct_pivot():
    # Test phase correction with different pivot points
    nmr_data = create_test_1d_data(
        size=128, complex=True, domain_type="frequency")

    # Create test data
    data_array = np.ones(128, dtype=np.complex128)
    nmr_data.data = data_array

    # Apply first-order phase correction with pivot at beginning
    result_start = phase_correct(
        nmr_data, dim=0, ph0=0.0, ph1=360.0, pivot=0
    )
    angles_start = np.angle(result_start.data)

    # Calculate expected angles using the same formula as in the code
    indices = np.arange(128)
    expected_angles_start = np.deg2rad(360.0) * (indices - 0) / (128 - 1)

    # Normalize both actual and expected angles to the same range
    angles_start_normalized = np.angle(np.exp(1j * angles_start))
    expected_start_normalized = np.angle(np.exp(1j * expected_angles_start))

    np.testing.assert_array_almost_equal(
        angles_start_normalized, expected_start_normalized, decimal=4
    )

    # Apply first-order phase correction with pivot at end
    result_end = phase_correct(
        nmr_data, dim=0, ph0=0.0, ph1=360.0, pivot=127
    )
    angles_end = np.angle(result_end.data)

    # Calculate expected angles using the same formula as in the code
    expected_angles_end = np.deg2rad(360.0) * (indices - 127) / (128 - 1)

    # Normalize both actual and expected angles to the same range
    angles_end_normalized = np.angle(np.exp(1j * angles_end))
    expected_end_normalized = np.angle(np.exp(1j * expected_angles_end))

    np.testing.assert_array_almost_equal(
        angles_end_normalized, expected_end_normalized, decimal=4
    )


def test_phase_correct_error_handling():
    # Test error handling in phase_correct
    nmr_data = create_test_1d_data(
        size=128, complex=True, domain_type="frequency")

    # Test with invalid dimension
    with pytest.raises(ValueError):
        phase_correct(nmr_data, dim=1)  # Dimension out of range

    # Test with invalid pivot point
    with pytest.raises(ValueError):
        phase_correct(nmr_data, dim=0, pivot=-1)  # Pivot too low

    with pytest.raises(ValueError):
        phase_correct(nmr_data, dim=0, pivot=128)  # Pivot too high


# Test correct_digital_filter_phase function
def test_correct_digital_filter_phase_basic():
    # Test basic digital filter phase correction
    nmr_data = create_test_1d_data(
        size=128, complex=True, domain_type="frequency")
    original_data = nmr_data.data.copy()

    # Apply correction with group delay
    result = correct_digital_filter_phase(nmr_data, dim=0, group_delay=10.0)

    # Verify data has been modified
    assert not np.array_equal(result.data, original_data)

    # Check that metadata has been updated
    assert "phase_correction" in result.dimensions[0].domain_metadata
    correction = result.dimensions[0].domain_metadata["phase_correction"][-1]
    assert correction["type"] == "digital_filter"
    assert correction["group_delay"] == 10.0


def test_correct_digital_filter_phase_metadata():
    # Test group delay extraction from metadata
    nmr_data = create_test_1d_data(
        size=128, complex=True, domain_type="frequency")

    # Add TopSpin-style metadata with group delay
    nmr_data.source_format = "topspin"
    nmr_data.metadata = {"parameters": {"direct": {"GRPDLY": 5.0}}}

    # Apply correction without specifying group delay
    result = correct_digital_filter_phase(nmr_data, dim=0)

    # Check that group delay was extracted from metadata
    correction = result.dimensions[0].domain_metadata["phase_correction"][-1]
    assert correction["group_delay"] == 5.0


def test_correct_digital_filter_phase_indirect_dimension():
    # Test correction for indirect dimensions
    nmr_data = create_test_1d_data(
        size=128, complex=True, domain_type="frequency")

    # Add TopSpin-style metadata with group delay for indirect dimension
    nmr_data.source_format = "topspin"
    nmr_data.metadata = {"parameters": {"indirect1": {"GRPDLY": 7.5}}}

    # Apply correction (should use default since dim 0 doesn't have indirect1
    # metadata)
    result = correct_digital_filter_phase(nmr_data, dim=0)
    correction = result.dimensions[0].domain_metadata["phase_correction"][-1]
    assert correction["group_delay"] == 0.0  # Default value


def test_correct_digital_filter_phase_error_handling():
    # Test error handling in correct_digital_filter_phase
    nmr_data = create_test_1d_data(
        size=128, complex=True, domain_type="frequency")

    # Test with invalid dimension
    with pytest.raises(ValueError):
        correct_digital_filter_phase(nmr_data, dim=1)  # Dimension out of range


# --- Helper to create a synthetic spectrum with known phase ---

def create_phased_spectrum(size=512, ph0_deg=0.0, ph1_deg=0.0, n_peaks=5, seed=42):
    """Create a synthetic NMR spectrum with known phase distortion.

    Generates Lorentzian peaks at random positions, applies a known
    (ph0, ph1) phase rotation, and returns both the distorted NMRData
    and the true phase values.
    """
    rng = np.random.RandomState(seed)
    freq = np.linspace(-1000, 1000, size)

    # Build a clean absorption-mode spectrum (all-positive Lorentzians)
    spectrum = np.zeros(size, dtype=np.complex128)
    for _ in range(n_peaks):
        center = rng.uniform(-800, 800)
        width = rng.uniform(5, 30)
        amplitude = rng.uniform(50, 200)
        lorentz = amplitude * width / (width + 1j * (freq - center))
        spectrum += lorentz

    # Apply known phase distortion
    ph0_rad = np.deg2rad(ph0_deg)
    ph1_rad = np.deg2rad(ph1_deg)
    indices = np.arange(size) / max(size - 1, 1)
    phase = ph0_rad + ph1_rad * indices
    distorted = spectrum * np.exp(-1j * phase)  # negative = distortion

    dims = [
        DimensionInfo(
            size=size,
            is_complex=True,
            domain_type="frequency",
            can_ft=False,
            axis_generator=LinearGenerator(start=-1000.0, step=2000.0 / size),
        )
    ]
    nmr_data = NMRData(data=distorted, dimensions=dims)
    return nmr_data, ph0_deg, ph1_deg


# --- Autophase tests ---

class TestAutophaseACME:
    """Tests for ACME entropy-minimization autophase."""

    def test_pure_zero_order(self):
        """ACME should recover a pure zero-order phase shift."""
        data, true_ph0, _ = create_phased_spectrum(
            ph0_deg=45.0, ph1_deg=0.0, n_peaks=5)
        result = autophase(data, method="acme")

        # After correction, real part should be mostly positive
        positive_frac = np.sum(result.data.real > 0) / len(result.data)
        assert positive_frac > 0.7, f"Only {positive_frac:.0%} positive after ACME"

    def test_mixed_phase(self):
        """ACME should recover both zero- and first-order phase."""
        data, _, _ = create_phased_spectrum(
            ph0_deg=30.0, ph1_deg=60.0, n_peaks=8, size=1024)
        result = autophase(data, method="acme")

        # Negative area should be small after correction
        real = result.data.real
        neg_area = np.sum(real[real < 0] ** 2)
        total_area = np.sum(real ** 2)
        neg_frac = neg_area / total_area
        assert neg_frac < 0.05, f"Negative fraction {neg_frac:.2%} too large"

    def test_large_phase(self):
        """ACME should handle large phase distortions."""
        data, _, _ = create_phased_spectrum(
            ph0_deg=170.0, ph1_deg=120.0, n_peaks=6, size=1024)
        result = autophase(data, method="acme")

        real = result.data.real
        neg_area = np.sum(real[real < 0] ** 2)
        total_area = np.sum(real ** 2)
        assert neg_area / total_area < 0.10

    def test_metadata_recorded(self):
        """Autophase should record method and parameters in metadata."""
        data, _, _ = create_phased_spectrum(ph0_deg=20.0, ph1_deg=10.0)
        result = autophase(data, method="acme")

        corrections = result.dimensions[0].domain_metadata["phase_correction"]
        last = corrections[-1]
        assert last["type"] == "autophase"
        assert last["method"] == "acme"
        assert "ph0" in last
        assert "ph1" in last

    def test_returns_copy(self):
        """Autophase must not modify the original data."""
        data, _, _ = create_phased_spectrum(ph0_deg=45.0, ph1_deg=0.0)
        original = data.data.copy()
        _ = autophase(data, method="acme")
        np.testing.assert_array_equal(data.data, original)


class TestAutophasePeakMinima:
    """Tests for peak-minima autophase method."""

    def test_pure_zero_order(self):
        """Peak-minima should recover a pure zero-order phase shift."""
        data, _, _ = create_phased_spectrum(
            ph0_deg=90.0, ph1_deg=0.0, n_peaks=5)
        result = autophase(data, method="peak_minima")

        positive_frac = np.sum(result.data.real > 0) / len(result.data)
        assert positive_frac > 0.7

    def test_mixed_phase(self):
        """Peak-minima should handle mixed phase."""
        data, _, _ = create_phased_spectrum(
            ph0_deg=30.0, ph1_deg=40.0, n_peaks=6, size=1024)
        result = autophase(data, method="peak_minima")

        real = result.data.real
        neg_area = np.sum(real[real < 0] ** 2)
        total_area = np.sum(real ** 2)
        assert neg_area / total_area < 0.10


class TestAutophaseEdgeCases:
    """Edge cases and error handling."""

    def test_invalid_method(self):
        """Unknown method should raise ValueError."""
        data, _, _ = create_phased_spectrum()
        with pytest.raises(ValueError, match="Unknown autophase method"):
            autophase(data, method="nonexistent")

    def test_invalid_dimension(self):
        """Out-of-range dimension should raise ValueError."""
        data, _, _ = create_phased_spectrum()
        with pytest.raises(ValueError):
            autophase(data, dim=5)

    def test_with_initial_guess(self):
        """User-provided initial guess should be accepted."""
        data, _, _ = create_phased_spectrum(ph0_deg=30.0, ph1_deg=0.0)
        result = autophase(data, method="acme", ph0_init=25.0, ph1_init=0.0)
        assert result is not data


# Tests for JEOL Delta digital filter group delay

def test_correct_digital_filter_phase_delta_metadata():
    """Test group delay extraction from Delta metadata."""
    nmr_data = create_test_1d_data(
        size=128, complex=True, domain_type="frequency")

    nmr_data.source_format = "delta"
    nmr_data.metadata = {"digital_filter_group_delay": 19.5}

    result = correct_digital_filter_phase(nmr_data, dim=0)
    correction = result.dimensions[0].domain_metadata["phase_correction"][-1]
    assert correction["group_delay"] == 19.5


def test_remove_digital_filter_basic():
    """Test time-domain circular shift removes pre-echo."""
    nmr_data = create_test_1d_data(
        size=128, complex=True, domain_type="time")
    original = nmr_data.data.copy()

    # Simulate pre-echo: first 10 points are near-zero, rest is signal
    nmr_data.data[:10] = 0.001 * (np.random.rand(10) + 1j * np.random.rand(10))
    nmr_data.data[10:] = 10.0 * (np.random.rand(118) + 1j * np.random.rand(118))

    result = remove_digital_filter(nmr_data, group_delay=10.0)

    # After shift, first point should be what was at index 10 (large signal)
    assert np.abs(result.data[0]) > 1.0
    # Last 10 points should be zeroed
    np.testing.assert_array_equal(result.data[-10:], 0)
    # Metadata should record the correction
    correction = result.dimensions[0].domain_metadata["phase_correction"][-1]
    assert correction["type"] == "digital_filter_removal"
    assert correction["shift"] == 10


def test_remove_digital_filter_auto_metadata():
    """Test auto group delay extraction from Delta metadata."""
    nmr_data = create_test_1d_data(
        size=128, complex=True, domain_type="time")
    nmr_data.source_format = "delta"
    nmr_data.metadata = {"digital_filter_group_delay": 5.7}

    result = remove_digital_filter(nmr_data, dim=0)
    correction = result.dimensions[0].domain_metadata["phase_correction"][-1]
    # 5.7 rounds to 6
    assert correction["shift"] == 6


def test_remove_digital_filter_zero_delay():
    """Test that zero group delay returns unchanged copy."""
    nmr_data = create_test_1d_data(
        size=128, complex=True, domain_type="time")
    original = nmr_data.data.copy()

    result = remove_digital_filter(nmr_data, group_delay=0.0)
    np.testing.assert_array_equal(result.data, original)
