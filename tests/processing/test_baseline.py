"""Tests for baseline correction module."""

import numpy as np
import pytest

from nmrkit.core.data import NMRData, DimensionInfo, LinearGenerator
from nmrkit.processing.baseline import (
    _asls_1d,
    _airpls_1d,
    _polynomial_1d,
    _center_baseline,
    baseline_correct,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def create_test_1d_data(
    size: int = 512,
    complex_data: bool = True,
    domain_type: str = "frequency",
) -> NMRData:
    """Create a simple test NMRData object."""
    rng = np.random.default_rng(42)
    if complex_data:
        data_array = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    else:
        data_array = rng.standard_normal(size)

    dims = [
        DimensionInfo(
            size=size,
            is_complex=complex_data,
            domain_type=domain_type,
            spectral_width=5000.0,
            observation_frequency=400.0,
            nucleus="1H",
            axis_generator=LinearGenerator(start=0.0, step=1.0 / 5000.0),
        )
    ]
    return NMRData(data=data_array, dimensions=dims)


def make_lorentzian_peak(x: np.ndarray, center: float, amplitude: float, width: float) -> np.ndarray:
    """Generate a Lorentzian line shape."""
    return amplitude * width**2 / ((x - center)**2 + width**2)


def create_spectrum_with_baseline(
    size: int = 1024,
    n_peaks: int = 5,
    baseline_type: str = "polynomial",
    noise_level: float = 0.01,
    seed: int = 42,
):
    """Create a synthetic spectrum with known peaks and known baseline.

    Returns:
        (NMRData, true_baseline): Tuple of NMR data and the ground truth baseline.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(size, dtype=float)

    # Generate peaks
    spectrum = np.zeros(size)
    peak_positions = np.linspace(size * 0.1, size * 0.9, n_peaks)
    for pos in peak_positions:
        amplitude = rng.uniform(5.0, 20.0)
        width = rng.uniform(3.0, 8.0)
        spectrum += make_lorentzian_peak(x, pos, amplitude, width)

    # Generate baseline
    if baseline_type == "polynomial":
        # Quadratic baseline: a*x^2 + b*x + c
        a = rng.uniform(-1e-5, 1e-5)
        b = rng.uniform(-0.01, 0.01)
        c = rng.uniform(-1.0, 1.0)
        true_baseline = a * x**2 + b * x + c
    elif baseline_type == "sinusoidal":
        true_baseline = 2.0 * np.sin(2 * np.pi * x / size) + 0.5 * np.cos(4 * np.pi * x / size)
    elif baseline_type == "constant":
        true_baseline = np.full(size, 3.0)
    else:
        true_baseline = np.zeros(size)

    # Combine: spectrum with baseline + noise
    real_part = spectrum + true_baseline + noise_level * rng.standard_normal(size)
    imag_part = noise_level * rng.standard_normal(size)
    data_array = real_part + 1j * imag_part

    dims = [
        DimensionInfo(
            size=size,
            is_complex=True,
            domain_type="frequency",
            spectral_width=5000.0,
            observation_frequency=400.0,
            nucleus="1H",
            axis_generator=LinearGenerator(start=0.0, step=1.0 / 5000.0),
        )
    ]
    nmr_data = NMRData(data=data_array, dimensions=dims)
    return nmr_data, true_baseline


# ===========================================================================
# Category 1: Algorithm unit tests (internal functions)
# ===========================================================================

class TestAsls1D:
    """Tests for the _asls_1d internal function."""

    def test_flat_baseline(self):
        """Flat spectrum should produce near-zero baseline."""
        spectrum = np.ones(256) * 5.0
        baseline = _asls_1d(spectrum, lambda_=1e4, p=0.01)
        # Baseline should match the constant value
        np.testing.assert_allclose(baseline, spectrum, atol=0.5)

    def test_polynomial_baseline_recovery(self):
        """Should recover a smooth polynomial baseline under peaks."""
        n = 1024
        x = np.arange(n, dtype=float)
        true_bl = 0.5e-5 * x**2 - 0.01 * x + 2.0

        # Add peaks
        spectrum = true_bl.copy()
        for pos in [200, 400, 600, 800]:
            spectrum += make_lorentzian_peak(x, pos, 10.0, 5.0)

        baseline = _asls_1d(spectrum, lambda_=1e6, p=0.001)

        # Check that baseline is close to true baseline (not peaks)
        # Exclude peak regions for comparison
        peak_mask = np.zeros(n, dtype=bool)
        for pos in [200, 400, 600, 800]:
            peak_mask[max(0, pos - 30):min(n, pos + 30)] = True
        non_peak = ~peak_mask

        rms_error = np.sqrt(np.mean((baseline[non_peak] - true_bl[non_peak])**2))
        assert rms_error < 1.0, f"RMS error {rms_error:.3f} too large"

    def test_lambda_effect(self):
        """Higher lambda should produce smoother baseline."""
        rng = np.random.default_rng(42)
        spectrum = rng.standard_normal(512) + 5.0

        bl_low = _asls_1d(spectrum, lambda_=1e2)
        bl_high = _asls_1d(spectrum, lambda_=1e7)

        # Smoothness measured by second derivative magnitude
        roughness_low = np.sum(np.diff(bl_low, n=2)**2)
        roughness_high = np.sum(np.diff(bl_high, n=2)**2)
        assert roughness_high < roughness_low

    def test_short_spectrum(self):
        """Very short spectrum should return zeros."""
        spectrum = np.array([1.0, 2.0, 3.0])
        baseline = _asls_1d(spectrum)
        np.testing.assert_array_equal(baseline, np.zeros(3))


class TestAirpls1D:
    """Tests for the _airpls_1d internal function."""

    def test_basic_recovery(self):
        """Should recover a smooth baseline from spectrum with peaks."""
        n = 512
        x = np.arange(n, dtype=float)
        true_bl = 0.01 * np.sin(2 * np.pi * x / n)

        spectrum = true_bl.copy()
        for pos in [128, 256, 384]:
            spectrum += make_lorentzian_peak(x, pos, 8.0, 4.0)

        baseline = _airpls_1d(spectrum, lambda_=1e5)

        # Baseline should be below peaks
        residual = spectrum - baseline
        # Most of the residual should be positive (peaks above baseline)
        assert np.mean(residual > -0.5) > 0.8

    def test_short_spectrum(self):
        """Very short spectrum should return zeros."""
        spectrum = np.array([1.0, 2.0])
        baseline = _airpls_1d(spectrum)
        np.testing.assert_array_equal(baseline, np.zeros(2))

    def test_zero_spectrum(self):
        """All-zero spectrum should return zeros."""
        spectrum = np.zeros(256)
        baseline = _airpls_1d(spectrum)
        np.testing.assert_array_equal(baseline, np.zeros(256))


class TestPolynomial1D:
    """Tests for the _polynomial_1d internal function."""

    def test_constant_baseline(self):
        """Should recover a constant offset."""
        n = 512
        x = np.arange(n, dtype=float)
        offset = 5.0
        spectrum = np.full(n, offset)
        # Add a few peaks
        for pos in [100, 300]:
            spectrum += make_lorentzian_peak(x, pos, 10.0, 5.0)

        baseline = _polynomial_1d(spectrum, order=2)
        # In non-peak regions, baseline should be close to offset
        non_peak = np.ones(n, dtype=bool)
        for pos in [100, 300]:
            non_peak[max(0, pos - 30):min(n, pos + 30)] = False

        np.testing.assert_allclose(
            np.mean(baseline[non_peak]), offset, atol=1.0
        )

    def test_quadratic_baseline(self):
        """Should recover a quadratic baseline."""
        n = 1024
        x = np.arange(n, dtype=float)
        true_bl = 1e-5 * (x - n / 2)**2

        spectrum = true_bl.copy()
        for pos in [200, 500, 800]:
            spectrum += make_lorentzian_peak(x, pos, 15.0, 5.0)

        baseline = _polynomial_1d(spectrum, order=3, max_iter=50)

        # Check non-peak regions
        non_peak = np.ones(n, dtype=bool)
        for pos in [200, 500, 800]:
            non_peak[max(0, pos - 40):min(n, pos + 40)] = False

        rms_error = np.sqrt(np.mean((baseline[non_peak] - true_bl[non_peak])**2))
        assert rms_error < 1.0, f"RMS error {rms_error:.3f} too large"

    def test_short_spectrum(self):
        """Very short spectrum should return zeros."""
        spectrum = np.array([1.0, 2.0, 3.0])
        baseline = _polynomial_1d(spectrum)
        np.testing.assert_array_equal(baseline, np.zeros(3))


# ===========================================================================
# Category 1b: Centering tests (_center_baseline)
# ===========================================================================

class TestCenterBaseline:
    """Tests for the _center_baseline offset correction."""

    def test_global_removes_constant_offset(self):
        """Global centering should remove a constant offset from noise."""
        rng = np.random.default_rng(7)
        spec = (rng.normal(50, 10, 1000) + 0j)
        centered = _center_baseline(spec, local=False)
        assert abs(np.median(centered.real)) < 2.0

    def test_local_removes_regional_drift(self):
        """Local centering should handle slowly-varying offset."""
        x = np.arange(4000, dtype=float)
        drift = 30 * x / 4000  # 0 at left, +30 at right
        noise = np.random.default_rng(8).normal(0, 5, 4000)
        spec = (drift + noise) + 0j

        centered = _center_baseline(spec, local=True)
        left_mean = centered.real[:500].mean()
        right_mean = centered.real[3500:].mean()
        assert abs(left_mean) < 5.0
        assert abs(right_mean) < 5.0

    def test_preserves_peaks(self):
        """Centering should not destroy peak signals."""
        rng = np.random.default_rng(9)
        spec = rng.normal(10, 5, 1000)
        spec[500] = 5000  # big peak
        centered = _center_baseline(spec + 0j, local=False)
        assert centered.real[500] > 4000

    def test_refine_better_flatness(self):
        """Local centering should produce flatter noise than global."""
        x = np.arange(4096, dtype=float)
        drift = 20 * np.sin(2 * np.pi * x / 4096)
        noise = np.random.default_rng(10).normal(0, 5, 4096)
        spec = (drift + noise) + 0j

        global_c = _center_baseline(spec, local=False)
        local_c = _center_baseline(spec, local=True)

        def seg_std(real, n_seg=8):
            segs = np.array_split(real, n_seg)
            return np.std([s.mean() for s in segs])

        assert seg_std(local_c.real) < seg_std(global_c.real)

    def test_few_noise_points_no_crash(self):
        """Spectrum where almost all points are 'peaks' should not crash."""
        spec = np.ones(100) * 5000 + 0j
        result = _center_baseline(spec, local=False)
        assert result is not None
        assert len(result) == 100


# ===========================================================================
# Category 2: Public API tests (baseline_correct)
# ===========================================================================

class TestBaselineCorrect:
    """Tests for the public baseline_correct function."""

    def test_returns_copy(self):
        """Result should be a new object, not the input."""
        data = create_test_1d_data(256)
        result = baseline_correct(data)
        assert result is not data
        assert result.data is not data.data

    def test_preserves_shape(self):
        """Output shape should match input shape."""
        data = create_test_1d_data(512)
        result = baseline_correct(data)
        assert result.data.shape == data.data.shape

    def test_default_method_is_asls(self):
        """Default method should be 'asls'."""
        data = create_test_1d_data(256)
        result = baseline_correct(data)
        assert result.dimensions[0].domain_metadata["baseline_correction"]["method"] == "asls"

    @pytest.mark.parametrize("method", ["asls", "airpls", "polynomial", "poly"])
    def test_all_methods_run(self, method):
        """All supported methods should execute without error."""
        data = create_test_1d_data(256)
        result = baseline_correct(data, method=method)
        assert result.data.shape == data.data.shape
        meta = result.dimensions[0].domain_metadata["baseline_correction"]
        expected_method = "polynomial" if method == "poly" else method
        assert meta["method"] == expected_method or meta["method"] == method

    def test_metadata_update(self):
        """Should record correction parameters in metadata."""
        data = create_test_1d_data(256)
        result = baseline_correct(data, method="asls", lambda_=1e4, p=0.05)
        meta = result.dimensions[0].domain_metadata["baseline_correction"]
        assert meta["method"] == "asls"
        assert meta["lambda_"] == 1e4
        assert meta["p"] == 0.05

    def test_complex_data_imag_unchanged(self):
        """Imaginary part should not be affected by baseline subtraction."""
        data, _ = create_spectrum_with_baseline(size=512, noise_level=0.0)
        # Set imaginary part to known values
        data.data = data.data.real + 1j * np.arange(512, dtype=float)

        result = baseline_correct(data, method="asls")

        # Imaginary part should be identical
        np.testing.assert_array_equal(result.data.imag, data.data.imag)

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError."""
        data = create_test_1d_data(256)
        with pytest.raises(ValueError, match="Unknown baseline correction method"):
            baseline_correct(data, method="nonexistent")

    def test_invalid_dimension_raises(self):
        """Out-of-range dimension should raise ValueError."""
        data = create_test_1d_data(256)
        with pytest.raises(ValueError):
            baseline_correct(data, dim=5)

    def test_extra_kwargs_ignored(self):
        """Unrecognized kwargs should be silently ignored."""
        data = create_test_1d_data(256)
        # Should not raise
        result = baseline_correct(data, method="asls", bogus_param=42)
        assert result.data.shape == data.data.shape

    def test_refine_default_true(self):
        """Default refine should be True."""
        data = create_test_1d_data(256)
        result = baseline_correct(data)
        meta = result.dimensions[0].domain_metadata["baseline_correction"]
        assert meta["refine"] is True

    def test_refine_false_recorded(self):
        """refine=False should be recorded in metadata."""
        data = create_test_1d_data(256)
        result = baseline_correct(data, refine=False)
        meta = result.dimensions[0].domain_metadata["baseline_correction"]
        assert meta["refine"] is False

    def test_refine_improves_flatness(self):
        """refine=True should produce flatter noise than refine=False."""
        rng = np.random.default_rng(42)
        n = 4096
        x = np.arange(n, dtype=float)
        # Quadratic baseline + regional drift + noise (no peaks)
        bl = 500 + 0.001 * (x - n/2)**2 + 20 * np.sin(2*np.pi*x/n)
        spec = bl + rng.normal(0, 10, n)

        dims = [DimensionInfo(
            size=n, is_complex=True, domain_type="frequency",
            axis_generator=LinearGenerator(start=0.0, step=1.0)
        )]
        data = NMRData(data=spec + 0j, dimensions=dims)

        r_global = baseline_correct(data, method="asls", lambda_=1e6, refine=False)
        r_local = baseline_correct(data, method="asls", lambda_=1e6, refine=True)

        # Segment-wise flatness: std of segment means (lower = flatter)
        def seg_flatness(real, n_seg=10):
            segs = np.array_split(real, n_seg)
            return np.std([s.mean() for s in segs])

        assert seg_flatness(r_local.data.real) <= seg_flatness(r_global.data.real)


# ===========================================================================
# Category 3: NMR-specific accuracy tests
# ===========================================================================

class TestBaselineAccuracy:
    """Tests verifying baseline correction quality on synthetic NMR spectra."""

    def test_constant_offset_removal(self):
        """Constant baseline offset should be removed."""
        data, true_bl = create_spectrum_with_baseline(
            size=1024, baseline_type="constant", noise_level=0.01
        )
        result = baseline_correct(data, method="asls", lambda_=1e5)

        # After correction, mean of non-peak regions should be near zero
        corrected_real = result.data.real
        # Use the lower 20% of values as proxy for baseline regions
        sorted_vals = np.sort(corrected_real)
        baseline_region = sorted_vals[:len(sorted_vals) // 5]
        assert abs(np.mean(baseline_region)) < 1.0

    def test_polynomial_baseline_removal(self):
        """Polynomial baseline should be substantially reduced."""
        data, true_bl = create_spectrum_with_baseline(
            size=1024, baseline_type="polynomial", noise_level=0.01
        )
        before_std = np.std(data.data.real - true_bl)  # This is ~noise level
        result = baseline_correct(data, method="asls", lambda_=1e6, p=0.001)

        # The corrected spectrum's baseline variance should be reduced
        corrected_real = result.data.real
        # Baseline region: bottom 20% of values
        sorted_vals = np.sort(corrected_real)
        baseline_region = sorted_vals[:len(sorted_vals) // 5]
        assert np.std(baseline_region) < np.std(true_bl) + 1.0

    def test_peaks_preserved(self):
        """Peak heights should be approximately preserved after correction."""
        data, true_bl = create_spectrum_with_baseline(
            size=1024, n_peaks=3, baseline_type="polynomial", noise_level=0.001
        )
        result = baseline_correct(data, method="asls")

        # Maximum value should still be similar (peaks not clipped)
        original_max = data.data.real.max()
        corrected_max = result.data.real.max()
        # Corrected max may differ by the baseline value at that point, but
        # should still be positive and substantial
        assert corrected_max > original_max * 0.3


# ===========================================================================
# Category 4: Multi-dimensional tests
# ===========================================================================

class TestBaselineMultiDim:
    """Tests for baseline correction on multi-dimensional data."""

    def test_2d_data_dim0(self):
        """Should correct each row independently along dim 0."""
        rng = np.random.default_rng(42)
        n0, n1 = 256, 4
        data_array = rng.standard_normal((n0, n1)) + 5.0 + 1j * rng.standard_normal((n0, n1))

        dims = [
            DimensionInfo(size=n0, is_complex=True, domain_type="frequency",
                         axis_generator=LinearGenerator(start=0.0, step=1.0)),
            DimensionInfo(size=n1, is_complex=True, domain_type="frequency",
                         axis_generator=LinearGenerator(start=0.0, step=1.0)),
        ]
        data = NMRData(data=data_array, dimensions=dims)

        result = baseline_correct(data, dim=0, method="asls")
        assert result.data.shape == (n0, n1)
        # Baseline correction metadata should be on dim 0
        assert "baseline_correction" in result.dimensions[0].domain_metadata

    def test_2d_data_dim1(self):
        """Should correct each column independently along dim 1."""
        rng = np.random.default_rng(42)
        n0, n1 = 4, 256
        data_array = rng.standard_normal((n0, n1)) + 3.0 + 1j * rng.standard_normal((n0, n1))

        dims = [
            DimensionInfo(size=n0, is_complex=True, domain_type="frequency",
                         axis_generator=LinearGenerator(start=0.0, step=1.0)),
            DimensionInfo(size=n1, is_complex=True, domain_type="frequency",
                         axis_generator=LinearGenerator(start=0.0, step=1.0)),
        ]
        data = NMRData(data=data_array, dimensions=dims)

        result = baseline_correct(data, dim=1, method="polynomial", order=3)
        assert result.data.shape == (n0, n1)
        assert "baseline_correction" in result.dimensions[1].domain_metadata


# ===========================================================================
# Category 5: Integration test
# ===========================================================================

class TestBaselineIntegration:
    """Integration tests verifying baseline correction in the processing chain."""

    def test_after_phase_correction(self):
        """Baseline correction should work on phase-corrected data."""
        from nmrkit.processing.phase import phase_correct

        data, _ = create_spectrum_with_baseline(
            size=512, baseline_type="polynomial", noise_level=0.01
        )
        # Apply phase correction first
        phased = phase_correct(data, ph0=10.0, ph1=5.0)
        # Then baseline correction
        result = baseline_correct(phased, method="asls")

        assert result.data.shape == data.data.shape
        assert "baseline_correction" in result.dimensions[0].domain_metadata
