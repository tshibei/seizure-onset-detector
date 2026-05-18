import numpy as np

from seizure_onset_detector.features import (
    band_power,
    compute_line_length,
    extract_features,
    feature_names,
)


def test_line_length_flat():
    """Line length of a flat signal should be zero."""
    window = np.array([[1, 1, 1], [2, 2, 2]])
    ll = compute_line_length(window)
    assert np.all(ll == 0)

def test_band_power():
    """Band power on a synthetic 10 Hz sine wave is concentrated in the alpha band."""
    BANDS = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 80),
    }
    fs = 256
    signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, fs))[None, :]  # (1, fs)
    freqs = np.fft.rfftfreq(signal.shape[1], d=1/fs)
    power_spectrum = np.abs(np.fft.rfft(signal, axis=1))**2

    powers = {band: band_power(power_spectrum, freqs, rng)[0] for band, rng in BANDS.items()}
    assert powers["alpha"] == max(powers.values()), powers

def test_extract_features_shape_and_finite():
    """Smoke test: extract_features produces the expected shape and no NaN/inf."""
    fs = 256
    window = np.random.randn(8, fs)  # 1 second, 8 channels
    
    features = extract_features(window, fs)
    names = feature_names()

    assert features.shape == (len(names),)
    assert np.all(np.isfinite(features)), "Features contain NaN or inf"
