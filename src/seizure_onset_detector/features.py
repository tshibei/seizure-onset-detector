"""Feature extraction for iEEG signals.

Per window, compute per-channel features and pool across channels (mean and max).
Each window: (n_channels, n_samples) → feature vector of length 2 * n_features_per_channel.
"""
import numpy as np

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 80),
}


# ---------- per-channel features ----------

def compute_line_length(window):
    """Line length per channel. window: (n_channels, n_samples). Returns (n_channels,)."""
    return np.sum(np.abs(np.diff(window, axis=1)), axis=1)


def hjorth_parameters(window):
    """Hjorth activity, mobility, complexity per channel.
    window: (n_channels, n_samples). Returns 3 arrays of shape (n_channels,)."""
    d1 = np.diff(window, axis=1)
    d2 = np.diff(d1, axis=1)
    
    var_signal = np.var(window, axis=1)
    var_d1 = np.var(d1, axis=1)
    var_d2 = np.var(d2, axis=1)
    
    # Guard against division by zero (flat channels)
    safe_var_signal = np.where(var_signal > 0, var_signal, 1)
    safe_var_d1 = np.where(var_d1 > 0, var_d1, 1)
    
    activity = var_signal
    mobility = np.sqrt(var_d1 / safe_var_signal)
    complexity = np.sqrt(var_d2 / safe_var_d1) / np.where(mobility > 0, mobility, 1)
    
    # Zero out where input was degenerate
    mobility = np.where(var_signal > 0, mobility, 0)
    complexity = np.where((var_signal > 0) & (var_d1 > 0), complexity, 0)
    
    return activity, mobility, complexity


def compute_spectrum(window, fs):
    """Hann-windowed power spectrum per channel.
    window: (n_channels, n_samples). Returns freqs (n_freqs,) and power (n_channels, n_freqs)."""
    n = window.shape[1]
    hann = np.hanning(n)
    windowed = window * hann  # broadcasts over channels
    fft_vals = np.fft.rfft(windowed, axis=1)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    return freqs, np.abs(fft_vals) ** 2


def band_power(power_spectrum, freqs, band):
    """Power in a frequency band per channel.
    power_spectrum: (n_channels, n_freqs). Returns (n_channels,)."""
    mask = (freqs >= band[0]) & (freqs < band[1])
    return power_spectrum[:, mask].sum(axis=1)


def spectral_edge_frequency(power_spectrum, freqs, edge_percent=0.95):
    """SEF per channel: frequency below which edge_percent of power lies.
    power_spectrum: (n_channels, n_freqs). Returns (n_channels,)."""
    total = power_spectrum.sum(axis=1)
    cumulative = np.cumsum(power_spectrum, axis=1)
    threshold = edge_percent * total[:, None]  # (n_channels, 1)
    # For each channel, find first index where cumulative >= threshold
    idx = (cumulative >= threshold).argmax(axis=1)
    idx = np.where(total > 0, idx, 0)  # flat channels → 0 Hz
    return freqs[idx]


def energy_ratio(band_powers):
    """Bartolomei (2008) energy ratio: fast-band / slow-band power per channel.
    
    Delta excluded from denominator to match Bartolomei's bands (~3.5-12.4 Hz slow)
    and avoid contamination by drift/movement artifacts.
    
    band_powers: dict mapping band name → (n_channels,) array. Returns (n_channels,)."""
    low = band_powers["theta"] + band_powers["alpha"]
    high = band_powers["beta"] + band_powers["gamma"]
    return high / (low + 1e-10)


# ---------- per-window pipeline ----------

def feature_names():
    """Names of features in the order returned by extract_features."""
    per_channel = (
        ["line_length"]
        + [f"bp_{band}" for band in BANDS]
        + ["sef95"]
        + ["hjorth_activity", "hjorth_mobility", "hjorth_complexity"]
        + ["energy_ratio"]
    )
    return [f"{name}_mean" for name in per_channel] + [f"{name}_max" for name in per_channel]


def extract_features(window, fs):
    """Extract per-window feature vector by computing per-channel features and pooling.
    
    window: (n_channels, n_samples) array.
    Returns: 1D feature vector of length 2 * n_features_per_channel.
    """
    freqs, power_spectrum = compute_spectrum(window, fs)
    
    band_powers = {name: band_power(power_spectrum, freqs, band) 
                   for name, band in BANDS.items()}
    
    activity, mobility, complexity = hjorth_parameters(window)
    
    # Stack per-channel features: shape (n_features, n_channels)
    per_channel = np.stack([
        compute_line_length(window),
        *band_powers.values(),
        spectral_edge_frequency(power_spectrum, freqs),
        activity, mobility, complexity,
        energy_ratio(band_powers),
    ])
    
    # Pool across channels: mean and max
    return np.concatenate([per_channel.mean(axis=1), per_channel.max(axis=1)])


def extract_features_batch(windows, fs):
    """Apply extract_features to a stack of windows.
    
    windows: (n_windows, n_channels, n_samples) array.
    Returns: (n_windows, n_features) feature matrix.
    """
    return np.stack([extract_features(w, fs) for w in windows])