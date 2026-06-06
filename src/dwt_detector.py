"""
dwt_detector.py — DWT Pressure Transient Burst Detector
=========================================================
Detects abrupt pipe burst onset using Discrete Wavelet Transform detail
coefficients on high-resolution pressure time series.

Physical basis
--------------
A pipe burst creates a rapid pressure transient: a near-step-function drop
at the burst node, propagating as a pressure wave at acoustic speed
(~1200 m/s in water, modified by pipe wall elasticity).

At sensor sampling rates of 15-min (typical SCADA), the transient manifests
as a sudden change in pressure level between consecutive samples — not a
high-frequency acoustic wave. DWT detail coefficients at level 1-2 respond
to these inter-sample step changes, localizing the burst onset timestep.

For field deployment at ≥1 Hz sampling: level 1 detail captures the
acoustic transient directly (target frequency: acoustic wave / network size).
At 15-min SCADA resolution: level 1 detail is effectively a difference
operator — the burst onset timestep is where |cD1| >> baseline.

Threshold: 3.5σ using MAD estimator (robust to outliers, unlike std).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    import pywt
    _PYWT_AVAILABLE = True
except ImportError:
    _PYWT_AVAILABLE = False


def _haar_detail_level1(signal: np.ndarray) -> np.ndarray:
    """Haar DWT level-1 detail: (x[2k] - x[2k+1]) / sqrt(2)."""
    n = (len(signal) // 2) * 2
    c = signal[:n]
    return (c[0::2] - c[1::2]) / np.sqrt(2)


def dwt_detail_coefficients(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
) -> List[np.ndarray]:
    """
    Compute DWT detail coefficients at multiple levels.

    Parameters
    ----------
    signal : np.ndarray
        1-D pressure time series [m H₂O].
    wavelet : str
        Wavelet family. 'db4' for smooth pressure signals; 'haar' for
        step-change dominated signals. Default: 'db4'.
    level : int
        Number of decomposition levels. At 15-min timestep:
        - Level 1: changes within 30 min
        - Level 2: changes within 1 hour
        - Level 3: changes within 2 hours
        Default: 3.

    Returns
    -------
    list of np.ndarray
        Detail coefficients [cD1, cD2, ..., cD_level].
        cD1 is finest (most sensitive to abrupt changes).
    """
    if _PYWT_AVAILABLE and wavelet != 'haar':
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        # coeffs = [cA_n, cD_n, ..., cD_1]; return details only, finest last
        return list(reversed(coeffs[1:]))  # [cD1, cD2, ..., cD_n]
    else:
        # Haar fallback
        details = []
        current = signal.copy()
        for _ in range(level):
            d = _haar_detail_level1(current)
            details.append(d)
            n = (len(current) // 2) * 2
            current = (current[:n:2] + current[1:n:2]) / np.sqrt(2)
        return details  # [cD1, cD2, ..., cD_level]


def detect_burst_onset(
    pressure_series: np.ndarray,
    timestamps_h: np.ndarray,
    wavelet: str = "db4",
    detail_level: int = 1,
    threshold_sigma: float = 3.5,
) -> Tuple[Optional[float], float]:
    """
    Detect burst onset time from a pressure time series.

    Parameters
    ----------
    pressure_series : np.ndarray
        Pressure at one sensor node [m H₂O].
    timestamps_h : np.ndarray
        Time axis [hours], same length as pressure_series.
    wavelet : str
        DWT wavelet. Default: 'db4'.
    detail_level : int
        Which detail level to threshold. Level 1 for burst detection.
    threshold_sigma : float
        MAD-based threshold multiplier. 3.5σ ≈ 0.02% false alarm for Gaussian.

    Returns
    -------
    onset_time_h : float or None
        Estimated burst onset time [hours]. None if no burst detected.
    max_detail_magnitude : float
        Maximum detail coefficient magnitude (useful for alarm ranking).
    """
    details = dwt_detail_coefficients(pressure_series, wavelet, level=detail_level)
    cD1 = details[0]

    # MAD-based threshold (robust to sustained leak that shifts mean)
    median_d = np.median(cD1)
    mad = np.median(np.abs(cD1 - median_d))
    sigma_robust = mad / 0.6745  # MAD → σ conversion
    threshold = threshold_sigma * max(sigma_robust, 1e-6)

    exceedance = np.where(np.abs(cD1) > threshold)[0]

    if len(exceedance) == 0:
        return None, float(np.max(np.abs(cD1)))

    # Earliest exceedance = burst onset
    onset_coeff_idx = exceedance[0]
    # DWT decimation: level-1 detail has N/2 coefficients
    # Map coefficient index back to original timestep
    decimation = 2 ** detail_level
    onset_sample = onset_coeff_idx * decimation + decimation // 2
    onset_sample = min(onset_sample, len(timestamps_h) - 1)

    return float(timestamps_h[onset_sample]), float(np.max(np.abs(cD1)))


def scan_all_sensors(
    pressures: Dict[str, np.ndarray],
    timestamps_h: np.ndarray,
    wavelet: str = "db4",
    threshold_sigma: float = 3.5,
    cooldown_hours: float = 6.0,
) -> List[dict]:
    """
    Run burst detection across all sensor nodes. Returns a sorted list of
    detected events with estimated onset times.

    Parameters
    ----------
    pressures : dict {sensor_id: pressure_array}
    timestamps_h : np.ndarray
    wavelet : str
    threshold_sigma : float
    cooldown_hours : float
        Minimum time between consecutive alarms on the same sensor.

    Returns
    -------
    list of dict:
        {'sensor': str, 'onset_h': float, 'magnitude': float, 'alarm': bool}
    """
    events = []
    for sensor_id, p_series in pressures.items():
        onset_h, magnitude = detect_burst_onset(
            p_series, timestamps_h, wavelet=wavelet,
            threshold_sigma=threshold_sigma,
        )
        events.append({
            'sensor': sensor_id,
            'onset_h': onset_h,
            'magnitude': float(magnitude),
            'alarm': onset_h is not None,
        })

    # Sort by magnitude (strongest transient first)
    events.sort(key=lambda x: -x['magnitude'])
    return events
