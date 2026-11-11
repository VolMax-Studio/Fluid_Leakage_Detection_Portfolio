"""
leak_predictor.py — Pressure Baseline Predictor + Residual Alarm
=================================================================
Trains a Random Forest regressor on leak-free historical pressure data
to predict expected nodal pressures from operational features. A leak
manifests as a sustained negative residual (measured < predicted).

Feature engineering
-------------------
Features encode the dominant physical drivers of nodal pressure:
    hour_sin, hour_cos   — diurnal demand cycle (circular encoding)
    day_sin, day_cos     — weekly cycle
    inlet_flow           — total network flow (N1 sensor) [L/s]
    pressure_N1          — upstream pressure head [m]
    lag_1, lag_2         — autoregressive terms (inertia in demand)

Target: pressure at each sensor node [m H₂O].

Detection logic
---------------
Residual r(t) = P_measured(t) - P_predicted(t)

A leak causes r(t) < 0 persistently. Detection threshold:
    alarm if r(t) < -k * σ_baseline
where σ_baseline is residual std during training (no-leak) period,
and k is the detection multiplier (default k=2.5).

This is equivalent to a CUSUM (cumulative sum) control chart trigger
when the leak is sustained over multiple timesteps.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings


def _build_features(
    timestamps_h: np.ndarray,
    inlet_flow: np.ndarray,
    pressure_inlet: np.ndarray,
    lag: int = 2,
) -> np.ndarray:
    """
    Build feature matrix from operational time series.

    Parameters
    ----------
    timestamps_h : np.ndarray, shape (T,)
    inlet_flow   : np.ndarray, shape (T,)   — flow at N1 [L/s]
    pressure_inlet : np.ndarray, shape (T,) — pressure at N1 [m H₂O]
    lag : int — number of autoregressive lags for inlet flow

    Returns
    -------
    X : np.ndarray, shape (T - lag, n_features)
    """
    hour = timestamps_h % 24
    day  = (timestamps_h / 24) % 7

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin  = np.sin(2 * np.pi * day / 7)
    day_cos  = np.cos(2 * np.pi * day / 7)

    T = len(timestamps_h)
    start = lag  # first valid index (lags require previous samples)

    features = [
        hour_sin[start:],
        hour_cos[start:],
        day_sin[start:],
        day_cos[start:],
        inlet_flow[start:],
        pressure_inlet[start:],
    ]
    for l in range(1, lag + 1):
        features.append(inlet_flow[start - l : T - l])

    return np.column_stack(features)


class LeakPredictor:
    """
    Per-sensor Random Forest pressure predictor with residual alarm logic.

    Usage
    -----
    predictor = LeakPredictor()
    predictor.fit(train_data)                    # use leak-free period
    alarms = predictor.predict_and_alarm(data)   # full simulation
    """

    def __init__(
        self,
        n_estimators: int = 100,
        alarm_sigma_multiplier: float = 2.5,
        alarm_consecutive_steps: int = 4,
        lag: int = 2,
        random_state: int = 42,
        detector_type: str = "consecutive",
        cusum_drift: float = 0.2,
        cusum_threshold: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        n_estimators : int
            RF trees. 100 is sufficient for this feature dimensionality.
        alarm_sigma_multiplier : float
            Alarm threshold = -k * residual_std from training data (for consecutive detector).
        alarm_consecutive_steps : int
            Number of consecutive threshold exceedances required to trigger
            a formal alarm (reduces false positives from noise spikes).
        lag : int
            Number of lagged inlet-flow features.
        detector_type : str
            Alarm detection method. Choices: "consecutive" or "cusum".
        cusum_drift : float
            CUSUM drift/allowance parameter (k) in standard deviations.
        cusum_threshold : float, optional
            CUSUM alarm threshold (H) in standard deviations. If None, it is
            auto-tuned on the training (leak-free) period to prevent false alarms.
        """
        self.k = alarm_sigma_multiplier
        self.n_consec = alarm_consecutive_steps
        self.lag = lag
        self.detector_type = detector_type
        self.cusum_drift = cusum_drift
        self.cusum_threshold = cusum_threshold
        self.cusum_thresholds: Dict[str, float] = {}
        self.models: Dict[str, RandomForestRegressor] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.residual_std: Dict[str, float] = {}
        self.sensor_ids: List[str] = []
        self._is_fitted = False

    def fit(
        self,
        timestamps_h: np.ndarray,
        pressures: Dict[str, np.ndarray],
        flows: Dict[str, np.ndarray],
        training_fraction: float = 0.40,
    ) -> None:
        """
        Fit a per-sensor RF model using the first training_fraction of data
        (assumed to be leak-free by construction in the synthetic simulation;
        this does NOT hold for L-Town 2018 — see AUDIT_2026-11.md).

        Parameters
        ----------
        timestamps_h : np.ndarray
        pressures : dict {sensor_id: pressure_array}
        flows : dict {sensor_id: flow_array}
        training_fraction : float
            Fraction of data used for training. Default: 0.40 (40%).
        """
        T = len(timestamps_h)
        T_train = int(T * training_fraction)
        self.sensor_ids = list(pressures.keys())

        # Retrieve inlet flow dynamically (sum of flows if multiple, or N1/n1)
        inlet_flow = flows.get("N1", flows.get("n1", None))
        if inlet_flow is None:
            if len(flows) > 0:
                inlet_flow = sum(flows.values())
            else:
                inlet_flow = np.zeros(T)
                
        # Retrieve inlet pressure dynamically (N1/n1)
        pressure_N1 = pressures.get("N1", pressures.get("n1", np.zeros(T)))
        
        lag = self.lag

        X_train = _build_features(
            timestamps_h[:T_train],
            inlet_flow[:T_train],
            pressure_N1[:T_train],
            lag=lag,
        )

        for sensor_id in self.sensor_ids:
            y_train = pressures[sensor_id][lag:T_train]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)

            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_scaled, y_train)

            # Compute in-sample residual std (baseline noise floor)
            y_pred_train = model.predict(X_scaled)
            residuals_train = y_train - y_pred_train
            sigma = float(np.std(residuals_train))
            self.residual_std[sensor_id] = sigma

            # If CUSUM is selected, auto-tune the threshold (H) on the training set
            if self.detector_type == "cusum":
                # Standardize training residuals
                z_train = residuals_train / (sigma if sigma > 1e-6 else 1.0)
                S_train = 0.0
                max_S_train = 0.0
                for val in z_train:
                    # We subtract val because negative residuals (pressure drop) increase the sum
                    S_train = max(0.0, S_train - val - self.cusum_drift)
                    if S_train > max_S_train:
                        max_S_train = S_train
                
                if self.cusum_threshold is not None:
                    self.cusum_thresholds[sensor_id] = self.cusum_threshold
                else:
                    # Auto-tune: peak training value + safety margin of 5.0 standard deviations
                    self.cusum_thresholds[sensor_id] = max_S_train + 5.0

            self.models[sensor_id] = model
            self.scalers[sensor_id] = scaler

        self._is_fitted = True

    def predict_and_alarm(
        self,
        timestamps_h: np.ndarray,
        pressures: Dict[str, np.ndarray],
        flows: Dict[str, np.ndarray],
    ) -> Dict[str, dict]:
        """
        Generate pressure predictions and residual-based leak alarms.

        Returns
        -------
        dict {sensor_id: {
            'predicted'   : np.ndarray,
            'residuals'   : np.ndarray,
            'threshold'   : float,
            'alarm_mask'  : np.ndarray (bool),
            'first_alarm_h': float or None,
        }}
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_and_alarm()")

        T = len(timestamps_h)
        lag = self.lag
        
        # Retrieve inlet flow dynamically
        inlet_flow = flows.get("N1", flows.get("n1", None))
        if inlet_flow is None:
            if len(flows) > 0:
                inlet_flow = sum(flows.values())
            else:
                inlet_flow = np.zeros(T)
                
        # Retrieve inlet pressure dynamically
        pressure_N1 = pressures.get("N1", pressures.get("n1", np.zeros(T)))

        X_full = _build_features(timestamps_h, inlet_flow, pressure_N1, lag=lag)

        results = {}
        for sensor_id in self.sensor_ids:
            model  = self.models[sensor_id]
            scaler = self.scalers[sensor_id]
            sigma  = self.residual_std[sensor_id]

            X_scaled = scaler.transform(X_full)
            y_pred = model.predict(X_scaled)

            # Align with timestamps (lag offset)
            y_meas = pressures[sensor_id][lag:]
            residuals = y_meas - y_pred

            # Alarm: residual < -k*sigma for n_consec consecutive timesteps, or CUSUM
            neg_threshold = -self.k * sigma
            alarm_mask = np.zeros(len(residuals), dtype=bool)

            if self.detector_type == "cusum":
                # CUSUM sequential alarm logic
                z = residuals / (sigma if sigma > 1e-6 else 1.0)
                H = self.cusum_thresholds[sensor_id]
                S = 0.0
                for i in range(len(z)):
                    S = max(0.0, S - z[i] - self.cusum_drift)
                    if S > H:
                        alarm_mask[i] = True
                first_alarm_idx = np.argmax(alarm_mask) if alarm_mask.any() else None
            else:
                # Consecutive exceedance logic
                below = (residuals < neg_threshold).astype(int)
                consec = 0
                for i in range(len(below)):
                    if below[i]:
                        consec += 1
                        if consec >= self.n_consec:
                            alarm_mask[i] = True
                    else:
                        consec = 0
                first_alarm_idx = np.argmax(alarm_mask) if alarm_mask.any() else None

            first_alarm_h = (
                float(timestamps_h[lag + first_alarm_idx])
                if first_alarm_idx is not None else None
            )

            # Pad predictions back to original length with NaN for lag period
            y_pred_full = np.full(T, np.nan)
            y_pred_full[lag:] = y_pred
            residuals_full = np.full(T, np.nan)
            residuals_full[lag:] = residuals
            alarm_full = np.zeros(T, dtype=bool)
            alarm_full[lag:] = alarm_mask

            results[sensor_id] = {
                'predicted': y_pred_full,
                'residuals': residuals_full,
                'threshold': float(-H * sigma) if self.detector_type == "cusum" else float(neg_threshold),
                'alarm_mask': alarm_full,
                'first_alarm_h': first_alarm_h,
            }

        return results

    def detection_summary(
        self,
        alarm_results: Dict[str, dict],
        leak_events: List[dict],
        timestamps_h: np.ndarray,
    ) -> dict:
        """
        Compute event-level detection metrics.

        KNOWN LIMITATIONS (AUDIT_2026-11.md) — kept unchanged so the published
        v2.0.0 numbers stay reproducible:
          * only the FIRST alarm of each sensor over the whole record is scored;
            later alarms (and alarms that never clear) are ignored;
          * true positives count unique EVENTS while false positives count
            SENSORS, so precision mixes two units;
          * events inside the training window stay in the recall denominator.

        An event is detected if at least one sensor alarm falls within
        [event_start, event_start + 48h]. Each event is counted at most once
        (multiple sensor alarms for the same event = 1 TP, not N).

        Returns Precision, Recall, mean time-to-detection [hours].
        """
        true_starts = {e['node']: e['start_hour'] for e in leak_events}
        n_events = len(leak_events)

        # Build alarm list: (sensor_id, alarm_time_h)
        alarm_list = [
            (sid, res['first_alarm_h'])
            for sid, res in alarm_results.items()
            if res['first_alarm_h'] is not None
        ]
        n_alarms = len(alarm_list)

        # For each alarm: does it match a ground truth event?
        detected_events: dict = {}   # event_node → earliest alarm time
        false_positive_count = 0

        for sensor_id, alarm_h in alarm_list:
            matched_node = None
            for node, t_start in true_starts.items():
                if t_start <= alarm_h <= t_start + 48.0:
                    matched_node = node
                    break

            if matched_node:
                # Keep earliest alarm per event (TTD = time from start to first detection)
                if matched_node not in detected_events:
                    detected_events[matched_node] = alarm_h
                else:
                    detected_events[matched_node] = min(detected_events[matched_node], alarm_h)
            else:
                false_positive_count += 1

        true_positives = len(detected_events)
        detection_delays = [
            detected_events[node] - true_starts[node]
            for node in detected_events
        ]

        precision = true_positives / max(true_positives + false_positive_count, 1)
        recall    = true_positives / max(n_events, 1)
        mean_ttd  = float(np.mean(detection_delays)) if detection_delays else np.nan

        return {
            'precision': precision,
            'recall': recall,
            'f1': 2 * precision * recall / max(precision + recall, 1e-9),
            'mean_ttd_hours': mean_ttd,
            'n_alarms': n_alarms,
            'true_positives': true_positives,
            'false_positives': false_positive_count,
            'n_ground_truth_events': n_events,
        }
