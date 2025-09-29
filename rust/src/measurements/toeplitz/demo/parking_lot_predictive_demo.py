#!/usr/bin/env python3
"""
Parking Lot Predictive Demo for the Toeplitz Mechanism

This demo measures the utility of a single-hour regression model that consumes continual
DP prefix sums from the Toeplitz mechanism. For a chosen hour-of-day we:

- release one noisy prefix stream per epsilon across the entire history once
- derive multi-scale rolling features (optionally lagged and horizon-shifted) by
  differencing those DP prefixes
- train on the early timeline and score the remainder, collecting regression error metrics
  on the total number of available parking spots alongside naïve baselines
- optionally, plot the learning trajectory to `demo/imgs/`

Notes:
- This script calls the Rust Toeplitz mechanism via the Toeplitz Python API through FFI.
- Sensitivity is 1 per appended label (event-level adjacency).

Run example:
  python3 measurements/toeplitz/demo/parking_lot_predictive_demo.py --hour 16 --epsilons \
    0.1 0.2 0.5 --window-grid 12 18 --lags 1 2 --sample-step 5 --max-samples 200 --trials 5 \
    --dow-mod7 --minute-cyc --plot
"""

from __future__ import annotations

import argparse
import csv
import importlib
import ctypes
import math
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple


# ================================================================
# Set up and parse OpenDP Toeplitz supports input Python callables
# ================================================================
def _ensure_local_opendp_on_path() -> None:
    """Currently assumes this demo lives in the same directory as the Toeplitz in OpenDP."""
    for parent in Path(__file__).resolve().parents:
        python_src = parent / "python" / "src"
        if (python_src / "opendp").exists():
            python_src_str = str(python_src)
            if python_src_str not in sys.path:
                sys.path.insert(0, python_src_str)
            break


def _import_opendp_module(module_name: str):
    """Fallback to locally importing OpenDP if not already on sys.path."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        _ensure_local_opendp_on_path()
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Unable to import '{module_name}'. Ensure OpenDP is installed or build the repo (pip install -e python)."
            ) from exc


dp: Any = _import_opendp_module("opendp.prelude")


EXPERIMENTAL_CONFIG = {
    # Set default values unless otherwise overridden via CLI
    "epsilons": [0.2, 0.5, 1.0],  # CLI override flag (--epsilons)
    "window_days": 14,  # --window-days
    "window_days_grid": [14],  # --window-grid / --window-days
    "trials": 10,  # --trials
    "monotonic": False,  # --monotonic
    "seed": 0,  # --seed
    "train_ratio": 0.5,  # --train-ratio
    "l2": 1e-3,  # --l2
    "horizon": 0,  # --horizon
    "lag_steps": (1, 2),  # --lags
    "dow_mod7": False,  # --dow-mod7
    "dow_offset": 0,  # --dow-offset
    "minute_cyc": False,  # --minute-cyc
}


_toeplitz_ffi_cache: Optional[Tuple[Any, ...]] = None


def _load_toeplitz_ffi() -> Tuple[Any, ...]:
    global _toeplitz_ffi_cache
    if _toeplitz_ffi_cache is None:
        lib_mod = _import_opendp_module("opendp._lib")
        convert_mod = _import_opendp_module("opendp._convert")
        data_mod = _import_opendp_module("opendp._data")

        unwrap = lib_mod.unwrap
        any_ptr = lib_mod.AnyObjectPtr

        toeplitz_instantiate = lib_mod.lib.opendp_measurements__toeplitz_continual_new_i64
        toeplitz_instantiate.argtypes = [ctypes.c_double, ctypes.c_uint8]
        toeplitz_instantiate.restype = lib_mod.FfiResult

        toeplitz_append_count_on_new_timestamp = lib_mod.lib.opendp_measurements__toeplitz_append_count_on_new_timestamp_i64
        toeplitz_append_count_on_new_timestamp.argtypes = [any_ptr, ctypes.c_longlong]
        toeplitz_append_count_on_new_timestamp.restype = lib_mod.FfiResult

        toeplitz_fetch_privacy_preserving_sub_interval_sum = lib_mod.lib.opendp_measurements__toeplitz_fetch_privacy_preserving_sub_interval_sum_i64
        toeplitz_fetch_privacy_preserving_sub_interval_sum.argtypes = [any_ptr, ctypes.c_size_t, ctypes.c_size_t]
        toeplitz_fetch_privacy_preserving_sub_interval_sum.restype = lib_mod.FfiResult

        _toeplitz_ffi_cache = (
            unwrap,
            any_ptr,
            data_mod.object_free,
            convert_mod.c_to_py,
            toeplitz_instantiate,
            toeplitz_append_count_on_new_timestamp,
            toeplitz_fetch_privacy_preserving_sub_interval_sum,
        )
    return _toeplitz_ffi_cache


class ContinualToeplitzStream:
    def __init__(self, *, scale: float, monotonic: bool) -> None:
        (
            self._unwrap,
            self._any_ptr,
            self._object_free,
            self._c_to_py,
            toeplitz_instantiate,
            toeplitz_append_count_on_new_timestamp,
            toeplitz_fetch_privacy_preserving_sub_interval_sum,
        ) = _load_toeplitz_ffi()
        self._toeplitz_append_count_on_new_timestamp = toeplitz_append_count_on_new_timestamp
        self._toeplitz_fetch_privacy_preserving_sub_interval_sum = toeplitz_fetch_privacy_preserving_sub_interval_sum
        self._handle = None
        self._handle = self._unwrap(
            toeplitz_instantiate(ctypes.c_double(scale), ctypes.c_uint8(1 if monotonic else 0)),
            self._any_ptr,
        )

    def append_count_on_new_timestamp(self, value: int) -> None:
        res = self._unwrap(
            self._toeplitz_append_count_on_new_timestamp(self._handle, ctypes.c_longlong(int(value))),
            self._any_ptr,
        )
        del res

    def fetch_privacy_preserving_sub_interval_sum(self, start_time: int, end_time: int) -> float:
        res = self._unwrap(
            self._toeplitz_fetch_privacy_preserving_sub_interval_sum(
                self._handle,
                ctypes.c_size_t(start_time),
                ctypes.c_size_t(end_time),
            ),
            self._any_ptr,
        )
        value = self._c_to_py(res)
        del res
        return float(value)

    def close_toeplitz_stream(self) -> None:
        if self._handle is not None:
            handle = self._handle
            self._handle = None
            del handle

    def __del__(self) -> None:  # pragma: no cover - cleanup
        self.close_toeplitz_stream()



# ===============================================================
# Predictive pipeline -- loading data, measuring utility, results
# ===============================================================

# Specs:
# Parking lot availability:
#   - we derive the number of available spots (status==0) for the last (within the minute) reading
#     observed for each spot.
#   - Additional temporal features (cyclical day-of-week/minute-of-hour) are derived only from index
#     arithmetic, so no extra data loading is required.


Timestamp = Tuple[int, int, int, int, int]  # (Year, Month, Date, Hour, Minute)


def load_hourly_available_counts(
    dataset_dir: Path,
    spot_filter: Optional[Iterable[int]] = None,
) -> Tuple[List[Tuple[Timestamp, int]], int]:
    """Load rows and derive available spot counts per minute (by the last recorded)."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    allowed_spots: Optional[Set[int]] = set(map(int, spot_filter)) if spot_filter else None

    # Map (Y, M, D, H, M) -> {SpotID: (second, status)} to keep last reading per spot
    last_by_slot: Dict[Tuple[int, int, int, int, int], Dict[int, Tuple[int, int]]] = {}
    observed_spots: Set[int] = set()

    csv_files: List[Path] = []
    subdirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    if subdirs:
        for subdir in subdirs:
            csv_files.extend(sorted(subdir.glob("*.csv")))
    else:
        csv_files = sorted(dataset_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    print(f"Loading {len(csv_files)} files for aggregate counts...", flush=True)
    for file_idx, path in enumerate(csv_files, start=1):
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    sid = int(row["SpotID"])     # Identifies a particular parking spot
                    if allowed_spots is not None and sid not in allowed_spots:
                        continue
                    year = int(row["Year"])      # e.g., 2025
                    month = int(row["Month"])    # 1..12
                    day = int(row["Date"])       # 1..31
                    hour = int(row["Hour"])      # 0..23
                    minute = int(row["Minute"])  # 0..59
                    second = int(row["Second"])  # 0..59
                    status = int(row["Status"])  # 0/1
                except Exception:
                    continue

                key = (
                    year,
                    month,
                    day,
                    hour,
                    minute,
                )
                slot_map = last_by_slot.setdefault(key, {})
                prev = slot_map.get(sid)
                if prev is None or second >= prev[0]:
                    slot_map[sid] = (second, status)
                    observed_spots.add(sid)

        if file_idx % max(1, len(csv_files) // 5) == 0 or file_idx == len(csv_files):
            print(f"  Processed {file_idx}/{len(csv_files)} files", flush=True)

    # Build chronological sequence
    keys_sorted = sorted(last_by_slot.keys())
    result: List[Tuple[Timestamp, int]] = []
    for key in keys_sorted:
        slot_map = last_by_slot[key]
        available = sum(1 for (_, status) in slot_map.values() if status == 0)
        result.append((key, available))

    return result, len(observed_spots)



# -------------------------------
# Training utilities
# -------------------------------

@dataclass
class WindowMetrics:
    rmse: float
    mae: float
    bias: float
    r2: float
    baseline_mean_rmse: float
    baseline_mean_mae: float
    baseline_mean_r2: float
    baseline_locf_rmse: float
    baseline_locf_mae: float
    baseline_locf_r2: float
    baseline_dp_rmse: float
    baseline_dp_mae: float
    baseline_dp_r2: float
    train_count: int
    test_count: int


@dataclass
class StreamingSeries:
    avg_prediction: List[float]
    mean_absolute_error: List[float]
    cumulative_rmse: List[float]
    cumulative_mae: List[float]
    cumulative_bias: List[float]


@dataclass
class WindowResults:
    summary: WindowMetrics
    timeline: StreamingSeries

def predict_linear_value(weights: List[float], features: List[float]) -> float:
    return weights[0] + sum(w * x for w, x in zip(weights[1:], features))


def mean_squared_error(predictions: List[float], targets: List[float]) -> float:
    if not targets:
        return 0.0
    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(targets)


def mean_absolute_error(predictions: List[float], targets: List[float]) -> float:
    if not targets:
        return 0.0
    return sum(abs(p - t) for p, t in zip(predictions, targets)) / len(targets)


def r2_score(predictions: List[float], targets: List[float]) -> float:
    if not targets:
        return float("nan")
    mean_target = sum(targets) / len(targets)
    ss_res = sum((p - t) ** 2 for p, t in zip(predictions, targets))
    ss_tot = sum((t - mean_target) ** 2 for t in targets)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rmse(predictions: List[float], targets: List[float]) -> float:
    return math.sqrt(mean_squared_error(predictions, targets))


def _gaussian_elimination(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(a)
    # Forward elimination with partial pivoting
    for i in range(n):
        # Pivot
        pivot_row = max(range(i, n), key=lambda r: abs(a[r][i]))
        if abs(a[pivot_row][i]) < 1e-12:
            raise ValueError("Singular matrix in ridge solver")
        if pivot_row != i:
            a[i], a[pivot_row] = a[pivot_row], a[i]
            b[i], b[pivot_row] = b[pivot_row], b[i]
        pivot = a[i][i]
        # Normalize row
        factor = pivot
        for j in range(i, n):
            a[i][j] /= factor
        b[i] /= factor
        # Eliminate
        for r in range(n):
            if r == i:
                continue
            coeff = a[r][i]
            if coeff == 0.0:
                continue
            for c in range(i, n):
                a[r][c] -= coeff * a[i][c]
            b[r] -= coeff * b[i]
    return b[:]


def train_ridge_regression(
    features: List[List[float]],
    targets: List[float],
    l2: float,
) -> List[float]:
    """Classical least-squares regression with L2 regularization."""
    if not features:
        raise ValueError("Cannot train regression model without features.")
    p = len(features[0]) + 1  # include intercept
    XtX = [[0.0] * p for _ in range(p)]
    XtY = [0.0] * p
    for row, target in zip(features, targets):
        extended = [1.0] + row
        for i in range(p):
            XtY[i] += extended[i] * target
            for j in range(p):
                XtX[i][j] += extended[i] * extended[j]
    for j in range(1, p):
        XtX[j][j] += l2
    XtX_copy = [row[:] for row in XtX]
    XtY_copy = XtY[:]
    weights = _gaussian_elimination(XtX_copy, XtY_copy)
    return weights


def extract_hourly_counts(entries: List[Tuple[Timestamp, int]], hour: int) -> List[int]:
    """Filter aggregate counts for the requested hour across the time series."""
    return [count for ((_, _, _, h, _), count) in entries if h == hour]


def evaluate_single_hour_regression(
    counts_by_step: List[int],
    max_available: int,
    config: Mapping[str, Any],
) -> Tuple[List[int], Dict[float, Dict[int, WindowResults]]]:
    """Evaluate epsilon-configured Toeplitz streams with regression models.

    For each epsilon we obtain a noisy prefix stream and derive rolling window features
    (DP-perturbed window averages). A simple linear regression model is trained to map those
    features to the total number of available parking spots, using an early portion of the
    timeline for training and the remainder for evaluation. Metrics are averaged across
    Monte Carlo trials.
    """
    results: Dict[float, Dict[int, WindowResults]] = {}
    epsilons = config["epsilons"]
    window_grid = sorted(set(config.get("window_days_grid", [config["window_days"]])))
    trials = config["trials"]
    monotonic = config["monotonic"]
    train_ratio = config.get("train_ratio", 0.7)
    l2 = config.get("l2", 1e-3)
    horizon = max(0, int(config.get("horizon", 0)))
    lag_steps_cfg = config.get("lag_steps", (1, 2))
    if isinstance(lag_steps_cfg, int):
        lag_steps = [lag_steps_cfg]
    else:
        lag_steps = [int(x) for x in lag_steps_cfg]
    lag_steps = sorted({lag for lag in lag_steps if lag > 0})
    dow_mod7 = bool(config.get("dow_mod7", False))
    dow_offset = int(config.get("dow_offset", 0)) % 7
    minute_cyc = bool(config.get("minute_cyc", False))
    sample_step = max(1, int(config.get("sample_step", 1)))
    max_samples = config.get("max_samples")

    if (dow_mod7 or minute_cyc) and 60 % sample_step != 0:
        print(
            "Warning: sample_step does not divide 60; disabling index-derived DOW/Minute features.",
            flush=True,
        )
        dow_mod7 = False
        minute_cyc = False

    if sample_step > 1:
        counts_by_step = counts_by_step[::sample_step]

    if isinstance(max_samples, int) and max_samples > 0 and len(counts_by_step) > max_samples:
        counts_by_step = counts_by_step[-max_samples:]

    n = len(counts_by_step)

    if n == 0 or not window_grid:
        return counts_by_step, results

    def _mean(values: List[float]) -> float:
        filtered = [v for v in values if not math.isnan(v)]
        if not filtered:
            return float("nan")
        return sum(filtered) / len(filtered)
    print(
        f"Prepared {n} samples (step={sample_step}, max={max_samples or '∞'})",
        flush=True,
    )

    points_per_day = max(1, int(round(60.0 / sample_step))) if sample_step > 0 else 1
    count_feature_len = 1 + len(lag_steps)

    for eps_idx, eps in enumerate(epsilons, start=1):
        if eps <= 0:
            continue

        rmse_vals: Dict[int, List[float]] = {window: [] for window in window_grid}
        mae_vals: Dict[int, List[float]] = {window: [] for window in window_grid}
        bias_vals: Dict[int, List[float]] = {window: [] for window in window_grid}
        r2_vals: Dict[int, List[float]] = {window: [] for window in window_grid}
        baseline_mean_rmse_vals: Dict[int, List[float]] = {window: [] for window in window_grid}
        baseline_mean_mae_vals: Dict[int, List[float]] = {window: [] for window in window_grid}
        baseline_mean_r2_vals: Dict[int, List[float]] = {window: [] for window in window_grid}
        baseline_locf_rmse_vals: Dict[int, List[float]] = {window: [] for window in window_grid}
        baseline_locf_mae_vals: Dict[int, List[float]] = {window: [] for window in window_grid}
        baseline_locf_r2_vals: Dict[int, List[float]] = {window: [] for window in window_grid}
        baseline_dp_rmse_vals: Dict[int, List[float]] = {window: [] for window in window_grid}
        baseline_dp_mae_vals: Dict[int, List[float]] = {window: [] for window in window_grid}
        baseline_dp_r2_vals: Dict[int, List[float]] = {window: [] for window in window_grid}
        train_counts: Dict[int, Optional[int]] = {window: None for window in window_grid}
        test_counts: Dict[int, Optional[int]] = {window: None for window in window_grid}

        per_step_pred_sums: Dict[int, List[float]] = {window: [0.0] * n for window in window_grid}
        per_step_abs_error_sums: Dict[int, List[float]] = {window: [0.0] * n for window in window_grid}
        per_step_sq_error_sums: Dict[int, List[float]] = {window: [0.0] * n for window in window_grid}
        per_step_bias_sums: Dict[int, List[float]] = {window: [0.0] * n for window in window_grid}
        per_step_counts: Dict[int, List[int]] = {window: [0] * n for window in window_grid}

        print(f"Evaluating epsilon {eps} ({eps_idx}/{len(epsilons)})", flush=True)
        for trial in range(trials):
            if trial % max(1, trials // 5) == 0 or trial == trials - 1:
                print(f"  Trial {trial + 1}/{trials}", flush=True)
            print("    Building Toeplitz continual stream...", flush=True)
            stream = ContinualToeplitzStream(scale=1.0 / eps, monotonic=monotonic)
            start_time = time.perf_counter()
            for value in counts_by_step:
                stream.append_count_on_new_timestamp(value)
            elapsed_stream = time.perf_counter() - start_time
            print(
                f"    Stream ready (steps={n}) in {elapsed_stream:.2f}s",
                flush=True,
            )

            try:
                dp_prefix: List[float] = [0.0] * n
                if n > 1:
                    for prefix_idx in range(1, n):
                        if prefix_idx % max(1, n // 5) == 0 or prefix_idx == n - 1:
                            print(
                                f"    Prefetching DP prefix {prefix_idx}/{n - 1}",
                                flush=True,
                            )
                        dp_prefix[prefix_idx] = stream.fetch_privacy_preserving_sub_interval_sum(1, prefix_idx)

                dp_means_by_window: Dict[int, List[Optional[float]]] = {}
                for win_idx, window in enumerate(window_grid, start=1):
                    if win_idx == 1:
                        print(f"    Window {window} (1/{len(window_grid)})", flush=True)
                    elif win_idx % max(1, len(window_grid) // 3) == 0 or win_idx == len(window_grid):
                        print(f"    Window {window} ({win_idx}/{len(window_grid)})", flush=True)
                    dp_means: List[Optional[float]] = [None] * n
                    start_t = window + horizon
                    if start_t >= n:
                        dp_means_by_window[window] = dp_means
                        continue
                    for t in range(start_t, n):
                        end_time = t - horizon
                        if t % max(1, n // 5) == 0 or t == n - 1:
                            print(
                                f"      Window {window}: processed {t}/{n} timesteps",
                                flush=True,
                            )
                        start_idx = end_time - window + 1
                        if start_idx < 1:
                            continue
                        window_sum = dp_prefix[end_time] - dp_prefix[start_idx - 1]
                        dp_mean = window_sum / float(window)
                        dp_means[t] = max(0.0, min(float(max_available), dp_mean))
                    dp_means_by_window[window] = dp_means

                feature_scale = float(max_available)
                target_scale = float(max_available)

                for window in window_grid:
                    dp_means = dp_means_by_window[window]
                    feature_rows_raw: List[List[float]] = []
                    target_rows_raw: List[float] = []
                    indices: List[int] = []
                    for idx in range(n):
                        value_opt = dp_means[idx]
                        if value_opt is None:
                            continue
                        features = [float(value_opt)]
                        feasible = True
                        for lag in lag_steps:
                            lag_idx = idx - lag
                            if lag_idx < 0:
                                feasible = False
                                break
                            lag_value_opt = dp_means[lag_idx]
                            if lag_value_opt is None:
                                feasible = False
                                break
                            features.append(float(lag_value_opt))
                        if not feasible:
                            continue
                        if dow_mod7:
                            day_idx = idx // points_per_day
                            dow = (dow_offset + day_idx) % 7
                            features.extend(
                                [
                                    math.sin(2.0 * math.pi * dow / 7.0),
                                    math.cos(2.0 * math.pi * dow / 7.0),
                                    1.0 if dow in (5, 6) else 0.0,
                                ]
                            )
                        if minute_cyc:
                            minute_position = (idx * sample_step) % 60.0
                            features.extend(
                                [
                                    math.sin(2.0 * math.pi * minute_position / 60.0),
                                    math.cos(2.0 * math.pi * minute_position / 60.0),
                                ]
                            )
                        feature_rows_raw.append(features)
                        target_rows_raw.append(float(counts_by_step[idx]))
                        indices.append(idx)

                    if len(feature_rows_raw) < 2:
                        continue

                    total = len(feature_rows_raw)
                    split = max(1, min(total - 1, int(total * train_ratio)))
                    train_features_raw = feature_rows_raw[:split]
                    test_features_raw = feature_rows_raw[split:]
                    if not test_features_raw:
                        continue
                    train_targets_raw = target_rows_raw[:split]
                    test_targets_raw = target_rows_raw[split:]
                    test_indices = indices[split:]

                    train_counts[window] = (
                        split if train_counts[window] is None else train_counts[window]
                    )
                    test_counts[window] = (
                        len(test_targets_raw)
                        if test_counts[window] is None
                        else test_counts[window]
                    )

                    def _normalize_rows(rows: List[List[float]]) -> List[List[float]]:
                        normalized: List[List[float]] = []
                        for row in rows:
                            norm_row = []
                            for idx_feat, value in enumerate(row):
                                norm_value = value / feature_scale if idx_feat < count_feature_len else value
                                norm_row.append(norm_value)
                            normalized.append(norm_row)
                        return normalized

                    train_features = _normalize_rows(train_features_raw)
                    test_features = _normalize_rows(test_features_raw)
                    train_targets = [value / target_scale for value in train_targets_raw]

                    weights = train_ridge_regression(
                        train_features,
                        train_targets,
                        l2=l2,
                    )

                    preds: List[float] = []
                    for feats in test_features:
                        raw_pred = predict_linear_value(weights, feats) * target_scale
                        preds.append(max(0.0, min(float(max_available), raw_pred)))

                    rmse_val = rmse(preds, test_targets_raw)
                    mae_val = mean_absolute_error(preds, test_targets_raw)
                    bias_val = (
                        sum(p - t for p, t in zip(preds, test_targets_raw)) / len(preds)
                        if preds
                        else 0.0
                    )
                    r2_val = r2_score(preds, test_targets_raw)

                    baseline_mean = sum(train_targets_raw) / len(train_targets_raw)
                    baseline_mean_preds = [baseline_mean] * len(test_targets_raw)
                    baseline_mean_rmse = rmse(baseline_mean_preds, test_targets_raw)
                    baseline_mean_mae = mean_absolute_error(baseline_mean_preds, test_targets_raw)
                    baseline_mean_r2 = r2_score(baseline_mean_preds, test_targets_raw)

                    baseline_locf_preds: List[float] = []
                    for idx_global in test_indices:
                        prev_idx = idx_global - max(1, horizon)
                        if prev_idx < 0:
                            baseline_locf_preds.append(baseline_mean)
                        else:
                            baseline_locf_preds.append(float(counts_by_step[prev_idx]))
                    baseline_locf_rmse = rmse(baseline_locf_preds, test_targets_raw)
                    baseline_locf_mae = mean_absolute_error(baseline_locf_preds, test_targets_raw)
                    baseline_locf_r2 = r2_score(baseline_locf_preds, test_targets_raw)

                    baseline_dp_preds: List[float] = []
                    for idx_global in test_indices:
                        value_opt = dp_means[idx_global]
                        if value_opt is None:
                            baseline_dp_preds = baseline_mean_preds[:]
                            break
                        baseline_dp_preds.append(float(value_opt))
                    baseline_dp_rmse = rmse(baseline_dp_preds, test_targets_raw)
                    baseline_dp_mae = mean_absolute_error(baseline_dp_preds, test_targets_raw)
                    baseline_dp_r2 = r2_score(baseline_dp_preds, test_targets_raw)

                    rmse_vals[window].append(rmse_val)
                    mae_vals[window].append(mae_val)
                    bias_vals[window].append(bias_val)
                    r2_vals[window].append(r2_val)
                    baseline_mean_rmse_vals[window].append(baseline_mean_rmse)
                    baseline_mean_mae_vals[window].append(baseline_mean_mae)
                    baseline_mean_r2_vals[window].append(baseline_mean_r2)
                    baseline_locf_rmse_vals[window].append(baseline_locf_rmse)
                    baseline_locf_mae_vals[window].append(baseline_locf_mae)
                    baseline_locf_r2_vals[window].append(baseline_locf_r2)
                    baseline_dp_rmse_vals[window].append(baseline_dp_rmse)
                    baseline_dp_mae_vals[window].append(baseline_dp_mae)
                    baseline_dp_r2_vals[window].append(baseline_dp_r2)

                    for idx_global, pred_value, target_value in zip(
                        test_indices, preds, test_targets_raw
                    ):
                        error = pred_value - target_value
                        per_step_pred_sums[window][idx_global] += pred_value
                        per_step_abs_error_sums[window][idx_global] += abs(error)
                        per_step_sq_error_sums[window][idx_global] += error ** 2
                        per_step_bias_sums[window][idx_global] += error
                        per_step_counts[window][idx_global] += 1
            finally:
                stream.close_toeplitz_stream()

        per_eps_metrics: Dict[int, WindowResults] = {}
        for window in window_grid:
            trial_count = len(rmse_vals[window])
            if trial_count == 0:
                continue

            summary = WindowMetrics(
                rmse=_mean(rmse_vals[window]),
                mae=_mean(mae_vals[window]),
                bias=_mean(bias_vals[window]),
                r2=_mean(r2_vals[window]),
                baseline_mean_rmse=_mean(baseline_mean_rmse_vals[window]),
                baseline_mean_mae=_mean(baseline_mean_mae_vals[window]),
                baseline_mean_r2=_mean(baseline_mean_r2_vals[window]),
                baseline_locf_rmse=_mean(baseline_locf_rmse_vals[window]),
                baseline_locf_mae=_mean(baseline_locf_mae_vals[window]),
                baseline_locf_r2=_mean(baseline_locf_r2_vals[window]),
                baseline_dp_rmse=_mean(baseline_dp_rmse_vals[window]),
                baseline_dp_mae=_mean(baseline_dp_mae_vals[window]),
                baseline_dp_r2=_mean(baseline_dp_r2_vals[window]),
                train_count=train_counts[window] or 0,
                test_count=test_counts[window] or 0,
            )

            avg_prediction_ts: List[float] = []
            mean_abs_error_ts: List[float] = []
            cumulative_rmse_ts: List[float] = []
            cumulative_mae_ts: List[float] = []
            cumulative_bias_ts: List[float] = []

            cum_sq_error = 0.0
            cum_abs_error = 0.0
            cum_bias = 0.0
            cum_total = 0

            for idx in range(n):
                count = per_step_counts[window][idx]
                if count > 0:
                    avg_prediction_ts.append(per_step_pred_sums[window][idx] / count)
                    mean_abs_error_ts.append(per_step_abs_error_sums[window][idx] / count)
                    cum_sq_error += per_step_sq_error_sums[window][idx]
                    cum_abs_error += per_step_abs_error_sums[window][idx]
                    cum_bias += per_step_bias_sums[window][idx]
                    cum_total += count
                    cumulative_rmse_ts.append(
                        math.sqrt(cum_sq_error / cum_total) if cum_total > 0 else float("nan")
                    )
                    cumulative_mae_ts.append(
                        cum_abs_error / cum_total if cum_total > 0 else float("nan")
                    )
                    cumulative_bias_ts.append(
                        cum_bias / cum_total if cum_total > 0 else float("nan")
                    )
                else:
                    avg_prediction_ts.append(float("nan"))
                    mean_abs_error_ts.append(float("nan"))
                    cumulative_rmse_ts.append(
                        math.sqrt(cum_sq_error / cum_total) if cum_total > 0 else float("nan")
                    )
                    cumulative_mae_ts.append(
                        cum_abs_error / cum_total if cum_total > 0 else float("nan")
                    )
                    cumulative_bias_ts.append(
                        cum_bias / cum_total if cum_total > 0 else float("nan")
                    )

            timeline = StreamingSeries(
                avg_prediction=avg_prediction_ts,
                mean_absolute_error=mean_abs_error_ts,
                cumulative_rmse=cumulative_rmse_ts,
                cumulative_mae=cumulative_mae_ts,
                cumulative_bias=cumulative_bias_ts,
            )

            per_eps_metrics[window] = WindowResults(summary=summary, timeline=timeline)

        if per_eps_metrics:
            results[eps] = per_eps_metrics

    return counts_by_step, results


# ======================
# Single-hour end-to-end
# ======================

def run_single_hour_evaluation(
    dataset_dir: Path,
    hour: int,
    config: Mapping[str, Any],
    spot_ids: Optional[Iterable[int]] = None,
) -> Tuple[List[int], Dict[float, Dict[int, WindowResults]], int]:
    """Load data, extract a single hour, and run the predictive evaluation."""
    spot_list = sorted({int(s) for s in spot_ids}) if spot_ids is not None else None
    entries, max_spots = load_hourly_available_counts(dataset_dir, spot_list)
    subset_hint = "" if not spot_list else f" for spots {spot_list}"

    if not entries or max_spots == 0:
        raise ValueError(f"No entries found{subset_hint} in {dataset_dir}")

    counts_single_hour = extract_hourly_counts(entries, hour)
    if not counts_single_hour:
        raise ValueError(f"No samples for hour={hour}{subset_hint}.")

    print(
        f"Extracted {len(counts_single_hour)} raw samples for hour {hour}{subset_hint}",
        flush=True,
    )
    processed_counts, results = evaluate_single_hour_regression(
        counts_by_step=counts_single_hour,
        max_available=max_spots,
        config=config,
    )
    return processed_counts, results, max_spots


# =======
# Results
# =======


def print_single_hour_summary(
    hour: int,
    variant: str,
    subset_label: str,
    max_available: int,
    results: Dict[float, Dict[int, WindowResults]],
) -> None:
    print(
        f"\nSingle-hour rolling regressor ({variant}), hour={hour}, subset={subset_label}, max_spots={max_available}"
    )
    if not results:
        print("No results (insufficient samples for the selected hour).")
        return
    print(
        "epsilon | window | scale | train | test |    rmse |   mae |  bias |   r2 | mean_rmse | locf_rmse |  dp_rmse"
    )
    for eps in sorted(results.keys()):
        metrics_by_window = results[eps]
        if not metrics_by_window:
            continue
        scale = 1.0 / eps if eps > 0 else float("inf")
        for window in sorted(metrics_by_window.keys()):
            summary = metrics_by_window[window].summary
            print(
                f"{eps:7.3f} | {window:6d} | {scale:5.2f} | {summary.train_count:5d} | {summary.test_count:4d} | "
                f"{summary.rmse:8.3f} | {summary.mae:6.3f} | {summary.bias:6.3f} | {summary.r2:5.3f} | "
                f"{summary.baseline_mean_rmse:8.3f} | {summary.baseline_locf_rmse:9.3f} | {summary.baseline_dp_rmse:8.3f}"
            )


def plot_single_hour_timelines(
    hour: int,
    variant: str,
    counts_by_step: List[int],
    results: Dict[float, Dict[int, WindowResults]],
    output_dir: Path,
    subset_label: str,
    max_available: int,
) -> None:
    if not counts_by_step or not results:
        return
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError as exc:  # pragma: no cover - optional dependency
        if not getattr(plot_single_hour_timelines, "_warned", False):
            print(f"Matplotlib not available; skipping plots ({exc}).")
            plot_single_hour_timelines._warned = True  # type: ignore[attr-defined]
        return

    subset_slug = subset_label.lower().replace(" ", "_").replace(",", "")
    output_dir = output_dir / subset_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    time_axis = list(range(len(counts_by_step)))
    true_series = counts_by_step

    for eps in sorted(results.keys()):
        window_results = results[eps]
        if not window_results:
            continue

        windows = sorted(window_results.keys())
        cmap = cm.get_cmap("tab10", len(windows))
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axes[0].step(time_axis, true_series, where="post", color="black", alpha=0.3, label="True count")
        for idx, window in enumerate(windows):
            color = cmap(idx)
            timeline = window_results[window].timeline
            axes[0].plot(time_axis, timeline.avg_prediction, color=color, label=f"win={window}")
        axes[0].set_ylabel("Available spots")
        axes[0].set_ylim(-0.05 * max_available, max_available + 1)
        axes[0].set_title(f"Hour {hour} ε={eps:.3f} ({variant}, {subset_label})")
        axes[0].legend(loc="upper right", fontsize=8)

        for idx, window in enumerate(windows):
            color = cmap(idx)
            timeline = window_results[window].timeline
            axes[1].plot(time_axis, timeline.mean_absolute_error, color=color, label=f"win={window}")
        axes[1].set_ylabel("Mean abs error")
        axes[1].set_ylim(bottom=0)

        for idx, window in enumerate(windows):
            color = cmap(idx)
            timeline = window_results[window].timeline
            axes[2].plot(time_axis, timeline.cumulative_rmse, color=color, label=f"rmse win={window}")
            axes[2].plot(
                time_axis,
                timeline.cumulative_mae,
                color=color,
                linestyle=":",
                label=f"mae win={window}"
            )
            axes[2].plot(
                time_axis,
                timeline.cumulative_bias,
                color=color,
                linestyle="--",
                label=f"bias win={window}"
            )
        axes[2].set_ylabel("Cum RMSE/MAE/Bias")
        axes[2].set_xlabel("Timestep")
        axes[2].legend(loc="upper right", fontsize=8)

        for ax in axes:
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

        fig.tight_layout()
        variant_slug = variant.lower().replace(" ", "_")
        eps_slug = f"{eps:.3f}".replace("-", "m").replace(".", "p")
        filename = f"hour{hour}_eps{eps_slug}_{subset_slug}_{variant_slug}.png"
        fig.savefig(str(output_dir / filename), dpi=200)
        plt.close(fig)



# ===================
# CLI & main function
# ===================

def main() -> None:
    parser = argparse.ArgumentParser(description="Toeplitz DP predictive demo on parking lot datasets")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).parent / "datasets",
        help="Path to datasets directory containing CSV files",
    )
    parser.add_argument(
        "--spot-ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional SpotIDs to include when aggregating counts (default: all spots)",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=list(EXPERIMENTAL_CONFIG["epsilons"]),
        help="List of epsilon values to evaluate (sensitivity=1)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=EXPERIMENTAL_CONFIG["trials"],
        help="Monte Carlo trials per epsilon",
    )
    parser.add_argument(
        "--monotonic",
        action="store_true",
        default=EXPERIMENTAL_CONFIG["monotonic"],
        help="Use monotonic post-processing (isotonic regression)",
    )
    parser.add_argument("--hour", type=int, default=7, help="Hour-of-day (0-23) to evaluate")
    parser.add_argument(
        "--window-days",
        type=int,
        default=EXPERIMENTAL_CONFIG["window_days"],
        help="Rolling window size in days/occurrences for training",
    )
    parser.add_argument(
        "--window-grid",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of window sizes (days/occurrences) to evaluate",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=EXPERIMENTAL_CONFIG["train_ratio"],
        help="Fraction of chronological samples used for training the regression model",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=EXPERIMENTAL_CONFIG["l2"],
        help="L2 regularization strength for regression weights",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=0,
        help="Forecast horizon in samples (0=nowcast, 1=one-step-ahead)",
    )
    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Lag offsets (in samples) of the smallest window mean to include as features",
    )
    parser.add_argument(
        "--dow-mod7",
        action="store_true",
        help="Add cyclic day-of-week features derived from index arithmetic",
    )
    parser.add_argument(
        "--dow-offset",
        type=int,
        default=0,
        help="Offset applied to the derived weekday (0=Monday). Only used with --dow-mod7",
    )
    parser.add_argument(
        "--minute-cyc",
        action="store_true",
        help="Add cyclic minute-of-hour features derived from index arithmetic",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate timeline plots for the single-hour results",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path(__file__).parent / "imgs",
        help="Directory to store generated plots",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=EXPERIMENTAL_CONFIG["seed"],
        help="Base RNG seed for reproducibility",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of samples (keep the most recent)",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=1,
        help="Keep every Nth sample to thin the timeline",
    )

    args = parser.parse_args()

    # Ensure necessary feature flags are enabled in Python
    dp.enable_features("contrib", "contrib-continual")

    spot_ids = (
        sorted({int(s) for s in args.spot_ids}) if args.spot_ids is not None else None
    )
    window_grid = args.window_grid if args.window_grid is not None else [args.window_days]

    experiment_cfg = dict(EXPERIMENTAL_CONFIG)
    experiment_cfg.update(
        {
            "epsilons": list(args.epsilons),
            "window_days": args.window_days,
            "trials": args.trials,
            "monotonic": args.monotonic,
            "seed": args.seed,
            "window_days_grid": list(window_grid),
            "train_ratio": args.train_ratio,
            "l2": args.l2,
            "max_samples": args.max_samples,
            "sample_step": args.sample_step,
            "horizon": args.horizon,
            "lag_steps": list(args.lags),
            "dow_mod7": args.dow_mod7,
            "dow_offset": args.dow_offset,
            "minute_cyc": args.minute_cyc,
        }
    )

    variant = "Monotonic" if args.monotonic else "Baseline"
    subset_label = (
        "all-spots"
        if spot_ids is None
        else "spots-" + "-".join(str(s) for s in spot_ids)
    )

    print(f"\n=== Aggregate ({subset_label}) ===")
    try:
        counts_processed, res_a, max_spots = run_single_hour_evaluation(
            dataset_dir=args.dataset_dir,
            hour=args.hour,
            config=experiment_cfg,
            spot_ids=spot_ids,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"{exc}")
        return

    print(
        f"Single-hour evaluation: hour={args.hour}, samples={len(counts_processed)}"
    )
    print_single_hour_summary(
        hour=args.hour,
        variant=variant,
        subset_label=subset_label,
        max_available=max_spots,
        results=res_a,
    )
    if args.plot:
        plot_single_hour_timelines(
            hour=args.hour,
            variant=variant,
            counts_by_step=counts_processed,
            results=res_a,
            output_dir=args.plot_dir,
            subset_label=subset_label,
            max_available=max_spots,
        )


if __name__ == "__main__":
    main()
