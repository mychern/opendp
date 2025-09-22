#!/usr/bin/env python3
"""
Parking Lot Predictive Demo for the Toeplitz Mechanism (predictive focus)

This demo evaluates two predictive applications using continual DP aggregates
from the Toeplitz mechanism over the parking datasets:

A) Single-hour rolling-average predictor (per spot, per hour-of-day):
   - For a chosen spot and hour, predicts availability for the next day’s same hour
     using a rolling window of the previous Y days.
   - Training statistic: the average availability at that hour across the last Y days,
     obtained from DP prefix sums via the Toeplitz continual release.

B) Per-hour classifier trained via DP sums (per spot):
   - For each hour-of-day h, continually maintain DP sums of availability labels.
   - Predict availability for an incoming (hour=h) example using the DP mean for that hour
     computed from a rolling window of the last Y occurrences of that hour.

We report streaming prediction performance (accuracy, Brier score, log-loss) over epsilons.

Notes:
- This demo calls the Rust Toeplitz mechanism via the OpenDP Python FFI.
- Sensitivity is 1 per appended label (event-level adjacency).

Run examples:
  python rust/src/measurements/toeplitz/demo/parking_lot_predictive_demo.py \
    --app both --epsilons 0.2 0.5 1.0 --trials 10 --window-days 14 \
    --spot-id 5 --monotonic

  python rust/src/measurements/toeplitz/demo/parking_lot_predictive_demo.py \
    --app single-hour --hour 8 --window-days 14 --epsilons 0.5 --trials 20
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import accumulate
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


def _ensure_local_opendp_on_path() -> None:
    """Allow running the demo from a source checkout without installing the Python package."""
    for parent in Path(__file__).resolve().parents:
        python_src = parent / "python" / "src"
        if (python_src / "opendp").exists():
            python_src_str = str(python_src)
            if python_src_str not in sys.path:
                sys.path.insert(0, python_src_str)
            break


def _import_opendp_module(module_name: str):
    """Import an OpenDP module, falling back to the local source tree."""
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
_opendp_mod = _import_opendp_module("opendp.mod")
Measurement = _opendp_mod.Measurement


EXPERIMENTAL_CONFIG = {
    "epsilons": [0.2, 0.5, 1.0],
    "window_days": 14,
    "window_days_grid": [14],
    "trials": 10,
    "monotonic": False,
    "seed": 0,
}


# -------------------------------
# Data loading and aggregation
# -------------------------------

Timestamp = Tuple[int, int, int, int, int]  # (Year, Month, Date, Hour, Minute)


def load_spot_hourly_labels(dataset_dir: Path, spot_id: int) -> List[Tuple[Timestamp, int]]:
    """
    Load rows for a specific spot and derive an hourly availability label.

    For each day/hour (Y, M, D, H), take the last reading in that hour and map
    Status -> availability label as: available=1 if Status==0, else 0.

    Returns a chronological list of ((Y, M, D, H), label) entries.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Map (Y, M, D, H, M) -> (second, status) to keep last reading within the minute
    last_by_slot: Dict[Tuple[int, int, int, int, int], Tuple[int, int]] = {}

    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    for day_idx, path in enumerate(csv_files):
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    sid = int(row["SpotID"])  # which spot
                    if sid != spot_id:
                        continue
                    year = int(row["Year"])  # e.g., 2025
                    month = int(row["Month"])  # 1..12
                    day = int(row["Date"])  # 1..31
                    hour = int(row["Hour"])  # 0..23
                    minute = int(row["Minute"])  # 0..59
                    second = int(row["Second"])  # 0..59
                    status = int(row["Status"])  # 0/1
                except Exception:
                    continue

                # Treat each CSV file as a successive day to provide a richer timeline
                adjusted_date = date(year, month, day) + timedelta(days=day_idx)
                key = (
                    adjusted_date.year,
                    adjusted_date.month,
                    adjusted_date.day,
                    hour,
                    minute,
                )
                prev = last_by_slot.get(key)
                if prev is None or second >= prev[0]:
                    last_by_slot[key] = (second, status)

    # Build chronological sequence
    keys_sorted = sorted(last_by_slot.keys())
    result: List[Tuple[Timestamp, int]] = []
    for (y, m, d, h, minute) in keys_sorted:
        _, status = last_by_slot[(y, m, d, h, minute)]
        # availability label: 1 if status==0 (vacant), else 0 (occupied)
        label = 1 if status == 0 else 0
        result.append(((y, m, d, h, minute), label))

    return result


def _dp_sum_last_y(vec: List[int], epsilon: float, monotonic: bool, cache: Dict[int, Measurement]) -> int:
    """Compute DP sum of a vector via Rust Toeplitz (one-shot), returning the last prefix value."""
    n = len(vec)
    if n == 0:
        return 0
    if n not in cache:
        domain = dp.vector_domain(dp.atom_domain(T=int), size=n)
        metric = dp.l1_distance(T=int)
        # enforce_monotonicity toggles isotonic regression in the Rust measurement
        cache[n] = dp.m.make_toeplitz(
            domain,
            metric,
            scale=1.0 / epsilon,
            enforce_monotonicity=monotonic,
        )
    meas = cache[n]
    prefix = meas.invoke(vec)
    return int(prefix[-1])


# -------------------------------
# Evaluation utilities
# -------------------------------

@dataclass
class Metrics:
    accuracy: float
    brier: float
    logloss: float


@dataclass
class WindowMetrics:
    accuracy: float
    brier: float
    logloss: float
    mae: float
    window_mae: float
    window_rmse: float
    window_bias: float
    window_max_error: float


@dataclass
class StreamingSeries:
    avg_prob: List[float]
    avg_abs_error: List[float]
    mean_accuracy: List[float]
    cumulative_accuracy: List[float]
    window_bias: List[float]
    window_mae: List[float]
    window_rmse: List[float]


@dataclass
class WindowResults:
    summary: WindowMetrics
    timeline: StreamingSeries


def brier_score(probs: List[float], labels: List[int]) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / n


def log_loss(probs: List[float], labels: List[int], eps: float = 1e-6) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    s = 0.0
    for p, y in zip(probs, labels):
        p = min(max(p, eps), 1 - eps)
        s += - (y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / n


def evaluate_single_hour_prediction(
    labels_by_day: List[int],
    config: Mapping[str, Any],
) -> Dict[float, Dict[int, WindowResults]]:
    """Application A: rolling predictor that reuses a single Toeplitz continual release.

    We release one DP prefix stream per epsilon and measure downstream utility across a
    grid of rolling-window horizons. For each horizon we track standard predictive
    metrics (accuracy, Brier, log-loss, MAE) *and* Toeplitz-centric statistics that
    quantify how far the noisy window sums deviate from the true sums (bias, MAE, RMSE,
    worst-case error). This highlights how the continual mechanism supports arbitrary
    sub-interval queries over long deployments.
    """
    results: Dict[float, Dict[int, WindowResults]] = {}
    epsilons = config["epsilons"]
    window_grid = sorted(set(config.get("window_days_grid", [config["window_days"]])))
    trials = config["trials"]
    monotonic = config["monotonic"]
    n = len(labels_by_day)

    if n == 0 or not window_grid:
        return results

    domain = dp.vector_domain(dp.atom_domain(T=int), size=n)
    metric = dp.l1_distance(T=int)
    true_prefix = list(accumulate(labels_by_day))

    for eps in epsilons:
        if eps <= 0:
            continue
        measurement = dp.m.make_toeplitz(
            domain,
            metric,
            scale=1.0 / eps,
            enforce_monotonicity=monotonic,
        )
        acc_vals: Dict[int, List[float]] = {window: [] for window in window_grid}
        brier_vals: Dict[int, List[float]] = {window: [] for window in window_grid}
        log_vals: Dict[int, List[float]] = {window: [] for window in window_grid}
        mae_vals: Dict[int, List[float]] = {window: [] for window in window_grid}
        window_mae_vals: Dict[int, List[float]] = {window: [] for window in window_grid}
        window_rmse_vals: Dict[int, List[float]] = {window: [] for window in window_grid}
        window_bias_vals: Dict[int, List[float]] = {window: [] for window in window_grid}
        window_max_tracker: Dict[int, float] = {window: 0.0 for window in window_grid}

        per_step_pred_sums: Dict[int, List[float]] = {window: [0.0] * n for window in window_grid}
        per_step_hits: Dict[int, List[int]] = {window: [0] * n for window in window_grid}
        per_step_abs_err_sums: Dict[int, List[float]] = {window: [0.0] * n for window in window_grid}
        per_step_window_error_sums: Dict[int, List[float]] = {window: [0.0] * n for window in window_grid}
        per_step_window_abs_err_sums: Dict[int, List[float]] = {window: [0.0] * n for window in window_grid}
        per_step_window_sq_err_sums: Dict[int, List[float]] = {window: [0.0] * n for window in window_grid}
        per_step_window_counts: Dict[int, List[int]] = {window: [0] * n for window in window_grid}

        for _ in range(trials):
            dp_prefix = list(map(float, measurement.invoke(labels_by_day)))
            preds_by_window: Dict[int, List[float]] = {window: [] for window in window_grid}
            errors_by_window: Dict[int, List[Optional[float]]] = {window: [] for window in window_grid}

            for t, y in enumerate(labels_by_day):
                for window in window_grid:
                    denom = min(window, t)
                    error: Optional[float]
                    if denom > 0:
                        right = t - 1
                        dp_window_sum = dp_prefix[right]
                        true_window_sum = true_prefix[right]
                        left = t - denom
                        if left > 0:
                            dp_window_sum -= dp_prefix[left - 1]
                            true_window_sum -= true_prefix[left - 1]
                        error = (dp_window_sum - true_window_sum) / float(denom)
                        raw_mean = dp_window_sum / float(denom)
                        p_hat = max(0.0, min(1.0, raw_mean))
                    else:
                        p_hat = 0.5
                        error = None

                    preds_by_window[window].append(p_hat)
                    errors_by_window[window].append(error)

                    per_step_pred_sums[window][t] += p_hat
                    if (1 if p_hat >= 0.5 else 0) == y:
                        per_step_hits[window][t] += 1
                    per_step_abs_err_sums[window][t] += abs(p_hat - y)
                    if error is not None:
                        per_step_window_error_sums[window][t] += error
                        per_step_window_abs_err_sums[window][t] += abs(error)
                        per_step_window_sq_err_sums[window][t] += error * error
                        per_step_window_counts[window][t] += 1

            for window in window_grid:
                preds = preds_by_window[window]
                accuracy = sum((1 if p >= 0.5 else 0) == y for p, y in zip(preds, labels_by_day)) / len(labels_by_day)
                brier = brier_score(preds, labels_by_day)
                ll = log_loss(preds, labels_by_day)
                mae = sum(abs(p - y) for p, y in zip(preds, labels_by_day)) / len(labels_by_day)
                errors = [e for e in errors_by_window[window] if e is not None]
                if errors:
                    window_mae = sum(abs(e) for e in errors) / len(errors)
                    window_rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
                    window_bias = sum(errors) / len(errors)
                    window_max = max(abs(e) for e in errors)
                else:
                    window_mae = 0.0
                    window_rmse = 0.0
                    window_bias = 0.0
                    window_max = 0.0

                acc_vals[window].append(accuracy)
                brier_vals[window].append(brier)
                log_vals[window].append(ll)
                mae_vals[window].append(mae)
                window_mae_vals[window].append(window_mae)
                window_rmse_vals[window].append(window_rmse)
                window_bias_vals[window].append(window_bias)
                window_max_tracker[window] = max(window_max_tracker[window], window_max)

        per_eps_metrics: Dict[int, WindowResults] = {}
        for window in window_grid:
            trial_count = len(acc_vals[window])
            if trial_count == 0:
                continue

            summary = WindowMetrics(
                accuracy=sum(acc_vals[window]) / trial_count,
                brier=sum(brier_vals[window]) / trial_count,
                logloss=sum(log_vals[window]) / trial_count,
                mae=sum(mae_vals[window]) / trial_count,
                window_mae=sum(window_mae_vals[window]) / trial_count,
                window_rmse=sum(window_rmse_vals[window]) / trial_count,
                window_bias=sum(window_bias_vals[window]) / trial_count,
                window_max_error=window_max_tracker[window],
            )

            avg_prob_ts = [s / trial_count for s in per_step_pred_sums[window]]
            avg_abs_err_ts = [s / trial_count for s in per_step_abs_err_sums[window]]
            mean_accuracy_ts = [hits / trial_count for hits in per_step_hits[window]]
            cumulative_accuracy_ts: List[float] = []
            cumulative_hits = 0
            for idx, hits in enumerate(per_step_hits[window]):
                cumulative_hits += hits
                cumulative_accuracy_ts.append(
                    cumulative_hits / float(trial_count * (idx + 1))
                )

            window_bias_ts: List[float] = []
            window_mae_ts: List[float] = []
            window_rmse_ts: List[float] = []
            for idx, count in enumerate(per_step_window_counts[window]):
                if count == 0:
                    window_bias_ts.append(0.0)
                    window_mae_ts.append(0.0)
                    window_rmse_ts.append(0.0)
                    continue
                window_bias_ts.append(per_step_window_error_sums[window][idx] / count)
                window_mae_ts.append(per_step_window_abs_err_sums[window][idx] / count)
                window_rmse_ts.append(
                    math.sqrt(per_step_window_sq_err_sums[window][idx] / count)
                )

            timeline = StreamingSeries(
                avg_prob=avg_prob_ts,
                avg_abs_error=avg_abs_err_ts,
                mean_accuracy=mean_accuracy_ts,
                cumulative_accuracy=cumulative_accuracy_ts,
                window_bias=window_bias_ts,
                window_mae=window_mae_ts,
                window_rmse=window_rmse_ts,
            )

            per_eps_metrics[window] = WindowResults(summary=summary, timeline=timeline)

        if per_eps_metrics:
            results[eps] = per_eps_metrics

    return results


def evaluate_per_hour_classifier(
    hourly_events: List[Tuple[int, int]],
    config: Mapping[str, Any],
) -> Dict[float, Metrics]:
    """Application B: per-hour classifier trained via DP sums (via Rust FFI).

    For each hour h, predict using DP mean over the last Y occurrences of hour h.
    """
    results: Dict[float, Metrics] = {}
    epsilons = config["epsilons"]
    window_days = config["window_days"]
    trials = config["trials"]
    monotonic = config["monotonic"]
    for eps in epsilons:
        if eps <= 0:
            continue
        meas_cache_by_hour: List[Dict[int, Measurement]] = [dict() for _ in range(24)]
        acc_vals: List[float] = []
        brier_vals: List[float] = []
        log_vals: List[float] = []
        for _ in range(trials):
            preds: List[float] = []
            trues: List[int] = []
            per_hour_labels: List[List[int]] = [[] for _ in range(24)]
            for h, y in hourly_events:
                hist = per_hour_labels[h]
                denom = min(window_days, len(hist))
                if denom > 0:
                    vec = hist[-denom:]
                    dp_sum = _dp_sum_last_y(vec, eps, monotonic, meas_cache_by_hour[h])
                    p_hat = max(0.0, min(1.0, dp_sum / float(denom)))
                else:
                    p_hat = 0.5
                preds.append(p_hat)
                trues.append(y)
                hist.append(y)
            acc = sum((1 if p >= 0.5 else 0) == y for p, y in zip(preds, trues)) / len(trues)
            brier = brier_score(preds, trues)
            ll = log_loss(preds, trues)
            acc_vals.append(acc)
            brier_vals.append(brier)
            log_vals.append(ll)
        results[eps] = Metrics(
            accuracy=sum(acc_vals) / len(acc_vals),
            brier=sum(brier_vals) / len(brier_vals),
            logloss=sum(log_vals) / len(log_vals),
        )
    return results


def print_single_hour_summary(
    spot_id: int,
    hour: int,
    variant: str,
    results: Dict[float, Dict[int, WindowResults]],
) -> None:
    print(f"\nApplication A: Single-hour rolling predictor ({variant}), hour={hour}, spot={spot_id}")
    if not results:
        print("No results (insufficient samples for the selected hour).")
        return
    print(
        "epsilon | window | scale | accuracy | brier | logloss | mae | "
        "win_mae | win_rmse | win_bias | win_max"
    )
    for eps in sorted(results.keys()):
        metrics_by_window = results[eps]
        if not metrics_by_window:
            continue
        scale = 1.0 / eps if eps > 0 else float("inf")
        for window in sorted(metrics_by_window.keys()):
            summary = metrics_by_window[window].summary
            print(
                f"{eps:7.3f} | {window:6d} | {scale:5.2f} | "
                f"{summary.accuracy:8.3f} | {summary.brier:6.4f} | {summary.logloss:8.4f} | "
                f"{summary.mae:6.3f} | {summary.window_mae:7.3f} | {summary.window_rmse:8.3f} | "
                f"{summary.window_bias:8.3f} | {summary.window_max_error:7.3f}"
            )


def print_per_hour_summary(
    spot_id: int,
    variant: str,
    results: Dict[float, Metrics],
) -> None:
    print(f"\nApplication B: Per-hour classifier via DP sums ({variant}), spot={spot_id}")
    if not results:
        print("No results (insufficient events).")
        return
    print("epsilon | scale | accuracy | brier | logloss")
    for eps in sorted(results.keys()):
        scale = 1.0 / eps if eps > 0 else float("inf")
        m = results[eps]
        print(f"{eps:7.3f} | {scale:5.2f} | {m.accuracy:8.3f} | {m.brier:5.3f} | {m.logloss:7.3f}")


def plot_single_hour_timelines(
    spot_id: int,
    hour: int,
    variant: str,
    labels_by_day: List[int],
    results: Dict[float, Dict[int, WindowResults]],
    output_dir: Path,
) -> None:
    if not labels_by_day or not results:
        return
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError as exc:  # pragma: no cover - optional dependency
        if not getattr(plot_single_hour_timelines, "_warned", False):
            print(f"Matplotlib not available; skipping plots ({exc}).")
            plot_single_hour_timelines._warned = True  # type: ignore[attr-defined]
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    time_axis = list(range(len(labels_by_day)))
    true_series = labels_by_day

    for eps in sorted(results.keys()):
        window_results = results[eps]
        if not window_results:
            continue

        windows = sorted(window_results.keys())
        cmap = cm.get_cmap("tab10", len(windows))
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axes[0].step(time_axis, true_series, where="post", color="black", alpha=0.3, label="True label")
        for idx, window in enumerate(windows):
            color = cmap(idx)
            timeline = window_results[window].timeline
            axes[0].plot(time_axis, timeline.avg_prob, color=color, label=f"win={window}")
        axes[0].set_ylabel("Avg prob")
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].set_title(f"Spot {spot_id} hour {hour} ε={eps:.3f} ({variant})")
        axes[0].legend(loc="upper right", fontsize=8)

        for idx, window in enumerate(windows):
            color = cmap(idx)
            timeline = window_results[window].timeline
            axes[1].plot(time_axis, timeline.cumulative_accuracy, color=color, label=f"win={window}")
        axes[1].set_ylabel("Cumulative accuracy")
        axes[1].set_ylim(0.0, 1.02)

        axes[2].axhline(0.0, color="black", linewidth=0.5, alpha=0.3)
        for idx, window in enumerate(windows):
            color = cmap(idx)
            timeline = window_results[window].timeline
            axes[2].plot(time_axis, timeline.window_rmse, color=color, label=f"win={window}")
            axes[2].plot(
                time_axis,
                timeline.window_bias,
                color=color,
                linestyle="--",
            )
        axes[2].set_ylabel("Window RMSE (solid) / bias (dashed)")
        axes[2].set_xlabel("Timestep (days)")

        for ax in axes:
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

        fig.tight_layout()
        variant_slug = variant.lower().replace(" ", "_")
        eps_slug = f"{eps:.3f}".replace("-", "m").replace(".", "p")
        filename = f"spot{spot_id}_hour{hour}_eps{eps_slug}_{variant_slug}.png"
        fig.savefig(str(output_dir / filename), dpi=200)
        plt.close(fig)
# -------------------------------
# CLI
# -------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Toeplitz DP predictive demo on parking lot datasets")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).parent / "datasets",
        help="Path to datasets directory containing CSV files",
    )
    parser.add_argument(
        "--app",
        choices=["single-hour", "per-hour", "both"],
        default="both",
        help="Which application to run: single-hour rolling average or per-hour classifier",
    )
    parser.add_argument(
        "--spot-ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of SpotIDs to evaluate (overrides --spot-id)",
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
    parser.add_argument("--spot-id", type=int, default=5, help="SpotID to analyze")
    parser.add_argument("--hour", type=int, default=7, help="Hour-of-day (0-23) for single-hour app")
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
        help="Optional list of window sizes (days/occurrences) to evaluate for Application A",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate timeline plots for Application A results",
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

    args = parser.parse_args()

    # Ensure necessary feature flags are enabled in Python
    dp.enable_features("contrib", "contrib-continual")

    # Monkey-patch make_toeplitz if the local OpenDP build does not expose it yet
    if not hasattr(dp.m, "make_toeplitz"):
        import ctypes

        opendp_lib = _import_opendp_module("opendp._lib")
        lib = opendp_lib.lib
        FfiResult = opendp_lib.FfiResult
        unwrap = opendp_lib.unwrap

        def _ffi_make_toeplitz(input_domain, input_metric, *, scale: float, enforce_monotonicity: bool = True) -> Measurement:
            fn = lib.opendp_measurements__make_toeplitz
            fn.argtypes = [type(input_domain), type(input_metric), ctypes.c_double, ctypes.c_bool, ctypes.c_char_p]
            fn.restype = FfiResult
            res = fn(
                input_domain,
                input_metric,
                scale,
                enforce_monotonicity,
                b"ZeroConcentratedDivergence",
            )
            return unwrap(res, Measurement)

        setattr(dp.m, "make_toeplitz", _ffi_make_toeplitz)

    spot_ids = args.spot_ids if args.spot_ids is not None else [args.spot_id]
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
        }
    )

    variant = "Monotonic" if args.monotonic else "Baseline"

    for spot_id in spot_ids:
        print(f"\n=== Spot {spot_id} ===")
        try:
            entries = load_spot_hourly_labels(args.dataset_dir, spot_id)
        except FileNotFoundError as exc:
            print(f"{exc}")
            continue
        if not entries:
            print(f"No entries found for SpotID={spot_id} in {args.dataset_dir}")
            continue

        print(f"Loaded {len(entries)} hourly labels for SpotID={spot_id}")

        labels_single_hour: List[int] = [
            label for ((_, _, _, h, _), label) in entries if h == args.hour
        ]
        print(f"Single-hour app: hour={args.hour}, samples={len(labels_single_hour)}")

        hourly_events: List[Tuple[int, int]] = [
            (h, label) for ((_, _, _, h, _), label) in entries
        ]
        print(f"Per-hour app: total events={len(hourly_events)}")

        if args.app in ("single-hour", "both"):
            res_a = evaluate_single_hour_prediction(
                labels_by_day=labels_single_hour,
                config=experiment_cfg,
            )
            print_single_hour_summary(
                spot_id=spot_id,
                hour=args.hour,
                variant=variant,
                results=res_a,
            )
            if args.plot:
                plot_dir = args.plot_dir / f"spot_{spot_id}"
                plot_single_hour_timelines(
                    spot_id=spot_id,
                    hour=args.hour,
                    variant=variant,
                    labels_by_day=labels_single_hour,
                    results=res_a,
                    output_dir=plot_dir,
                )

        if args.app in ("per-hour", "both"):
            res_b = evaluate_per_hour_classifier(
                hourly_events=hourly_events,
                config=experiment_cfg,
            )
            print_per_hour_summary(
                spot_id=spot_id,
                variant=variant,
                results=res_b,
            )


if __name__ == "__main__":
    main()
