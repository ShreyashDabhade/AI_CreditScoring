from __future__ import annotations

from math import sqrt
import sqlite3
from pathlib import Path
from typing import Any

from src.db_manager import resolve_database_path


DEFAULT_DRIFT_LOOKBACK = 100
DEFAULT_DRIFT_BASELINE_MEAN = 0.21
DEFAULT_DRIFT_ALERT_THRESHOLD = 0.10
DEFAULT_DRIFT_WATCH_THRESHOLD = 0.05
DEFAULT_DRIFT_MIN_SAMPLE = 30


def _load_recent_probabilities(
    db_path: str | Path | None,
    *,
    limit: int,
) -> list[float]:
    resolved_path = Path(resolve_database_path(db_path))
    if not resolved_path.exists():
        return []

    with sqlite3.connect(str(resolved_path)) as conn:
        rows = conn.execute(
            """
            SELECT probability
            FROM score_runs
            WHERE probability IS NOT NULL
            ORDER BY scored_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [float(row[0]) for row in rows if row[0] is not None]


def compute_probability_drift_snapshot(
    db_path: str | Path | None,
    *,
    baseline_mean: float = DEFAULT_DRIFT_BASELINE_MEAN,
    lookback: int = DEFAULT_DRIFT_LOOKBACK,
    alert_threshold: float = DEFAULT_DRIFT_ALERT_THRESHOLD,
    watch_threshold: float = DEFAULT_DRIFT_WATCH_THRESHOLD,
    min_sample_size: int = DEFAULT_DRIFT_MIN_SAMPLE,
) -> dict[str, Any]:
    probabilities = _load_recent_probabilities(db_path, limit=lookback)
    sample_size = len(probabilities)

    live_mean = sum(probabilities) / sample_size if sample_size else None
    if sample_size > 1 and live_mean is not None:
        variance = sum((value - live_mean) ** 2 for value in probabilities) / sample_size
        live_std = sqrt(variance)
    else:
        live_std = None

    deviation_ratio = None
    deviation_pct = None
    if live_mean is not None and baseline_mean:
        deviation_ratio = (live_mean - baseline_mean) / baseline_mean
        deviation_pct = deviation_ratio * 100.0

    if sample_size < min_sample_size:
        status = "watch"
        alert_message = f"Monitoring from {sample_size} recent score runs. More live volume is needed before drift can be assessed confidently."
    elif deviation_ratio is None:
        status = "watch"
        alert_message = "Baseline comparison is unavailable."
    elif abs(deviation_ratio) >= alert_threshold:
        status = "alert"
        alert_message = "Re-calibration Alert: live default probability has drifted materially from the configured training baseline."
    elif abs(deviation_ratio) >= watch_threshold:
        status = "watch"
        alert_message = "Live scoring is drifting away from baseline and should be monitored."
    else:
        status = "stable"
        alert_message = "Live scoring remains close to the configured baseline."

    return {
        "baseline_mean": baseline_mean,
        "live_mean": live_mean,
        "live_std": live_std,
        "sample_size": sample_size,
        "lookback": lookback,
        "deviation_ratio": deviation_ratio,
        "deviation_pct": deviation_pct,
        "status": status,
        "alert_message": alert_message,
        "min_sample_size": min_sample_size,
        "alert_threshold_pct": alert_threshold * 100.0,
        "watch_threshold_pct": watch_threshold * 100.0,
        "recalibration_alert": status == "alert",
    }
