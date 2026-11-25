#!/usr/bin/env python3
"""
Continuous CHB-MIT EEG-CASCADE evaluation using a fixed-duration rolling
30-s cumulative log-likelihood score.

This script reuses the validated model implementation in
chbmit_dynml_ictal_detection_BD_v4_fixed.py and changes only the evaluation
protocol needed for continuous monitoring:

1) Hold out one entire EDF before model development.
2) Build training/validation interictal 30-s scans only from the other EDFs.
3) Fit StandardScaler, PCA, R4 DynML, and residual Gaussian exactly as before.
4) Convert each 30-s validation sequence to ONE scalar score by summing all
   one-step residual log-likelihoods in that sequence.
5) Set one scalar validation threshold at the requested percentile.
6) Run through the held-out EDF chronologically and compute pointwise
   one-step residual log-likelihoods.
7) Form a rolling sum containing exactly the same number of likelihood steps
   as a 30-s validation sequence (normally 25 with 5-s chunks, 1-s stride,
   horizon=1).
8) Apply the fixed scalar threshold and persistence rule continuously.
9) Count seizure detections in [onset, onset + detection_window_seconds].
10) Count false-alarm episodes only when the confirmed alarm time lies in
    clean interictal time using the same +/- seizure exclusion buffers as the
    submitted CHB-MIT analysis.

The held-out EDF is never used for scaler/PCA fitting, DynML fitting,
residual-Gaussian fitting, or threshold calibration.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_core(core_script: Path):
    core_script = Path(core_script).resolve()
    if not core_script.exists():
        raise FileNotFoundError(f"Core script not found: {core_script}")
    spec = importlib.util.spec_from_file_location("chbmit_core", str(core_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import core script: {core_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_edf(patient_dir: Path, value: str, edf_df: pd.DataFrame, seizures) -> Path:
    value = str(value).strip()
    if value.lower() == "auto":
        if not seizures:
            raise RuntimeError("No annotated seizures available for automatic EDF selection")
        counts: Dict[Path, int] = {}
        for s in seizures:
            counts[Path(s.edf_path)] = counts.get(Path(s.edf_path), 0) + 1
        # Prefer the seizure EDF with the largest number of annotated seizures;
        # break ties by EDF name for reproducibility.
        selected = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].name))[0][0]
        return selected

    p = Path(value)
    if p.exists():
        return p.resolve()

    p2 = patient_dir / value
    if p2.exists():
        return p2.resolve()

    matches = edf_df.loc[edf_df["edf_name"].astype(str) == value, "edf"].tolist()
    if len(matches) == 1:
        return Path(matches[0]).resolve()

    raise FileNotFoundError(
        f"Could not resolve --continuous_edf={value!r}. "
        f"Use --list_edfs to see available files."
    )


def print_edf_table(edf_df: pd.DataFrame, seizures) -> None:
    seizure_counts: Dict[str, int] = {}
    seizure_onsets: Dict[str, List[float]] = {}
    for s in seizures:
        name = Path(s.edf_path).name
        seizure_counts[name] = seizure_counts.get(name, 0) + 1
        seizure_onsets.setdefault(name, []).append(float(s.onset_sec))

    rows = []
    for _, row in edf_df.iterrows():
        name = str(row["edf_name"])
        onsets = seizure_onsets.get(name, [])
        rows.append(
            {
                "edf_name": name,
                "duration_min": float(row["duration_sec"]) / 60.0
                if np.isfinite(row["duration_sec"])
                else math.nan,
                "n_seizures": seizure_counts.get(name, 0),
                "seizure_onsets_sec": ",".join(f"{x:.1f}" for x in onsets),
            }
        )
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))


def build_development_scans(
    core,
    patient: str,
    edf_df: pd.DataFrame,
    seizures,
    heldout_edf: Path,
    args,
):
    """Create train/validation 30-s clean-interictal scans from non-heldout EDFs."""
    heldout_edf = heldout_edf.resolve()

    dev_edf_df = edf_df[
        edf_df["edf"].map(lambda x: Path(x).resolve() != heldout_edf)
    ].copy()
    dev_seizures = [s for s in seizures if Path(s.edf_path).resolve() != heldout_edf]

    if dev_edf_df.empty:
        raise RuntimeError("No development EDFs remain after holding out the continuous-test EDF")

    # Reuse the original scan construction, but do not allocate a sampled test set.
    split_args = copy.copy(args)
    split_args.test_fraction = 0.0

    # Preserve the original train:validation ratio (0.50:0.25 = 2:1) while
    # using all available development scans after the EDF holdout.
    original_train_fraction = max(0.0, 1.0 - float(args.val_fraction) - float(args.test_fraction))
    denom = original_train_fraction + float(args.val_fraction)
    if denom <= 0:
        raise ValueError("Invalid train/validation fractions")
    split_args.val_fraction = float(args.val_fraction) / denom

    train_scans, val_scans, _, _ = core.make_patient_scans(
        patient, dev_edf_df, dev_seizures, split_args
    )

    return train_scans, val_scans


def validation_scalar_threshold(
    core,
    val_seqs,
    model,
    mu,
    var,
    washout: int,
    pred_horizon: int,
    percentile: float,
    chunk_seconds: float,
) -> Tuple[float, int, np.ndarray]:
    """
    Each 30-s validation sequence contributes one total cumulative LL score.
    Returns one scalar threshold and the number of LL steps per 30-s sequence.
    """
    scores = []
    lengths = []

    for seq in val_seqs:
        ll, _ = core.seq_loglik(
            seq, model, mu, var, washout, pred_horizon, chunk_seconds
        )
        if len(ll) == 0:
            continue
        scores.append(float(np.sum(ll)))
        lengths.append(int(len(ll)))

    if not scores:
        raise RuntimeError("No validation log-likelihood scores available")

    unique_lengths = sorted(set(lengths))
    if len(unique_lengths) != 1:
        raise RuntimeError(
            "Validation sequences do not have equal LL lengths: "
            f"{unique_lengths}. Continuous rolling calibration requires equal "
            "fixed-duration validation sequences."
        )

    rolling_steps = unique_lengths[0]
    threshold = float(np.percentile(np.asarray(scores), float(percentile)))
    return threshold, rolling_steps, np.asarray(scores, dtype=float)


def build_continuous_sequence(core, patient: str, edf_path: Path, duration_sec: float, args):
    scan = core.Scan(
        patient=patient,
        edf_path=edf_path,
        kind="continuous_test",
        start_sec=0.0,
        end_sec=float(duration_sec),
    )
    cache = {}
    seq = core.build_feature_sequence(scan, cache, args)
    if seq is None:
        raise RuntimeError(f"Could not build continuous feature sequence for {edf_path.name}")
    return seq


def rolling_sum(ll: np.ndarray, times: np.ndarray, rolling_steps: int):
    ll = np.asarray(ll, dtype=float)
    times = np.asarray(times, dtype=float)
    rolling_steps = int(rolling_steps)

    if rolling_steps <= 0:
        raise ValueError("rolling_steps must be positive")
    if len(ll) < rolling_steps:
        return np.asarray([]), np.asarray([])

    cs = np.concatenate(([0.0], np.cumsum(ll)))
    scores = cs[rolling_steps:] - cs[:-rolling_steps]
    score_times = times[rolling_steps - 1 :]
    return scores, score_times


def persistence_episodes(
    scores: np.ndarray,
    times: np.ndarray,
    threshold: float,
    persistence_steps: int,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Count one alarm episode per contiguous anomalous run."""
    scores = np.asarray(scores, dtype=float)
    times = np.asarray(times, dtype=float)
    flags = scores < float(threshold)

    episodes: List[Dict[str, float]] = []
    run = 0
    in_episode = False

    for i, flag in enumerate(flags):
        if bool(flag):
            run += 1
            if run >= persistence_steps and not in_episode:
                first_i = i - persistence_steps + 1
                episodes.append(
                    {
                        "first_crossing_time_sec": float(times[first_i]),
                        "confirmed_alarm_time_sec": float(times[i]),
                        "score_at_confirmation": float(scores[i]),
                    }
                )
                in_episode = True
        else:
            run = 0
            in_episode = False

    return flags, episodes


def point_in_intervals(t: float, intervals: Sequence[Tuple[float, float]]) -> bool:
    return any(float(a) <= float(t) <= float(b) for a, b in intervals)


def classify_alarms(
    alarms_df: pd.DataFrame,
    seizures,
    duration_sec: float,
    core,
    args,
    detection_window_seconds: float,
):
    # Clean interictal monitoring time uses the same +/- seizure buffers as the manuscript.
    blocked = [
        (
            float(s.onset_sec) - float(args.interictal_pre_buffer_seconds),
            float(s.end_sec) + float(args.interictal_post_buffer_seconds),
        )
        for s in seizures
    ]
    clean_intervals = core.complement_intervals(float(duration_sec), blocked)
    clean_hours = sum(max(0.0, b - a) for a, b in clean_intervals) / 3600.0

    seizure_rows = []
    detected_alarm_indices = set()

    if alarms_df.empty:
        alarm_times = np.asarray([], dtype=float)
    else:
        alarm_times = alarms_df["confirmed_alarm_time_sec"].to_numpy(dtype=float)

    for s in seizures:
        start = float(s.onset_sec)
        end = min(float(duration_sec), start + float(detection_window_seconds))
        idx = np.where((alarm_times >= start) & (alarm_times <= end))[0]
        if len(idx):
            first_idx = int(idx[0])
            first_alarm = float(alarm_times[first_idx])
            detected_alarm_indices.add(first_idx)
            detected = True
            delay = first_alarm - start
        else:
            first_alarm = math.nan
            detected = False
            delay = math.nan

        seizure_rows.append(
            {
                "seizure_id": s.seizure_id,
                "onset_sec": start,
                "end_sec": float(s.end_sec),
                "detection_window_end_sec": end,
                "detected": detected,
                "confirmed_alarm_time_sec": first_alarm,
                "detection_delay_seconds": delay,
            }
        )

    if alarms_df.empty:
        classified = alarms_df.copy()
        false_alarm_count = 0
    else:
        classified = alarms_df.copy()
        labels = []
        for i, row in classified.iterrows():
            t = float(row["confirmed_alarm_time_sec"])
            if i in detected_alarm_indices:
                labels.append("seizure_detection")
            elif point_in_intervals(t, clean_intervals):
                labels.append("interictal_false_alarm")
            else:
                labels.append("excluded_seizure_buffer")
        classified["classification"] = labels
        false_alarm_count = int((classified["classification"] == "interictal_false_alarm").sum())

    seizure_columns = [
        "seizure_id",
        "onset_sec",
        "end_sec",
        "detection_window_end_sec",
        "detected",
        "confirmed_alarm_time_sec",
        "detection_delay_seconds",
    ]
    seizure_df = pd.DataFrame(seizure_rows, columns=seizure_columns)

    sensitivity = (
        float(seizure_df["detected"].mean()) if len(seizure_df) else math.nan
    )

    delays = (
        seizure_df.loc[
            seizure_df["detected"],
            "detection_delay_seconds"
        ].to_numpy(dtype=float)
        if len(seizure_df)
        else np.asarray([], dtype=float)
    )

    far_per_hour = false_alarm_count / clean_hours if clean_hours > 0 else math.nan

    metrics = {
        "n_seizures": int(len(seizure_df)),
        "n_detected_seizures": int(seizure_df["detected"].sum()) if len(seizure_df) else 0,
        "seizure_sensitivity": sensitivity,
        "mean_detection_delay_seconds": float(np.mean(delays)) if len(delays) else math.nan,
        "median_detection_delay_seconds": float(np.median(delays)) if len(delays) else math.nan,
        "clean_interictal_hours": float(clean_hours),
        "interictal_false_alarm_episodes": int(false_alarm_count),
        "false_alarms_per_hour": float(far_per_hour),
    }

    return classified, seizure_df, clean_intervals, metrics


def run(args) -> None:
    core = load_core(Path(args.core_script))

    patient = args.patient.lower().strip()
    patient_dir = Path(args.data_root) / patient
    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

    edf_df, seizures = core.index_patient(patient_dir, patient)
    if edf_df.empty:
        raise RuntimeError(f"No EDF files indexed for {patient}")

    if args.list_edfs:
        print_edf_table(edf_df, seizures)
        return

    heldout_edf = resolve_edf(patient_dir, args.continuous_edf, edf_df, seizures)
    row = edf_df[edf_df["edf"].map(lambda x: Path(x).resolve() == heldout_edf)]
    if row.empty:
        raise RuntimeError(f"Held-out EDF is not indexed for {patient}: {heldout_edf}")
    duration_sec = float(row.iloc[0]["duration_sec"])
    if not np.isfinite(duration_sec):
        raise RuntimeError(f"Invalid EDF duration for {heldout_edf.name}")

    test_seizures = [s for s in seizures if Path(s.edf_path).resolve() == heldout_edf]

    print("=" * 80)
    print("CHB-MIT CONTINUOUS EEG-CASCADE")
    print(f"Patient:             {patient}")
    print(f"Held-out EDF:        {heldout_edf.name}")
    print(f"EDF duration:        {duration_sec / 60.0:.2f} min")
    print(f"Seizures in EDF:     {len(test_seizures)}")
    print(f"Rössler regime:      R{args.regime_id}")
    print(f"Reservoir seed:      {args.res_seed}")
    print("=" * 80)

    train_scans, val_scans = build_development_scans(
        core, patient, edf_df, seizures, heldout_edf, args
    )

    print(f"Development train scans: {len(train_scans)}")
    print(f"Development val scans:   {len(val_scans)}")

    train_raw = core.scans_to_features(train_scans, args)
    val_raw = core.scans_to_features(val_scans, args)

    if len(train_raw) < int(args.min_train_scans):
        raise RuntimeError(f"Insufficient training feature sequences: {len(train_raw)}")
    if len(val_raw) < int(args.min_val_scans):
        raise RuntimeError(f"Insufficient validation feature sequences: {len(val_raw)}")

    train_mat = np.vstack([s.features for s in train_raw])
    scaler = StandardScaler().fit(train_mat)
    pca_n = min(
        int(args.pca_components),
        train_mat.shape[1],
        max(1, train_mat.shape[0] - 1),
    )
    pca = PCA(n_components=pca_n, random_state=args.data_seed).fit(
        scaler.transform(train_mat)
    )

    train = core.transform_sequences(train_raw, scaler, pca)
    val = core.transform_sequences(val_raw, scaler, pca)

    Xtr, Ytr = core.collect_xy(train, args.washout, args.pred_horizon)
    model = core.RosslerDynML(pca_n, args)
    model.fit(Xtr, Ytr)

    mu, var = core.fit_diag_gaussian_from_validation(
        val, model, args.washout, args.pred_horizon
    )

    threshold, rolling_steps, val_scores = validation_scalar_threshold(
        core,
        val,
        model,
        mu,
        var,
        args.washout,
        args.pred_horizon,
        args.threshold_percentile,
        args.chunk_seconds,
    )

    print(f"PCA components:       {pca_n}")
    print(f"Validation 30-s scores: {len(val_scores)}")
    print(f"Rolling LL steps:     {rolling_steps}")
    print(f"Validation threshold: {threshold:.6f}")

    continuous_raw = build_continuous_sequence(
        core, patient, heldout_edf, duration_sec, args
    )
    continuous = core.transform_sequences([continuous_raw], scaler, pca)[0]

    ll, ll_times = core.seq_loglik(
        continuous,
        model,
        mu,
        var,
        args.washout,
        args.pred_horizon,
        args.chunk_seconds,
    )
    scores, score_times = rolling_sum(ll, ll_times, rolling_steps)

    if len(scores) == 0:
        raise RuntimeError("Continuous EDF is too short for the rolling score")

    persistence_steps = max(
        1, int(math.ceil(float(args.persistence_seconds) / float(args.stride_seconds)))
    )
    flags, alarms = persistence_episodes(
        scores, score_times, threshold, persistence_steps
    )
    alarms_df = pd.DataFrame(alarms)

    classified_alarms, seizure_df, clean_intervals, metrics = classify_alarms(
        alarms_df,
        test_seizures,
        duration_sec,
        core,
        args,
        args.detection_window_seconds,
    )

    out_dir = Path(args.out_dir) / patient / heldout_edf.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_df = pd.DataFrame(
        {
            "time_sec": score_times,
            "time_min": score_times / 60.0,
            "rolling_loglik_30s": scores,
            "threshold": threshold,
            "below_threshold": flags.astype(int),
        }
    )
    trace_df.to_csv(out_dir / "continuous_rolling_trace.csv", index=False)
    pd.DataFrame({"validation_30s_loglik": val_scores}).to_csv(
        out_dir / "validation_30s_scores.csv", index=False
    )
    classified_alarms.to_csv(out_dir / "continuous_alarm_episodes.csv", index=False)
    seizure_df.to_csv(out_dir / "continuous_seizure_detection.csv", index=False)

    summary = {
        "patient": patient,
        "heldout_edf": heldout_edf.name,
        "duration_seconds": duration_sec,
        "regime_id": int(args.regime_id),
        "data_seed": int(args.data_seed),
        "res_seed": int(args.res_seed),
        "pca_components": int(pca_n),
        "chunk_seconds": float(args.chunk_seconds),
        "stride_seconds": float(args.stride_seconds),
        "rolling_steps": int(rolling_steps),
        "rolling_span_nominal_seconds": float(args.segment_seconds),
        "threshold_percentile": float(args.threshold_percentile),
        "scalar_validation_threshold": float(threshold),
        "persistence_seconds": float(args.persistence_seconds),
        "detection_window_seconds": float(args.detection_window_seconds),
        **metrics,
    }
    (out_dir / "continuous_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )

    # Main continuous-monitoring figure.
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(
        score_times / 60.0,
        scores,
        linewidth=1.0,
        label="30-s rolling cumulative log-likelihood",
    )
    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=2.0,
        label=f"Validation {args.threshold_percentile:g}th-percentile threshold",
    )

    for i, s in enumerate(test_seizures):
        ax.axvspan(
            float(s.onset_sec) / 60.0,
            min(duration_sec, float(s.onset_sec) + float(args.detection_window_seconds)) / 60.0,
            alpha=0.16,
            label="Seizure-onset evaluation window" if i == 0 else None,
        )
        ax.axvline(
            float(s.onset_sec) / 60.0,
            linestyle=":",
            linewidth=1.6,
            label="Annotated seizure onset" if i == 0 else None,
        )

    if not classified_alarms.empty:
        for j, row_alarm in classified_alarms.iterrows():
            tmin = float(row_alarm["confirmed_alarm_time_sec"]) / 60.0
            cls = row_alarm.get("classification", "alarm")
            if cls == "seizure_detection":
                marker = "o"
            elif cls == "interictal_false_alarm":
                marker = "x"
            else:
                marker = "."
            ax.scatter(
                [tmin],
                [float(row_alarm["score_at_confirmation"])],
                marker=marker,
                s=55,
                zorder=5,
            )

    ax.set_xlabel("Time from start of held-out EDF (min)")
    ax.set_ylabel("30-s rolling cumulative log-likelihood")
    ax.set_title(
        f"CHB-MIT {patient}: continuous held-out EEG ({heldout_edf.name})"
    )
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "continuous_monitoring.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "continuous_monitoring.pdf", bbox_inches="tight")
    plt.close(fig)

    print("\nRESULTS")
    print("-" * 80)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("-" * 80)
    print(f"Saved outputs to: {out_dir.resolve()}")


def parse_args():
    p = argparse.ArgumentParser(
        description="CHB-MIT continuous EEG-CASCADE using a rolling 30-s log-likelihood score"
    )

    p.add_argument(
        "--core_script",
        default="chbmit_dynml_ictal_detection_BD_v4_fixed.py",
        help="Existing validated CHB-MIT ictal model script",
    )
    p.add_argument("--data_root", default="/data/halders2/EEG_anomaly/Data")
    p.add_argument("--patient", default="chb01")
    p.add_argument(
        "--continuous_edf",
        default="auto",
        help="EDF basename/path to hold out completely, or 'auto'",
    )
    p.add_argument("--list_edfs", action="store_true")
    p.add_argument(
        "--out_dir",
        default="chbmit_continuous_rolling30s_results",
    )

    # Keep manuscript preprocessing/model defaults.
    p.add_argument("--l_freq", type=float, default=0.5)
    p.add_argument("--h_freq", type=float, default=40.0)
    p.add_argument("--segment_seconds", type=float, default=30.0)
    p.add_argument("--ictal_start_seconds", type=float, default=0.0)
    p.add_argument("--ictal_scan_seconds", type=float, default=30.0)
    p.add_argument("--chunk_seconds", type=float, default=5.0)
    p.add_argument("--stride_seconds", type=float, default=1.0)
    p.add_argument("--min_sequence_chunks", type=int, default=20)

    p.add_argument("--interictal_pre_buffer_seconds", type=float, default=3600.0)
    p.add_argument("--interictal_post_buffer_seconds", type=float, default=3600.0)
    p.add_argument("--interictal_scan_stride_seconds", type=float, default=30.0)
    p.add_argument("--max_interictal_scans_per_patient", type=int, default=120)
    p.add_argument("--val_fraction", type=float, default=0.25)
    p.add_argument("--test_fraction", type=float, default=0.25)
    p.add_argument("--min_train_scans", type=int, default=5)
    p.add_argument("--min_val_scans", type=int, default=2)

    p.add_argument("--pca_components", type=int, default=30)
    p.add_argument("--threshold_percentile", type=float, default=5.0)
    p.add_argument("--persistence_seconds", type=float, default=3.0)
    p.add_argument("--detection_window_seconds", type=float, default=30.0)

    p.add_argument("--regime_id", type=int, default=4)
    p.add_argument("--data_seed", type=int, default=17)
    p.add_argument("--res_seed", type=int, default=1107)
    p.add_argument("--num_reservoirs", type=int, default=10)
    p.add_argument("--integration_time", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--tau_min", type=float, default=0.8)
    p.add_argument("--tau_max", type=float, default=2.0)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--washout", type=int, default=0)
    p.add_argument("--pred_horizon", type=int, default=1)
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
