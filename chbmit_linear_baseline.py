#!/usr/bin/env python3
"""Direct ridge forecasting baseline using the supplied CHB-MIT v4 core.

No source edits, reservoir construction, ODE integration, or BD calculation.
Additional core-script arguments can be passed after the wrapper arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd


class LinearRidge:
    """Y = X @ W, no intercept; same unnormalized ridge objective as v4."""

    def __init__(self, ridge):
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be positive and finite")
        self.ridge = float(ridge)
        self.Wout = None

    def fit(self, X, Y):
        X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
        if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y) or not len(X):
            raise ValueError("Invalid training arrays")
        if not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise ValueError("Nonfinite training values")
        A = X.T @ X
        A.flat[::A.shape[0] + 1] += self.ridge
        self.Wout = np.linalg.solve(A, X.T @ Y)
        return self

    def predict(self, X):
        if self.Wout is None:
            raise RuntimeError("Linear baseline has not been fitted")
        return np.asarray(X) @ self.Wout


def load_core(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("linear_baseline_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    return core


def overlap_count(left, right):
    return sum(
        Path(a.edf_path).resolve() == Path(b.edf_path).resolve()
        and max(a.start_sec, b.start_sec) < min(a.end_sec, b.end_sec)
        for a in left for b in right
    )


def scan_id(scan):
    return scan.seizure_id or f"{Path(scan.edf_path).stem}_{scan.start_sec:.3f}"


def prepare(core, args, patient, out):
    patient_dir = Path(args.data_root) / patient
    if not patient_dir.is_dir():
        raise FileNotFoundError(patient_dir)
    index, seizures = core.index_patient(patient_dir, patient)
    if index.empty:
        raise RuntimeError("No indexed EDFs")
    train, val, test, positive = core.make_patient_scans(patient, index, seizures, args)
    scans = dict(train=train, validation=val, test=test, positive=positive)
    names = list(scans)
    audit = [dict(left=a, right=b, overlapping_pairs=overlap_count(scans[a], scans[b]))
             for i, a in enumerate(names) for b in names[i + 1:]]
    pd.DataFrame(audit).to_csv(out / "selected_window_overlap_audit.csv", index=False)
    if any(r["overlapping_pairs"] for r in audit):
        raise RuntimeError("Cross-split selected-window overlap; see audit CSV")

    raw, manifest = {}, []
    for name, selected in scans.items():
        raw[name] = core.scans_to_features(selected, args)
        kept_ids = {s.scan_id for s in raw[name]}
        selected_ids = [scan_id(s) for s in selected]
        if len(set(selected_ids)) != len(selected_ids):
            raise RuntimeError(f"Duplicate selected IDs in {name}")
        if len(kept_ids) != len(raw[name]):
            raise RuntimeError(f"Duplicate feature sequence IDs in {name}")
        for s, sid in zip(selected, selected_ids):
            manifest.append(dict(patient=patient, split=name, edf=str(s.edf_path),
                                 start_sec=s.start_sec, end_sec=s.end_sec,
                                 sequence_id=sid, retained=sid in kept_ids))
        print(f"{patient} {name}: selected={len(selected)} retained={len(raw[name])}", flush=True)
    pd.DataFrame(manifest).to_csv(out / "window_manifest.csv", index=False)
    if len(raw["train"]) < args.min_train_scans or len(raw["validation"]) < args.min_val_scans:
        raise RuntimeError("Insufficient retained train/validation sequences")
    if not raw["test"] or not raw["positive"]:
        raise RuntimeError("No retained negative or positive test sequences")

    train_mat = np.vstack([s.features for s in raw["train"]])
    scaler = core.StandardScaler().fit(train_mat)
    d = min(int(args.pca_components), train_mat.shape[1], max(1, train_mat.shape[0] - 1))
    pca = core.PCA(n_components=d, random_state=args.data_seed).fit(scaler.transform(train_mat))
    transformed = {name: core.transform_sequences(seqs, scaler, pca)
                   for name, seqs in raw.items()}
    return transformed, scaler, pca


def evaluate(core, args, task, patient, seqs, model, mu, var, threshold, persistence, out):
    steps = max(1, int(math.ceil(persistence / args.stride_seconds)))
    neg_rows, pos_rows, timing = [], [], []
    fp_scans = fp_episodes = tp = 0
    hours = 0.0
    for s in seqs["test"]:
        ll, times = core.seq_loglik(s, model, mu, var, args.washout, args.pred_horizon, args.chunk_seconds)
        alarm, first, confirmed, nflag = core.alarm_from_ll(ll, times, threshold, steps)
        episodes = core.alarm_episode_count(ll, threshold, steps)
        fp_scans += int(alarm)
        fp_episodes += episodes
        if len(s.chunk_starts_sec):
            hours += max(0.0, s.chunk_starts_sec[-1] + args.chunk_seconds - s.chunk_starts_sec[0]) / 3600.0
        neg_rows.append(dict(patient=patient, scan_id=s.scan_id, alarm=bool(alarm),
                             alarm_episodes=episodes, first_crossing_time_sec=first,
                             confirmed_alarm_time_sec=confirmed, n_flagged_steps=nflag))
    for s in seqs["positive"]:
        ll, times = core.seq_loglik(s, model, mu, var, args.washout, args.pred_horizon, args.chunk_seconds)
        alarm, first, confirmed, nflag = core.alarm_from_ll(ll, times, threshold, steps)
        if task == "ictal":
            value = float(confirmed - s.onset_sec) if alarm and confirmed is not None else math.nan
            valid = bool(np.isfinite(value) and value >= 0)
            timing_col = "detection_delay_seconds"
        else:
            valid = bool(alarm and confirmed is not None and np.isfinite(s.onset_sec) and confirmed < s.onset_sec)
            value = float(s.onset_sec - confirmed) if valid else math.nan
            timing_col = "warning_lead_time_seconds"
        if valid:
            tp += 1
            timing.append(value)
        row = dict(patient=patient, seizure_id=s.scan_id, onset_sec=s.onset_sec,
                   alarm=valid, raw_alarm=bool(alarm), first_crossing_time_sec=first,
                   confirmed_alarm_time_sec=confirmed, n_flagged_steps=nflag,
                   alarm_persistence_seconds=persistence)
        row[timing_col] = value if valid else math.nan
        if task == "preictal":
            row["warning_lead_time_minutes"] = value / 60.0 if valid else math.nan
        pos_rows.append(row)
    npos, nneg = len(seqs["positive"]), len(seqs["test"])
    sens, spec = tp / npos, (nneg - fp_scans) / nneg
    summary = dict(patient=patient, model="direct_linear_ridge", task=task,
                   data_seed=args.data_seed, ridge=args.ridge, fit_intercept=False,
                   alarm_persistence_seconds=persistence, pca_components=len(mu),
                   train_scans=len(seqs["train"]), val_scans=len(seqs["validation"]),
                   interictal_test_scans=nneg, interictal_fp_scans=fp_scans,
                   interictal_tn_scans=nneg - fp_scans, interictal_test_hours=hours,
                   false_alarm_episodes=fp_episodes, false_alarms_per_hour=fp_episodes / hours,
                   sensitivity=sens, specificity=spec, balanced_accuracy=(sens + spec) / 2,
                   positive_total=npos, positive_tp=tp, positive_fn=npos - tp)
    mean_t = float(np.mean(timing)) if timing else math.nan
    median_t = float(np.median(timing)) if timing else math.nan
    if task == "ictal":
        summary.update(ictal_total=npos, ictal_tp=tp, ictal_fn=npos - tp,
                       ictal_sensitivity=sens, ictal_delay_mean_seconds=mean_t,
                       ictal_delay_median_seconds=median_t)
    else:
        summary.update(preictal_total=npos, preictal_tp=tp, preictal_fn=npos - tp,
                       preictal_sensitivity=sens, warning_lead_mean_seconds=mean_t,
                       warning_lead_median_seconds=median_t,
                       warning_lead_mean_minutes=mean_t / 60.0,
                       warning_lead_median_minutes=median_t / 60.0)
    suffix = str(float(persistence)).replace(".", "p")
    pd.DataFrame(neg_rows).to_csv(out / f"{patient}_interictal_test_scans_pers{suffix}s.csv", index=False)
    pd.DataFrame(pos_rows).to_csv(out / f"{patient}_{task}_events_pers{suffix}s.csv", index=False)
    return summary


def run_case(core, args, task, patient, output_root, source):
    out = Path(output_root).resolve() / task / patient
    out.mkdir(parents=True, exist_ok=True)
    # Do not mix a rerun with stale summaries from an earlier attempt.
    summary_path = out / "all_patient_summary.csv"
    if summary_path.exists():
        raise FileExistsError(f"Existing baseline summary: {summary_path}; choose a new --output_root")
    try:
        metadata = dict(model="direct_linear_ridge", task=task, arguments=vars(args),
                        core_script=str(source), core_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        numpy_version=np.__version__, pandas_version=pd.__version__)
        (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
        seqs, scaler, pca = prepare(core, args, patient, out)
        X, Y = core.collect_xy(seqs["train"], args.washout, args.pred_horizon)
        model = LinearRidge(args.ridge).fit(X, Y)
        mu, var = core.fit_diag_gaussian_from_validation(seqs["validation"], model, args.washout, args.pred_horizon)
        threshold = core.validation_threshold(seqs["validation"], model, mu, var, args.washout,
                                              args.pred_horizon, args.threshold_percentile, args.chunk_seconds)
        np.savez_compressed(out / "linear_model.npz", W=model.Wout, scaler_mean=scaler.mean_,
                            scaler_scale=scaler.scale_, pca_components=pca.components_, pca_mean=pca.mean_,
                            residual_mu=mu, residual_var=var, validation_threshold=threshold)
        summaries = [evaluate(core, args, task, patient, seqs, model, mu, var, threshold, p, out)
                     for p in core.parse_float_list(args.alarm_persistence_seconds)]
        pd.DataFrame(summaries).to_csv(summary_path, index=False)
        print(pd.DataFrame(summaries).to_string(index=False), flush=True)
        print(f"Saved: {summary_path}", flush=True)
    except Exception:
        (out / "error.txt").write_text(traceback.format_exc())
        raise


def main():
    wrapper = argparse.ArgumentParser(description=__doc__, add_help=False)
    wrapper.add_argument("--task", required=True, choices=["ictal", "preictal"])
    wrapper.add_argument("--core_script", required=True)
    wrapper.add_argument("--output_root", default="chbmit_linear_baseline_results")
    opts, remainder = wrapper.parse_known_args()
    if any(s == "--out_dir" or s.startswith("--out_dir=") for s in remainder):
        raise ValueError("Use --output_root, not the core script's --out_dir")
    source = Path(opts.core_script).resolve()
    core = load_core(source)
    if opts.task == "preictal" and not any(
        s == "--interictal_scan_stride_seconds" or s.startswith("--interictal_scan_stride_seconds=")
        for s in remainder
    ):
        remainder += ["--interictal_scan_stride_seconds", "1800"]
    old_argv = sys.argv
    try:
        sys.argv = [str(source)] + remainder
        args = core.parse_args()
    finally:
        sys.argv = old_argv
    expected_seconds = 30.0 if opts.task == "ictal" else 1800.0
    if args.segment_seconds != expected_seconds or args.pred_horizon != 1:
        raise ValueError("Task/core mismatch: expected matching sequence duration and h=1")
    if opts.task == "preictal" and args.interictal_scan_stride_seconds < args.segment_seconds:
        raise ValueError("Preictal baseline requires nonoverlapping interictal selection")
    for patient in args.patient.split(","):
        run_case(core, args, opts.task, patient.strip().lower(), opts.output_root, source)


if __name__ == "__main__":
    main()
