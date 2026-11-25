#!/usr/bin/env python3
"""
Patient-level continuous CHB-MIT EEG-CASCADE evaluation.

Design
------
For one CHB-MIT case folder:
1. Hold out ALL seizure-containing EDFs for continuous testing.
2. Also hold out a deterministic fraction of seizure-free EDFs for continuous FAR.
3. Use only the remaining EDFs for model development.
4. Reuse the original CHB-MIT scan builder on development EDFs, preserving the
   original max-120 interictal pool and 50/25/25 train/validation/sampled-test
   split. The sampled-test portion is ignored in this continuous analysis.
5. Fit StandardScaler, PCA, R4 DynML, and residual Gaussian exactly as in the
   original analysis.
6. Convert each 30-s validation sequence to one total log-likelihood score and
   use its requested percentile as ONE scalar threshold.
7. Process each held-out EDF independently and chronologically; form a rolling
   sum with the same number of log-likelihood steps as a 30-s validation scan.
8. Apply 3-s and/or 5-s persistence continuously.
9. Compute empirical finite-time Bowen-Dinaburg (BD) entropy once per fitted
   patient x regime x reservoir-seed x N model, using development/training states only.
10. Aggregate seizure sensitivity/delay and false-alarm episodes per clean
    monitored hour across held-out EDFs. EDFs are never concatenated.

Requires, in the same directory by default:
  chbmit_dynml_ictal_detection_BD_v4_fixed.py
  chbmit_continuous_rolling30s.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_module(path: Path, name: str):
    path = path.resolve()
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec_module so decorators such as @dataclass can
    # resolve cls.__module__ through sys.modules during module execution.
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


def parse_float_list(text: str) -> List[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one persistence value")
    return vals


def choose_heldout_edfs(edf_df, seizures, no_seizure_fraction: float, seed: int):
    seizure_edfs = sorted({Path(s.edf_path).resolve() for s in seizures})
    all_edfs = [Path(x).resolve() for x in edf_df["edf"].tolist()]
    seizure_set = set(seizure_edfs)
    seizure_free = sorted([p for p in all_edfs if p not in seizure_set], key=lambda p: p.name)

    if not 0.0 <= no_seizure_fraction <= 1.0:
        raise ValueError("--test_noseizure_fraction must be in [0, 1]")

    n_free = len(seizure_free)
    if n_free == 0 or no_seizure_fraction == 0:
        selected_free = []
    else:
        n_hold = int(round(no_seizure_fraction * n_free))
        n_hold = max(1, n_hold)
        # Leave at least one seizure-free EDF for development whenever possible.
        if n_free > 1:
            n_hold = min(n_hold, n_free - 1)
        else:
            n_hold = 0
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_free, size=n_hold, replace=False)) if n_hold else np.array([], dtype=int)
        selected_free = [seizure_free[int(i)] for i in idx]

    heldout = sorted(seizure_edfs + selected_free, key=lambda p: p.name)
    dev = sorted([p for p in all_edfs if p not in set(heldout)], key=lambda p: p.name)
    return heldout, dev, seizure_edfs, selected_free


def subset_edf_df(edf_df: pd.DataFrame, paths: List[Path]) -> pd.DataFrame:
    keep = {p.resolve() for p in paths}
    return edf_df[edf_df["edf"].map(lambda x: Path(x).resolve() in keep)].copy()


def run(args):
    core = load_module(Path(args.core_script), "chbmit_core")
    helper = load_module(Path(args.helper_script), "chbmit_cont_helper")

    patient = args.patient.strip().lower()
    patient_dir = Path(args.data_root) / patient
    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

    edf_df, seizures = core.index_patient(patient_dir, patient)
    if edf_df.empty:
        raise RuntimeError(f"No EDFs found for {patient}")

    heldout_edfs, dev_edfs, seizure_edfs, heldout_free = choose_heldout_edfs(
        edf_df, seizures, args.test_noseizure_fraction, args.data_seed
    )
    if not heldout_edfs:
        raise RuntimeError(f"{patient}: no held-out EDFs selected")
    if not dev_edfs:
        raise RuntimeError(f"{patient}: no development EDFs remain")

    dev_df = subset_edf_df(edf_df, dev_edfs)
    dev_set = {p.resolve() for p in dev_edfs}
    dev_seizures = [s for s in seizures if Path(s.edf_path).resolve() in dev_set]

    # Preserve the original 50/25/25 split within development data. The sampled
    # development-test scans are deliberately ignored; continuous whole-EDF data
    # are the test set for this analysis.
    train_scans, val_scans, unused_sampled_test_scans, _ = core.make_patient_scans(
        patient, dev_df, dev_seizures, args
    )

    train_raw = core.scans_to_features(train_scans, args)
    val_raw = core.scans_to_features(val_scans, args)

    if len(train_raw) < args.min_train_scans:
        raise RuntimeError(f"{patient}: insufficient train sequences ({len(train_raw)})")
    if len(val_raw) < args.min_val_scans:
        raise RuntimeError(f"{patient}: insufficient validation sequences ({len(val_raw)})")

    train_mat = np.vstack([s.features for s in train_raw])
    scaler = core.StandardScaler().fit(train_mat)
    pca_n = min(int(args.pca_components), train_mat.shape[1], max(1, train_mat.shape[0] - 1))
    pca = core.PCA(n_components=pca_n, random_state=args.data_seed).fit(scaler.transform(train_mat))

    train = core.transform_sequences(train_raw, scaler, pca)
    val = core.transform_sequences(val_raw, scaler, pca)

    Xtr, Ytr = core.collect_xy(train, args.washout, args.pred_horizon)
    model = core.RosslerDynML(pca_n, args)
    model.fit(Xtr, Ytr)

    mu, var = core.fit_diag_gaussian_from_validation(val, model, args.washout, args.pred_horizon)
    threshold, rolling_steps, validation_scores = helper.validation_scalar_threshold(
        core, val, model, mu, var, args.washout, args.pred_horizon,
        args.threshold_percentile, args.chunk_seconds
    )

    # Empirical finite-time Bowen-Dinaburg entropy.
    # This is computed ONCE for the fitted patient x regime x seed x N model,
    # from development/training-derived PCA states Xtr. It does not use held-out
    # continuous EDFs and is not recomputed for each persistence value.
    bd_rng = np.random.default_rng(args.res_seed + 104729)
    bd = model.bd_entropy(
        Xtr,
        q=args.bd_q,
        eps=args.bd_epsilon,
        n_steps=args.bd_steps,
        greedy_repeats=args.bd_greedy_repeats,
        rng=bd_rng,
    )
    param_summary = model.parameter_summary()
    proj_diag = model.projection_diagnostics(Xtr)

    out_root = Path(args.out_dir) / patient / f"seed_{args.res_seed}"
    out_root.mkdir(parents=True, exist_ok=True)

    bd_row = {
        "patient": patient,
        "analysis": "chbmit_patient_level_continuous_rolling30s_bd",
        "data_seed": args.data_seed,
        "res_seed": args.res_seed,
        "regime_id": args.regime_id,
        "regime_name": model.regime["name"],
        "num_reservoirs": model.N,
        "pca_components": pca_n,
        "integration_time": args.integration_time,
        "dt": args.dt,
        "tau_min": args.tau_min,
        "tau_max": args.tau_max,
        "train_scans": int(len(train_raw)),
        "validation_scans": int(len(val_raw)),
        **bd,
        **proj_diag,
        **param_summary,
    }
    pd.DataFrame([bd_row]).to_csv(
        out_root / f"{patient}_rossler_bd_entropy.csv", index=False
    )
    (out_root / f"{patient}_rossler_bd_entropy.json").write_text(
        json.dumps(bd_row, indent=2, sort_keys=True)
    )

    if args.bd_only:
        print("=" * 80)
        print("CHB-MIT CONTINUOUS MODEL: BD-ONLY MODE")
        print(f"Patient:             {patient}")
        print(f"Regime:              R{args.regime_id} ({model.regime['name']})")
        print(f"Reservoir seed:      {args.res_seed}")
        print(f"Reservoir count N:   {model.N}")
        print(f"BD entropy:          {bd['bd_entropy']}")
        print(f"Separated count:     {bd['bd_separated_count']}/{bd['bd_q']}")
        print(f"BD saturated:        {bd['bd_saturated']}")
        print(f"Saved to:            {out_root.resolve()}")
        print("=" * 80)
        return

    split_rows = []
    seizure_names = {p.name for p in seizure_edfs}
    free_names = {p.name for p in heldout_free}
    dev_names = {p.name for p in dev_edfs}
    for _, row in edf_df.iterrows():
        name = str(row["edf_name"])
        if name in seizure_names:
            role = "continuous_test_seizure_edf"
        elif name in free_names:
            role = "continuous_test_seizure_free_edf"
        elif name in dev_names:
            role = "development_edf"
        else:
            role = "other"
        split_rows.append({
            "patient": patient,
            "edf_name": name,
            "duration_sec": float(row["duration_sec"]),
            "role": role,
        })
    pd.DataFrame(split_rows).to_csv(out_root / "edf_split.csv", index=False)
    pd.DataFrame({"validation_30s_score": validation_scores}).to_csv(
        out_root / "validation_30s_scores.csv", index=False
    )

    heldout_set = {p.resolve() for p in heldout_edfs}
    heldout_seizures = [s for s in seizures if Path(s.edf_path).resolve() in heldout_set]
    seizures_by_edf: Dict[Path, List] = {}
    for s in heldout_seizures:
        seizures_by_edf.setdefault(Path(s.edf_path).resolve(), []).append(s)

    persistence_values = parse_float_list(args.persistence_seconds)
    per_persistence = {p: {"edf": [], "alarms": [], "seizures": []} for p in persistence_values}

    print("=" * 80)
    print("CHB-MIT PATIENT-LEVEL CONTINUOUS EEG-CASCADE")
    print(f"Patient:                         {patient}")
    print(f"Total EDFs:                      {len(edf_df)}")
    print(f"Seizure-containing held-out:     {len(seizure_edfs)}")
    print(f"Seizure-free held-out:           {len(heldout_free)}")
    print(f"Development EDFs:                {len(dev_edfs)}")
    print(f"Development train scans:         {len(train_raw)}")
    print(f"Development validation scans:    {len(val_raw)}")
    print(f"Unused sampled dev-test scans:   {len(unused_sampled_test_scans)}")
    print(f"PCA components:                  {pca_n}")
    print(f"Rössler regime:                  R{args.regime_id}")
    print(f"Reservoir seed:                  {args.res_seed}")
    print(f"Rolling LL steps:                {rolling_steps}")
    print(f"Validation threshold:            {threshold:.6f}")
    print("=" * 80)

    # Process each EDF independently. No synthetic concatenation across files.
    for edf_path in heldout_edfs:
        row = edf_df[edf_df["edf"].map(lambda x: Path(x).resolve() == edf_path.resolve())]
        if row.empty:
            raise RuntimeError(f"Missing EDF metadata for {edf_path}")
        duration = float(row.iloc[0]["duration_sec"])
        seq_raw = helper.build_continuous_sequence(core, patient, edf_path, duration, args)
        seq = core.transform_sequences([seq_raw], scaler, pca)[0]
        ll, ll_times = core.seq_loglik(
            seq, model, mu, var, args.washout, args.pred_horizon, args.chunk_seconds
        )
        scores, score_times = helper.rolling_sum(ll, ll_times, rolling_steps)
        if len(scores) == 0:
            print(f"[WARN] {edf_path.name}: no rolling scores")
            continue

        seizures_here = sorted(seizures_by_edf.get(edf_path.resolve(), []), key=lambda s: s.onset_sec)

        trace_base = pd.DataFrame({
            "time_sec": score_times,
            "rolling_loglik_30s": scores,
            "threshold": threshold,
        })
        trace_base.to_csv(out_root / f"{edf_path.stem}_rolling_trace.csv", index=False)

        for pers in persistence_values:
            psteps = max(1, int(math.ceil(float(pers) / float(args.stride_seconds))))
            flags, episodes = helper.persistence_episodes(scores, score_times, threshold, psteps)
            alarms_df = pd.DataFrame(
                episodes,
                columns=["first_crossing_time_sec", "confirmed_alarm_time_sec", "score_at_confirmation"],
            )
            classified, seizure_df, clean_intervals, metrics = helper.classify_alarms(
                alarms_df, seizures_here, duration, core, args, args.detection_window_seconds
            )

            role = "seizure_edf" if seizures_here else "seizure_free_edf"
            per_persistence[pers]["edf"].append({
                "patient": patient,
                "res_seed": args.res_seed,
                "persistence_seconds": pers,
                "edf_name": edf_path.name,
                "role": role,
                "duration_seconds": duration,
                **metrics,
            })

            if not classified.empty:
                c = classified.copy()
                c.insert(0, "edf_name", edf_path.name)
                c.insert(0, "patient", patient)
                per_persistence[pers]["alarms"].append(c)
            if not seizure_df.empty:
                s = seizure_df.copy()
                s.insert(0, "edf_name", edf_path.name)
                s.insert(0, "patient", patient)
                per_persistence[pers]["seizures"].append(s)

    all_summaries = []
    for pers in persistence_values:
        edf_metrics = pd.DataFrame(per_persistence[pers]["edf"])
        alarms_all = pd.concat(per_persistence[pers]["alarms"], ignore_index=True) if per_persistence[pers]["alarms"] else pd.DataFrame()
        seizures_all = pd.concat(per_persistence[pers]["seizures"], ignore_index=True) if per_persistence[pers]["seizures"] else pd.DataFrame()

        total_seizures = int(edf_metrics["n_seizures"].sum()) if len(edf_metrics) else 0
        total_detected = int(edf_metrics["n_detected_seizures"].sum()) if len(edf_metrics) else 0
        sensitivity = total_detected / total_seizures if total_seizures else math.nan
        clean_hours = float(edf_metrics["clean_interictal_hours"].sum()) if len(edf_metrics) else 0.0
        false_alarms = int(edf_metrics["interictal_false_alarm_episodes"].sum()) if len(edf_metrics) else 0
        far = false_alarms / clean_hours if clean_hours > 0 else math.nan

        if len(seizures_all):
            delays = seizures_all.loc[seizures_all["detected"], "detection_delay_seconds"].to_numpy(dtype=float)
        else:
            delays = np.asarray([], dtype=float)

        summary = {
            "patient": patient,
            "analysis": "chbmit_patient_level_continuous_rolling30s",
            "data_seed": args.data_seed,
            "res_seed": args.res_seed,
            "regime_id": args.regime_id,
            "regime_name": model.regime["name"],
            "num_reservoirs": model.N,
            "pca_components": pca_n,
            "threshold_percentile": args.threshold_percentile,
            "scalar_validation_threshold": threshold,
            "rolling_steps": rolling_steps,
            "rolling_span_nominal_seconds": args.segment_seconds,
            "persistence_seconds": pers,
            "detection_window_seconds": args.detection_window_seconds,
            "n_total_edfs": int(len(edf_df)),
            "n_development_edfs": int(len(dev_edfs)),
            "n_continuous_test_edfs": int(len(heldout_edfs)),
            "n_test_seizure_edfs": int(len(seizure_edfs)),
            "n_test_seizure_free_edfs": int(len(heldout_free)),
            "train_scans": int(len(train_raw)),
            "validation_scans": int(len(val_raw)),
            "unused_sampled_dev_test_scans": int(len(unused_sampled_test_scans)),
            "n_seizures": total_seizures,
            "n_detected_seizures": total_detected,
            "seizure_sensitivity": sensitivity,
            "mean_detection_delay_seconds": float(np.mean(delays)) if len(delays) else math.nan,
            "median_detection_delay_seconds": float(np.median(delays)) if len(delays) else math.nan,
            "clean_interictal_hours": clean_hours,
            "interictal_false_alarm_episodes": false_alarms,
            "false_alarms_per_hour": far,
            **bd,
            **proj_diag,
            **param_summary,
        }
        all_summaries.append(summary)

        suf = str(pers).replace(".", "p")
        edf_metrics.to_csv(out_root / f"per_edf_metrics_pers{suf}s.csv", index=False)
        alarms_all.to_csv(out_root / f"all_alarm_episodes_pers{suf}s.csv", index=False)
        seizures_all.to_csv(out_root / f"all_seizure_detections_pers{suf}s.csv", index=False)

        print("\n" + "-" * 80)
        print(f"PATIENT SUMMARY | persistence={pers:g} s")
        print(f"Seizures detected:       {total_detected}/{total_seizures}")
        print(f"Sensitivity:             {sensitivity}")
        print(f"Mean detection delay:    {summary['mean_detection_delay_seconds']}")
        print(f"Clean interictal hours:  {clean_hours:.3f}")
        print(f"False-alarm episodes:    {false_alarms}")
        print(f"False alarms/hour:       {far}")
        print("-" * 80)

    pd.DataFrame(all_summaries).to_csv(out_root / "patient_continuous_summary.csv", index=False)
    (out_root / "patient_continuous_summary.json").write_text(json.dumps(all_summaries, indent=2, sort_keys=True))
    print(f"\nSaved patient-level outputs to: {out_root.resolve()}")


def parse_args():
    p = argparse.ArgumentParser(description="CHB-MIT patient-level continuous rolling-30s evaluation")
    p.add_argument("--core_script", default="chbmit_dynml_ictal_detection_BD_v4_fixed.py")
    p.add_argument("--helper_script", default="chbmit_continuous_rolling30s.py")
    p.add_argument("--data_root", default="/data/halders2/EEG_anomaly/Data")
    p.add_argument("--patient", default="chb01")
    p.add_argument("--out_dir", default="chbmit_continuous_patient_level_results")

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
    p.add_argument("--persistence_seconds", default="3,5")
    p.add_argument("--detection_window_seconds", type=float, default=30.0)
    p.add_argument("--test_noseizure_fraction", type=float, default=0.25)

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

    # Empirical finite-time Bowen-Dinaburg entropy settings.
    # Defaults match the BD-enabled sampled CHB-MIT implementation.
    p.add_argument("--bd_q", type=int, default=64)
    p.add_argument("--bd_epsilon", type=float, default=0.05)
    p.add_argument("--bd_steps", type=int, default=30)
    p.add_argument("--bd_greedy_repeats", type=int, default=25)
    p.add_argument(
        "--bd_only",
        action="store_true",
        help=(
            "Fit the development model and compute/save BD entropy, then stop "
            "before continuous held-out EDF evaluation. Useful when continuous "
            "performance results already exist."
        ),
    )

    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
