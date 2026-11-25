#!/usr/bin/env python3
"""
CHB-MIT patient-specific ictal anomaly detection using Rössler DynML
and empirical finite-time Bowen-Dinaburg separated-orbit entropy.

Design
------
- Train: clean interictal 30-s scans only.
- Validation: clean interictal 30-s scans only.
  * fit diagonal Gaussian residual model
  * build time-matched cumulative-loglikelihood thresholds
- Negative test: held-out clean interictal 30-s scans.
- Positive test: fixed 30-s windows beginning at annotated seizure onset.
- EEG features: 18-channel common bipolar montage x 15 engineered features = 270.
- StandardScaler and PCA are fit on training interictal data only.
- DynML input projection: z0 = R x, with NO tanh and NO additive bias.
- Each reservoir unit is an independent Rössler system with sampled (a,b,c,tau).
- Final 3N-dimensional reservoir state alone is used by the linear readout.
- Multiple persistence values can be evaluated in a single model run so that the
  expensive EEG preprocessing / DynML fitting / BD calculation is not repeated.

Empirical BD entropy
--------------------
For Q training-derived input states, the same random projection and Rössler
reservoir used by DynML generate Q finite-time trajectories. The normalized
Bowen-Dinaburg distance is

    d_n(i,j) = max_t ||z_i(t)-z_j(t)||_2 / sqrt(3N)

and the separated-set size is approximated by repeated randomized greedy
selection. The reported finite-time estimate is

    h_BD(n, eps) = log(S_Q(n, eps)) / n.

This is a finite-time empirical diagnostic, not an exact topological entropy.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import mne
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires mne.") from exc

from scipy.integrate import solve_ivp
from scipy.signal import welch
from scipy.stats import entropy as scipy_entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# EEG configuration
# -----------------------------------------------------------------------------
COMMON18 = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


# -----------------------------------------------------------------------------
# Six candidate Rössler dynamical regimes.
# Fixed N and fixed readout dimension across all regimes.
# These ranges are intended to span low-complexity/periodic neighborhoods
# through canonical and broader chaotic neighborhoods. The empirical BD result,
# not the regime label, is used to quantify finite-time dynamical richness.
# -----------------------------------------------------------------------------
# ROSSLER_REGIMES = {
#     0:  dict(name="low_complexity",        a=(0.08, 0.12), b=(0.08, 0.12), c=(1.80, 2.30)),
#     1:  dict(name="low_periodic",          a=(0.12, 0.16), b=(0.10, 0.14), c=(2.80, 3.40)),
#     2:  dict(name="higher_periodic",       a=(0.16, 0.20), b=(0.14, 0.18), c=(3.80, 4.40)),
#     3:  dict(name="transition",            a=(0.18, 0.22), b=(0.18, 0.22), c=(4.70, 5.20)),
#     4:  dict(name="canonical_chaotic",     a=(0.18, 0.22), b=(0.18, 0.22), c=(5.50, 5.90)),
#     5:  dict(name="broader_chaotic_test",  a=(0.30, 0.36), b=(0.18, 0.22), c=(5.50, 5.90)),
#     6:  dict(name="very_low_test",         a=(0.05, 0.08), b=(0.05, 0.08), c=(1.00, 1.50)),
#     7:  dict(name="low_test",              a=(0.07, 0.10), b=(0.07, 0.10), c=(1.50, 2.00)),
#     8:  dict(name="midlow_test",           a=(0.10, 0.14), b=(0.08, 0.12), c=(3.00, 3.50)),
#     9:  dict(name="chaotic_test_1",        a=(0.22, 0.26), b=(0.18, 0.22), c=(5.50, 6.00)),
#     10: dict(name="chaotic_test_2",        a=(0.36, 0.42), b=(0.18, 0.22), c=(5.50, 6.00)),
#     11: dict(name="chaotic_test_3",        a=(0.42, 0.50), b=(0.18, 0.22), c=(5.50, 6.00)),
# }

ROSSLER_REGIMES = {
    0:  dict(name="low_complexity",        a=(0.08, 0.12), b=(0.08, 0.12), c=(1.80, 2.30)),
    1:  dict(name="low_periodic",          a=(0.12, 0.16), b=(0.10, 0.14), c=(2.80, 3.40)),
    2:  dict(name="higher_periodic",       a=(0.16, 0.20), b=(0.14, 0.18), c=(3.80, 4.40)),
    3:  dict(name="transition",            a=(0.18, 0.22), b=(0.18, 0.22), c=(4.70, 5.20)),
    4:  dict(name="canonical_chaotic",     a=(0.18, 0.22), b=(0.18, 0.22), c=(5.50, 5.90)),
    5:  dict(name="broader_chaotic_test",  a=(0.30, 0.36), b=(0.18, 0.22), c=(5.50, 5.90)),
    6:  dict(name="very_low_test",         a=(0.05, 0.08), b=(0.05, 0.08), c=(1.00, 1.50)),
    7:  dict(name="low_test",              a=(0.07, 0.10), b=(0.07, 0.10), c=(1.50, 2.00)),
    8:  dict(name="midlow_test",           a=(0.10, 0.14), b=(0.08, 0.12), c=(3.00, 3.50)),
    9:  dict(name="chaotic_test_1",        a=(0.22, 0.26), b=(0.18, 0.22), c=(5.50, 6.00)),
    10: dict(name="chaotic_test_2",        a=(0.36, 0.42), b=(0.18, 0.22), c=(5.50, 6.00)),
    11: dict(name="chaotic_test_3",        a=(0.42, 0.50), b=(0.18, 0.22), c=(5.50, 6.00)),

    12: dict(name="R4_a_left",             a=(0.16, 0.20), b=(0.18, 0.22), c=(5.50, 5.90)),
    13: dict(name="R4_a_right",            a=(0.20, 0.24), b=(0.18, 0.22), c=(5.50, 5.90)),
    14: dict(name="R4_b_left",             a=(0.18, 0.22), b=(0.16, 0.20), c=(5.50, 5.90)),
    15: dict(name="R4_b_right",            a=(0.18, 0.22), b=(0.20, 0.24), c=(5.50, 5.90)),
    16: dict(name="R4_c_left",             a=(0.18, 0.22), b=(0.18, 0.22), c=(5.30, 5.70)),
    17: dict(name="R4_c_right",            a=(0.18, 0.22), b=(0.18, 0.22), c=(5.70, 6.10)),

    18: dict(name="a_mid_1", a=(0.24, 0.30), b=(0.18, 0.22), c=(5.50, 6.00)),
    19: dict(name="a_mid_2", a=(0.28, 0.34), b=(0.18, 0.22), c=(5.50, 6.00)),
    20: dict(name="a_mid_3", a=(0.32, 0.38), b=(0.18, 0.22), c=(5.50, 6.00)),
    21: dict(name="a_mid_4", a=(0.38, 0.44), b=(0.18, 0.22), c=(5.50, 6.00)),
    22: dict(name="a_high_1", a=(0.46, 0.54), b=(0.18, 0.22), c=(5.50, 6.00)),
}


@dataclass
class Seizure:
    patient: str
    edf_path: Path
    seizure_id: str
    onset_sec: float
    end_sec: float


@dataclass
class Scan:
    patient: str
    edf_path: Path
    kind: str
    start_sec: float
    end_sec: float
    seizure_id: str = ""
    onset_sec: float = math.nan
    seizure_end_sec: float = math.nan


@dataclass
class FeatureSequence:
    patient: str
    kind: str
    scan_id: str
    chunk_starts_sec: np.ndarray
    features: np.ndarray
    onset_sec: float = math.nan


# -----------------------------------------------------------------------------
# CHB-MIT annotation / channel handling
# -----------------------------------------------------------------------------
def norm_ch(name: str) -> str:
    s = str(name).upper().strip()
    s = s.replace("EEG", "").replace(" ", "").replace("_", "").replace(".", "")
    s = s.replace("--", "-")
    s = s.replace("T3", "T7").replace("T4", "T8").replace("T5", "P7").replace("T6", "P8")
    for suffix in ["-REF", "-LE", "-AR", "-AVG", "-A1", "-A2"]:
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    s = s.replace("REF", "")
    s = re.sub(r"-\d+$", "", s)
    return s.strip("-")


def patient_from_path(path: Path) -> str:
    p = path.parent.name
    if not re.match(r"chb\d+", p, re.I):
        p = next((x for x in path.parts if re.match(r"chb\d+", x, re.I)), path.parent.name)
    return p.lower()


def parse_chb_summary(summary_path: Path, patient: str) -> List[Seizure]:
    text = summary_path.read_text(errors="ignore").splitlines()
    file_re = re.compile(r"File Name:\s*(\S+\.edf)", re.I)
    numbered_re = re.compile(r"Seizure\s+(\d+)\s+(Start|End)\s+Time:\s*([0-9.]+)\s*seconds", re.I)
    unnumbered_re = re.compile(r"Seizure\s+(Start|End)\s+Time:\s*([0-9.]+)\s*seconds", re.I)

    seizures: List[Seizure] = []
    current_file: Optional[str] = None
    starts: Dict[int, float] = {}
    ends: Dict[int, float] = {}
    unnumbered_idx = 1

    def flush() -> None:
        nonlocal starts, ends, unnumbered_idx
        if current_file is None:
            return
        for k in sorted(set(starts) & set(ends)):
            st = float(starts[k])
            en = float(ends[k])
            if en > st:
                edf_path = summary_path.parent / current_file
                seizures.append(
                    Seizure(
                        patient=patient,
                        edf_path=edf_path,
                        seizure_id=f"{patient}_{Path(current_file).stem}_s{k:03d}",
                        onset_sec=st,
                        end_sec=en,
                    )
                )
        starts = {}
        ends = {}
        unnumbered_idx = 1

    for line in text:
        m = file_re.search(line)
        if m:
            flush()
            current_file = m.group(1).strip()
            continue
        if current_file is None:
            continue

        m = numbered_re.search(line)
        if m:
            idx = int(m.group(1))
            typ = m.group(2).lower()
            val = float(m.group(3))
            (starts if typ == "start" else ends)[idx] = val
            continue

        m = unnumbered_re.search(line)
        if m:
            typ = m.group(1).lower()
            val = float(m.group(2))
            if typ == "start":
                starts[unnumbered_idx] = val
            else:
                ends[unnumbered_idx] = val
                unnumbered_idx += 1

    flush()
    return seizures


def index_patient(patient_dir: Path, patient: str) -> Tuple[pd.DataFrame, List[Seizure]]:
    summaries = sorted(patient_dir.glob("*summary*.txt"))
    seizures: List[Seizure] = []
    for summ in summaries:
        seizures.extend(parse_chb_summary(summ, patient))

    rows = []
    for edf in sorted(patient_dir.glob("*.edf")):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.read_raw_edf(str(edf), preload=False, verbose="ERROR")
            sfreq = float(raw.info["sfreq"])
            duration = float(raw.n_times / sfreq)
            n_channels = len(raw.ch_names)
        except Exception:
            sfreq = math.nan
            duration = math.nan
            n_channels = math.nan
        rows.append({
            "patient": patient,
            "edf": str(edf),
            "edf_name": edf.name,
            "sfreq": sfreq,
            "duration_sec": duration,
            "n_channels": n_channels,
        })
    return pd.DataFrame(rows), seizures


def load_common18_edf(edf_path: Path, l_freq: float, h_freq: float) -> Tuple[np.ndarray, float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
        raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose="ERROR")

    sfreq = float(raw.info["sfreq"])
    duration = float(raw.n_times / sfreq)

    direct = {norm_ch(ch): ch for ch in raw.ch_names}

    electrodes: Dict[str, str] = {}
    for ch in raw.ch_names:
        key = norm_ch(ch)

        # Interpret the apparent "01" label as O1 in these two recordings.
        if (
            edf_path.name.lower() in {"chb12_28.edf", "chb12_29.edf"}
            and key == "01"
        ):
            key = "O1"

        if "-" not in key and key not in electrodes:
            electrodes[key] = ch

    rows = []
    missing = []
    for pair in COMMON18:
        key = norm_ch(pair)
        if key in direct:
            rows.append(raw.get_data(picks=[direct[key]])[0])
            continue
        a, b = key.split("-")

        if a in electrodes and b in electrodes:
            da = raw.get_data(picks=[electrodes[a]])[0]
            db = raw.get_data(picks=[electrodes[b]])[0]
            rows.append(da - db)

        elif f"{a}-CS2" in direct and f"{b}-CS2" in direct:
            # Reconstruct the bipolar derivation from a common CS2 reference.
            da = raw.get_data(picks=[direct[f"{a}-CS2"]])[0]
            db = raw.get_data(picks=[direct[f"{b}-CS2"]])[0]
            rows.append(da - db)

        else:
            missing.append(pair)

    if missing:
        raise ValueError(f"Missing common-18 channels {missing} in {edf_path.name}")

    return np.vstack(rows).astype(np.float32), sfreq, duration


# -----------------------------------------------------------------------------
# Scan construction
# -----------------------------------------------------------------------------
def merge_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    clean = sorted((max(0.0, float(a)), max(0.0, float(b))) for a, b in intervals if b > a)
    if not clean:
        return []
    merged = [list(clean[0])]
    for a, b in clean[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(float(a), float(b)) for a, b in merged]


def complement_intervals(duration: float, blocked: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    blocked2 = merge_intervals((max(0.0, a), min(duration, b)) for a, b in blocked if b > 0 and a < duration)
    out = []
    cur = 0.0
    for a, b in blocked2:
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < duration:
        out.append((cur, duration))
    return [(a, b) for a, b in out if b > a]


def make_patient_scans(
    patient: str,
    edf_df: pd.DataFrame,
    seizures: Sequence[Seizure],
    args: argparse.Namespace,
) -> Tuple[List[Scan], List[Scan], List[Scan], List[Scan]]:
    by_edf: Dict[Path, List[Seizure]] = {}
    for s in seizures:
        by_edf.setdefault(s.edf_path, []).append(s)

    seg = float(args.segment_seconds)
    interictal: List[Scan] = []
    for _, row in edf_df.iterrows():
        edf = Path(row["edf"])
        dur = float(row["duration_sec"])
        if not np.isfinite(dur) or dur < seg:
            continue
        blocked = []
        for s in by_edf.get(edf, []):
            blocked.append((
                s.onset_sec - float(args.interictal_pre_buffer_seconds),
                s.end_sec + float(args.interictal_post_buffer_seconds),
            ))
        for a, b in complement_intervals(dur, blocked):
            t = a
            while t + seg <= b + 1e-9:
                interictal.append(Scan(patient, edf, "interictal", t, t + seg))
                t += float(args.interictal_scan_stride_seconds)

    rng = np.random.default_rng(args.data_seed)
    rng.shuffle(interictal)
    if args.max_interictal_scans_per_patient > 0:
        interictal = interictal[: int(args.max_interictal_scans_per_patient)]

    n = len(interictal)
    n_val = int(round(args.val_fraction * n))
    n_test = int(round(args.test_fraction * n))
    if n >= 3:
        n_val = max(1, n_val)
        n_test = max(1, n_test)
    if n_val + n_test >= n:
        n_val = max(0, min(n // 3, n - 1))
        n_test = max(0, min(n // 3, n - n_val - 1))

    val = interictal[:n_val]
    test = interictal[n_val:n_val + n_test]
    train = interictal[n_val + n_test:]

    ictal: List[Scan] = []
    for s in seizures:
        row = edf_df[edf_df["edf"] == str(s.edf_path)]
        if row.empty:
            continue
        dur = float(row.iloc[0]["duration_sec"])
        a = s.onset_sec + float(args.ictal_start_seconds)
        b = a + float(args.ictal_scan_seconds)
        # Fixed-duration onset window; skip if a full scan is unavailable.
        if a >= 0 and b <= dur + 1e-9 and (b - a) >= seg - 1e-9:
            ictal.append(Scan(patient, s.edf_path, "ictal", a, b, s.seizure_id, s.onset_sec, s.end_sec))
        elif args.verbose:
            print(f"[ICTAL DROP] {s.seizure_id}: requested=({a:.2f},{b:.2f}) EDF_dur={dur:.2f}")

    return train, val, test, ictal


# -----------------------------------------------------------------------------
# Engineered EEG features (15/channel x 18 channels = 270)
# -----------------------------------------------------------------------------
def hjorth_params(x: np.ndarray) -> Tuple[float, float, float]:
    dx = np.diff(x)
    ddx = np.diff(dx)
    var0 = float(np.var(x)) + 1e-12
    var1 = float(np.var(dx)) + 1e-12
    var2 = float(np.var(ddx)) + 1e-12
    mobility = math.sqrt(var1 / var0)
    complexity = math.sqrt(var2 / var1) / (mobility + 1e-12)
    return var0, mobility, complexity


def bandpowers(x: np.ndarray, sfreq: float) -> Dict[str, float]:
    nperseg = min(len(x), int(max(2 * sfreq, 64)))
    freqs, pxx = welch(x, fs=sfreq, nperseg=nperseg)
    out = {}
    for name, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        if np.any(m):
            out[name] = float(np.trapezoid(pxx[m], freqs[m]) + 1e-12)
        else:
            out[name] = 1e-12
    return out


def engineered_feature_vector(seg: np.ndarray, sfreq: float) -> np.ndarray:
    feats: List[float] = []
    for ch in range(seg.shape[0]):
        x = np.asarray(seg[ch], dtype=np.float64)
        rms = math.sqrt(float(np.mean(x * x)) + 1e-12)
        line_len = float(np.mean(np.abs(np.diff(x))))
        ptp = float(np.ptp(x))
        h_act, h_mob, h_comp = hjorth_params(x)
        bp = bandpowers(x, sfreq)
        total_bp = sum(bp.values()) + 1e-12
        p = np.asarray(list(bp.values()), dtype=float) / total_bp
        spec_ent = float(scipy_entropy(p + 1e-12) / math.log(len(p)))
        zcr = float(np.mean(np.diff(np.signbit(x)) != 0))
        feats.extend([
            rms, line_len, ptp, h_act, h_mob, h_comp,
            math.log(bp["delta"]), math.log(bp["theta"]), math.log(bp["alpha"]),
            math.log(bp["beta"]), math.log(bp["gamma"]),
            bp["theta"] / (bp["beta"] + 1e-12),
            bp["alpha"] / (bp["beta"] + 1e-12),
            spec_ent, zcr,
        ])
    arr = np.asarray(feats, dtype=np.float64)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def build_feature_sequence(
    scan: Scan,
    cache: Dict[Path, Tuple[np.ndarray, float, float]],
    args: argparse.Namespace,
) -> Optional[FeatureSequence]:
    if scan.edf_path not in cache:
        cache[scan.edf_path] = load_common18_edf(scan.edf_path, args.l_freq, args.h_freq)
    data, sfreq, duration = cache[scan.edf_path]
    if scan.start_sec < 0 or scan.end_sec > duration + 1e-9:
        return None

    chunk_n = int(round(args.chunk_seconds * sfreq))
    stride_n = int(round(args.stride_seconds * sfreq))
    start_samp = int(round(scan.start_sec * sfreq))
    end_samp = int(round(scan.end_sec * sfreq))
    starts = np.arange(start_samp, end_samp - chunk_n + 1, stride_n, dtype=int)
    if len(starts) < args.min_sequence_chunks:
        return None

    feats = []
    times = []
    for ss in starts:
        feats.append(engineered_feature_vector(data[:, ss:ss + chunk_n], sfreq))
        times.append(ss / sfreq)

    scan_id = scan.seizure_id if scan.seizure_id else f"{scan.edf_path.stem}_{scan.start_sec:.3f}"
    return FeatureSequence(
        patient=scan.patient,
        kind=scan.kind,
        scan_id=scan_id,
        chunk_starts_sec=np.asarray(times, dtype=float),
        features=np.vstack(feats),
        onset_sec=float(scan.onset_sec),
    )


def scans_to_features(scans: Sequence[Scan], args: argparse.Namespace) -> List[FeatureSequence]:
    cache: Dict[Path, Tuple[np.ndarray, float, float]] = {}
    out = []
    for sc in scans:
        try:
            fs = build_feature_sequence(sc, cache, args)
            if fs is not None:
                out.append(fs)
        except Exception as exc:
            if args.verbose:
                print(f"[WARN] skipped {sc.kind} {sc.edf_path.name}: {exc}")
    return out


def transform_sequences(
    seqs: Sequence[FeatureSequence], scaler: StandardScaler, pca: PCA
) -> List[FeatureSequence]:
    out = []
    for s in seqs:
        z = pca.transform(scaler.transform(s.features))
        out.append(FeatureSequence(s.patient, s.kind, s.scan_id, s.chunk_starts_sec, z, s.onset_sec))
    return out


def make_xy(seq: np.ndarray, washout: int, pred_horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = seq.shape[0]
    idx = np.arange(washout, n - pred_horizon)
    if len(idx) == 0:
        return np.empty((0, seq.shape[1])), np.empty((0, seq.shape[1])), idx
    return seq[idx], seq[idx + pred_horizon], idx + pred_horizon


def collect_xy(seqs: Sequence[FeatureSequence], washout: int, pred_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for s in seqs:
        X, Y, _ = make_xy(s.features, washout, pred_horizon)
        if len(X):
            xs.append(X)
            ys.append(Y)
    if not xs:
        raise RuntimeError("No usable training transition pairs")
    return np.vstack(xs), np.vstack(ys)


# -----------------------------------------------------------------------------
# Rössler DynML: z0 = R x; NO tanh, NO bias, readout from final 3N state only.
# -----------------------------------------------------------------------------
class RosslerDynML:
    def __init__(self, input_dim: int, args: argparse.Namespace):
        self.input_dim = int(input_dim)
        self.N = int(args.num_reservoirs)
        self.regime_id = int(args.regime_id)
        self.regime = ROSSLER_REGIMES[self.regime_id]
        self.integration_time = float(args.integration_time)
        self.dt = float(args.dt)
        self.ridge = float(args.ridge)
        self.seed = int(args.res_seed)

        rng = np.random.default_rng(self.seed)

        # Fixed bounded random linear projection R in R^(3N x d).
        # No nonlinear activation or additive bias is used.
        self.R = rng.uniform(
            low=-0.0025,
            high=0.0025,
            size=(3 * self.N, self.input_dim),
        )

        self.a = rng.uniform(*self.regime["a"], size=self.N)
        self.b = rng.uniform(*self.regime["b"], size=self.N)
        self.c = rng.uniform(*self.regime["c"], size=self.N)
        self.tau = rng.uniform(float(args.tau_min), float(args.tau_max), size=self.N)
        self.Wout: Optional[np.ndarray] = None

    def _rhs(self, state: np.ndarray) -> np.ndarray:
        # state shape (..., N, 3)
        x = state[..., 0]
        y = state[..., 1]
        z = state[..., 2]
        dx = (-y - z) / self.tau
        dy = (x + self.a * y) / self.tau
        dz = (self.b + z * (x - self.c)) / self.tau
        return np.stack([dx, dy, dz], axis=-1)

    def _solve_one(self, z0: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
        """Integrate one 3N-dimensional Rössler initial condition with adaptive solve_ivp."""
        z0 = np.asarray(z0, dtype=np.float64).reshape(self.N, 3)
        t_eval = np.asarray(t_eval, dtype=float)
        if t_eval.size == 0 or np.any(t_eval <= 0):
            raise ValueError("t_eval must contain positive integration times")
        t_end = float(np.max(t_eval))

        def rhs_flat(t: float, y: np.ndarray) -> np.ndarray:
            state = y.reshape(self.N, 3)
            return self._rhs(state).reshape(-1)

        sol = solve_ivp(
            rhs_flat,
            (0.0, t_end),
            z0.reshape(-1),
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9,
        )
        if not sol.success:
            raise FloatingPointError(f"Rössler solve_ivp failed: {sol.message}")
        if not np.all(np.isfinite(sol.y)):
            raise FloatingPointError("Non-finite Rössler state encountered during solve_ivp integration")
        return sol.y.T

    def initial_states(self, X: np.ndarray) -> np.ndarray:
        # Mapping: z0 = R x. X: batch x d -> batch x N x 3.
        # No nonlinear activation or additive bias is used.
        z0 = X @ self.R.T
        if not np.all(np.isfinite(z0)):
            raise FloatingPointError("Non-finite projected Rössler initial condition")
        return z0.reshape(len(X), self.N, 3).astype(np.float64)

    def projection_diagnostics(self, X: np.ndarray) -> Dict[str, float]:
        z0 = self.initial_states(X).reshape(len(X), -1)
        absz = np.abs(z0).ravel()
        return {
            "projection_r_min": float(np.min(self.R)),
            "projection_r_max": float(np.max(self.R)),
            "z0_abs_mean": float(np.mean(absz)),
            "z0_abs_median": float(np.median(absz)),
            "z0_abs_q90": float(np.quantile(absz, 0.90)),
            "z0_abs_q99": float(np.quantile(absz, 0.99)),
            "z0_abs_max": float(np.max(absz)),
        }

    def final_states(self, X: np.ndarray) -> np.ndarray:
        state0 = self.initial_states(X).reshape(len(X), 3 * self.N)
        out = np.empty_like(state0, dtype=np.float64)
        t_eval = np.asarray([self.integration_time], dtype=float)
        for i, z0 in enumerate(state0):
            out[i] = self._solve_one(z0, t_eval)[-1]
        return out

    def trajectory_states(self, X: np.ndarray, n_steps: int) -> np.ndarray:
        """Return Q x n_steps x (3N) adaptive-solver trajectories for empirical BD."""
        n_steps = max(1, int(n_steps))
        # Preserve the existing BD horizon: n_steps samples separated by dt.
        t_eval = self.dt * np.arange(1, n_steps + 1, dtype=float)
        state0 = self.initial_states(X).reshape(len(X), 3 * self.N)
        traj = np.empty((len(X), n_steps, 3 * self.N), dtype=np.float64)
        for i, z0 in enumerate(state0):
            traj[i] = self._solve_one(z0, t_eval)
        return traj

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        Phi = self.final_states(X)  # ONLY final reservoir state, dimension 3N
        A = Phi.T @ Phi
        A.flat[::A.shape[0] + 1] += self.ridge
        B = Phi.T @ Y
        self.Wout = np.linalg.solve(A, B)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.Wout is None:
            raise RuntimeError("DynML readout is not fit")
        return self.final_states(X) @ self.Wout

    def parameter_summary(self) -> Dict[str, float]:
        return {
            "rossler_a_mean": float(np.mean(self.a)),
            "rossler_a_std": float(np.std(self.a, ddof=1)) if self.N > 1 else 0.0,
            "rossler_b_mean": float(np.mean(self.b)),
            "rossler_b_std": float(np.std(self.b, ddof=1)) if self.N > 1 else 0.0,
            "rossler_c_mean": float(np.mean(self.c)),
            "rossler_c_std": float(np.std(self.c, ddof=1)) if self.N > 1 else 0.0,
            "tau_mean": float(np.mean(self.tau)),
            "tau_std": float(np.std(self.tau, ddof=1)) if self.N > 1 else 0.0,
        }

    def bd_entropy(
        self,
        X_ref: np.ndarray,
        q: int,
        eps: float,
        n_steps: int,
        greedy_repeats: int,
        rng: np.random.Generator,
    ) -> Dict[str, float]:
        if len(X_ref) == 0:
            return {
                "bd_entropy": math.nan,
                "bd_separated_count": 0,
                "bd_q": 0,
                "bd_epsilon": float(eps),
                "bd_steps": int(n_steps),
                "bd_finite_sample_bound": math.nan,
                "bd_saturated": False,
            }

        q_eff = min(int(q), len(X_ref))
        idx = rng.choice(len(X_ref), size=q_eff, replace=False)
        T = self.trajectory_states(X_ref[idx], n_steps=n_steps)  # Q x n x 3N

        # Pairwise Bowen-Dinaburg distances.
        # d_n(i,j) = max_t ||z_i(t)-z_j(t)||_2 / sqrt(3N)
        denom = math.sqrt(3.0 * self.N)
        D = np.zeros((q_eff, q_eff), dtype=np.float64)
        for i in range(q_eff):
            diff = T[i+1:] - T[i][None, :, :]
            if len(diff):
                per_t = np.linalg.norm(diff, axis=2) / denom
                vals = np.max(per_t, axis=1)
                D[i, i+1:] = vals
                D[i+1:, i] = vals

        tri = D[np.triu_indices(q_eff, k=1)]
        finite_tri = tri[np.isfinite(tri)]
        if len(finite_tri):
            dist_diag = {
                "bd_distance_min": float(np.min(finite_tri)),
                "bd_distance_q10": float(np.quantile(finite_tri, 0.10)),
                "bd_distance_q25": float(np.quantile(finite_tri, 0.25)),
                "bd_distance_median": float(np.median(finite_tri)),
                "bd_distance_q75": float(np.quantile(finite_tri, 0.75)),
                "bd_distance_q90": float(np.quantile(finite_tri, 0.90)),
                "bd_distance_max": float(np.max(finite_tri)),
            }
        else:
            dist_diag = {k: math.nan for k in [
                "bd_distance_min", "bd_distance_q10", "bd_distance_q25",
                "bd_distance_median", "bd_distance_q75", "bd_distance_q90", "bd_distance_max"
            ]}

        best = 0
        for _ in range(max(1, int(greedy_repeats))):
            order = rng.permutation(q_eff)
            selected: List[int] = []
            for i in order:
                if not selected or np.all(D[i, selected] > eps):
                    selected.append(int(i))
            best = max(best, len(selected))

        h = math.log(max(best, 1)) / max(1, int(n_steps))
        bound = math.log(max(q_eff, 1)) / max(1, int(n_steps))
        return {
            "bd_entropy": float(h),
            "bd_separated_count": int(best),
            "bd_q": int(q_eff),
            "bd_epsilon": float(eps),
            "bd_steps": int(n_steps),
            "bd_finite_sample_bound": float(bound),
            "bd_saturated": bool(best >= q_eff),
            **dist_diag,
        }


# -----------------------------------------------------------------------------
# Residual likelihood and alarm logic
# -----------------------------------------------------------------------------
def residuals_for_sequences(
    seqs: Sequence[FeatureSequence], model: RosslerDynML, washout: int, pred_horizon: int
) -> np.ndarray:
    parts = []
    for s in seqs:
        X, Y, _ = make_xy(s.features, washout, pred_horizon)
        if len(X):
            parts.append(Y - model.predict(X))
    if not parts:
        raise RuntimeError("No residuals available")
    return np.vstack(parts)


def fit_diag_gaussian_from_validation(
    val_seqs: Sequence[FeatureSequence], model: RosslerDynML, washout: int, pred_horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    R = residuals_for_sequences(val_seqs, model, washout, pred_horizon)
    mu = np.mean(R, axis=0)
    var = np.var(R, axis=0) + 1e-6
    return mu, var


def seq_loglik(
    seq: FeatureSequence,
    model: RosslerDynML,
    mu: np.ndarray,
    var: np.ndarray,
    washout: int,
    pred_horizon: int,
    chunk_seconds: float,
) -> Tuple[np.ndarray, np.ndarray]:
    X, Y, y_idx = make_xy(seq.features, washout, pred_horizon)
    if len(X) == 0:
        return np.asarray([]), np.asarray([])
    R = Y - model.predict(X)
    ll_dim = -0.5 * (np.log(2.0 * np.pi * var) + ((R - mu) ** 2) / var)
    ll = np.sum(ll_dim, axis=1)
    # Online availability time: the target EEG feature is only available
    # after the full target chunk has been observed.
    times = seq.chunk_starts_sec[y_idx] + float(chunk_seconds)
    return ll, times


def validation_threshold(
    val_seqs: Sequence[FeatureSequence],
    model: RosslerDynML,
    mu: np.ndarray,
    var: np.ndarray,
    washout: int,
    pred_horizon: int,
    percentile: float,
    chunk_seconds: float,
) -> np.ndarray:
    curves = []
    for s in val_seqs:
        ll, _ = seq_loglik(s, model, mu, var, washout, pred_horizon, chunk_seconds)
        if len(ll):
            curves.append(np.cumsum(ll))
    if not curves:
        raise RuntimeError("No validation cumulative-loglikelihood curves")
    min_len = min(len(c) for c in curves)
    M = np.vstack([c[:min_len] for c in curves])
    return np.percentile(M, float(percentile), axis=0)


def alarm_from_ll(
    ll: np.ndarray,
    times: np.ndarray,
    thr: np.ndarray,
    persistence_steps: int,
) -> Tuple[bool, Optional[float], Optional[float], int]:
    """Return both the first crossing in the qualifying run and the time
    at which persistence is actually confirmed. The confirmed time is the
    primary online alarm time.
    """
    if len(ll) == 0:
        return False, None, None, 0
    m = min(len(ll), len(thr), len(times))
    flags = np.cumsum(ll[:m]) < thr[:m]
    run = 0
    for i, flag in enumerate(flags):
        run = run + 1 if bool(flag) else 0
        if run >= persistence_steps:
            first_idx = i - persistence_steps + 1
            return (
                True,
                float(times[first_idx]),
                float(times[i]),
                int(np.sum(flags)),
            )
    return False, None, None, int(np.sum(flags))


def alarm_episode_count(ll: np.ndarray, thr: np.ndarray, persistence_steps: int) -> int:
    if len(ll) == 0:
        return 0
    m = min(len(ll), len(thr))
    flags = np.cumsum(ll[:m]) < thr[:m]
    episodes = 0
    run = 0
    in_episode = False
    for flag in flags:
        if flag:
            run += 1
            if run >= persistence_steps and not in_episode:
                episodes += 1
                in_episode = True
        else:
            run = 0
            in_episode = False
    return int(episodes)


def parse_float_list(text: str) -> List[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one numeric value")
    return vals


# -----------------------------------------------------------------------------
# Patient run
# -----------------------------------------------------------------------------
def run_patient(patient: str, args: argparse.Namespace) -> List[Dict[str, object]]:
    patient_dir = Path(args.data_root) / patient
    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

    edf_df, seizures = index_patient(patient_dir, patient)
    train_scans, val_scans, test_scans, ictal_scans = make_patient_scans(patient, edf_df, seizures, args)

    if args.verbose:
        print(
            f"{patient}: seizures={len(seizures)} train_scans={len(train_scans)} "
            f"val_scans={len(val_scans)} test_scans={len(test_scans)} ictal_scans={len(ictal_scans)}"
        )

    train_raw = scans_to_features(train_scans, args)
    val_raw = scans_to_features(val_scans, args)
    test_raw = scans_to_features(test_scans, args)
    ictal_raw = scans_to_features(ictal_scans, args)

    if len(train_raw) < args.min_train_scans:
        raise RuntimeError(f"{patient}: insufficient train feature sequences ({len(train_raw)})")
    if len(val_raw) < args.min_val_scans:
        raise RuntimeError(f"{patient}: insufficient validation feature sequences ({len(val_raw)})")
    if len(ictal_raw) == 0:
        raise RuntimeError(f"{patient}: no valid fixed 30-s ictal windows")

    train_mat = np.vstack([s.features for s in train_raw])
    scaler = StandardScaler().fit(train_mat)
    pca_n = min(int(args.pca_components), train_mat.shape[1], max(1, train_mat.shape[0] - 1))
    pca = PCA(n_components=pca_n, random_state=args.data_seed).fit(scaler.transform(train_mat))

    train = transform_sequences(train_raw, scaler, pca)
    val = transform_sequences(val_raw, scaler, pca)
    test = transform_sequences(test_raw, scaler, pca)
    ictal = transform_sequences(ictal_raw, scaler, pca)

    Xtr, Ytr = collect_xy(train, args.washout, args.pred_horizon)
    model = RosslerDynML(pca_n, args)
    model.fit(Xtr, Ytr)

    # EEG manuscript protocol: fit residual Gaussian on validation interictal errors.
    mu, var = fit_diag_gaussian_from_validation(val, model, args.washout, args.pred_horizon)
    thr = validation_threshold(
        val, model, mu, var, args.washout, args.pred_horizon, args.threshold_percentile, args.chunk_seconds
    )

    # Compute empirical BD once per patient x regime x seed.
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

    out_dir = Path(args.out_dir)
    patient_out = out_dir / patient
    patient_out.mkdir(parents=True, exist_ok=True)

    # Save reservoir realization once.
    reservoir_table = pd.DataFrame({
        "unit": np.arange(model.N),
        "a": model.a,
        "b": model.b,
        "c": model.c,
        "tau": model.tau,
    })
    reservoir_table.to_csv(patient_out / f"{patient}_reservoir_parameters.csv", index=False)

    bd_row = {
        "patient": patient,
        "run_id": args.run_id,
        "regime_id": args.regime_id,
        "regime_name": model.regime["name"],
        "res_seed": args.res_seed,
        "data_seed": args.data_seed,
        "pca_components": pca_n,
        "num_reservoirs": model.N,
        "integration_time": args.integration_time,
        "dt": args.dt,
        **bd,
        **proj_diag,
        **param_summary,
    }
    pd.DataFrame([bd_row]).to_csv(patient_out / f"{patient}_rossler_bd_entropy.csv", index=False)

    summaries: List[Dict[str, object]] = []
    persistence_values = parse_float_list(args.alarm_persistence_seconds)

    for persistence_sec in persistence_values:
        persistence_steps = max(1, int(math.ceil(persistence_sec / args.stride_seconds)))

        inter_rows = []
        fp_scans = 0
        fp_episodes = 0
        inter_hours = 0.0
        for s in test:
            ll, times = seq_loglik(s, model, mu, var, args.washout, args.pred_horizon, args.chunk_seconds)
            alarm, first_crossing_time, confirmed_alarm_time, nflag = alarm_from_ll(ll, times, thr, persistence_steps)
            episodes = alarm_episode_count(ll, thr, persistence_steps)
            fp_scans += int(alarm)
            fp_episodes += episodes
            if len(s.chunk_starts_sec):
                duration = (s.chunk_starts_sec[-1] + args.chunk_seconds) - s.chunk_starts_sec[0]
                inter_hours += max(0.0, duration) / 3600.0
            inter_rows.append({
                "patient": patient,
                "scan_id": s.scan_id,
                "alarm_persistence_seconds": persistence_sec,
                "alarm": bool(alarm),
                "first_crossing_time_sec": first_crossing_time,
                "confirmed_alarm_time_sec": confirmed_alarm_time,
                "n_flagged_steps": nflag,
                "alarm_episodes": episodes,
            })

        ictal_rows = []
        tp = 0
        delays: List[float] = []
        for s in ictal:
            ll, times = seq_loglik(s, model, mu, var, args.washout, args.pred_horizon, args.chunk_seconds)
            alarm, first_crossing_time, confirmed_alarm_time, nflag = alarm_from_ll(ll, times, thr, persistence_steps)
            if alarm and confirmed_alarm_time is not None and np.isfinite(s.onset_sec):
                delay = float(confirmed_alarm_time - s.onset_sec)
                valid = delay >= 0.0
            else:
                delay = math.nan
                valid = False
            if valid:
                tp += 1
                delays.append(delay)
            ictal_rows.append({
                "patient": patient,
                "seizure_id": s.scan_id,
                "alarm_persistence_seconds": persistence_sec,
                "alarm": bool(valid),
                "raw_alarm": bool(alarm),
                "first_crossing_time_sec": first_crossing_time,
                "confirmed_alarm_time_sec": confirmed_alarm_time,
                "detection_delay_seconds": delay,
                "n_flagged_steps": nflag,
            })

        n_ictal = len(ictal)
        n_test = len(test)
        sensitivity = tp / n_ictal if n_ictal else math.nan
        specificity = (n_test - fp_scans) / n_test if n_test else math.nan
        balanced_accuracy = np.nanmean([sensitivity, specificity])
        far_per_hour = fp_episodes / inter_hours if inter_hours > 0 else math.nan

        summary = {
            "patient": patient,
            "run_id": args.run_id,
            "analysis": "chbmit_ictal_vs_interictal_rossler_dynml",
            "feature_mode": "engineered",
            "regime_id": args.regime_id,
            "regime_name": model.regime["name"],
            "data_seed": args.data_seed,
            "res_seed": args.res_seed,
            "pca_components": pca_n,
            "threshold_percentile": args.threshold_percentile,
            "alarm_persistence_seconds": persistence_sec,
            "segment_seconds": args.segment_seconds,
            "ictal_start_seconds": args.ictal_start_seconds,
            "ictal_scan_seconds": args.ictal_scan_seconds,
            "chunk_seconds": args.chunk_seconds,
            "stride_seconds": args.stride_seconds,
            "pred_horizon": args.pred_horizon,
            "washout": args.washout,
            "num_reservoirs": model.N,
            "integration_time": args.integration_time,
            "dt": args.dt,
            "tau_min": args.tau_min,
            "tau_max": args.tau_max,
            "train_scans": len(train),
            "val_scans": len(val),
            "interictal_test_scans": n_test,
            "interictal_fp_scans": fp_scans,
            "interictal_tn_scans": max(0, n_test - fp_scans),
            "specificity": float(specificity),
            "false_alarm_episodes": fp_episodes,
            "interictal_test_hours": float(inter_hours),
            "false_alarms_per_hour": float(far_per_hour),
            "ictal_total": n_ictal,
            "ictal_tp": tp,
            "ictal_fn": max(0, n_ictal - tp),
            "ictal_sensitivity": float(sensitivity),
            "balanced_accuracy": float(balanced_accuracy),
            "ictal_delay_mean_seconds": float(np.mean(delays)) if delays else math.nan,
            "ictal_delay_median_seconds": float(np.median(delays)) if delays else math.nan,
            "ictal_delay_std_seconds": float(np.std(delays, ddof=1)) if len(delays) > 1 else (0.0 if len(delays) == 1 else math.nan),
            **bd,
            **proj_diag,
            **param_summary,
        }
        summaries.append(summary)

        suffix = str(persistence_sec).replace(".", "p")
        pd.DataFrame(inter_rows).to_csv(
            patient_out / f"{patient}_interictal_test_scans_pers{suffix}s.csv", index=False
        )
        pd.DataFrame(ictal_rows).to_csv(
            patient_out / f"{patient}_ictal_events_pers{suffix}s.csv", index=False
        )

    (patient_out / f"{patient}_run_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True))
    return summaries


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CHB-MIT ictal Rössler-DynML + empirical BD entropy")
    p.add_argument("--data_root", default="/data/halders2/EEG_anomaly/Data")
    p.add_argument("--patient", default="chb01", help="Single patient or comma-separated patient list")
    p.add_argument("--out_dir", default="chbmit_ictal_rossler_bd_results/debug")
    p.add_argument("--run_id", default="debug")

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
    p.add_argument(
        "--alarm_persistence_seconds",
        default="3,5",
        help="Comma-separated values evaluated from one fitted model, e.g. 3,5",
    )

    p.add_argument("--regime_id", type=int, default=4, choices=sorted(ROSSLER_REGIMES))
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

    p.add_argument("--bd_q", type=int, default=64)
    p.add_argument("--bd_epsilon", type=float, default=0.05)
    p.add_argument("--bd_steps", type=int, default=30)
    p.add_argument("--bd_greedy_repeats", type=int, default=25)

    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.tau_min <= 0 or args.tau_max <= 0 or args.tau_max < args.tau_min:
        raise ValueError("Require 0 < tau_min <= tau_max")
    if args.integration_time <= 0 or args.dt <= 0:
        raise ValueError("integration_time and dt must be positive")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, object]] = []

    patients = [x.strip().lower() for x in args.patient.split(",") if x.strip()]
    for patient in patients:
        try:
            all_rows.extend(run_patient(patient, args))
        except Exception as exc:
            print(f"[ERROR] {patient}: {exc}")
            for pers in parse_float_list(args.alarm_persistence_seconds):
                all_rows.append({
                    "patient": patient,
                    "run_id": args.run_id,
                    "regime_id": args.regime_id,
                    "data_seed": args.data_seed,
                    "res_seed": args.res_seed,
                    "alarm_persistence_seconds": pers,
                    "error": str(exc),
                })

    out_csv = Path(args.out_dir) / "all_patient_summary.csv"
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
