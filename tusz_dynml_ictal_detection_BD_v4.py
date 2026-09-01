#!/usr/bin/env python3
"""
TUSZ patient-specific ictal anomaly detection using corrected Rössler DynML v4 and empirical finite-time BD entropy.

Analysis 1 design:
  train: interictal/background 30-sec windows only
  validation: interictal/background 30-sec windows only, for cumulative-loglik threshold
  negative test: held-out interictal/background 30-sec windows, for FP and FA/hour
  positive test: ictal onset windows, seizure onset to onset + 30 sec

Notes:
  - TUSZ is treated here as an ictal seizure-detection dataset.
  - This script does not do preictal early warning.
  - Ictal windows are fixed relative to annotated onset and are not truncated at annotated seizure end;
    they are clipped only by EDF duration. This is intentional because median TUSZ seizure duration is short.
  - Uses common-18 bipolar montage. If bipolar channels are not directly present, it subtracts
    matching referential electrodes.

Compact outputs only:
  all_patient_summary.csv
  <patient>_run_summary.json
  <patient>_ictal_events.csv
  <patient>_interictal_test_scans.csv
  <patient>_rossler_bd_entropy.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import mne
except Exception as exc:
    raise RuntimeError("This script requires mne. Use your dynml_ecg environment on Biowulf.") from exc

from scipy.integrate import solve_ivp
from scipy.signal import welch
from scipy.stats import entropy as scipy_entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# COMMON18 = [
#     "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
#     "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
#     "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
#     "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
#     "FZ-CZ", "CZ-PZ",
# ]

COMMON16 = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
]

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

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
}

SEIZURE_LABELS = {
    "seiz", "fnsz", "gnsz", "spsz", "cpsz", "absz",
    "tnsz", "tcsz", "mysz", "atsz", "clsz", "seizure",
}


@dataclass
class Seizure:
    patient: str
    edf_path: Path
    seizure_id: str
    onset_sec: float
    end_sec: float
    label: str = "seiz"


@dataclass
class Scan:
    patient: str
    edf_path: Path
    region: str
    scan_id: str
    start_sec: float
    end_sec: float
    onset_sec: Optional[float] = None
    end_seizure_sec: Optional[float] = None
    seizure_label: str = ""


@dataclass
class FeatureSequence:
    patient: str
    region: str
    scan_id: str
    chunk_starts_sec: np.ndarray
    features: np.ndarray
    onset_sec: Optional[float] = None
    end_seizure_sec: Optional[float] = None
    seizure_label: str = ""


def norm_ch(name: str) -> str:
    s = str(name).upper().strip()
    s = s.replace("EEG", "")
    s = s.replace(" ", "").replace("_", "").replace(".", "")
    s = s.replace("-LE", "").replace("-REF", "").replace("REF", "")
    s = re.sub(r"-\d+$", "", s)
    s = s.replace("T3", "T7").replace("T4", "T8").replace("T5", "P7").replace("T6", "P8")
    return s.strip("-")


def parse_tusz_path(p: Path) -> Dict[str, str]:
    parts = p.parts
    out = dict(split="unknown", patient="unknown", session="unknown", montage="unknown")
    if "edf" in parts:
        i = parts.index("edf")
        if len(parts) > i + 5:
            out["split"] = parts[i + 1]
            out["patient"] = parts[i + 2]
            out["session"] = parts[i + 3]
            out["montage"] = parts[i + 4]
    return out


def annotation_candidates_for_edf(edf_path: Path) -> List[Path]:
    stem = edf_path.with_suffix("")
    return [Path(str(stem) + ".csv_bi"), Path(str(stem) + ".csv")]


def read_tusz_annotation(path: Path) -> pd.DataFrame:
    cols = ["channel", "start_time", "stop_time", "label", "confidence"]
    if (not path.exists()) or path.stat().st_size == 0:
        return pd.DataFrame(columns=cols)
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 4:
                continue
            joined = ",".join(parts).lower()
            if "channel" in joined and "start" in joined and "stop" in joined:
                continue
            start = pd.to_numeric(parts[1], errors="coerce")
            stop = pd.to_numeric(parts[2], errors="coerce")
            if pd.isna(start) or pd.isna(stop):
                continue
            rows.append({
                "channel": parts[0],
                "start_time": float(start),
                "stop_time": float(stop),
                "label": str(parts[3]).strip().lower(),
                "confidence": pd.to_numeric(parts[4], errors="coerce") if len(parts) >= 5 else np.nan,
            })
    return pd.DataFrame(rows, columns=cols)


def is_seizure_label(x: str) -> bool:
    x = str(x).lower().strip()
    return (x in SEIZURE_LABELS) or ("seiz" in x) or x.endswith("sz")


def get_seizure_events_for_edf(edf_path: Path, patient: str) -> List[Seizure]:
    ann = None
    ann_path = None
    for p in annotation_candidates_for_edf(edf_path):
        if p.exists():
            tmp = read_tusz_annotation(p)
            if len(tmp):
                ev = tmp[tmp["label"].apply(is_seizure_label)].dropna(subset=["start_time", "stop_time"])
                if len(ev):
                    ann = tmp
                    ann_path = p
                    break
    if ann is None:
        return []
    df = ann[ann["label"].apply(is_seizure_label)].dropna(subset=["start_time", "stop_time"]).copy()
    if df.empty:
        return []
    events = (
        df.groupby(["start_time", "stop_time", "label"], as_index=False)
          .size()
          .sort_values(["start_time", "stop_time"])
    )
    seizures = []
    for j, r in events.iterrows():
        onset = float(r["start_time"])
        end = float(r["stop_time"])
        if end <= onset:
            continue
        sid = f"{edf_path.stem}_s{len(seizures)+1:03d}"
        seizures.append(Seizure(patient, edf_path, sid, onset, end, str(r["label"])))
    return seizures


def index_tusz(data_root: Path, montage_filter: str = "01_tcp_ar") -> Tuple[pd.DataFrame, List[Seizure]]:
    edfs = sorted(data_root.rglob("*.edf"))
    rows = []
    all_seizures: List[Seizure] = []
    for edf in edfs:
        meta = parse_tusz_path(edf)
        if montage_filter and meta["montage"] != montage_filter:
            continue
        rows.append(dict(edf=str(edf), **meta))
        all_seizures.extend(get_seizure_events_for_edf(edf, meta["patient"]))
    return pd.DataFrame(rows), all_seizures


def load_common16_edf(edf_path: Path, l_freq: float, h_freq: float) -> Tuple[np.ndarray, float, List[str], float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
        raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    duration = raw.n_times / sfreq
    ch_lookup: Dict[str, str] = {}
    for ch in raw.ch_names:
        key = norm_ch(ch)
        if key and key not in ch_lookup:
            ch_lookup[key] = ch
    rows = []
    ok_names = []
    missing = []
    for pair in COMMON16:
        pair_key = norm_ch(pair)
        a, b = pair_key.split("-")
        if pair_key in ch_lookup:
            rows.append(raw.get_data(picks=[ch_lookup[pair_key]])[0])
            ok_names.append(pair)
        elif a in ch_lookup and b in ch_lookup:
            data = raw.get_data(picks=[ch_lookup[a], ch_lookup[b]])
            rows.append(data[0] - data[1])
            ok_names.append(pair)
        else:
            missing.append(pair)
    if missing:
        available = ", ".join(list(ch_lookup.keys())[:50])
        raise ValueError(f"Missing common16 {missing}; available normalized channels include: {available}")
    return np.asarray(rows, dtype=np.float32), sfreq, ok_names, duration


def merge_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    intervals = sorted([(float(a), float(b)) for a, b in intervals if b > a])
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for a, b in intervals[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def safe_intervals(duration: float, seizures: Sequence[Seizure], pre_buffer: float, post_buffer: float) -> List[Tuple[float, float]]:
    blocked = []
    for s in seizures:
        blocked.append((max(0.0, s.onset_sec - pre_buffer), min(duration, s.end_sec + post_buffer)))
    blocked = merge_intervals(blocked)
    clean = []
    cur = 0.0
    for a, b in blocked:
        if a > cur:
            clean.append((cur, a))
        cur = max(cur, b)
    if cur < duration:
        clean.append((cur, duration))
    return clean


def scans_from_interval(patient: str, edf_path: Path, start: float, end: float, seg_len: float, stride: float, region: str) -> List[Scan]:
    scans = []
    t = start
    idx = 0
    while t + seg_len <= end + 1e-6:
        scans.append(Scan(patient, edf_path, region, f"{edf_path.stem}_{region}_{idx:05d}", t, t + seg_len))
        t += stride
        idx += 1
    return scans


def hjorth_params(x: np.ndarray) -> Tuple[float, float, float]:
    dx = np.diff(x)
    ddx = np.diff(dx)
    var0 = np.var(x) + 1e-12
    var1 = np.var(dx) + 1e-12
    var2 = np.var(ddx) + 1e-12
    mobility = math.sqrt(var1 / var0)
    complexity = math.sqrt(var2 / var1) / (mobility + 1e-12)
    return float(var0), float(mobility), float(complexity)


def bandpowers(x: np.ndarray, sfreq: float) -> Dict[str, float]:
    nperseg = min(len(x), int(max(sfreq * 2, 64)))
    freqs, pxx = welch(x, fs=sfreq, nperseg=nperseg)
    out = {}
    for name, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        out[name] = float(np.trapezoid(pxx[m], freqs[m]) + 1e-12)
    return out


def engineered_feature_vector(seg: np.ndarray, sfreq: float) -> np.ndarray:
    feats = []
    for ch in range(seg.shape[0]):
        x = np.asarray(seg[ch], dtype=np.float64)
        rms = math.sqrt(float(np.mean(x * x)) + 1e-12)
        line_len = float(np.mean(np.abs(np.diff(x))))
        ptp = float(np.ptp(x))
        h_act, h_mob, h_comp = hjorth_params(x)
        bp = bandpowers(x, sfreq)
        total_bp = sum(bp.values()) + 1e-12
        p = np.array(list(bp.values()), dtype=float) / total_bp
        spec_ent = float(scipy_entropy(p + 1e-12) / math.log(len(p)))
        zcr = float(np.mean(np.diff(np.signbit(x)) != 0))
        feats.extend([
            rms, line_len, ptp, h_act, h_mob, h_comp,
            math.log(bp["delta"]), math.log(bp["theta"]), math.log(bp["alpha"]), math.log(bp["beta"]), math.log(bp["gamma"]),
            bp["theta"] / (bp["beta"] + 1e-12), bp["alpha"] / (bp["beta"] + 1e-12), spec_ent, zcr,
        ])
    return np.asarray(feats, dtype=np.float32)


def build_feature_sequence(scan: Scan, cache: Dict[Path, Tuple[np.ndarray, float, List[str], float]], args) -> Optional[FeatureSequence]:
    if scan.edf_path not in cache:
        cache[scan.edf_path] = load_common16_edf(scan.edf_path, args.l_freq, args.h_freq)
    data, sfreq, _, duration = cache[scan.edf_path]
    if scan.start_sec < 0 or scan.end_sec > duration:
        return None
    chunk = int(round(args.chunk_seconds * sfreq))
    stride = int(round(args.stride_seconds * sfreq))
    start_samp = int(round(scan.start_sec * sfreq))
    end_samp = int(round(scan.end_sec * sfreq))
    starts = np.arange(start_samp, end_samp - chunk + 1, stride, dtype=int)
    if len(starts) < args.min_sequence_chunks:
        return None
    feats = []
    chunk_starts = []
    for ss in starts:
        feats.append(engineered_feature_vector(data[:, ss:ss + chunk], sfreq))
        chunk_starts.append(ss / sfreq)
    return FeatureSequence(scan.patient, scan.region, scan.scan_id, np.asarray(chunk_starts), np.vstack(feats), scan.onset_sec, scan.end_seizure_sec, scan.seizure_label)


def make_xy(seq: np.ndarray, washout: int, pred_horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = seq.shape[0]
    idx = np.arange(washout, n - pred_horizon)
    if len(idx) <= 0:
        return np.empty((0, seq.shape[1])), np.empty((0, seq.shape[1])), idx
    return seq[idx], seq[idx + pred_horizon], idx + pred_horizon


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

        # CASCADE/DynML-style fixed random linear projection R in R^(3N x d).
        # No projection scaling, no tanh, and no additive bias.
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
        """Integrate one 3N-dimensional Rössler initial condition with CASCADE-style solve_ivp."""
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
        # CASCADE/DynML mapping: z0 = R x, with no nonlinear activation or bias.
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
        # Preserve the EEG BD definition: n_steps samples separated by dt.
        # Only the numerical ODE solver changes from fixed RK4 to adaptive solve_ivp.
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



def transform_sequences(seqs: Sequence[FeatureSequence], scaler: StandardScaler, pca: PCA) -> List[FeatureSequence]:
    out = []
    for s in seqs:
        z = pca.transform(scaler.transform(s.features))
        out.append(FeatureSequence(s.patient, s.region, s.scan_id, s.chunk_starts_sec, z, s.onset_sec, s.end_seizure_sec, s.seizure_label))
    return out


def collect_xy(seqs: Sequence[FeatureSequence], washout: int, pred_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    Xs, Ys = [], []
    for s in seqs:
        X, Y, _ = make_xy(s.features, washout, pred_horizon)
        if len(X):
            Xs.append(X)
            Ys.append(Y)
    if not Xs:
        raise RuntimeError("No training XY rows. Check scan durations, chunk/stride, or washout.")
    return np.vstack(Xs), np.vstack(Ys)


def residuals_for_sequences(seqs, model, washout, pred_horizon):
    parts = []
    for s in seqs:
        X, Y, _ = make_xy(s.features, washout, pred_horizon)
        if len(X):
            parts.append(Y - model.predict(X))
    if not parts:
        raise RuntimeError("No residuals available")
    return np.vstack(parts)


def fit_diag_gaussian_from_validation(val_seqs, model, washout, pred_horizon):
    R = residuals_for_sequences(val_seqs, model, washout, pred_horizon)
    return R.mean(axis=0), R.var(axis=0) + 1e-6


def loglik_residual(model, X, Y, mu, var):
    R = Y - model.predict(X)
    ll_dim = -0.5 * (np.log(2 * np.pi * var) + ((R - mu) ** 2) / var)
    return ll_dim.sum(axis=1)


def seq_loglik(seq, model, mu, var, washout, pred_horizon, chunk_seconds):
    X, Y, y_idx = make_xy(seq.features, washout, pred_horizon)
    if len(X) == 0:
        return np.array([]), np.array([])
    ll = loglik_residual(model, X, Y, mu, var)
    # Online time: target feature is available only after the target EEG chunk ends.
    return ll, seq.chunk_starts_sec[y_idx] + float(chunk_seconds)


def validation_threshold(val_seqs, model, mu, var, washout, pred_horizon, percentile, chunk_seconds):
    curves = []
    for s in val_seqs:
        ll, _ = seq_loglik(s, model, mu, var, washout, pred_horizon, chunk_seconds)
        if len(ll):
            curves.append(np.cumsum(ll))
    if not curves:
        raise RuntimeError("No validation curves available")
    min_len = min(len(c) for c in curves)
    M = np.vstack([c[:min_len] for c in curves])
    return np.percentile(M, percentile, axis=0)


def alarm_from_ll(ll, times, thr, persistence_steps):
    if len(ll) == 0:
        return False, None, None, 0
    m = min(len(ll), len(thr), len(times))
    flags = np.cumsum(ll[:m]) < thr[:m]
    run = 0
    for i, flag in enumerate(flags):
        run = run + 1 if bool(flag) else 0
        if run >= persistence_steps:
            first_i = i - persistence_steps + 1
            return True, float(times[first_i]), float(times[i]), int(flags.sum())
    return False, None, None, int(flags.sum())


def alarm_episodes(ll, thr, persistence_steps):
    if len(ll) == 0:
        return 0
    m = min(len(ll), len(thr))
    flags = np.cumsum(ll[:m]) < thr[:m]
    episodes = 0; run = 0; in_ep = False
    for f in flags:
        if f:
            run += 1
            if run >= persistence_steps and not in_ep:
                episodes += 1; in_ep = True
        else:
            run = 0; in_ep = False
    return int(episodes)


def parse_float_list(text):
    vals = [float(x.strip()) for x in str(text).split(',') if x.strip()]
    if not vals:
        raise ValueError('Expected at least one persistence value')
    return vals

def prepare_patient_scans(patient: str, args, edf_index: pd.DataFrame, all_seizures: List[Seizure]) -> Tuple[List[Scan], List[Scan], List[Scan], List[Scan], List[Seizure]]:
    patient_edfs = [Path(x) for x in edf_index.loc[edf_index["patient"] == patient, "edf"].tolist()]
    patient_seizures = [s for s in all_seizures if s.patient == patient]
    by_file: Dict[Path, List[Seizure]] = {}
    for s in patient_seizures:
        by_file.setdefault(s.edf_path, []).append(s)

    clean_all: List[Scan] = []
    ictal_scans: List[Scan] = []
    duration_cache: Dict[Path, float] = {}

    for edf in patient_edfs:
        try:
            _, _, _, dur = load_common16_edf(edf, args.l_freq, args.h_freq)
            duration_cache[edf] = dur
        except Exception as exc:
            if args.verbose:
                print(f"[WARN] skipped EDF during scan prep: {patient} {edf.name}: {exc}")
            continue

        seizures_here = by_file.get(edf, [])
        clean = safe_intervals(dur, seizures_here, args.interictal_pre_buffer_seconds, args.interictal_post_buffer_seconds)
        for a, b in clean:
            clean_all.extend(scans_from_interval(patient, edf, a, b, args.segment_seconds, args.interictal_scan_stride_seconds, "interictal"))

        for s in seizures_here:
            ist = s.onset_sec
            ien = min(dur, ist + args.segment_seconds)
            if ien - ist >= args.segment_seconds - 1e-6:
                ictal_scans.append(Scan(patient, edf, "ictal", s.seizure_id, ist, ien, s.onset_sec, s.end_sec, s.label))

    rng = np.random.default_rng(args.data_seed)
    rng.shuffle(clean_all)
    if args.max_interictal_scans_per_patient > 0 and len(clean_all) > args.max_interictal_scans_per_patient:
        clean_all = clean_all[:args.max_interictal_scans_per_patient]

    n = len(clean_all)
    n_train = int(round(args.train_fraction * n))
    n_val = int(round(args.val_fraction * n))
    n_train = max(1, n_train) if n > 0 else 0
    n_val = max(1, n_val) if n - n_train > 0 else 0
    train_scans = clean_all[:n_train]
    val_scans = clean_all[n_train:n_train + n_val]
    test_scans = clean_all[n_train + n_val:]
    return train_scans, val_scans, test_scans, ictal_scans, patient_seizures


def scans_to_features(scans: Sequence[Scan], args) -> List[FeatureSequence]:
    cache: Dict[Path, Tuple[np.ndarray, float, List[str], float]] = {}
    out: List[FeatureSequence] = []
    for sc in scans:
        try:
            fs = build_feature_sequence(sc, cache, args)
            if fs is not None:
                out.append(fs)
        except Exception as exc:
            if args.verbose:
                print(f"[WARN] skipped {sc.region} {sc.edf_path.name} {sc.scan_id}: {exc}")
    return out


def run_patient(patient: str, args, edf_index: pd.DataFrame, all_seizures: List[Seizure]) -> List[Dict[str, object]]:
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    patient_out = out_dir / patient; patient_out.mkdir(exist_ok=True)
    train_scans, val_scans, test_scans, ictal_scans, patient_seizures = prepare_patient_scans(patient, args, edf_index, all_seizures)
    if args.verbose:
        print(f"{patient}: scans train={len(train_scans)} val={len(val_scans)} test={len(test_scans)} ictal={len(ictal_scans)} seizures={len(patient_seizures)}")
    train_raw=scans_to_features(train_scans,args); val_raw=scans_to_features(val_scans,args); test_raw=scans_to_features(test_scans,args); ictal_raw=scans_to_features(ictal_scans,args)
    if len(train_raw)==0 or len(val_raw)==0: raise RuntimeError(f"{patient}: insufficient train/validation interictal feature sequences")
    if len(ictal_raw)==0: raise RuntimeError(f"{patient}: no usable ictal onset feature sequences")
    train_mat=np.vstack([s.features for s in train_raw]); scaler=StandardScaler().fit(train_mat)
    pca_n=min(args.pca_components, train_mat.shape[0]-1, train_mat.shape[1])
    if pca_n<2: raise RuntimeError(f"{patient}: insufficient rows for PCA")
    pca=PCA(n_components=pca_n,random_state=args.data_seed).fit(scaler.transform(train_mat))
    train=transform_sequences(train_raw,scaler,pca); val=transform_sequences(val_raw,scaler,pca); test=transform_sequences(test_raw,scaler,pca); ictal=transform_sequences(ictal_raw,scaler,pca)
    Xtr,Ytr=collect_xy(train,args.washout,args.pred_horizon)
    model=RosslerDynML(pca_n,args); model.fit(Xtr,Ytr)
    mu,var=fit_diag_gaussian_from_validation(val,model,args.washout,args.pred_horizon)
    thr=validation_threshold(val,model,mu,var,args.washout,args.pred_horizon,args.threshold_percentile,args.chunk_seconds)
    bd_rng=np.random.default_rng(args.res_seed+104729)
    bd=model.bd_entropy(Xtr,q=args.bd_q,eps=args.bd_epsilon,n_steps=args.bd_steps,greedy_repeats=args.bd_greedy_repeats,rng=bd_rng)
    proj=model.projection_diagnostics(Xtr); params=model.parameter_summary()
    pd.DataFrame({'unit':np.arange(model.N),'a':model.a,'b':model.b,'c':model.c,'tau':model.tau}).to_csv(patient_out/f"{patient}_reservoir_parameters.csv",index=False)
    bd_row=dict(patient=patient,run_id=args.run_id,regime_id=args.regime_id,regime_name=model.regime['name'],res_seed=args.res_seed,data_seed=args.data_seed,pca_components=pca_n,num_reservoirs=model.N,integration_time=args.integration_time,dt=args.dt,**bd,**proj,**params)
    pd.DataFrame([bd_row]).to_csv(patient_out/f"{patient}_rossler_bd_entropy.csv",index=False)
    summaries=[]
    for persistence_sec in parse_float_list(args.alarm_persistence_seconds):
        psteps=max(1,int(math.ceil(persistence_sec/args.stride_seconds)))
        fp_scans=0; fp_eps=0; inter_hours=0.0; inter_rows=[]
        for s in test:
            ll,times=seq_loglik(s,model,mu,var,args.washout,args.pred_horizon,args.chunk_seconds)
            alarm,first_t,confirm_t,nflag=alarm_from_ll(ll,times,thr,psteps); eps=alarm_episodes(ll,thr,psteps)
            fp_scans+=int(alarm); fp_eps+=eps
            if len(s.chunk_starts_sec): inter_hours+=max(0.0,(s.chunk_starts_sec[-1]+args.chunk_seconds)-s.chunk_starts_sec[0])/3600.0
            inter_rows.append(dict(patient=patient,scan_id=s.scan_id,alarm_persistence_seconds=persistence_sec,alarm=bool(alarm),first_crossing_time_sec=first_t,confirmed_alarm_time_sec=confirm_t,n_flagged_steps=nflag,alarm_episodes=eps))
        tp=0; delays=[]; ictal_rows=[]
        for s in ictal:
            ll,times=seq_loglik(s,model,mu,var,args.washout,args.pred_horizon,args.chunk_seconds)
            alarm,first_t,confirm_t,nflag=alarm_from_ll(ll,times,thr,psteps)
            delay=float(confirm_t-s.onset_sec) if alarm and confirm_t is not None and s.onset_sec is not None else math.nan
            valid=bool(np.isfinite(delay) and delay>=0)
            if valid: tp+=1; delays.append(delay)
            ictal_rows.append(dict(patient=patient,seizure_id=s.scan_id,seizure_label=s.seizure_label,onset_sec=s.onset_sec,seizure_end_sec=s.end_seizure_sec,alarm_persistence_seconds=persistence_sec,alarm=valid,raw_alarm=bool(alarm),first_crossing_time_sec=first_t,confirmed_alarm_time_sec=confirm_t,detection_delay_seconds=delay,n_flagged_steps=nflag))
        n_i=len(ictal); n_t=len(test); sens=tp/n_i if n_i else math.nan; spec=(n_t-fp_scans)/n_t if n_t else math.nan; ba=np.nanmean([sens,spec]); far=fp_eps/inter_hours if inter_hours>0 else math.nan
        summary=dict(patient=patient,run_id=args.run_id,analysis='tusz_ictal_detection_rossler_dynml_v4',feature_mode='engineered',regime_id=args.regime_id,regime_name=model.regime['name'],data_seed=args.data_seed,res_seed=args.res_seed,pca_components=pca_n,threshold_percentile=args.threshold_percentile,alarm_persistence_seconds=persistence_sec,segment_seconds=args.segment_seconds,chunk_seconds=args.chunk_seconds,stride_seconds=args.stride_seconds,pred_horizon=args.pred_horizon,washout=args.washout,num_reservoirs=model.N,integration_time=args.integration_time,dt=args.dt,tau_min=args.tau_min,tau_max=args.tau_max,train_scans=len(train),val_scans=len(val),interictal_test_scans=n_t,interictal_fp_scans=fp_scans,interictal_tn_scans=max(0,n_t-fp_scans),specificity=float(spec),false_alarm_episodes=fp_eps,interictal_test_hours=float(inter_hours),false_alarms_per_hour=float(far),ictal_total=n_i,ictal_tp=tp,ictal_fn=max(0,n_i-tp),ictal_sensitivity=float(sens),balanced_accuracy=float(ba),ictal_delay_mean_seconds=float(np.mean(delays)) if delays else math.nan,ictal_delay_median_seconds=float(np.median(delays)) if delays else math.nan,**bd,**proj,**params)
        summaries.append(summary)
        suf=str(persistence_sec).replace('.','p')
        pd.DataFrame(inter_rows).to_csv(patient_out/f"{patient}_interictal_test_scans_pers{suf}s.csv",index=False)
        pd.DataFrame(ictal_rows).to_csv(patient_out/f"{patient}_ictal_events_pers{suf}s.csv",index=False)
    (patient_out/f"{patient}_run_summary.json").write_text(json.dumps(summaries,indent=2,sort_keys=True))
    return summaries

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="TUSZ ictal Rössler-DynML v4 + empirical BD")
    p.add_argument('--data_root',default='/data/halders2/EEG_anomaly/TUH_EEG_data/v2.0.6'); p.add_argument('--out_dir',default='tusz_ictal_rossler_bd_v4_results'); p.add_argument('--run_id',default='manual_run'); p.add_argument('--patients',default=''); p.add_argument('--min_seizures',type=int,default=31); p.add_argument('--montage',default='')
    p.add_argument('--l_freq',type=float,default=0.5); p.add_argument('--h_freq',type=float,default=40.0); p.add_argument('--segment_seconds',type=float,default=30.0); p.add_argument('--chunk_seconds',type=float,default=5.0); p.add_argument('--stride_seconds',type=float,default=1.0); p.add_argument('--min_sequence_chunks',type=int,default=20)
    p.add_argument('--interictal_pre_buffer_seconds',type=float,default=120.0); p.add_argument('--interictal_post_buffer_seconds',type=float,default=120.0); p.add_argument('--interictal_scan_stride_seconds',type=float,default=60.0); p.add_argument('--max_interictal_scans_per_patient',type=int,default=200); p.add_argument('--train_fraction',type=float,default=0.50); p.add_argument('--val_fraction',type=float,default=0.25)
    p.add_argument('--data_seed',type=int,default=17); p.add_argument('--res_seed',type=int,default=1107); p.add_argument('--pca_components',type=int,default=50); p.add_argument('--num_reservoirs',type=int,default=10); p.add_argument('--regime_id',type=int,default=4,choices=sorted(ROSSLER_REGIMES)); p.add_argument('--integration_time',type=float,default=20.0); p.add_argument('--dt',type=float,default=0.05); p.add_argument('--tau_min',type=float,default=0.8); p.add_argument('--tau_max',type=float,default=2.0); p.add_argument('--ridge',type=float,default=1e-4); p.add_argument('--washout',type=int,default=0); p.add_argument('--pred_horizon',type=int,default=1); p.add_argument('--threshold_percentile',type=float,default=5.0); p.add_argument('--alarm_persistence_seconds',default='3,5')
    p.add_argument('--bd_q',type=int,default=64); p.add_argument('--bd_epsilon',type=float,default=0.05); p.add_argument('--bd_steps',type=int,default=30); p.add_argument('--bd_greedy_repeats',type=int,default=25); p.add_argument('--verbose',action='store_true'); return p.parse_args()

def main() -> None:
    args=parse_args(); Path(args.out_dir).mkdir(parents=True,exist_ok=True)
    if args.tau_min<=0 or args.tau_max<args.tau_min or args.integration_time<=0: raise ValueError('Invalid Rössler integration parameters')
    print('Indexing TUSZ EDFs and annotations...'); edf_index,all_seizures=index_tusz(Path(args.data_root),args.montage)
    if edf_index.empty: raise RuntimeError(f'No EDFs found under {args.data_root} with montage={args.montage}')
    seizure_counts=pd.Series([s.patient for s in all_seizures]).value_counts() if all_seizures else pd.Series(dtype=int)
    patients=[x.strip() for x in args.patients.split(',') if x.strip()] if args.patients.strip() else sorted(seizure_counts[seizure_counts>=args.min_seizures].index.tolist())
    print(f'EDFs indexed: {len(edf_index)}'); print(f'Seizures indexed: {len(all_seizures)}'); print(f'Patients selected: {len(patients)}'); print(','.join(patients))
    rows=[]
    for patient in patients:
        try: rows.extend(run_patient(patient,args,edf_index,all_seizures))
        except Exception as exc:
            print(f'[ERROR] {patient}: {exc}')
            for pers in parse_float_list(args.alarm_persistence_seconds): rows.append(dict(patient=patient,run_id=args.run_id,regime_id=args.regime_id,data_seed=args.data_seed,res_seed=args.res_seed,alarm_persistence_seconds=pers,error=str(exc)))
    out_csv=Path(args.out_dir)/'all_patient_summary.csv'; pd.DataFrame(rows).to_csv(out_csv,index=False); print(f'Saved {out_csv}')

if __name__=='__main__': main()
