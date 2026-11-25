#!/usr/bin/env python3
from pathlib import Path
from itertools import product

SCRIPT = "chbmit_dynml_preictal_35to5_BD_v4.py"
DATA_ROOT = "/data/halders2/EEG_anomaly/Data"

# OUT_ROOT = "chbmit_preictal_35to5_rossler_bd_v4_results"
# SWARM_OUT = "swarm_chbmit_preictal_35to5_BD_v4.swarm"
# LOG_ROOT = "swarm_logs_chbmit_preictal_35to5_bd_v4"

# New directories for the non-overlap run for revision
OUT_ROOT = "chbmit_preictal_35to5_nonoverlap_rossler_bd_v4_results"
SWARM_OUT = "swarm_chbmit_preictal_35to5_nonoverlap_BD_v4.swarm"
LOG_ROOT = "swarm_logs_chbmit_preictal_35to5_nonoverlap_bd_v4"

PATIENTS = [f"chb{i:02d}" for i in range(1, 25)]
REGIME_IDS = list(range(12))
DATA_SEEDS = [17]
RES_SEEDS = [1107, 1249, 1388, 1523, 1697]

# Long-warning analysis: SOP=30 min, SPH=5 min -> -35 to -5 min.
PREICTAL_START_MINUTES = 35
PREICTAL_END_MINUTES = 5
SEGMENT_SECONDS = 1800
CHUNK_SECONDS = 15
STRIDE_SECONDS = 5
MIN_SEQUENCE_CHUNKS = 20
PERSISTENCE_SECONDS = "60"

# Keep the v4 DynML/BD settings fixed.
PCA_COMPONENTS = 30
THRESHOLD_PERCENTILE = 5
INTERICTAL_PRE_BUFFER_SECONDS = 3600
INTERICTAL_POST_BUFFER_SECONDS = 3600

# INTERICTAL_SCAN_STRIDE_SECONDS = 300
INTERICTAL_SCAN_STRIDE_SECONDS = 1800 # changed to remove overlap

MAX_INTERICTAL_SCANS_PER_PATIENT = 120
NUM_RESERVOIRS = 10
PRED_HORIZON = 1
WASHOUT = 0
INTEGRATION_TIME = 20.0
DT = 0.05
TAU_MIN = 0.8
TAU_MAX = 2.0
RIDGE = 1e-4
BD_Q = 64
BD_EPSILON = 0.05
BD_STEPS = 30
BD_GREEDY_REPEATS = 25


def main():
    Path(LOG_ROOT).mkdir(parents=True, exist_ok=True)
    lines = []
    for patient, regime, data_seed, res_seed in product(PATIENTS, REGIME_IDS, DATA_SEEDS, RES_SEEDS):
        run_id = f"{patient}_preictal35to5_reg{regime}_ds{data_seed}_rs{res_seed}"
        out_dir = f"{OUT_ROOT}/{run_id}"
        cmd = (
            f"python3 {SCRIPT} "
            f"--data_root {DATA_ROOT} "
            f"--patient {patient} "
            f"--out_dir {out_dir} "
            f"--run_id {run_id} "
            f"--preictal_start_minutes {PREICTAL_START_MINUTES} "
            f"--preictal_end_minutes {PREICTAL_END_MINUTES} "
            f"--segment_seconds {SEGMENT_SECONDS} "
            f"--chunk_seconds {CHUNK_SECONDS} "
            f"--stride_seconds {STRIDE_SECONDS} "
            f"--min_sequence_chunks {MIN_SEQUENCE_CHUNKS} "
            f"--interictal_pre_buffer_seconds {INTERICTAL_PRE_BUFFER_SECONDS} "
            f"--interictal_post_buffer_seconds {INTERICTAL_POST_BUFFER_SECONDS} "
            f"--interictal_scan_stride_seconds {INTERICTAL_SCAN_STRIDE_SECONDS} "
            f"--max_interictal_scans_per_patient {MAX_INTERICTAL_SCANS_PER_PATIENT} "
            f"--pca_components {PCA_COMPONENTS} "
            f"--threshold_percentile {THRESHOLD_PERCENTILE} "
            f"--alarm_persistence_seconds {PERSISTENCE_SECONDS} "
            f"--regime_id {regime} "
            f"--data_seed {data_seed} "
            f"--res_seed {res_seed} "
            f"--num_reservoirs {NUM_RESERVOIRS} "
            f"--pred_horizon {PRED_HORIZON} "
            f"--washout {WASHOUT} "
            f"--integration_time {INTEGRATION_TIME} "
            f"--dt {DT} "
            f"--tau_min {TAU_MIN} "
            f"--tau_max {TAU_MAX} "
            f"--ridge {RIDGE} "
            f"--bd_q {BD_Q} "
            f"--bd_epsilon {BD_EPSILON} "
            f"--bd_steps {BD_STEPS} "
            f"--bd_greedy_repeats {BD_GREEDY_REPEATS}"
        )
        lines.append(cmd)

    Path(SWARM_OUT).write_text("\n".join(lines) + "\n")
    print(f"Wrote {SWARM_OUT} with {len(lines)} jobs")
    print(f"Results root: {OUT_ROOT}")
    print(f"Log root:     {LOG_ROOT}")
    print("Submit with:")
    print(f"swarm -f {SWARM_OUT} -g 12 -t 4 --time=6:00:00 --logdir {LOG_ROOT}")


if __name__ == "__main__":
    main()
