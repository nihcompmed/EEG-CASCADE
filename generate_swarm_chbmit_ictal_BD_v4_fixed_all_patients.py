#!/usr/bin/env python3
from pathlib import Path
from itertools import product

# Main analysis script tested in the 30-job chb01 pilot.
SCRIPT = "chbmit_dynml_ictal_detection_BD_v4_fixed.py"
DATA_ROOT = "/data/halders2/EEG_anomaly/Data"

# Keep results and logs completely separate.
OUT_ROOT = Path("chbmit_ictal_rossler_bd_v4_results")
LOG_ROOT = Path("swarm_logs_chbmit_ictal_bd_v4")
SWARM_OUT = Path("swarm_chbmit_ictal_BD_v4_all_patients.swarm")

# Same CHB-MIT cohort convention used in the earlier ictal generator.
PATIENTS = [f"chb{i:02d}" for i in range(1, 25)]

# Fixed experimental settings.
REGIME_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
DATA_SEEDS = [17]
RES_SEEDS = [1107, 1249, 1388, 1523, 1697]

PCA_COMPONENTS = 30
THRESHOLD_PERCENTILE = 5.0
# Both persistence values are evaluated inside ONE fitted model/run.
PERSISTENCE_SECONDS = "3,5"

SEGMENT_SECONDS = 30.0
ICTAL_START_SECONDS = 0.0
ICTAL_SCAN_SECONDS = 30.0
CHUNK_SECONDS = 5.0
STRIDE_SECONDS = 1.0
MIN_SEQUENCE_CHUNKS = 20

INTERICTAL_PRE_BUFFER_SECONDS = 3600.0
INTERICTAL_POST_BUFFER_SECONDS = 3600.0
INTERICTAL_SCAN_STRIDE_SECONDS = 30.0
MAX_INTERICTAL_SCANS_PER_PATIENT = 120
VAL_FRACTION = 0.25
TEST_FRACTION = 0.25

NUM_RESERVOIRS = 10
INTEGRATION_TIME = 20.0
DT = 0.05
TAU_MIN = 0.8
TAU_MAX = 2.0
RIDGE = 1e-4
WASHOUT = 0
PRED_HORIZON = 1

BD_Q = 64
BD_EPSILON = 0.05
BD_STEPS = 30
BD_GREEDY_REPEATS = 25


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    lines = []

    for patient, regime_id, data_seed, res_seed in product(
        PATIENTS, REGIME_IDS, DATA_SEEDS, RES_SEEDS
    ):
        run_id = f"{patient}_ictal_reg{regime_id}_ds{data_seed}_rs{res_seed}"
        out_dir = OUT_ROOT / run_id

        cmd = (
            f"python3 {SCRIPT} "
            f"--data_root {DATA_ROOT} "
            f"--patient {patient} "
            f"--out_dir {out_dir} "
            f"--run_id {run_id} "
            f"--segment_seconds {SEGMENT_SECONDS} "
            f"--ictal_start_seconds {ICTAL_START_SECONDS} "
            f"--ictal_scan_seconds {ICTAL_SCAN_SECONDS} "
            f"--chunk_seconds {CHUNK_SECONDS} "
            f"--stride_seconds {STRIDE_SECONDS} "
            f"--min_sequence_chunks {MIN_SEQUENCE_CHUNKS} "
            f"--interictal_pre_buffer_seconds {INTERICTAL_PRE_BUFFER_SECONDS} "
            f"--interictal_post_buffer_seconds {INTERICTAL_POST_BUFFER_SECONDS} "
            f"--interictal_scan_stride_seconds {INTERICTAL_SCAN_STRIDE_SECONDS} "
            f"--max_interictal_scans_per_patient {MAX_INTERICTAL_SCANS_PER_PATIENT} "
            f"--val_fraction {VAL_FRACTION} "
            f"--test_fraction {TEST_FRACTION} "
            f"--pca_components {PCA_COMPONENTS} "
            f"--threshold_percentile {THRESHOLD_PERCENTILE} "
            f"--alarm_persistence_seconds {PERSISTENCE_SECONDS} "
            f"--regime_id {regime_id} "
            f"--data_seed {data_seed} "
            f"--res_seed {res_seed} "
            f"--num_reservoirs {NUM_RESERVOIRS} "
            f"--integration_time {INTEGRATION_TIME} "
            f"--dt {DT} "
            f"--tau_min {TAU_MIN} "
            f"--tau_max {TAU_MAX} "
            f"--ridge {RIDGE} "
            f"--washout {WASHOUT} "
            f"--pred_horizon {PRED_HORIZON} "
            f"--bd_q {BD_Q} "
            f"--bd_epsilon {BD_EPSILON} "
            f"--bd_steps {BD_STEPS} "
            f"--bd_greedy_repeats {BD_GREEDY_REPEATS}"
        )
        lines.append(cmd)

    SWARM_OUT.write_text("\n".join(lines) + "\n")

    print(f"Wrote {SWARM_OUT} with {len(lines)} jobs")
    print(f"Results root: {OUT_ROOT}")
    print(f"Log root:     {LOG_ROOT}")
    print()
    print("Each job evaluates BOTH 3 s and 5 s persistence from the same fitted model.")
    print()
    print("Submit with:")
    print(
        f"swarm -f {SWARM_OUT} -g 12 -t 4 --time=6:00:00 "
        f"--logdir {LOG_ROOT}"
    )


if __name__ == "__main__":
    main()
