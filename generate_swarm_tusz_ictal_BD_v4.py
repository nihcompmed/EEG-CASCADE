#!/usr/bin/env python3
from pathlib import Path
from itertools import product

SCRIPT = "tusz_dynml_ictal_detection_BD_v4.py"
DATA_ROOT = "/data/halders2/EEG_anomaly/TUH_EEG_data/v2.0.6"
OUT_ROOT = "tusz_ictal_rossler_bd_v4_results"
SWARM_OUT = "swarm_tusz_ictal_BD_v4.swarm"

# 16 patients
# PATIENTS = [
#     "aaaaapks", "aaaaaqtx", "aaaaahie", "aaaaajqo",
#     "aaaaajdn", "aaaaajru", "aaaaaovk", "aaaaabhz",
#     "aaaaardk", "aaaaampk", "aaaaahyp", "aaaaaprj",
#     "aaaaakfo", "aaaaafcf", "aaaaaqkb", "aaaaarsm",
# ]
# # 15 pateinst: removed "aaaaajdn" no valid interictal segments 
# PATIENTS = [
#     "aaaaapks", "aaaaaqtx", "aaaaahie", "aaaaajqo",
#     "aaaaajru", "aaaaaovk", "aaaaabhz",
#     "aaaaardk", "aaaaampk", "aaaaahyp", "aaaaaprj",
#     "aaaaakfo", "aaaaafcf", "aaaaaqkb", "aaaaarsm",
# ]

PATIENTS = ['aaaaapks', 'aaaaaqtx', 'aaaaahie', 'aaaaajqo', 'aaaaajru', 'aaaaaovk', 'aaaaardk', 'aaaaabhz', 'aaaaampk', 'aaaaaprj', 'aaaaafcf', 'aaaaaqkb', 'aaaaarsm', 'aaaaaedy', 'aaaaardf', 'aaaaaelb', 'aaaaacqc', 'aaaaatez', 'aaaaanme', 'aaaaaghb', 'aaaaaksg', 'aaaaamqq', 'aaaaaskf', 'aaaaatjh', 'aaaaates', 'aaaaarpv', 'aaaaaizz', 'aaaaafjw', 'aaaaaooo', 'aaaaadkj', 'aaaaarnq', 'aaaaaroo', 'aaaaaasy', 'aaaaajsl', 'aaaaakkm', 'aaaaakoe', 'aaaaamod', 'aaaaaasf', 'aaaaamoa', 'aaaaaqvx', 'aaaaamhj', 'aaaaarsp', 'aaaaaswc', 'aaaaalfs', 'aaaaamta', 'aaaaaqbb', 'aaaaaqtw', 'aaaaaqld', 'aaaaammu', 'aaaaasgd', 'aaaaapas', 'aaaaamnk', 'aaaaajfq', 'aaaaatds', 'aaaaaksh', 'aaaaaccj', 'aaaaaqrs', 'aaaaaron', 'aaaaasoc', 'aaaaajax', 'aaaaaayf', 'aaaaaarq', 'aaaaajmr']

# REGIME_IDS = list(range(12))
REGIME_IDS = [4]
DATA_SEEDS = [17]
RES_SEEDS = [1107, 1249, 1388, 1523, 1697]

PCA_COMPONENTS = 30
THRESHOLD_PERCENTILE = 5
PERSISTENCE_SECONDS = "3,5"
SEGMENT_SECONDS = 30
CHUNK_SECONDS = 5
STRIDE_SECONDS = 1
MIN_SEQUENCE_CHUNKS = 20
INTERICTAL_PRE_BUFFER_SECONDS = 120
INTERICTAL_POST_BUFFER_SECONDS = 120
INTERICTAL_SCAN_STRIDE_SECONDS = 60
MAX_INTERICTAL_SCANS_PER_PATIENT = 200
NUM_RESERVOIRS = 10
INTEGRATION_TIME = 20
DT = 0.05
TAU_MIN = 0.8
TAU_MAX = 2.0
RIDGE = 1e-4
BD_Q = 64
BD_EPSILON = 0.05
BD_STEPS = 30
BD_GREEDY_REPEATS = 25

lines = []
for patient, reg, ds, rs in product(PATIENTS, REGIME_IDS, DATA_SEEDS, RES_SEEDS):
    run_id = f"{patient}_ictal_reg{reg}_ds{ds}_rs{rs}"
    out_dir = f"{OUT_ROOT}/{run_id}"
    cmd = (
        f"python3 {SCRIPT} "
        f"--data_root {DATA_ROOT} --out_dir {out_dir} --run_id {run_id} "
        f"--patients {patient} --montage \"\" "
        f"--pca_components {PCA_COMPONENTS} --threshold_percentile {THRESHOLD_PERCENTILE} "
        f"--alarm_persistence_seconds {PERSISTENCE_SECONDS} --regime_id {reg} "
        f"--data_seed {ds} --res_seed {rs} --segment_seconds {SEGMENT_SECONDS} "
        f"--chunk_seconds {CHUNK_SECONDS} --stride_seconds {STRIDE_SECONDS} "
        f"--min_sequence_chunks {MIN_SEQUENCE_CHUNKS} "
        f"--interictal_pre_buffer_seconds {INTERICTAL_PRE_BUFFER_SECONDS} "
        f"--interictal_post_buffer_seconds {INTERICTAL_POST_BUFFER_SECONDS} "
        f"--interictal_scan_stride_seconds {INTERICTAL_SCAN_STRIDE_SECONDS} "
        f"--max_interictal_scans_per_patient {MAX_INTERICTAL_SCANS_PER_PATIENT} "
        f"--num_reservoirs {NUM_RESERVOIRS} --integration_time {INTEGRATION_TIME} --dt {DT} "
        f"--tau_min {TAU_MIN} --tau_max {TAU_MAX} --ridge {RIDGE} --washout 0 --pred_horizon 1 "
        f"--bd_q {BD_Q} --bd_epsilon {BD_EPSILON} --bd_steps {BD_STEPS} "
        f"--bd_greedy_repeats {BD_GREEDY_REPEATS}"
    )
    lines.append(cmd)

Path(SWARM_OUT).write_text("\n".join(lines) + "\n")
print(f"Wrote {SWARM_OUT} with {len(lines)} jobs")
print("Each job fits once and evaluates both 3-s and 5-s persistence.")
print("Submit:")
print(f"swarm -f {SWARM_OUT} -g 8 -t 4 --time=8:00:00 --logdir swarm_logs_tusz_ictal_bd_v4")
