#!/usr/bin/env python3

from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PYTHON = "/data/halders2/conda/envs/dynml_ecg/bin/python"

SCRIPT = (
    "/data/halders2/EEG_anomaly/"
    "chbmit_continuous_patient_level_BD.py"
)

CORE_SCRIPT = (
    "/data/halders2/EEG_anomaly/"
    "chbmit_dynml_ictal_detection_BD_v4_fixed.py"
)

HELPER_SCRIPT = (
    "/data/halders2/EEG_anomaly/"
    "chbmit_continuous_rolling30s.py"
)

DATA_ROOT = "/data/halders2/EEG_anomaly/Data"

# Keep BD outputs separate from the already-computed continuous results.
BASE_OUT_DIR = (
    "/data/halders2/EEG_anomaly/"
    "chbmit_continuous_BD_results"
)

SWARM_FILE = (
    "/data/halders2/EEG_anomaly/"
    "swarm_chbmit_continuous_BD.swarm"
)

LOG_DIR = Path(
    "/data/halders2/EEG_anomaly/"
    "swarm_logs_chbmit_continuous_BD"
)

# ============================================================
# EXPERIMENT GRID
# ============================================================

PATIENTS = [f"chb{i:02d}" for i in range(1, 25)]

# All currently defined regimes R0-R22.
# If you only want the newest high-a refinements, use:
# REGIME_IDS = [18, 19, 20, 21, 22]
REGIME_IDS = list(range(23))

RES_SEEDS = [
    1107,
    1249,
    1388,
    1523,
    1697,
]

N_VALUES = [5, 10, 20, 30, 40, 50]

# ============================================================
# BD SETTINGS
# These match the defaults used in the BD-enabled core analysis.
# ============================================================

BD_Q = 64
BD_EPSILON = 0.05
BD_STEPS = 30
BD_GREEDY_REPEATS = 25

# False = calculate EVERYTHING together:
#   BD entropy
#   continuous sensitivity
#   FAR/hour
#   detection delay
#
# True = fit model + calculate/save BD only,
#        without rescoring held-out continuous EDFs.
BD_ONLY = False

# ============================================================
# SWARM RESOURCE SETTINGS
# ============================================================

MEM_GB = 16
THREADS = 4
WALLTIME = "12:00:00"

# ============================================================
# GENERATE SWARM COMMANDS
# ============================================================

commands = []

for patient in PATIENTS:
    for regime_id in REGIME_IDS:
        for N in N_VALUES:
            for res_seed in RES_SEEDS:

                out_dir = (
                    f"{BASE_OUT_DIR}/"
                    f"N_{N}/"
                    f"regime_{regime_id}"
                )

                cmd = (
                    f"{PYTHON} {SCRIPT} "
                    f"--core_script {CORE_SCRIPT} "
                    f"--helper_script {HELPER_SCRIPT} "
                    f"--data_root {DATA_ROOT} "
                    f"--patient {patient} "
                    f"--regime_id {regime_id} "
                    f"--res_seed {res_seed} "
                    f"--num_reservoirs {N} "
                    f"--persistence_seconds 3,5 "
                    f"--bd_q {BD_Q} "
                    f"--bd_epsilon {BD_EPSILON} "
                    f"--bd_steps {BD_STEPS} "
                    f"--bd_greedy_repeats {BD_GREEDY_REPEATS} "
                    f"--out_dir {out_dir}"
                )

                if BD_ONLY:
                    cmd += " --bd_only"

                commands.append(cmd)

# ============================================================
# WRITE SWARM FILE
# ============================================================

swarm_path = Path(SWARM_FILE)
swarm_path.parent.mkdir(parents=True, exist_ok=True)
swarm_path.write_text("\n".join(commands) + "\n")

# Make log directory now so swarm has somewhere to write logs.
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SUMMARY
# ============================================================

expected_jobs = (
    len(PATIENTS)
    * len(REGIME_IDS)
    * len(N_VALUES)
    * len(RES_SEEDS)
)

print("=" * 72)
print("CHB-MIT CONTINUOUS + BD ENTROPY SWARM")
print("=" * 72)

print(f"Patients:             {len(PATIENTS)}")
print(f"Regimes:              {len(REGIME_IDS)}  {REGIME_IDS}")
print(f"N values:             {len(N_VALUES)}  {N_VALUES}")
print(f"Reservoir seeds:      {len(RES_SEEDS)}")
print(f"BD-only mode:         {BD_ONLY}")

if BD_ONLY:
    print("Run mode:             BD entropy only")
else:
    print("Run mode:             FULL continuous evaluation + BD entropy")

print(f"Total jobs:           {len(commands)}")
print(f"Expected jobs:        {expected_jobs}")

print(f"Swarm file:           {swarm_path}")
print(f"Output root:          {BASE_OUT_DIR}")
print(f"Log directory:        {LOG_DIR}")

# ============================================================
# PREVIEW COMMANDS
# ============================================================

print("\nFirst 3 commands:")
for cmd in commands[:3]:
    print(cmd)

print("\nLast 3 commands:")
for cmd in commands[-3:]:
    print(cmd)

# ============================================================
# PRINT SWARM SUBMISSION COMMAND
# ============================================================

swarm_submit_command = (
    f"swarm "
    f"-f {swarm_path} "
    f"-g {MEM_GB} "
    f"-t {THREADS} "
    f"--time={WALLTIME} "
    f"--logdir {LOG_DIR}"
)

print("\n" + "=" * 72)
print("SUBMIT WITH:")
print("=" * 72)
print(swarm_submit_command)
print("=" * 72)