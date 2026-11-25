#!/usr/bin/env python3
"""Generate one deterministic linear-baseline job per case and task."""
import argparse
from pathlib import Path
import shlex
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project_dir", default="/data/halders2/EEG_anomaly")
    p.add_argument("--data_root", default="/data/halders2/EEG_anomaly/Data")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--tasks", choices=["both", "ictal", "preictal"], default="both")
    p.add_argument("--patients", default=",".join(f"chb{i:02d}" for i in range(1, 25)))
    p.add_argument("--output_root", default="chbmit_linear_baseline_results")
    p.add_argument("--swarm_file", default="swarm_chbmit_linear_baseline.swarm")
    args = p.parse_args()
    project = Path(args.project_dir).resolve()
    runner = project / "chbmit_linear_baseline.py"
    tasks = ["ictal", "preictal"] if args.tasks == "both" else [args.tasks]
    patients = [x.strip() for x in args.patients.split(",") if x.strip()]
    if len(set(patients)) != len(patients) or not patients:
        raise ValueError("Patient list must be nonempty and unique")
    output = Path(args.output_root)
    if not output.is_absolute():
        output = project / output
    commands = []
    for task in tasks:
        core_name = ("chbmit_dynml_ictal_detection_BD_v4_fixed.py" if task == "ictal"
                     else "chbmit_dynml_preictal_35to5_BD_v4.py")
        core = project / core_name
        if not runner.is_file() or not core.is_file():
            raise FileNotFoundError(f"Place the runner and {core_name} in {project}")
        for patient in patients:
            cmd = [args.python, str(runner), "--task", task, "--core_script", str(core),
                   "--output_root", str(output), "--data_root", args.data_root,
                   "--patient", patient, "--data_seed", "17", "--pca_components", "30",
                   "--ridge", "0.0001", "--washout", "0", "--pred_horizon", "1",
                   "--threshold_percentile", "5", "--val_fraction", "0.25",
                   "--test_fraction", "0.25", "--max_interictal_scans_per_patient", "120",
                   "--interictal_pre_buffer_seconds", "3600",
                   "--interictal_post_buffer_seconds", "3600", "--l_freq", "0.5",
                   "--h_freq", "40", "--min_sequence_chunks", "20",
                   "--segment_seconds", "30" if task == "ictal" else "1800",
                   "--chunk_seconds", "5" if task == "ictal" else "15",
                   "--stride_seconds", "1" if task == "ictal" else "5",
                   "--interictal_scan_stride_seconds", "30" if task == "ictal" else "1800",
                   "--alarm_persistence_seconds", "3,5" if task == "ictal" else "60",
                   "--verbose"]
            if task == "preictal":
                cmd += ["--preictal_start_minutes", "35", "--preictal_end_minutes", "5"]
            commands.append(shlex.join(cmd))
    swarm = project / args.swarm_file
    if swarm.exists():
        raise FileExistsError(f"{swarm} exists; choose a new --swarm_file")
    swarm.write_text("\n".join(commands) + "\n")
    logs = project / (swarm.stem + "_logs")
    logs.mkdir(parents=True, exist_ok=True)
    print(f"Wrote {len(commands)} jobs: {swarm}")
    print(f"Results: {output}")
    print("Submit one test job per task before launching all jobs.")
    print(shlex.join(["swarm", "-f", str(swarm), "-g", "12", "-t", "4",
                      "--time=6:00:00", "--logdir", str(logs)]))


if __name__ == "__main__":
    main()
