from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STEPS = [
    ["python", "pipeline/01_data_loading.py"],
    ["python", "pipeline/02_eda.py"],
    ["python", "pipeline/03_preprocessing.py"],
    ["python", "pipeline/04_model_training.py"],
    ["python", "pipeline/05_evaluation.py"],
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full flight delay pipeline")
    parser.add_argument("--use-full-data", action="store_true")
    parser.add_argument("--sample-size", type=int, default=500000)
    args = parser.parse_args()

    for idx, cmd in enumerate(STEPS):
        run_cmd = cmd.copy()
        if idx == 0:
            if args.use_full_data:
                run_cmd.append("--use-full-data")
            else:
                run_cmd.extend(["--sample-size", str(args.sample_size)])

        print("Running:", " ".join(run_cmd))
        subprocess.run(run_cmd, check=True)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
