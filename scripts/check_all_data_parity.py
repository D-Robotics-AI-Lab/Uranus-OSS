"""Run the Uranus-data parity suite for every exported sample directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=Path("examples/data_v4"),
        help="directory containing numbered sample directories",
    )
    parser.add_argument(
        "--data-root",
        default="tos://uranus-data/las/lance-dataset/robotics/",
    )
    args = parser.parse_args()

    samples = sorted(path.parent for path in args.sample_root.glob("*/meta.json"))
    if not samples:
        raise SystemExit(f"no exported samples found under {args.sample_root}")
    for sample in samples:
        print(f"[parity] {sample}", flush=True)
        env = os.environ.copy()
        env["URANUS_PARITY_SAMPLE"] = str(sample)
        env["URANUS_DATA_ROOT"] = args.data_root
        subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "tests.test_data_v3_parity",
                "-v",
            ],
            check=True,
            env=env,
        )
    print(f"[parity] all {len(samples)} datasets passed", flush=True)


if __name__ == "__main__":
    main()
