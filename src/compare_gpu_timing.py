import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--parallel", required=True)
    parser.add_argument("--out", default="docs/results/gpu_timing_comparison.csv")
    args = parser.parse_args()

    serial = pd.read_csv(args.serial).iloc[0]
    parallel = pd.read_csv(args.parallel).iloc[0]

    serial_time = float(serial["wall_time_sec"])
    parallel_time = float(parallel["wall_time_sec"])
    speedup = serial_time / parallel_time if parallel_time > 0 else None
    efficiency = speedup / float(parallel["gpus"]) if speedup is not None else None

    row = {
        "serial_wall_time_sec": serial_time,
        "parallel_wall_time_sec": parallel_time,
        "speedup": speedup,
        "parallel_efficiency": efficiency,
        "serial_gpus": int(serial["gpus"]),
        "parallel_gpus": int(parallel["gpus"]),
        "parts": int(parallel["parts"]),
        "candidate": int(parallel["candidate"]),
        "max_tokens": int(parallel["max_tokens"]),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out, index=False)
    print(pd.DataFrame([row]))
    print("Saved:", out)


if __name__ == "__main__":
    main()
