import argparse
import json
import subprocess
from pathlib import Path

PYTHON = "/opt/python/bin/python"


def run(cmd):
    print("RUN:", cmd)
    subprocess.run(cmd, shell=True)


def read_json(path):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--candidate", type=int, required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    cid = args.candidate

    raw_log = outdir / f"verify_candidate_{cid}.json"
    safe_log = outdir / f"verify_candidate_{cid}_safe.json"
    final_log = outdir / f"pipeline_candidate_{cid}.json"

    run(
        f"{PYTHON} src/run_cadquery.py "
        f"--code {args.code} "
        f"--outdir {args.outdir} "
        f"--candidate {cid}"
    )

    raw = read_json(raw_log)

    if raw.get("success") is True:
        final = {
            "candidate": cid,
            "final_success": True,
            "selected_stage": "raw",
            "step_path": raw.get("step_path"),
            "stl_path": raw.get("stl_path"),
            "raw_error_type": None,
            "safe_error_type": None,
        }
        final_log.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(final, indent=2, ensure_ascii=False))
        return

    run(
        f"{PYTHON} src/run_cadquery_safe.py "
        f"--code {args.code} "
        f"--outdir {args.outdir} "
        f"--candidate {cid} "
        f"--safe"
    )

    safe = read_json(safe_log)

    final = {
        "candidate": cid,
        "final_success": safe.get("success") is True,
        "selected_stage": "safe" if safe.get("success") is True else "failed",
        "step_path": safe.get("step_path"),
        "stl_path": safe.get("stl_path"),
        "raw_error_type": raw.get("error_type"),
        "safe_error_type": safe.get("error_type"),
    }

    final_log.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
