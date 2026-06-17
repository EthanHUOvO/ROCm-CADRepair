import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

PYTHON = "/opt/python/bin/python"


def run(cmd, timeout_sec):
    print("RUN:", cmd)
    popen_kwargs = {"shell": True}
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)
    try:
        proc.wait(timeout=timeout_sec)
        return proc.returncode
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout_sec} sec")
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(2)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
        return "TimeoutExpired"


def read_json(path):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def write_timeout_json(path, cid, stage, timeout_sec):
    data = {
        "candidate": cid,
        "success": False,
        "stage": stage,
        "error_type": "TimeoutExpired",
        "error": f"CadQuery execution exceeded {timeout_sec} sec",
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=int(os.environ.get("CADQUERY_TIMEOUT_SEC", "180")),
        help="Timeout for each CadQuery execution stage.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    cid = args.candidate

    raw_log = outdir / f"verify_candidate_{cid}.json"
    safe_log = outdir / f"verify_candidate_{cid}_safe.json"
    final_log = outdir / f"pipeline_candidate_{cid}.json"

    raw_status = run(
        f"{PYTHON} src/run_cadquery.py "
        f"--code {args.code} "
        f"--outdir {args.outdir} "
        f"--candidate {cid}",
        args.timeout_sec,
    )

    if raw_status == "TimeoutExpired" and not raw_log.exists():
        raw = write_timeout_json(raw_log, cid, "raw", args.timeout_sec)
    else:
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

    safe_status = run(
        f"{PYTHON} src/run_cadquery_safe.py "
        f"--code {args.code} "
        f"--outdir {args.outdir} "
        f"--candidate {cid} "
        f"--safe",
        args.timeout_sec,
    )

    if safe_status == "TimeoutExpired" and not safe_log.exists():
        safe = write_timeout_json(safe_log, cid, "safe", args.timeout_sec)
    else:
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
