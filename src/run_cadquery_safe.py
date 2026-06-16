import argparse
import json
import re
import time
import traceback
from pathlib import Path

import cadquery as cq


def clean_code(raw_code: str) -> str:
    code = raw_code.replace("```python", "").replace("```", "").strip()

    lines = []
    start = False

    for line in code.splitlines():
        s = line.strip()
        if s.startswith("import ") or s.startswith("from ") or "cq." in line or "result" in line:
            start = True
        if start:
            lines.append(line)

    code = "\n".join(lines).strip()

    if not code:
        code = raw_code.replace("```python", "").replace("```", "").strip()

    return code


def remove_dangerous_ops(code: str) -> str:
    code = re.sub(r"\.shell\([^\)]*\)", "", code)
    code = re.sub(r"\.fillet\([^\)]*\)", "", code)
    code = re.sub(r"\.chamfer\([^\)]*\)", "", code)
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--candidate", type=int, default=0)
    parser.add_argument("--safe", action="store_true")
    args = parser.parse_args()

    code_path = Path(args.code)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    step_path = outdir / f"candidate_{args.candidate}.step"
    stl_path = outdir / f"candidate_{args.candidate}.stl"
    log_path = outdir / f"verify_candidate_{args.candidate}_safe.json"
    cleaned_code_path = outdir / f"candidate_{args.candidate}_cleaned.py"

    raw_code = code_path.read_text(encoding="utf-8")
    code = clean_code(raw_code)

    if args.safe:
        code = remove_dangerous_ops(code)

    cleaned_code_path.write_text(code, encoding="utf-8")

    namespace = {
        "cq": cq,
        "cadquery": cq,
    }

    success = False
    error_type = None
    error_message = None

    t0 = time.time()

    try:
        exec(code, namespace)

        result = namespace.get("result", None)
        if result is None:
            raise RuntimeError("No variable named result found in generated code.")

        cq.exporters.export(result, str(step_path))
        cq.exporters.export(result, str(stl_path))
        success = True

    except Exception:
        error_type = traceback.format_exc().splitlines()[-1].split(":")[0]
        error_message = traceback.format_exc()

    t1 = time.time()

    log = {
        "candidate": args.candidate,
        "safe_mode": args.safe,
        "original_code": str(code_path),
        "cleaned_code": str(cleaned_code_path),
        "success": success,
        "cadquery_time_sec": t1 - t0,
        "step_path": str(step_path) if success else None,
        "stl_path": str(stl_path) if success else None,
        "error_type": error_type,
        "error_message": error_message,
    }

    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    if success:
        print("Candidate success:", args.candidate)
        print("STEP:", step_path)
        print("STL:", stl_path)
    else:
        print("Candidate failed:", args.candidate)
        print("Error:", error_type)
        print("See:", log_path)


if __name__ == "__main__":
    main()
