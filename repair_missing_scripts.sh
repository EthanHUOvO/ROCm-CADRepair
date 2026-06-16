#!/bin/bash
set -e

mkdir -p src scripts outputs docs/results

echo "===== Create src/infer_one.py ====="
cat > src/infer_one.py <<'PY'
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration as ModelClass
except Exception:
    from transformers import AutoModelForCausalLM as ModelClass


def load_images(view_dir):
    view_dir = Path(view_dir)
    images = []
    for i in range(8):
        p = view_dir / f"view_{i}.png"
        if not p.exists():
            raise FileNotFoundError(f"Missing image: {p}")
        images.append(Image.open(p).convert("RGB"))
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/Zero-To-CAD-Qwen3-VL-2B")
    parser.add_argument("--views", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--candidate", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--do-sample", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    code_path = outdir / f"candidate_{args.candidate}.py"
    log_path = outdir / f"infer_candidate_{args.candidate}.json"

    print("[1/5] Loading images...")
    images = load_images(args.views)

    print("[2/5] Loading processor...")
    try:
        processor = AutoProcessor.from_pretrained(
            args.model,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
    except TypeError:
        processor = AutoProcessor.from_pretrained(
            args.model,
            trust_remote_code=True,
        )

    print("[3/5] Loading model on ROCm GPU...")
    try:
        model = ModelClass.from_pretrained(
            args.model,
            dtype=torch.float16,
            trust_remote_code=True,
        ).to("cuda")
    except TypeError:
        model = ModelClass.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to("cuda")

    model.eval()

    prompt = (
        "Generate robust executable CadQuery Python code for this CAD object. "
        "Only output Python code. "
        "The final CAD object must be stored in a variable named result. "
        "Prefer simple primitives such as box, cylinder, circle, extrude, union, cut, and hole. "
        "Avoid unnecessary shell, complex fillet, complex chamfer, loft, sweep, and fragile face selection. "
        "The code must be executable in CadQuery and exportable to STEP/STL. "
        "Do not include explanation or markdown."
    )

    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in images],
                {"type": "text", "text": prompt},
            ],
        }
    ]

    print("[4/5] Preparing input...")
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=text,
        images=images,
        return_tensors="pt",
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()

    print("[5/5] Running inference...")
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }

    if args.do_sample:
        gen_kwargs["temperature"] = args.temperature

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    torch.cuda.synchronize()
    t1 = time.time()

    generated = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0]

    code_path.write_text(generated, encoding="utf-8")

    log = {
        "candidate": args.candidate,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "inference_time_sec": t1 - t0,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "gpu_name": torch.cuda.get_device_name(0),
        "output_code": str(code_path),
    }

    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Generated code:", code_path)
    print("Inference time:", round(log["inference_time_sec"], 2), "sec")
    print("Peak VRAM:", round(log["peak_vram_gb"], 2), "GB")


if __name__ == "__main__":
    main()
PY

echo "===== Create src/run_cadquery.py ====="
cat > src/run_cadquery.py <<'PY'
import argparse
import json
import time
import traceback
from pathlib import Path

import cadquery as cq


def clean_code(raw_code: str) -> str:
    return raw_code.replace("```python", "").replace("```", "").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--candidate", type=int, default=0)
    args = parser.parse_args()

    code_path = Path(args.code)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    step_path = outdir / f"candidate_{args.candidate}.step"
    stl_path = outdir / f"candidate_{args.candidate}.stl"
    log_path = outdir / f"verify_candidate_{args.candidate}.json"

    raw_code = code_path.read_text(encoding="utf-8")
    code = clean_code(raw_code)

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
        "code": str(code_path),
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
PY

echo "===== Create src/run_cadquery_safe.py ====="
cat > src/run_cadquery_safe.py <<'PY'
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
PY

echo "===== Create src/verify_pipeline.py ====="
cat > src/verify_pipeline.py <<'PY'
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
PY

echo "===== Create src/summarize_pipeline.py ====="
cat > src/summarize_pipeline.py <<'PY'
import argparse
import json
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", required=True)
parser.add_argument("--csv", required=True)
args = parser.parse_args()

outdir = Path(args.outdir)
rows = []

for p in sorted(outdir.glob("pipeline_candidate_*.json")):
    rows.append(json.loads(p.read_text(encoding="utf-8")))

df = pd.DataFrame(rows)

if len(df) > 0:
    df = df.sort_values("candidate")

print(df)

if len(df) > 0:
    print("\nPipeline success:", int(df["final_success"].sum()), "/", len(df))
    print("\nSelected stages:")
    print(df["selected_stage"].value_counts())

df.to_csv(args.csv, index=False)
print("Saved:", args.csv)
PY

echo "===== Create src/error_analysis.py ====="
cat > src/error_analysis.py <<'PY'
import argparse
import json
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", required=True)
parser.add_argument("--csv", required=True)
args = parser.parse_args()

outdir = Path(args.outdir)
rows = []

for p in sorted(outdir.glob("verify_candidate*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    rows.append({
        "log": p.name,
        "candidate": d.get("candidate"),
        "safe_mode": d.get("safe_mode", False),
        "success": d.get("success"),
        "error_type": d.get("error_type"),
    })

df = pd.DataFrame(rows)
print(df)

if len(df) > 0:
    summary = (
        df[df["success"] == False]
        .groupby(["safe_mode", "error_type"])
        .size()
        .reset_index(name="count")
    )
else:
    summary = pd.DataFrame(columns=["safe_mode", "error_type", "count"])

print("\nError summary:")
print(summary)

df.to_csv(args.csv, index=False)
summary.to_csv(args.csv.replace(".csv", "_summary.csv"), index=False)
PY

echo "===== Create src/evaluate_geometry.py ====="
cat > src/evaluate_geometry.py <<'PY'
import argparse
from pathlib import Path

import pandas as pd
import trimesh


def load_mesh(path):
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def mesh_stats(path):
    mesh = load_mesh(path)
    bbox = mesh.bounds
    size = bbox[1] - bbox[0]

    return {
        "file": str(path),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bbox_x": float(size[0]),
        "bbox_y": float(size[1]),
        "bbox_z": float(size[2]),
        "volume": float(mesh.volume) if mesh.is_watertight else None,
        "watertight": bool(mesh.is_watertight),
    }


def rel_err(a, b):
    if b == 0 or b is None or a is None:
        return None
    return abs(a - b) / abs(b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    gt = mesh_stats(Path(args.gt))
    rows = []

    for stl in sorted(Path(args.pred_dir).glob("candidate*.stl")):
        pred = mesh_stats(stl)

        rows.append({
            "candidate_stl": str(stl),
            "vertices": pred["vertices"],
            "faces": pred["faces"],
            "watertight": pred["watertight"],
            "bbox_x": pred["bbox_x"],
            "bbox_y": pred["bbox_y"],
            "bbox_z": pred["bbox_z"],
            "gt_bbox_x": gt["bbox_x"],
            "gt_bbox_y": gt["bbox_y"],
            "gt_bbox_z": gt["bbox_z"],
            "bbox_x_error": rel_err(pred["bbox_x"], gt["bbox_x"]),
            "bbox_y_error": rel_err(pred["bbox_y"], gt["bbox_y"]),
            "bbox_z_error": rel_err(pred["bbox_z"], gt["bbox_z"]),
            "volume": pred["volume"],
            "gt_volume": gt["volume"],
            "volume_error": rel_err(pred["volume"], gt["volume"]),
        })

    df = pd.DataFrame(rows)
    print(df)
    df.to_csv(args.out, index=False)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
PY

echo "===== Create src/select_best_candidate.py ====="
cat > src/select_best_candidate.py <<'PY'
import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--geometry_csv", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

df = pd.read_csv(args.geometry_csv)

if len(df) == 0:
    raise RuntimeError("No candidates found in geometry csv.")

for col in ["bbox_x_error", "bbox_y_error", "bbox_z_error", "volume_error"]:
    if col not in df.columns:
        df[col] = None

df["bbox_mean_error"] = df[["bbox_x_error", "bbox_y_error", "bbox_z_error"]].mean(axis=1)
df["watertight_score"] = df["watertight"].apply(lambda x: 0 if bool(x) else 1)

df_sorted = df.sort_values(
    by=["watertight_score", "bbox_mean_error", "volume_error"],
    ascending=[True, True, True],
)

best = df_sorted.iloc[0]
print("Best candidate:")
print(best)

Path(args.out).write_text(best.to_json(indent=2, force_ascii=False), encoding="utf-8")
print("Saved:", args.out)
PY

echo "===== Missing scripts repaired. ====="
