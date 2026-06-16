import argparse
import json
import os
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
    parser.add_argument("--profile-label", default=None)
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
    if not torch.cuda.is_available():
        raise RuntimeError("No ROCm/CUDA device is visible to PyTorch.")

    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)

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
    input_tokens = int(inputs["input_ids"].shape[1])

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

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    torch.cuda.synchronize()
    t1 = time.time()

    total_tokens = int(output_ids.shape[1])
    generated_tokens = max(total_tokens - input_tokens, 0)
    inference_time_sec = t1 - t0
    generated_tokens_per_sec = (
        generated_tokens / inference_time_sec if inference_time_sec > 0 else None
    )

    generated = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0]

    code_path.write_text(generated, encoding="utf-8")

    log = {
        "candidate": args.candidate,
        "profile_label": args.profile_label,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "inference_time_sec": inference_time_sec,
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_sec": generated_tokens_per_sec,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_vram_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "gpu_name": device_name,
        "torch_version": torch.__version__,
        "torch_hip_version": getattr(torch.version, "hip", None),
        "device_count": torch.cuda.device_count(),
        "visible_devices": {
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "output_code": str(code_path),
    }

    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Generated code:", code_path)
    print("Inference time:", round(log["inference_time_sec"], 2), "sec")
    print("Generated tokens/sec:", round(log["generated_tokens_per_sec"], 2) if log["generated_tokens_per_sec"] else None)
    print("Peak VRAM:", round(log["peak_vram_gb"], 2), "GB")


if __name__ == "__main__":
    main()
