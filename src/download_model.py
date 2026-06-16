import os
from huggingface_hub import snapshot_download

repo_id = os.environ.get("Z2CAD_REPO", "ADSKAILab/Zero-To-CAD-Qwen3-VL-2B")
local_dir = os.environ.get("Z2CAD_MODEL_DIR", "./models/Zero-To-CAD-Qwen3-VL-2B")

print("Downloading model:")
print("repo_id:", repo_id)
print("local_dir:", local_dir)

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=4,
)

print("Model downloaded to:", local_dir)
