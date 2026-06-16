import torch
import time

print("===== PyTorch / ROCm Check =====")
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())

if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    raise RuntimeError("No ROCm GPU visible to PyTorch.")

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

print("\n===== Basic Tensor Test =====")
a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
torch.cuda.synchronize()
print(a + 1)
print("basic tensor ok")

print("\n===== FP16 Matmul Test =====")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
y = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

torch.cuda.synchronize()
t0 = time.time()
z = x @ y
torch.cuda.synchronize()
t1 = time.time()

print("matmul ok")
print("time:", round(t1 - t0, 4), "sec")
print("device:", z.device)
print("ROCm PyTorch check passed.")
