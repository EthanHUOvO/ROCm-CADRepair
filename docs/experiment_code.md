# 实验代码说明

本项目的实验代码围绕“本地化 CAD 工程 AI 助手”展开，形成从多视角图像输入、ROCm 推理、CadQuery 程序生成、STEP/STL 导出、几何质量评价、错误分析到工程师报告生成的完整流程。

## 1. 环境检查

```bash
export PY=/opt/python/bin/python
$PY src/check_rocm.py
$PY src/validate_cad_views.py --examples-dir examples --out docs/results/cad_view_validation.csv
```

`check_rocm.py` 用于确认 PyTorch ROCm、HIP 版本、GPU 数量和 FP16 矩阵乘法是否正常。`validate_cad_views.py` 用于检查 8 个 CAD 样例的 64 张视角图是否存在空白、裁切或主体异常。

## 2. 单零件本地 CAD 助手

```bash
MODEL=./models/Zero-To-CAD-Qwen3-VL-2B \
PY=$PY \
PART=part_003 \
NUM_CANDIDATES=5 \
MAX_TOKENS=1024 \
GPU=0 \
ENABLE_CHAMFER=1 \
bash scripts/run_local_cad_assistant.sh part_003
```

该脚本完成：

1. 读取 `examples/part_003/views` 中的 8 视角图像。
2. 调用 `src/infer_one.py` 在 ROCm GPU 上生成多个 CadQuery 候选程序。
3. 调用 `src/verify_pipeline.py` 进行 raw 执行和 safe repair 执行。
4. 调用 `src/evaluate_geometry.py` 计算 bbox、volume 和 watertight 指标。
5. 可选调用 `src/evaluate_chamfer_rocm.py` 在 GPU 上计算 Chamfer 距离。
6. 调用 `src/generate_assistant_report.py` 生成工程师报告。

输出位置：

```text
outputs/part_003/
assistant_reports/part_003_assistant_report.md
assistant_reports/assistant_summary.csv
```

## 3. 8 个样例完整 benchmark

```bash
MODEL=./models/Zero-To-CAD-Qwen3-VL-2B \
PY=$PY \
NUM_CANDIDATES=5 \
MAX_TOKENS=1024 \
ENABLE_CHAMFER=1 \
bash scripts/run_8parts_benchmark.sh
```

汇总结果：

```bash
$PY src/collect_inference_stats.py
$PY src/collect_overall_results.py
$PY src/analyze_success_gain.py
$PY src/analyze_geometry_quality.py
$PY src/build_competition_report.py
$PY src/generate_assistant_report.py --outputs-root outputs --outdir assistant_reports
```

核心输出：

```text
docs/results/competition_report.md
docs/results/overall_8parts_summary.csv
docs/results/inference_overall_stats.csv
docs/results/success_gain_analysis.csv
assistant_reports/*.md
```

## 4. raw vs safe repair 对比

raw vs safe repair 的数据来自每个 part 的：

```text
outputs/part_*/pipeline_summary.csv
```

汇总脚本：

```bash
$PY src/analyze_success_gain.py
```

该实验用于证明系统不是简单调用模型，而是通过 CAD 执行反馈和 safe repair 提高 STEP/STL 导出成功率。

## 5. 候选数量消融实验

```bash
MODEL=./models/Zero-To-CAD-Qwen3-VL-2B \
PY=$PY \
CANDIDATE_COUNTS="1 3 5 10" \
MAX_TOKENS=1024 \
GPU=0 \
bash scripts/run_candidate_ablation.sh
```

输出：

```text
docs/results/candidate_count_ablation.csv
outputs_ablation/candidates/
assistant_reports/candidate_ablation/
```

该实验用于比较不同候选数量对成功率、几何误差和推理成本的影响，说明多候选搜索的必要性。

## 6. token budget 消融实验

```bash
MODEL=./models/Zero-To-CAD-Qwen3-VL-2B \
PY=$PY \
TOKENS="512 1024 2048" \
GPU=0 \
bash scripts/run_token_ablation.sh
```

输出：

```text
outputs_ablation/tokens_512/
outputs_ablation/tokens_1024/
outputs_ablation/tokens_2048/
```

建议再用 `src/summarize_ablation.py` 汇总：

```bash
$PY src/summarize_ablation.py \
  --root outputs_ablation \
  --label-prefix tokens_ \
  --label-name max_tokens \
  --out docs/results/token_budget_ablation.csv
```

该实验用于分析生成长度对成功率、几何质量、推理延迟、显存和 tokens/s 的影响。

## 7. 单卡串行 vs 8 卡并行

单卡串行：

```bash
MODEL=./models/Zero-To-CAD-Qwen3-VL-2B \
PY=$PY \
GPU=0 \
MAX_TOKENS=512 \
CANDIDATE=0 \
bash scripts/run_serial_8parts_timing.sh
```

8 卡并行：

```bash
MODEL=./models/Zero-To-CAD-Qwen3-VL-2B \
PY=$PY \
MAX_TOKENS=512 \
CANDIDATE=0 \
bash scripts/run_parallel_8gpu_timing.sh
```

输出：

```text
docs/results/serial_8parts_timing.csv
docs/results/parallel_8gpu_timing.csv
docs/results/gpu_timing_comparison.csv
```

该实验用于体现 AMD Radeon 多 GPU 服务器在批量 CAD 生成任务中的吞吐优势。

## 8. clean vs noisy views 鲁棒性实验

```bash
MODEL=./models/Zero-To-CAD-Qwen3-VL-2B \
PY=$PY \
NUM_CANDIDATES=3 \
MAX_TOKENS=1024 \
GPU=0 \
bash scripts/run_noisy_views_ablation.sh
```

该脚本先通过 `src/create_noisy_views.py` 生成带有轻微旋转、亮度变化、对比度变化、模糊和噪声的输入图像，再分别对 clean views 和 noisy views 运行同样流程。

输出：

```text
examples_noisy/
outputs_ablation/noisy_views/
docs/results/noisy_view_ablation.csv
```

该实验用于说明系统从干净 CAD 渲染图走向真实工程图像时的鲁棒性边界。

## 9. ROCm profiling

```bash
PART=part_003 \
CANDIDATE=0 \
GPU=0 \
MAX_TOKENS=1024 \
PY=$PY \
bash scripts/run_rocm_profile.sh
```

输出：

```text
profiles/part_003_candidate0_gpu0/
```

其中包含 `rocm-smi` 的显存、温度、功耗和 GPU 利用率记录。该实验用于提供 ROCm 平台证据链。

## 10. 主要源代码对应关系

| 文件 | 功能 |
|---|---|
| `src/infer_one.py` | ROCm GPU 上加载 Zero-to-CAD 模型并生成 CadQuery 程序 |
| `src/verify_pipeline.py` | raw 执行与 safe repair 执行调度 |
| `src/run_cadquery.py` | 执行原始 CadQuery 程序并导出 STEP/STL |
| `src/run_cadquery_safe.py` | 对高风险 CadQuery 操作进行安全化处理后再次执行 |
| `src/evaluate_geometry.py` | 计算 watertight、bbox 误差和 volume 误差 |
| `src/evaluate_chamfer_rocm.py` | 使用 PyTorch/ROCm 计算预测 STL 与 GT STL 的 Chamfer 距离 |
| `src/select_best_candidate.py` | 根据几何指标选择最佳候选 |
| `src/generate_assistant_report.py` | 生成面向工程师的本地 CAD 助手报告 |
| `src/summarize_ablation.py` | 汇总候选数量、token、噪声视图等消融实验 |
| `scripts/run_local_cad_assistant.sh` | 单个零件的一键助手流程 |
| `scripts/run_8parts_benchmark.sh` | 8 个零件的完整 benchmark |
| `scripts/run_candidate_ablation.sh` | 候选数量消融 |
| `scripts/run_token_ablation.sh` | token budget 消融 |
| `scripts/run_serial_8parts_timing.sh` | 单 GPU 串行计时 |
| `scripts/run_parallel_8gpu_timing.sh` | 8 GPU 并行计时 |
| `scripts/run_noisy_views_ablation.sh` | clean/noisy views 鲁棒性实验 |
| `scripts/run_rocm_profile.sh` | ROCm profiling 和资源采样 |
