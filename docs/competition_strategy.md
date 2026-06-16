# 赛题二参赛扩充方案

## 1. 项目定位

当前项目建议定位为：

**基于 AMD ROCm 的 Zero-to-CAD 智能 CAD 重建、修复与验证系统。**

不要只强调“使用 Zero-to-CAD 生成 CAD”，而要强调：

- ROCm 负责加速多模态 CAD 代码生成；
- CadQuery/STEP/STL 负责把生成结果变成可执行工程对象；
- safe repair 负责提升可用 CAD 输出率；
- geometry evaluation 负责量化 bbox、体积、水密性和 Chamfer 距离；
- rocprof/rocm-smi 负责证明 AMD GPU 资源被真实使用并可分析。

这样能贴合赛题二的四个评分重点：项目实现、ROCm 组件使用、性能资源分析、可复现交付。

## 2. 对评分标准的对应关系

| 评分项 | 本项目材料 |
|---|---|
| 项目背景与挑战 | 从多视图图像恢复可编辑 CAD，难点是代码可执行性、几何一致性、推理成本 |
| 方案与实现 | Zero-to-CAD 推理、CadQuery 执行、safe repair、STEP/STL 导出、几何评估 |
| ROCm 组件 | PyTorch ROCm、HIP runtime、rocprof/rocprofv3、rocm-smi、ROCR_VISIBLE_DEVICES |
| 性能与资源分析 | latency、tokens/sec、peak VRAM、ROCm Chamfer、8 GPU 并行、token budget ablation |
| 复现与交付 | benchmark scripts、profile scripts、sample inputs、expected CSV/Markdown outputs |
| 创新开发 | safe repair + candidate selection + geometry quality scoring |
| 性能瓶颈定位 | 对比不同 token budget、候选数量、单 GPU/8 GPU 并行，结合 rocprof 数据分析 |

## 3. 建议补充的实验

### 实验 A：基础重建成功率

运行 8 个零件，每个零件生成 5 个候选：

```bash
MODEL=./models/Zero-To-CAD-Qwen3-VL-2B NUM_CANDIDATES=5 MAX_TOKENS=1024 \
bash scripts/run_8parts_benchmark.sh
```

报告指标：

- pipeline success rate
- raw success rate
- safe repair gain
- bbox mean error
- volume error
- normalized Chamfer distance
- watertight ratio

### 实验 B：token budget 消融

```bash
GPU=0 TOKENS="512 1024 2048" bash scripts/run_token_ablation.sh
```

报告问题：

- token 越多是否明显提升几何质量；
- token 越多延迟和显存增长多少；
- 推荐比赛 demo 使用哪个 token budget。

### 实验 C：ROCm profiling

```bash
PART=part_001 CANDIDATE=0 GPU=0 MAX_TOKENS=1024 \
bash scripts/run_rocm_profile.sh
```

报告材料：

- `profiles/*/rocprof*` kernel profile;
- `profiles/*/rocm_smi_before.txt`;
- `profiles/*/rocm_smi_after.txt`;
- `outputs/*/infer_candidate_*.json` 中的 HIP 版本、GPU 名称、VRAM、tokens/s。

### 实验 D：8 GPU 并行吞吐

```bash
MAX_TOKENS=512 CANDIDATE=0 bash scripts/run_8parts_parallel_8gpu.sh
```

报告问题：

- 8 个零件并行时总耗时；
- 每张 GPU 的任务隔离方式；
- 这体现了 Radeon PRO W 多卡平台对 CAD 科研批处理的价值。

## 4. 技术论文建议结构

1. 背景与挑战
   - 科研/CAD 设计中，从图像或扫描结果恢复可编辑 CAD 的需求；
   - 生成式模型能给出草稿，但工程可执行性和几何可靠性不足；
   - 本地 ROCm GPU 可以降低批量生成与验证的等待时间。

2. 系统架构
   - 输入：8 视角 PNG；
   - ROCm 推理：Zero-to-CAD/Qwen-VL 生成 CadQuery；
   - 执行验证：CadQuery raw run；
   - 修复验证：safe repair；
   - 几何评估：STL bbox、volume、watertight；
   - 结果汇总：CSV/Markdown 报告。

3. ROCm 实现
   - PyTorch ROCm 加载 FP16 模型；
   - `ROCR_VISIBLE_DEVICES` 控制多 GPU 并行；
   - `torch.cuda` API 在 ROCm 后端记录 VRAM；
   - rocprof/rocm-smi 进行资源剖析。

4. 实验与结果
   - 成功率表；
   - safe repair 增益表；
   - 几何质量表：bbox、volume、watertight、ROCm Chamfer；
   - token budget 消融图；
   - 单 GPU 与 8 GPU 并行吞吐；
   - profiler 截图。

5. 阶段性成果与规划
   - 目前已完成可复现原型；
   - 后续可加入真实工业零件数据；
   - 后续可做 CadQuery 语法约束解码、GPU 加速 mesh distance、ROCm kernel/算子级优化。

## 5. 可以进一步加分的方向

- 增加 GPU 加速几何评估：用 PyTorch/ROCm 计算 Chamfer Distance 或点云距离。
- 增加候选重排序模型：综合执行成功、几何误差、代码复杂度选择最佳候选。
- 增加 Dockerfile：固定 ROCm、PyTorch、CadQuery、trimesh 版本。
- 尝试向开源项目提交 issue/PR：例如补充 ROCm 上 Zero-to-CAD 推理文档或 CadQuery 生成验证脚本。
