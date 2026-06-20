# CLOVER 消融实验脚本

两套数据集实验彼此独立，每套固定使用 100 个 case：

- `run_wikitq_ablation.sh`：WikiTQ 机制分层子集；
- `run_tablebench_ablation.sh`：TableBench 子集，仅包含 `FactChecking` 与 `NumericalReasoning`。

抽样仅依据数据集元信息、问题文本、答案类型、表格规模和单元格表示形式，
不读取模型预测、运行轨迹或正确性结果。默认 manifest 位于
`benchmarks/ablation_cases/`，七个变体严格复用同一 case 集合。

脚本支持两种互补的固定子集：

- `representative`：原始任务类型分层子集，用于报告整体 ACC/成本趋势；
- `edge_opportunity`：Edge 机制增强子集，用于检验局部语义处理是否提高
  ACC、减少 Cloud 调用，以及节点修复和末端复核各自贡献多少。

当前消融脚本默认使用 `edge_opportunity`。WikiTQ 中 85 个 case 来自字段选择、
值归一化、短列表整理和小候选集，15 个 case 是确定性控制题；TableBench 中
70 个 case 来自局部语义机会，30 个是确定性控制题。每个 manifest 的
`.summary.json` 明确记录抽样依据，并标记
`uses_model_predictions=false` 和 `uses_answer_correctness=false`。

论文中应将它明确称为 “mechanism-focused, outcome-blind subset”，不要把该
子集上的 ACC 当作整个 benchmark 的无偏总体 ACC。若篇幅允许，建议同时报告
`representative` 子集作为稳健性对照。

## 运行

```bash
conda activate dl

bash benchmarks/run_wikitq_ablation.sh /path/to/edge-model
bash benchmarks/run_tablebench_ablation.sh /path/to/edge-model
```

每套脚本运行以下七个变体：

1. `full`：完整 CLOVER；
2. `no_edge`：关闭 Edge Agent、节点修复、节点复核与末端 Edge Review，其他静态和 Cloud 能力与 Full 保持一致；
3. `static`：仅关闭节点级 Edge Repair，保留末端 Edge Review；
4. `no_contract`：关闭 Edge Agent 输出契约验证；
5. `end_review`：关闭节点级 Edge Repair 和 Node Review，只保留末端 Edge Review；
6. `one_shot`：保留 Cloud 最终综合，但禁止 Cloud 生成后续 SQL/DAG action；
7. `cloud_finalize`：关闭静态与末端 Edge 最终化，统一交给 Cloud 综合。

`no_edge` 是验证 Edge 是否真正替代 Cloud 的无混杂对照组。它保留静态最终化、
Cloud synthesis 和 Cloud replan，因此与 Full 之间的 Cloud 调用差异可归因于
Edge 路径是否启用。

Full CLOVER 默认启用主动局部语义复核：

```text
CLOVER_EDGE_REVIEW_PROACTIVE=true
```

静态运行时仅在 evidence 闭合且规模受限时触发，包括单行多字段选择、至多五行的
候选选择、简单布尔组合、短列表整理，以及百分数、单位、引号、标签或尾部括号的
值归一化。Edge 输出必须引用 fact ID，并通过确定性操作重放；复核失败时继续原有
静态或 Cloud 路径。包含最高、排序、计数或跨行比较的列表问题不会进入主动
Edge 整理，而是保留在确定性或 Cloud 路径。

为了减少顺序偏差，默认使用 seed 对七个变体进行可复现打乱，实际顺序记录在
`variant_order.txt`。可通过以下变量指定固定顺序：

```bash
CLOVER_ABLATION_VARIANT_ORDER=full,no_edge,static,no_contract,end_review,one_shot,cloud_finalize
```

同一套实验只启动一次 vLLM 服务。每个变体开始前会执行一次不计入评测时间的
本地模型 warm-up。结果默认写入：

```text
benchmark/runs/<dataset>_ablation_<timestamp>/
```

目录中的 `sanity_check.json` 会校验固定 case 集、细粒度 feature flags、
`w/o Cloud Replan` 的重规划计数，以及 Cloud Finalization 的终止路径。
七个变体结束后，脚本会在终端直接打印汇总表，并生成：

```text
ablation_summary.md
ablation_summary.csv
ablation_summary.json
```

表中包含：

- 正确率、相对 Full CLOVER 的百分点变化和 Exact McNemar 配对检验；
- 相对 Full 的退化/恢复 case 数；
- 节点 Edge 运行/成功/step、节点复核和契约拒绝；
- 末端 Edge 调用/命中/上报、主动语义机会/命中与 Cloud 重规划次数；
- 最终答案来源、Cloud/本地模型 token、估算成本和耗时。
- `Full CLOVER` 与 `w/o Edge Agent` 的直接 Edge-to-Cloud 替代效应，包括
  Cloud 调用、synthesis、replan、token、成本和每题调用增量。
- `End-only Review` 对比 `w/o Edge Agent` 的终局/主动语义复核贡献，以及
  `Full CLOVER` 对比 `End-only Review` 的节点修复贡献。
- 每个变体相对 Full 的具体退化/恢复 case ID、含 retry case 的正确率以及
  runtime error 类型。

此外会生成：

```text
ablation_case_diagnostics.jsonl
ablation_discordant_cases.csv
```

前者保存每个 case 在七个变体下的正确性、答案来源、retry 和错误类型；后者只
保存 Full 与变体结果不同的配对 case，便于直接检查 Contract Verification 和
Cloud Replan 为什么出现反向增益。

如果七组实验已经跑完，只需要补生成汇总而不重跑：

```bash
python -m benchmarks.summarize_ablation_suite \
  --suite-root benchmark/runs/wikitq_ablation_<timestamp> \
  --dataset wikitq
```

## 重新生成固定子集

```bash
CLOVER_ABLATION_REGENERATE_MANIFEST=1 \
bash benchmarks/run_wikitq_ablation.sh /path/to/edge-model
```

默认参数：

```text
CLOVER_ABLATION_SIZE=100
CLOVER_ABLATION_SEED=20260619
CLOVER_ABLATION_SELECTION_POLICY=edge_opportunity
```

运行代表性对照子集：

```bash
CLOVER_ABLATION_SELECTION_POLICY=representative \
bash benchmarks/run_wikitq_ablation.sh /path/to/edge-model
```

两种 policy 使用不同的 manifest 文件名，不会互相覆盖：

```text
wikitq_representative_100_seed20260619.jsonl
wikitq_edge_opportunity_100_seed20260619.jsonl
```

TableBench 下载转换也固定只保留两类。已有旧转换数据需要重新生成时：

```bash
CLOVER_DATASET_OVERWRITE=1 \
bash benchmarks/download_datasets.sh --dataset tablebench --overwrite
```
