# CLOVER 消融实验脚本

两套实验彼此独立，每套固定使用 100 个 case：

- `run_wikitq_ablation.sh`：WikiTQ 机制分层子集；
- `run_tablebench_ablation.sh`：TableBench 子集，仅包含 `FactChecking` 与 `NumericalReasoning`。

抽样仅依据数据集元信息和问题文本，不读取模型预测或正确性结果。默认 manifest 位于 `benchmarks/ablation_cases/`，七个变体严格复用同一 case 集合。

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
- 末端 Edge 调用/命中/上报与 Cloud 重规划次数；
- 最终答案来源、Cloud/本地模型 token、估算成本和耗时。
- `Full CLOVER` 与 `w/o Edge Agent` 的直接 Edge-to-Cloud 替代效应，包括
  Cloud 调用、synthesis、replan、token、成本和每题调用增量。

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
```

TableBench 下载转换也固定只保留两类。已有旧转换数据需要重新生成时：

```bash
CLOVER_DATASET_OVERWRITE=1 \
bash benchmarks/download_datasets.sh --dataset tablebench --overwrite
```
