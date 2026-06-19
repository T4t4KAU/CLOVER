# CLOVER 消融实验脚本

两套实验彼此独立，每套固定使用 100 个 case：

- `run_wikitq_ablation.sh`：WikiTQ 机制分层子集；
- `run_tablebench_ablation.sh`：TableBench 子集，仅包含 `FactChecking` 与 `NumericalReasoning`。

抽样仅依据数据集元信息和问题文本，不读取模型预测或正确性结果。默认 manifest 位于 `benchmarks/ablation_cases/`，六个变体严格复用同一 case 集合。

## 运行

```bash
conda activate dl

bash benchmarks/run_wikitq_ablation.sh /path/to/edge-model
bash benchmarks/run_tablebench_ablation.sh /path/to/edge-model
```

每套脚本依次运行：

1. `full`
2. `static`
3. `no_contract`
4. `end_review`
5. `one_shot`
6. `cloud_finalize`

同一套实验只启动一次 vLLM 服务。结果默认写入：

```text
benchmark/runs/<dataset>_ablation_<timestamp>/
```

目录中的 `sanity_check.json` 会校验固定 case 集、feature flags、OneShot 后续云调用和 CloudFinalize 静态终止计数。

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
