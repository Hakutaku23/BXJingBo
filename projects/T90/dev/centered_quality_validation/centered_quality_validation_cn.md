# Centered Quality Validation 实验记录

## 记录范围

- 本记录仅对应 `projects/T90/dev/centered_quality_validation/`
- 这是一条与 `cleanroom_validation/` 平行的新 cleanroom 分支
- 当前 strongest threshold-oriented cleanroom 仅作为 baseline 锚点保留，不再在本分支中继续深挖

## 首轮冻结路线

- Baseline A: 保留现有 strongest threshold-oriented cleanroom 作为对照
- Treatment B: centered desirability only
- Treatment C: lower-risk + upper-risk + centered desirability

## 首轮固定输入边界

- DCS: `merge_data.csv + merge_data_otr.csv`
- 先做 strict sensor-identity de-dup
- 特征表示: simple causal `120 min` window stats
- fold-local `topk=40`
- `TimeSeriesSplit(5)`

## 首轮固定 centered-quality 定义

- 目标中心: `8.45`
- 半宽: `0.25`
- desirability:

`q(y) = max(0, 1 - |y - 8.45| / 0.25)`

## 首轮固定决策映射

1. `max_risk >= 0.50` -> `unacceptable`
2. `max_risk < 0.20` 且 `q_hat >= 0.80` -> `premium`
3. `max_risk < 0.35` 且 `q_hat >= 0.30` -> `acceptable`
4. 否则 -> `retest`

## 首轮 cleanroom 修正说明

- 第一版脚本在首跑时暴露出一个 cleanroom 错误: `true_desirability`、`target_lower_risk`、`target_upper_risk`、`center_core_flag`、`margin_in_spec_flag` 这些由标签直接派生的列被错误地留在了候选特征池里。
- 这会形成标签泄漏，因此该版结果已废弃，不作为正式结论保留。
- 接受版本已经在重新筛特征前显式剔除了这些列，并完成了重跑。

## 首轮 accepted run 数据摘要

- 对齐后的 feature rows: `2726`
- outer-test 实际计分样本: `2270`
- identity de-dup 后合并 DCS 位点: `64`
- 剔除 alias pair: `7`
- preclean 后保留特征: `446`
- 5 折入选位点并集: `54`
- `selection_count >= 4` 的稳定位点: `33`

## 首轮 accepted run 结果

### Baseline A

- threshold macro_f1: `0.2674`
- threshold balanced_accuracy: `0.3580`
- boundary warning AP: `0.1963`
- boundary high-confidence non-warning rate: `0.4848`

### Treatment B

- desirability MAE: `0.3744`
- desirability RMSE: `0.4532`
- desirability Spearman: `-0.0434`
- premium precision / recall: `0.1028 / 0.0207`

### Treatment C

- lower-risk AP: `0.0157`
- upper-risk AP: `0.1690`
- unacceptable recall: `0.7219`
- unacceptable false-clear rate: `0.0150`
- centered decision macro_f1_vs_truth: `0.1439`
- decision coverage: `0.8956`
- retest rate: `0.1044`
- covered balanced_accuracy: `0.3305`

## 当前判断

- 现阶段 strongest threshold-oriented cleanroom baseline 仍明显强于 centered-quality 首轮 treatment。
- 当前这版 centered desirability 定义与 decision map 还没有被支持为更优的现场任务表达。
- 因此这条新分支后续更应该继续检查“目标定义是否合理”，而不是马上切去更复杂模型或回到旧分支继续调 ordinal/cumulative。

## 当前产物位置

- 计划: `plans/centered_quality_validation.md`
- 配置: `configs/centered_quality_current_head.yaml`
- 脚本: `scripts/run_centered_quality_current_head.py`
- 审计: `reports/centered_quality_current_head_audit.md`
- 摘要: `reports/centered_quality_current_head/centered_quality_summary.md`
- 结果表: `artifacts/centered_quality_current_head/centered_quality_results.csv`
- 汇总 JSON: `artifacts/centered_quality_current_head/centered_quality_summary.json`
