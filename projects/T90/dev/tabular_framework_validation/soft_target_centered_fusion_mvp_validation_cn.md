# Soft Target + Centered Desirability 融合 MVP 首轮验证记录

## 1. 目标

在不重做大模型的前提下，验证一个最小双头融合方案是否值得继续：

- `soft target` 头：使用当前最优 `range_position`，输出 `p_fail_soft`
- `centered_desirability` 头：使用当前最优 `lag_plus_interaction_plus_quality`，输出 `q_center`
- 构造 centered 风险代理：`p_fail_center_proxy = 1 - clip(q_center, 0, 1)`
- 只新增一个轻量融合层，不改两条主头本身

本轮属于 `tail-calibration MVP`，不是完整的内层 OOF 融合。

## 2. 固定底座

- soft 头配置：
  [autogluon_stage2_soft_probability_x_enrichment_range_position_only.yaml](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/configs/autogluon_stage2_soft_probability_x_enrichment_range_position_only.yaml)
- centered 头配置：
  [autogluon_centered_desirability_outspec_eval.yaml](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/configs/autogluon_centered_desirability_outspec_eval.yaml)
- 融合脚本：
  [run_autogluon_soft_centered_fusion_mvp.py](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_soft_centered_fusion_mvp.py)
- 融合配置：
  [autogluon_soft_centered_fusion_mvp.yaml](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/configs/autogluon_soft_centered_fusion_mvp.yaml)

## 3. 方法

### 3.1 数据对齐

- 两条历史实验线的时间键不一致：
  - soft 侧天然有 `sample_time`
  - centered 侧以 `decision_time` 为主
- 本轮补了一个兼容对齐层：
  - soft 侧以 `sample_time` 作为 `align_time`
  - centered 侧优先用 `sample_time`，否则回退到 `decision_time`
- 共同样本数：`2726`

### 3.2 外层评估

- `TimeSeriesSplit(5)`
- 每个 outer train 再切一个尾部 calibration slice：
  - `calibration_fraction = 0.2`
  - `min_calibration_samples = 120`
  - `min_base_train_samples = 360`

### 3.3 融合器

本轮只做两个最小融合器：

1. 固定权重融合
   - `p_fail_fused = w * p_fail_soft + (1 - w) * p_fail_center_proxy`
   - 候选权重：`0.2 / 0.35 / 0.5 / 0.65 / 0.8`

2. 轻量 logistic 融合
   - 输入：
     - `p_fail_soft`
     - `p_fail_center_proxy`
     - `disagreement = |p_fail_soft - p_fail_center_proxy|`

### 3.4 评价指标

- `Brier`
- `hard_out_ap_diagnostic`
- `hard_out_auc_diagnostic`
- `false_clear_rate`
- 行动率诊断：
  - `low_risk_rate`
  - `high_risk_rate`
  - `retest_rate`

本轮阈值：

- `tau_low = 0.2`
- `tau_high = 0.6`

## 4. 结果

结果文件：

- 汇总：
  [soft_centered_fusion_mvp_summary.json](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/soft_centered_fusion_mvp/soft_centered_fusion_mvp_summary.json)
- 折内选择：
  [soft_centered_fusion_mvp_fold_selection.json](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/soft_centered_fusion_mvp/soft_centered_fusion_mvp_fold_selection.json)
- 明细预测：
  [soft_centered_fusion_mvp_predictions.csv](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/soft_centered_fusion_mvp/soft_centered_fusion_mvp_predictions.csv)
- 折指标：
  [soft_centered_fusion_mvp_fold_metrics.csv](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/soft_centered_fusion_mvp/soft_centered_fusion_mvp_fold_metrics.csv)
- 英文摘要：
  [soft_centered_fusion_mvp_summary.md](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/reports/soft_centered_fusion_mvp/soft_centered_fusion_mvp_summary.md)

### 4.1 聚合结果

| method | mean_brier | hard_out_ap | hard_out_auc | false_clear_rate | low_risk_rate | high_risk_rate | retest_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_fused | 0.1509 | 0.2167 | 0.5395 | 0.6255 | 0.6546 | 0.0000 | 0.3454 |
| soft_only | 0.2120 | 0.1902 | 0.4957 | 0.0078 | 0.0044 | 0.0445 | 0.9511 |
| fixed_weight_fused | 0.2383 | 0.1882 | 0.4905 | 0.0000 | 0.0000 | 0.0899 | 0.9101 |
| center_proxy_only | 0.4108 | 0.1805 | 0.4795 | 0.0000 | 0.0000 | 0.7749 | 0.2251 |

### 4.2 折内选择信号

- fixed-weight 在 5 折里都选到了 `0.8`
- 说明在当前候选集中，calibration slice 一直更偏向 soft 头
- logistic 融合确实学到了额外信息，但系数方向折间不稳定：
  - 有的折 soft/centered 同时为正
  - 有的折 centered 为负
  - 有的折 `disagreement` 为正，有的为负

这说明首轮 logistic 融合不是在学一个稳定的工程规则，而更像在不同折里做局部补偿。

## 5. 判断

### 5.1 可以确认的事情

- 双头融合并不是完全没信号。
- `logistic_fused` 在统计指标上明显优于单头：
  - `Brier` 更低
  - `hard_out AP/AUC` 更高

### 5.2 不能接受的地方

- `logistic_fused` 的 `false_clear_rate = 0.6255`
- 这说明它是靠大幅放松低风险判定换来的统计改进
- 工程上这版不能用

### 5.3 本轮结论

本轮 MVP **不支持**“直接把 soft 风险头和 centered 风险代理做无约束轻融合”。

更准确地说：

- `soft target` 适合继续做主风险头
- `centered_desirability` 更适合作为辅助质量头
- 但 centered 头当前不适合作为一个未加约束的第二个失败概率来源，直接参与概率融合

## 6. 下一步建议

如果继续，优先级建议如下：

1. 做有 guardrail 的融合，而不是无约束 logistic 融合  
   例如把 `false_clear_rate` 作为硬约束，只允许在不劣化安全性的前提下吸收 centered 信息。

2. 改成“分层使用”而不是“概率平均”  
   让 `soft target` 决定主风险，`centered_desirability` 只在 in-spec 或中风险区间参与排序/加权。

3. centered 只做 gating feature  
   不把 `p_fail_center_proxy` 直接作为概率，而是只输入 `q_center`、`center_gap` 等解释性变量。

---

## 8. 中风险区排序器验证

### 8.1 方法

这轮不再把 `centered_desirability` 当成第二个失败概率，而是改成 `soft target` 的中风险区排序器。

规则是：

- `soft target` 继续单独决定动作边界
  - `soft < tau_low` 仍是低风险
  - `soft >= tau_high` 仍是高风险
- 只有 `soft` 自己落在中风险带 `[tau_low, tau_high)` 时，才允许 `centered` 参与
- 在这批中风险样本里，用 `centered proxy risk` 做 isotonic 排序，再把排序分数映射回中风险带内部
- 这样动作层不会变：
  - `low_risk_rate` 不应该变
  - `high_risk_rate` 不应该变
  - `false_clear_rate` 不应该变
- 但如果 `centered` 真有增益，那么它应该提升中风险区内部的 `AP/AUC`

配置：
[autogluon_soft_centered_midrank.yaml](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/configs/autogluon_soft_centered_midrank.yaml)

结果文件：

- 汇总：
  [soft_centered_fusion_mvp_summary.json](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/soft_centered_midrank/soft_centered_fusion_mvp_summary.json)
- 折内信息：
  [soft_centered_fusion_mvp_fold_selection.json](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/soft_centered_midrank/soft_centered_fusion_mvp_fold_selection.json)

### 8.2 结果

`soft_midrank_centered` 与 `soft_only` 的动作率完全一致：

- `false_clear_rate`: `0.0078 -> 0.0078`
- `low_risk_rate`: `0.0044 -> 0.0044`
- `high_risk_rate`: `0.0445 -> 0.0445`
- `retest_rate`: `0.9511 -> 0.9511`

这符合“只做中风险区排序”的设定。

但排序质量没有变好，反而略变差：

- 全局 `hard_out AP`: `0.1902 -> 0.1865`
- 全局 `hard_out AUC`: `0.4957 -> 0.4923`
- 中风险区 `AP`: `0.2018 -> 0.1856`
- 中风险区 `AUC`: `0.5083 -> 0.4948`

### 8.3 折内信号

- 5 折全部真正使用了 `isotonic`
- calibration 中风险样本数分别是：
  - `90`
  - `175`
  - `272`
  - `352`
  - `453`

也就是说，这轮失败不是因为“样本太少，模型没有真正工作”，而是因为 `centered proxy` 在 `soft target` 已经筛出来的中风险带里，没有提供更好的 out-of-spec 排序信息。

### 8.4 判断

这轮结果不支持“把 `centered_desirability` 改成中风险区排序器”这条路。

更准确地说：

- `centered` 作为第二个失败概率不合适
- `centered` 作为中风险区 out-of-spec 排序器也没有被支持
- 它更像还是一个“中心质量”语义头，而不是“失败风险细排器”

因此，如果继续结合两条线，更值得试的是：

1. 让 `soft target` 负责 `fail risk`
2. 让 `centered_desirability` 只负责 `in-spec` 内部质量排序
3. 或者把 `centered` 改成少量 gating feature，而不是直接承担风险排序职责

当前最值得继续的方向不是重新加大融合器复杂度，而是先给融合层加工程 guardrail。

---

## 7. False-Clear Guardrail 追加验证

### 7.1 方法

在不改两条主头、不重训融合器的前提下，给 `logistic_fused` 新增了一层动作级 guardrail：

- 先得到原始 `logistic_fused` 风险分
- 只有当下面两个条件同时满足时，才允许样本真正进入 `low_risk`
  - `p_fail_soft <= soft_ceiling`
  - `p_fail_center_proxy <= center_ceiling`
- 否则即使 `logistic_fused < tau_low`，也会被退回 `retest`

新增配置：
[autogluon_soft_centered_fusion_mvp_false_clear_guardrail.yaml](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/configs/autogluon_soft_centered_fusion_mvp_false_clear_guardrail.yaml)

结果文件：

- 汇总：
  [soft_centered_fusion_mvp_summary.json](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/soft_centered_fusion_mvp_false_clear_guardrail/soft_centered_fusion_mvp_summary.json)
- 折内选择：
  [soft_centered_fusion_mvp_fold_selection.json](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/soft_centered_fusion_mvp_false_clear_guardrail/soft_centered_fusion_mvp_fold_selection.json)

### 7.2 guardrail 结果

| method | mean_brier | hard_out_ap | hard_out_auc | false_clear_rate | low_risk_rate | high_risk_rate | retest_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_fused_guardrailed | 0.1500 | 0.1966 | 0.5244 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| logistic_fused | 0.1509 | 0.2167 | 0.5395 | 0.6255 | 0.6546 | 0.0000 | 0.3454 |
| soft_only | 0.2120 | 0.1902 | 0.4957 | 0.0078 | 0.0044 | 0.0445 | 0.9511 |

### 7.3 折内 guardrail 选择

- 5 折里有 4 折选到最严格组合：
  - `soft_ceiling = 0.02`
  - `center_ceiling = 0.05`
- 只有 fold 2 稍微放宽到：
  - `soft_ceiling = 0.2`
  - `center_ceiling = 0.4`
- 但即使这样，test 侧聚合后仍然变成：
  - `low_risk_rate = 0.0`
  - `retest_rate = 1.0`

### 7.4 判断

这轮 guardrail 的结论非常明确：

- 它确实把 `false_clear_rate` 从 `0.6255` 压回到了 `0.0`
- 但代价是把整个融合器几乎完全“关掉”了
- 也就是说，在当前这版融合语义下，安全性一旦被严肃约束，融合器并不能保住可用的 `low_risk` 行动空间

因此，这轮结果**不支持**“继续沿无结构概率融合 + 外挂 guardrail”深挖。

更合理的下一步不是继续调更细的 ceiling 网格，而是换一种融合关系：

1. `soft target` 继续做主风险头
2. `centered_desirability` 不再直接转成第二个失败概率
3. 它只作为中风险区排序、in-spec 质量排序，或者作为 gating feature 参与决策
