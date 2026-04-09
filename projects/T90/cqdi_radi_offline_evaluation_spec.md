# CQDI 与 RADI 的计算方法与离线评估说明

## 1. 目的

本文定义两类面向工程部署的离线可用性指标：

- `CQDI`：用于 `centered_desirability`
- `RADI`：用于 `soft target`

两者不共享一个主指标，因为它们服务的工程目标不同：

- `centered_desirability` 更接近**中心质量优选 / 排序**
- `soft target` 更接近**风险评分 / 告警决策 / reject-retest 策略**

---

## 2. 统一评估前提

## 2.1 外层时间切分

```text
outer_cv = TimeSeriesSplit(n_splits=5)
```

## 2.2 内层阈值选择

```text
inner_cv = TimeSeriesSplit(n_splits=3)
```

阈值必须仅使用训练折内的 OOF 预测选择，不得看 outer test。

## 2.3 测量误差场景

```text
measurement_error_scenarios = [0.10, 0.15, 0.20]
```

所有部署指标都建议分场景输出，并以最差场景作为上线前主要参考。

## 2.4 公共常量

```text
center = 8.45
tolerance = 0.25
spec_low = 8.20
spec_high = 8.70
center_band_low = 8.35
center_band_high = 8.55
```

---

## 3. 公共样本级明细表

文件名建议：

```text
offline_eval_scored_rows.csv
```

## 3.1 字段说明

| 字段名 | 类型 | 说明 |
|---|---:|---|
| run_id | string | 本次评估唯一 ID |
| scheme_name | string | 方案名 |
| label_family | string | `centered_desirability` / `soft_target` |
| feature_recipe_name | string | 特征工程方案名 |
| model_framework | string | 如 `autogluon` |
| outer_fold_id | int | 外层 fold |
| sample_time | datetime | 样本时间 |
| t90_obs | float | 观测 T90 |
| measurement_error_scenario | float | `0.10 / 0.15 / 0.20` |
| target_centered_desirability_point | float | 点式 centered 标签 |
| target_centered_desirability_uncertain | float | 误差感知 centered 标签 |
| true_center_band_obs | int | 观测是否在 center band |
| true_center_band_prob | float | 误差感知 center band 概率 |
| true_outspec_obs | int | 观测是否越界 |
| true_outspec_prob | float | 误差感知越界概率 |
| pred_score_raw | float | 模型原始输出 |
| pred_score_clipped | float | `clip(pred_score_raw, 0, 1)` |
| pred_desirability_aligned | float | 对齐后的 desirability 分数 |
| pred_outspec_risk_aligned | float | 对齐后的 out-of-spec 风险分数 |
| preferred_threshold | float | centered 线使用 |
| preferred_flag | int | centered 线使用 |
| clear_threshold | float | soft 线使用 |
| alert_threshold | float | soft 线使用 |
| clear_flag | int | soft 线使用 |
| retest_flag | int | soft 线使用 |
| alert_flag | int | soft 线使用 |

---

## 4. 对齐分数规则

为了让两条线在共同外部业务诊断上可比较，定义统一对齐规则。

## 4.1 `centered_desirability` 线

```text
pred_desirability_aligned = clip(pred_score_raw, 0, 1)
pred_outspec_risk_aligned = 1 - clip(pred_score_raw, 0, 1)
```

## 4.2 `soft target` 线

```text
pred_outspec_risk_aligned = clip(pred_score_raw, 0, 1)
pred_desirability_aligned = 1 - clip(pred_score_raw, 0, 1)
```

> 说明：该对齐仅用于共同外部诊断，不改变模型本来的训练任务。

---

# 5. CQDI：Center-Quality Deployability Index

## 5.1 适用对象

- 标签族：`centered_desirability`

## 5.2 目标

回答问题：

> 当模型把某些样本打成高 desirability 时，这些样本是否真的富集中心优质工况，同时不显著混入越界风险？

## 5.3 优选区定义

在 inner OOF 预测上搜索 `preferred_threshold = τ_pref`，定义：

```text
preferred_flag = 1(pred_desirability_aligned >= τ_pref)
```

推荐候选网格：

```text
τ_pref ∈ {0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85}
```

---

## 5.4 centered 阈值候选表

文件名建议：

```text
offline_centered_threshold_candidates.csv
```

### 字段

| 字段名 | 类型 | 说明 |
|---|---:|---|
| run_id | string | 评估 ID |
| scheme_name | string | 方案名 |
| outer_fold_id | int | 外层 fold |
| inner_threshold_pref | float | 候选阈值 |
| train_preferred_rate | float | inner OOF 高分区覆盖率 |
| train_center_prob_mean_preferred | float | 高分区的 `mean(true_center_band_prob)` |
| train_outspec_prob_mean_preferred | float | 高分区的 `mean(true_outspec_prob)` |
| train_center_prob_base | float | 全样本 `mean(true_center_band_prob)` |
| train_center_lift | float | `train_center_prob_mean_preferred / train_center_prob_base` |
| train_hard_out_ap_aligned | float | `pred_outspec_risk_aligned` 对 `true_outspec_obs` 的 AP |
| train_gate_pass | int | 是否过门槛 |
| cqdi_train_candidate | float | 该阈值的训练候选分 |

---

## 5.5 CQDI 子分数

定义：

```text
center_score   = min(1, center_prob_mean_preferred / 0.50)
lift_score     = min(1, center_lift / 1.50)
safety_score   = min(1, 0.08 / max(outspec_prob_mean_preferred, 1e-6))
coverage_score = clip(1 - abs(preferred_rate - 0.20) / 0.12, 0, 1)
```

### 含义

- `center_score`：高分区里中心优质样本比例是否足够高
- `lift_score`：相较随机抽样，是否显著富集中心优质样本
- `safety_score`：高分区里越界污染是否足够低
- `coverage_score`：高分区覆盖率是否落在可执行区间，而不是太窄或太宽

---

## 5.6 CQDI 门槛项

定义：

```text
gate_pass = 1
  if hard_out_ap_aligned >= ap_floor
  and outspec_prob_mean_preferred <= 0.12
  and 0.08 <= preferred_rate <= 0.30
else 0
```

### `ap_floor` 设定建议

若当前项目没有正式统一门槛，可先取：

```text
ap_floor = 当前工程基线 hard_out_ap_diagnostic × 0.95
```

---

## 5.7 CQDI 公式

```text
CQDI = 100 * gate_pass * (
    0.40 * center_score
  + 0.25 * lift_score
  + 0.20 * safety_score
  + 0.15 * coverage_score
)
```

---

## 5.8 centered 外层 fold 汇总表

文件名建议：

```text
offline_centered_fold_summary.csv
```

### 字段

| 字段名 | 类型 | 说明 |
|---|---:|---|
| run_id | string | 评估 ID |
| scheme_name | string | 方案名 |
| outer_fold_id | int | fold |
| selected_preferred_threshold | float | 内层选出的阈值 |
| test_preferred_rate | float | 测试集高分区覆盖率 |
| test_center_prob_mean_preferred | float | 高分区 `mean(true_center_band_prob)` |
| test_outspec_prob_mean_preferred | float | 高分区 `mean(true_outspec_prob)` |
| test_center_prob_base | float | 全样本 `mean(true_center_band_prob)` |
| test_center_lift | float | 中心富集倍数 |
| test_hard_out_ap_aligned | float | hard-out AP diagnostic |
| test_hard_out_auc_aligned | float | hard-out AUC diagnostic |
| cqdi_fold | float | 本 fold 的 CQDI |
| gate_pass | int | 是否过门槛 |
| deployment_status_fold | string | `FAIL / CANARY / READY` |

---

## 5.9 centered 全局部署汇总表

文件名建议：

```text
offline_centered_deployability_summary.csv
```

### 字段

| 字段名 | 类型 | 说明 |
|---|---:|---|
| run_id | string | 评估 ID |
| scheme_name | string | 方案名 |
| label_family | string | 固定 `centered_desirability` |
| feature_recipe_name | string | 特征方案 |
| measurement_error_scenario | float | `0.10 / 0.15 / 0.20` |
| cqdi_mean | float | 平均 CQDI |
| cqdi_median | float | 中位数 CQDI |
| cqdi_p10 | float | 第 10 分位 CQDI |
| gate_pass_rate | float | 过门槛 fold 占比 |
| hard_out_ap_aligned_mean | float | 平均 hard-out AP |
| hard_out_auc_aligned_mean | float | 平均 hard-out AUC |
| recommended_status | string | `FAIL / CANARY / READY` |
| summary_note | string | 备注 |

### 推荐状态规则

```text
FAIL   : gate_pass_rate < 0.80 or cqdi_mean < 60
CANARY : gate_pass_rate >= 0.80 and cqdi_mean >= 70
READY  : gate_pass_rate >= 0.80 and cqdi_mean >= 80 and cqdi_p10 >= 70
```

---

# 6. RADI：Risk-Alert Deployability Index

## 6.1 适用对象

- 标签族：`soft_target`

## 6.2 目标

回答问题：

> 模型输出的 soft risk 能否被可靠用于 clear / retest / alert 的工程动作？

## 6.3 三段式动作区

在 inner OOF 预测上搜索两个阈值：

- `clear_threshold = τ_low`
- `alert_threshold = τ_high`

定义：

```text
clear_flag  = 1(pred_outspec_risk_aligned <= τ_low)
alert_flag  = 1(pred_outspec_risk_aligned >= τ_high)
retest_flag = 1 - clear_flag - alert_flag
```

推荐阈值网格：

```text
τ_low  ∈ {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40}
τ_high ∈ {0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90}
约束：τ_low < τ_high
```

---

## 6.4 soft 阈值候选表

文件名建议：

```text
offline_soft_threshold_candidates.csv
```

### 字段

| 字段名 | 类型 | 说明 |
|---|---:|---|
| run_id | string | 评估 ID |
| scheme_name | string | 方案名 |
| outer_fold_id | int | 外层 fold |
| inner_clear_threshold | float | clear 阈值 |
| inner_alert_threshold | float | alert 阈值 |
| train_clear_rate | float | clear 区占比 |
| train_retest_rate | float | retest 区占比 |
| train_alert_rate | float | alert 区占比 |
| train_clear_outspec_prob_mean | float | clear 区 `mean(true_outspec_prob)` |
| train_alert_outspec_recall_prob | float | `sum(true_outspec_prob * alert_flag) / sum(true_outspec_prob)` |
| train_alert_outspec_precision_prob | float | `sum(true_outspec_prob * alert_flag) / sum(alert_flag)` |
| train_outspec_prob_base | float | 全体 `mean(true_outspec_prob)` |
| train_alert_outspec_lift | float | `alert_outspec_precision_prob / train_outspec_prob_base` |
| train_soft_brier | float | `mean((pred_outspec_risk_aligned - true_outspec_prob)^2)` |
| train_hard_out_ap_aligned | float | 对 `true_outspec_obs` 的 AP |
| train_gate_pass | int | 是否过门槛 |
| radi_train_candidate | float | 该阈值组合的训练候选分 |

---

## 6.5 RADI 子分数

定义：

```text
clear_score  = min(1, 0.05 / max(clear_outspec_prob_mean, 1e-6))
recall_score = min(1, alert_outspec_recall_prob / 0.80)
lift_score   = min(1, alert_outspec_lift / 2.00)
retest_score = clip(1 - max(0, retest_rate - 0.25) / 0.25, 0, 1)
brier_score  = min(1, 0.10 / max(soft_brier, 1e-6))
```

### 含义

- `clear_score`：放行区是否足够安全
- `recall_score`：真实危险样本是否能被足够比例拦住
- `lift_score`：告警区是否真正富集危险样本
- `retest_score`：复检负担是否可控
- `brier_score`：概率分数本身是否足够稳定

---

## 6.6 RADI 门槛项

定义：

```text
gate_pass = 1
  if clear_outspec_prob_mean <= 0.08
  and alert_outspec_recall_prob >= 0.50
  and hard_out_ap_aligned >= ap_floor
else 0
```

---

## 6.7 RADI 公式

```text
RADI = 100 * gate_pass * (
    0.35 * clear_score
  + 0.30 * recall_score
  + 0.15 * lift_score
  + 0.10 * retest_score
  + 0.10 * brier_score
)
```

---

## 6.8 soft 外层 fold 汇总表

文件名建议：

```text
offline_soft_fold_summary.csv
```

### 字段

| 字段名 | 类型 | 说明 |
|---|---:|---|
| run_id | string | 评估 ID |
| scheme_name | string | 方案名 |
| outer_fold_id | int | fold |
| selected_clear_threshold | float | 内层选出的 clear 阈值 |
| selected_alert_threshold | float | 内层选出的 alert 阈值 |
| test_clear_rate | float | clear 区占比 |
| test_retest_rate | float | retest 区占比 |
| test_alert_rate | float | alert 区占比 |
| test_clear_outspec_prob_mean | float | clear 区越界概率均值 |
| test_alert_outspec_recall_prob | float | alert 召回 |
| test_alert_outspec_precision_prob | float | alert 精度 |
| test_outspec_prob_base | float | 全体越界概率均值 |
| test_alert_outspec_lift | float | alert lift |
| test_soft_brier | float | soft Brier |
| test_hard_out_ap_aligned | float | hard-out AP diagnostic |
| test_hard_out_auc_aligned | float | hard-out AUC diagnostic |
| radi_fold | float | 本 fold 的 RADI |
| gate_pass | int | 是否过门槛 |
| deployment_status_fold | string | `FAIL / CANARY / READY` |

---

## 6.9 soft 全局部署汇总表

文件名建议：

```text
offline_soft_deployability_summary.csv
```

### 字段

| 字段名 | 类型 | 说明 |
|---|---:|---|
| run_id | string | 评估 ID |
| scheme_name | string | 方案名 |
| label_family | string | 固定 `soft_target` |
| feature_recipe_name | string | 特征方案 |
| measurement_error_scenario | float | `0.10 / 0.15 / 0.20` |
| radi_mean | float | 平均 RADI |
| radi_median | float | 中位数 RADI |
| radi_p10 | float | 第 10 分位 RADI |
| gate_pass_rate | float | 过门槛 fold 占比 |
| hard_out_ap_aligned_mean | float | 平均 hard-out AP |
| hard_out_auc_aligned_mean | float | 平均 hard-out AUC |
| soft_brier_mean | float | 平均 soft Brier |
| recommended_status | string | `FAIL / CANARY / READY` |
| summary_note | string | 备注 |

### 推荐状态规则

```text
FAIL   : gate_pass_rate < 0.80 or radi_mean < 60
CANARY : gate_pass_rate >= 0.80 and radi_mean >= 70
READY  : gate_pass_rate >= 0.80 and radi_mean >= 80 and radi_p10 >= 70
```

---

# 7. 计算步骤（可直接落地）

## 7.1 样本级打分
对每个外层测试样本，记录：

1. `t90_obs`
2. 测量误差场景 `e`
3. `true_center_band_prob`
4. `true_outspec_prob`
5. 模型原始输出 `pred_score_raw`
6. 对齐后的：
   - `pred_desirability_aligned`
   - `pred_outspec_risk_aligned`

## 7.2 内层阈值搜索
- centered：搜索 `preferred_threshold`
- soft：搜索 `(clear_threshold, alert_threshold)`

## 7.3 外层计算 fold 指标
- centered：计算 `CQDI`
- soft：计算 `RADI`

## 7.4 跨 fold 聚合
输出：

- mean
- median
- p10
- gate_pass_rate
- recommended_status

## 7.5 跨误差场景聚合
建议额外输出：

```text
worst_case_cqdi = min(cqdi_mean over e in {0.10,0.15,0.20})
worst_case_radi = min(radi_mean over e in {0.10,0.15,0.20})
```

用于部署前最终保守判定。

---

# 8. 解释建议

## 8.1 centered 线
不要只说：

- `MAE = 0.30`

要同时说明：

- 高分区有多少真的落在 center band
- 高分区越界污染率是多少
- 是否守住 hard-out 安全底线

## 8.2 soft 线
不要只说：

- `soft_brier = 0.087`

要同时说明：

- clear 区是否足够安全
- alert 区是否能抓到足够多的危险样本
- retest 负担是否可执行

---

# 9. 最终建议

- `CQDI` 用来判断 `centered_desirability` 是否可用于**中心质量优选**
- `RADI` 用来判断 `soft target` 是否可用于**风险告警与动作策略**
- 两者都不应只依赖单一拟合误差
- 所有部署前判断都建议采用：
  - 多 fold
  - 多测量误差场景
  - 最差场景优先

如果只做一轮最小可用离线验收，建议最少输出这 5 个文件：

```text
offline_eval_scored_rows.csv
offline_centered_fold_summary.csv
offline_centered_deployability_summary.csv
offline_soft_fold_summary.csv
offline_soft_deployability_summary.csv
```
