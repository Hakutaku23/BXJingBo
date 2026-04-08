# AutoGluon 滞后性复核记录

## 1. 任务目的

本次工作不再沿用 cleanroom 的 threshold-oriented 指标口径，而是回到 `tabular_framework_validation` 分支，先梳理所有使用 `AutoGluon` 建模的实验，再复用这一分支已经建立的标签方法与评价指标，对 cleanroom 中已观察到的 lag 结论做一次交叉复核。

目标问题是：

> 在 `AutoGluon` 分支自己的标签语义下，当前在 cleanroom 中观察到的 `L60 / L180 / 分层 lag` 是否仍然成立，尤其是“约 3h 的滞后”是否能得到同口径支持。

## 2. 查找到的 AutoGluon 实验

本次通过 `scripts/run_autogluon_*.py` 对当前分支做了完整梳理，实际存在的 AutoGluon 实验包括：

- [run_autogluon_stage0_baseline.py](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_stage0_baseline.py)
- [run_autogluon_stage1_lag_scale.py](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_stage1_lag_scale.py)
- [run_autogluon_stage1_quickcheck.py](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_stage1_quickcheck.py)
- [run_autogluon_stage2_desirability.py](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_stage2_desirability.py)
- [run_autogluon_stage2_dynamic_morphology.py](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_stage2_dynamic_morphology.py)
- [run_autogluon_stage2_feature_engineering.py](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_stage2_feature_engineering.py)
- [run_autogluon_stage2_high_risk.py](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_stage2_high_risk.py)
- [run_autogluon_stage2_soft_probability.py](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_stage2_soft_probability.py)
- [run_autogluon_stage2_soft_probability_feature_distillation.py](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_stage2_soft_probability_feature_distillation.py)
- [run_autogluon_stage2_soft_probability_label_family.py](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_stage2_soft_probability_label_family.py)
- [run_autogluon_stage2_soft_probability_tuning.py](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_stage2_soft_probability_tuning.py)
- [run_autogluon_stage2_soft_probability_x_enrichment.py](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_stage2_soft_probability_x_enrichment.py)
- [run_autogluon_stage3_regime_state.py](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_stage3_regime_state.py)
- [run_autogluon_stage4_interactions.py](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_stage4_interactions.py)
- [run_autogluon_stage5_quality.py](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_stage5_quality.py)

其中，和 lag 最直接相关、同时又在后续阶段持续沿用的主标签线是：

- `centered_desirability`
- `five_bin`

对应历史说明分别见：

- [stage0_baseline_audit.md](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/reports/stage0_baseline_audit.md)
- [stage1_lag_scale_summary.md](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/reports/stage1_lag_scale_summary.md)
- [stage2_dynamic_summary.md](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/reports/stage2_dynamic_summary.md)
- [stage3_regime_summary.md](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/reports/stage3_regime_summary.md)
- [stage4_interaction_summary.md](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/reports/stage4_interaction_summary.md)

## 3. 本次复用的标签与指标

本次复核没有使用 cleanroom 的阈值三类指标，而是严格复用了 AutoGluon 线的标签定义和评价指标。

### 3.1 标签

- `centered_desirability = max(0, 1 - |y - 8.45| / 0.25)`
- `five_bin = [-inf, 8.0, 8.2, 8.7, 8.9, inf]`

说明：

- `stage0` 里还存在 `low_risk`、`high_risk` 两个二头风险标签。
- 但从 `stage1_lag_scale` 开始，按历史记录与用户决策，lag 线已经聚焦为 `centered_desirability + five_bin`。
- 因此本次 lag 复核也只沿这两条标签主线继续，不再混入 `low/high risk`。

### 3.2 指标

- `centered_desirability`
  - `MAE`
  - `RMSE`
  - `rank_correlation`
  - `in_spec_auc_from_desirability`
- `five_bin`
  - `macro_f1`
  - `balanced_accuracy`
  - `multiclass_log_loss`

## 4. 本次 lag 复核的实验方案

本次新增脚本与配置：

- [run_autogluon_lag_reality_check.py](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_lag_reality_check.py)
- [autogluon_lag_reality_check.yaml](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/configs/autogluon_lag_reality_check.yaml)

运行环境：

- `conda` 环境：`autoGluon`
- Python 可执行文件：`D:\miniconda3\envs\autoGluon\python.exe`

控变量方式：

- 仍使用 `merge_data.csv + merge_data_otr.csv`
- 仍使用 `TimeSeriesSplit(n_splits=5)`
- 仍使用 `120min` 统计特征
- 仍使用 `AutoGluon TabularPredictor`
- 仍使用 `medium_quality_faster_train`
- 仍限制为 `GBM + XGB`
- 所有 lag 变体先对齐到共同可评分样本，再统一比较

共同评分样本数：

- `2726`

本次比较的 lag 变体：

- `lag0_win120`
- `lag60_win120`
- `lag120_win120`
- `lag180_win120`
- `hybrid_l180_drop_FI_C53001`
- `hybrid_l180_drop_TI_CM53201`

其中两条分层 lag 直接来自 cleanroom 最后一轮确认：

- `hybrid_l180_drop_FI_C53001`
- `hybrid_l180_drop_TI_CM53201`

它们的语义是：

- 默认多数点位走 `60min lag`
- 少数在 cleanroom 中被识别为慢响应的点位走 `180min lag`

## 5. 结果文件

本次复核结果落盘在：

- [lag_reality_check_summary.json](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/lag_reality_check/lag_reality_check_summary.json)
- [lag_reality_check_results.csv](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/lag_reality_check/lag_reality_check_results.csv)
- [lag_reality_check_feature_catalog.csv](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/lag_reality_check/lag_reality_check_feature_catalog.csv)
- [lag_reality_check_common_samples.csv](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/lag_reality_check/lag_reality_check_common_samples.csv)
- [lag_reality_check_summary.md](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/reports/lag_reality_check/lag_reality_check_summary.md)

## 6. 关键结果

### 6.1 `centered_desirability`

各变体 AutoGluon 汇总结果：

- `lag0_win120`
  - `MAE = 0.3101`
  - `RMSE = 0.3524`
  - `rank_correlation = 0.0294`
  - `in_spec_auc = 0.5102`
- `lag60_win120`
  - `MAE = 0.3141`
  - `RMSE = 0.3542`
  - `rank_correlation = 0.0475`
  - `in_spec_auc = 0.5216`
- `lag120_win120`
  - `MAE = 0.3130`
  - `RMSE = 0.3535`
  - `rank_correlation = 0.0323`
  - `in_spec_auc = 0.5348`
- `lag180_win120`
  - `MAE = 0.3098`
  - `RMSE = 0.3511`
  - `rank_correlation = 0.0558`
  - `in_spec_auc = 0.5635`
- `hybrid_l180_drop_FI_C53001`
  - `MAE = 0.3091`
  - `RMSE = 0.3495`
  - `rank_correlation = 0.0399`
  - `in_spec_auc = 0.5546`
- `hybrid_l180_drop_TI_CM53201`
  - `MAE = 0.3134`
  - `RMSE = 0.3542`
  - `rank_correlation = 0.0166`
  - `in_spec_auc = 0.5259`

判断：

- 如果看 `centered_desirability`，确实能观察到一部分“长 lag 更有利”的迁移信号。
- `lag180_win120` 拿到了这一组里最好的 `rank_correlation` 和 `in_spec_auc`。
- `hybrid_l180_drop_FI_C53001` 拿到了这一组里最好的 `MAE` 和 `RMSE`。
- 但这种支持是“弱支持”，不是压倒性支持。

另外，历史 `stage1` 中 `centered_desirability` 的最佳 lag 包仍然是：

- `lag120_win60`
- `autogluon_mean_mae = 0.3046`

所以 AutoGluon 分支并没有把“约 3h”推成新的明确最优点，更像是在 centered 语义下给出了“长 lag 值得认真看”的旁证。

### 6.2 `five_bin`

各变体 AutoGluon 汇总结果：

- `lag0_win120`
  - `macro_f1 = 0.1835`
  - `balanced_accuracy = 0.2114`
  - `multiclass_log_loss = 0.9398`
- `lag60_win120`
  - `macro_f1 = 0.1819`
  - `balanced_accuracy = 0.2111`
  - `multiclass_log_loss = 0.9249`
- `lag120_win120`
  - `macro_f1 = 0.1827`
  - `balanced_accuracy = 0.2110`
  - `multiclass_log_loss = 0.9207`
- `lag180_win120`
  - `macro_f1 = 0.1787`
  - `balanced_accuracy = 0.2089`
  - `multiclass_log_loss = 0.9421`
- `hybrid_l180_drop_FI_C53001`
  - `macro_f1 = 0.1818`
  - `balanced_accuracy = 0.2108`
  - `multiclass_log_loss = 0.9304`
- `hybrid_l180_drop_TI_CM53201`
  - `macro_f1 = 0.1819`
  - `balanced_accuracy = 0.2111`
  - `multiclass_log_loss = 0.9209`

判断：

- `five_bin` 这条线上，并没有出现对 `3h` 的支持。
- `lag180_win120` 在三项指标上都弱于 `lag120_win120`，也弱于不少较短 lag 版本。
- 历史 `stage1` 中 `five_bin` 的最佳 lag 包就是 `lag120_win120`，本次复核也没有推翻它。
- 分层 lag 里，`hybrid_l180_drop_TI_CM53201` 的 `multiclass_log_loss = 0.9209` 几乎贴近 `lag120_win120 = 0.9207`，但并没有形成稳定超越。

## 7. 综合结论

当前最准确的结论是：

> 在 AutoGluon 分支自己的标签语义和评价指标下，`lag` 这一现象本身是存在的，但“约 3h 的滞后已经被确认”为时尚早。

更细一点地说：

- `centered_desirability` 对长 lag 更敏感，`L180` 和部分分层 lag 确实给出了弱正信号。
- `five_bin` 没有支持 `L180`，仍更偏向 `L120`。
- 因此，cleanroom 中“`L60` 偏整体、`L180` 偏边界谨慎”的结论，并没有在 AutoGluon 语义下被完整复制。
- AutoGluon 口径给出的更稳妥表述应当是：
  - 存在 lag sensitivity
  - centered 目标对更长 lag 有一定支持
  - 但 `3h` 不是当前被稳健确认的统一最优滞后

## 8. 对后续工作的建议

如果后续继续沿 AutoGluon 这条线推进，更合适的方向不是直接宣布 `3h` 成立，而是：

1. 把 lag 结论拆成“目标依赖”
   - `centered_desirability` 看长 lag
   - `five_bin` 保留 `120min` 参考
2. 不再只做全局 lag
   - 若继续验证 lag，应重点围绕少数慢响应点位做更克制的分层 lag
3. 在 AutoGluon 线上先做确认性实验，而不是再扩大搜索网格
   - 当前分层 lag 已经足够说明“值得继续”，但还不足以支持“已经确认”

## 9. centered_desirability-only 长 lag 精修

在上一轮结论明确为“只在 `centered_desirability` 下继续看长 lag，不再让 `five_bin` 牵制判断”之后，本分支继续做了一轮 centered-only 的 focused refinement。

### 9.1 新增配置与运行

- [autogluon_centered_desirability_long_lag_refinement.yaml](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/configs/autogluon_centered_desirability_long_lag_refinement.yaml)
- [run_autogluon_lag_reality_check.py](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_lag_reality_check.py)

新产物在：

- [lag_reality_check_summary.json](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/centered_desirability_long_lag_refinement/lag_reality_check_summary.json)
- [lag_reality_check_results.csv](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/centered_desirability_long_lag_refinement/lag_reality_check_results.csv)
- [lag_reality_check_summary.md](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/reports/centered_desirability_long_lag_refinement/lag_reality_check_summary.md)

### 9.2 实验边界

本轮只保留：

- `centered_desirability`

不再比较：

- `five_bin`

候选 lag 族包括：

- `lag0_win120`
- `lag120_win60`
- `lag180_win60`
- `lag240_win60`
- `lag300_win60`
- `lag360_win60`
- `lag480_win60`
- `lag120_win120`
- `lag180_win120`
- `lag240_win120`
- `lag300_win120`
- `hybrid_l180_drop_FI_C53001`

共同评分样本数：

- `2725`

### 9.3 关键结果

按 `AutoGluon mean MAE` 从好到差排序：

- `lag240_win120`
  - `MAE = 0.3053`
  - `RMSE = 0.3459`
  - `rank_correlation = 0.1133`
  - `in_spec_auc = 0.5846`
- `lag480_win60`
  - `MAE = 0.3063`
  - `RMSE = 0.3474`
  - `rank_correlation = 0.1195`
  - `in_spec_auc = 0.5931`
- `lag180_win60`
  - `MAE = 0.3066`
  - `RMSE = 0.3473`
  - `rank_correlation = 0.0770`
  - `in_spec_auc = 0.5643`
- `lag120_win120`
  - `MAE = 0.3072`
  - `RMSE = 0.3477`
  - `rank_correlation = 0.0766`
  - `in_spec_auc = 0.5548`
- `lag120_win60`
  - `MAE = 0.3095`
  - `RMSE = 0.3497`
  - `rank_correlation = 0.0398`
  - `in_spec_auc = 0.5295`

较弱的一组包括：

- `lag300_win60`
- `lag300_win120`
- `lag0_win120`
- `hybrid_l180_drop_FI_C53001`

其中 `hybrid_l180_drop_FI_C53001` 在 centered-only 口径下并没有延续上一轮的优势，说明 cleanroom 里对 threshold/boundary 任务有帮助的分层 lag，不一定会自然迁移到 centered 语义。

### 9.4 当前判断

这一轮比上一轮更明确地支持：

> centered_desirability 这条 AutoGluon 语义线，确实存在对更长 lag 的稳定偏好。

但这条支持不是“3h 已确认”，而是：

- `180min` 有效
- `240min` 更强
- `480min` 也仍然有强信号

所以当前更准确的结论应该写成：

> 在 `centered_desirability` 任务上，AutoGluon 分支支持“长 lag 家族值得继续”，而不是只支持“约 3h”这一个固定点。

### 9.5 当前最值得冻结的 centered lag 参考

如果只从 `centered_desirability` 出发，当前最值得作为后续 centered 参考输入包的是：

1. `lag240_win120`
   - 综合最均衡
   - `MAE / RMSE / rank_correlation / in_spec_auc` 都处在第一梯队
2. `lag480_win60`
   - 更偏“排序与区分能力”最强
   - `rank_correlation` 和 `in_spec_auc` 更高
3. `lag180_win60`
   - 仍保留“约 3h”的可解释候选身份
   - 但不再应被当作既定事实，只能视为长 lag 家族中的一个有效成员

## 10. 将 centered 长 lag 候选带回 Stage 6 centered-quality

在 centered-only 长 lag 精修之后，本分支继续做了下一步验证：

> 把新的 centered lag 参考输入包带回 `stage6 centered-quality`，检查更长 lag 是否也能作为 centered-quality 专用特征层的更优底座。

### 10.1 实现方式

为了避免重写 `stage6` 逻辑，这次只对脚本增加了一个很小的输入覆盖口：

- [run_autogluon_stage6_centered_quality.py](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_stage6_centered_quality.py)

新增了两个配置：

- [autogluon_stage6_centered_quality_lag240_win120.yaml](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/configs/autogluon_stage6_centered_quality_lag240_win120.yaml)
- [autogluon_stage6_centered_quality_lag480_win60.yaml](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/configs/autogluon_stage6_centered_quality_lag480_win60.yaml)

本次只改：

- `selected_stage1_variant`
- `tau_minutes`
- `window_minutes`

其余 `stage6` 语义保持不变：

- interaction 仍为 `flow_balance`
- quality package 仍为 `combined_quality`
- centered-quality packages 仍为
  - `center_stability`
  - `center_direction`
  - `centered_quality_full`

### 10.2 结果文件

`lag240_win120` 对照：

- [stage6_centered_quality_summary.json](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/stage6_centered_quality_lag240_win120/stage6_centered_quality_summary.json)
- [stage6_centered_quality_summary.md](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/reports/stage6_centered_quality_lag240_win120/stage6_centered_quality_summary.md)

`lag480_win60` 对照：

- [stage6_centered_quality_summary.json](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/stage6_centered_quality_lag480_win60/stage6_centered_quality_summary.json)
- [stage6_centered_quality_summary.md](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/reports/stage6_centered_quality_lag480_win60/stage6_centered_quality_summary.md)

### 10.3 对比结果

原始 `stage6` 最优参考是：

- 输入底座：`lag120_win60`
- 最优 centered package：`center_direction`
- `autogluon_current_mean_mae = 0.301895`
- `autogluon_centered_mean_mae = 0.300583`
- `delta_mae = -0.001313`

#### `lag240_win120` 作为 stage6 底座

最好的两个 centered package 为：

- `center_stability`
  - `autogluon_current_mean_mae = 0.307060`
  - `autogluon_centered_mean_mae = 0.304651`
  - `delta_mae = -0.002409`
- `centered_quality_full`
  - `autogluon_current_mean_mae = 0.307060`
  - `autogluon_centered_mean_mae = 0.304668`
  - `delta_mae = -0.002391`

解读：

- centered-quality 特征层本身仍然有效
- 但绝对最优没有超过原始 `stage6` 的 `0.300583`

#### `lag480_win60` 作为 stage6 底座

最好的 centered package 为：

- `center_direction`
  - `autogluon_current_mean_mae = 0.306817`
  - `autogluon_centered_mean_mae = 0.303368`
  - `delta_mae = -0.003448`

解读：

- 这是三组底座里“局部 centered package 增益”最大的一个
- 但它的绝对最优仍然没有超过原始 `stage6` 的 `0.300583`

### 10.4 当前判断

这一步的结论很明确：

> centered-only 里的长 lag 信号，不能直接平移成更好的 `stage6 centered-quality` 底座。

更细一点说：

- 长 lag 底座下，centered-quality 包仍然能带来局部增益
- 说明 centered-quality 特征与长 lag 并不冲突
- 但原始 `lag120_win60` 底座在 `stage6 centered-quality` 这一层仍然是绝对最优

所以现在最稳妥的工作结论应当写成：

- `centered_desirability` 回归本身支持继续看长 lag
- 但如果目标是 `stage6 centered-quality` 当前实现的整体最优底座，仍应保留 `lag120_win60`
- `lag240_win120` 和 `lag480_win60` 更适合作为 centered-only 旁路线索，而不是立即替换 `stage6` 主底座

## 11. 原因分析：为什么长 lag 对 centered-only 有利，却没有迁移成更好的 Stage 6 底座

当前我对这个现象的判断是：

> 长 lag 并不是“没用”，而是它的增益更偏向 `centered_desirability` 主回归本身；一旦进入 `stage6 centered-quality` 这一层，原本围绕 `lag120_win60` 验证出来的 interaction / quality / centered 专用特征协同关系会被部分打散。

可以拆成四点来看。

### 11.1 长 lag 先把底座拉弱了，centered package 只是在做补救

最直观的证据是三套 `stage6` 底座的 `autogluon_current_mean_mae`：

- 原始 `lag120_win60` 底座：`0.301895`
- `lag240_win120` 底座：`0.307060`
- `lag480_win60` 底座：`0.306817`

也就是说，在还没加 centered-quality 包之前，长 lag 底座本身已经比原始 `stage6` 底座更弱了。

之后 centered-quality 包确实能把它们往回拉：

- `lag240_win120 + center_stability`: `delta_mae = -0.002409`
- `lag240_win120 + centered_quality_full`: `delta_mae = -0.002391`
- `lag480_win60 + center_direction`: `delta_mae = -0.003448`

这说明 centered-quality 特征层对长 lag 仍然有帮助，但它们做的是“补救”，不是“把更弱底座反超成新的最优底座”。

### 11.2 Stage 6 的 centered 特征语义，本质上是“窗口内相对中心性”，它更依赖被选中的窗口本身是合适的

`stage6` 的 centered-quality 特征并不是直接预测 `8.45`，而是在每个窗口内部先标准化，再提取：

- `mean_abs_z`
- `center_band_ratio`
- `center_hold_ratio`
- `last_z`
- `late_shift_z`
- `tail_bias`
- `reversion_score`

实现位置见：

- [run_autogluon_stage6_centered_quality.py](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_stage6_centered_quality.py#L81)
- [run_autogluon_stage6_centered_quality.py](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_stage6_centered_quality.py#L129)

这类特征的前提是：

> 当前取到的那个时间窗，本身就应该与最终质量中心性有较直接的关系。

对 `lag120_win60` 来说，这个前提在历史 `stage1-6` 里已经被逐层验证过；但当窗口整体后移到 `240min` 甚至 `480min` 后，`last_z / late_shift_z / reversion_score` 描述的就更像“更早一段历史过程的相对位置”，而不一定还是“接近样品形成时刻的中心性线索”。  
所以 centered-only 回归还能从长 lag 的宏观关联里受益，但 centered-quality 这类“窗口内形态语义”未必还能同样成立。

### 11.3 长 lag 的收益更不稳定，集中在少数折段，而不是像原始底座那样更均衡

原始 `stage6 + center_direction` 的最优版本，5 折里是“前两折略亏、后三折持续回补”：

- fold 1: `+0.002181`
- fold 2: `+0.001515`
- fold 3: `-0.006803`
- fold 4: `-0.002089`
- fold 5: `-0.001367`

这是一个比较均衡的结构。

而长 lag 底座下的 centered package 更像“局部强、但跨折不稳”：

`lag240_win120 + center_stability`
- fold 1: `-0.003907`
- fold 2: `-0.009265`
- fold 3: `-0.004524`
- fold 4: `+0.000018`
- fold 5: `+0.005634`

`lag480_win60 + center_direction`
- fold 1: `-0.004612`
- fold 2: `-0.003737`
- fold 3: `+0.003729`
- fold 4: `-0.007828`
- fold 5: `-0.004793`

可以看到，长 lag 的改善更依赖具体折段；尤其 `lag240_win120` 在 fold 5 明显回吐，`lag480_win60` 在 fold 3 明显回吐。  
这说明长 lag 更像“特定工况下的强信号”，还没有形成稳定底座。

### 11.4 特征重要性显示：长 lag 底座更容易被少数强 current 特征主导，centered 特征并没有形成像原始底座那样的稳定协同

原始最优 `stage6` 的 feature importance 更像一种平衡结构：

- current lag 特征
- quality 特征
- centered directional 特征

同时都能排进前列。例子见：

- [stage6_centered_quality_feature_importance_full_center_direction.csv](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/stage6_centered_quality_feature_importance_full_center_direction.csv)

而在 `lag480_win60 + center_direction` 里，最重要的单个特征出现了明显“过强主导”：

- `TI_C54002_PV_F_CV__lag480_win60_last`
  - importance `0.025619`

它远高于后面其他特征，见：

- [stage6_centered_quality_feature_importance_full_center_direction.csv](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/stage6_centered_quality_lag480_win60/stage6_centered_quality_feature_importance_full_center_direction.csv)

这通常意味着：

- 模型抓到了一个强相关长 lag 信号
- 但整体结构更容易被少数 current 特征牵着走
- centered 专用特征是在做二级修正，而不是形成稳定的主导协同

`lag240_win120` 的 feature importance 则呈现另一种模式：

- centered 特征能进入前排
- 但当前 top features 仍主要是 lag240 的 current snapshot 与 quality 特征

对应文件：

- [stage6_centered_quality_feature_importance_full_center_stability.csv](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/stage6_centered_quality_lag240_win120/stage6_centered_quality_feature_importance_full_center_stability.csv)
- [stage6_centered_quality_feature_importance_full_centered_quality_full.csv](D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/stage6_centered_quality_lag240_win120/stage6_centered_quality_feature_importance_full_centered_quality_full.csv)

这进一步支持了前面的判断：

> 长 lag 底座下，centered-quality 仍有辅助价值，但还没有建立起比原始 `lag120_win60` 更稳的整体协同结构。

## 12. 当前最稳妥的结论

现在可以把这条线收紧成一句话：

> 长 lag 更像是 `centered_desirability` 主回归的一条有效旁路，而不是当前 `stage6 centered-quality` 的新主底座。

因此后续更合适的工作方向不是继续盲目扩 lag，而是二选一：

1. 保留 `lag120_win60` 作为 `stage6` 主底座，继续做 centered-quality 专用特征精修。
2. 把 `lag240_win120 / lag480_win60` 留在 centered-only 路线，单独研究它们为何对 centered 回归更敏感，而不要强行带入 `stage6` 主线。
