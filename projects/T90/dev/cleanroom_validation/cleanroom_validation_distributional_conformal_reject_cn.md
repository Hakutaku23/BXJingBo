# Cleanroom 实验记录——5-bin 软分布标签 + conformal/quantile 校准 + reject option

## 分支目的

这份记录单独对应一条新的 cleanroom 子路线：

- `soft 5-bin distribution`
- `split conformal / quantile uncertainty calibration`
- `retest / reject option`

它与前一份 `distributional_reject_option` 记录区分开，原因是这条线不再把“塌缩成业务三类”作为主评价对象，而是把“校准后的 5-bin 预测集”本身当作主要不确定性对象。

## 纠偏说明

2026-04-01 复核后确认，前一轮实现虽然已经进入了 `5-bin soft distribution`，但实验解释仍然过度围绕：

- 塌缩后的 `acceptable / warning / unacceptable`

来展开。

这会带来一个偏差：

- 即使 5-bin 输出本身已经更温和、更诚实，
- 一旦又被强制压回离散业务类，
- 边界模糊仍可能被重新硬切。

因此，这条新记录对应的实验主旨被收紧为：

> 先验证“校准后的 5-bin 预测集 + retest”是否能把边界模糊保留下来，并转成更合理的拒判逻辑，而不是先比较塌缩后三类是否立刻更高分。

## 冻结的首轮实验设计

- 输入边界不变：
  - `merge_data.csv + merge_data_otr.csv`
  - strict sensor-identity de-dup
  - `simple 120min causal stats`
  - fold-local `topk=40`
  - `TimeSeriesSplit(5)`
- bin 定义固定：
  - `<8.0`
  - `8.0-8.2`
  - `8.2-8.7`
  - `8.7-8.9`
  - `>=8.9`
- 监督形式固定：
  - 只对 `8.2 / 8.7`
  - `soft_radius = 0.05`
  - piecewise-linear soft labels
- 校准方式：
  - outer-train 内再切出 calibration tail
  - 使用 split conformal APS quantile
- reject 规则：
  - 先形成 5-bin prediction set
  - 若 set 只落入一个 business side，则直接输出该业务结论
  - 若 set 跨越多个 business side，则输出 `retest`

## 预期主指标

首轮优先看：

- hard-bin set coverage
- mean set size
- singleton rate
- ambiguous-business-set rate
- retest rate
- boundary retest rate
- non-boundary decision coverage

业务三类指标只作为次级观察量。

## 当前状态

- 新计划已冻结
- conformal 首轮脚本与配置已建立
- 首轮结果已补入本文件

---

## 2026-04-01 首轮执行记录：split conformal / quantile calibrated 5-bin + reject

### 本轮真正验证的对象

这次我按纠偏后的路线，明确把主对象改成了：

- `5-bin soft distribution`
- `split conformal APS quantile calibration`
- `prediction set -> unique business side else retest`

也就是说，这一轮不再把“最终三类分数”当作主任务，而是先看：

- 校准后的 5-bin 预测集是否合理
- `retest` 是否真的集中到模糊边界

### 实际实现

- 数据与特征边界保持不变：
  - `merge_data.csv + merge_data_otr.csv`
  - strict sensor-identity de-dup
  - `simple 120min causal stats`
  - fold-local `topk=40`
  - `TimeSeriesSplit(5)`
- soft label 规则保持不变：
  - 只对 `8.2 / 8.7`
  - `soft_radius = 0.05`
  - piecewise-linear redistribution
- 模型：
  - `5-bin multinomial logistic`
  - weighted sample replication
- conformal 层：
  - 每个 outer train fold 再切出 calibration tail
  - `alpha in {0.05, 0.10, 0.15, 0.20}`
  - 用 APS quantile 形成 5-bin prediction set
- reject 规则：
  - 若 prediction set 只对应一个 business side，则直接输出该 side
  - 若跨越多个 business side，则输出 `retest`

### 总体结果

pooled outer-test 结果如下：

- `hard_bin_coverage = 0.8656`
- `mean_set_size = 4.1855`
- `singleton_rate = 0.0696`
- `ambiguous_business_set_rate = 0.9225`
- `retest_rate = 0.9225`
- `boundary_retest_rate = 0.9268`
- `non_boundary_decision_coverage = 0.0796`

次级业务指标只保留作诊断：

- `covered_macro_f1 = 0.1705`
- `covered_balanced_accuracy = 0.2360`
- `covered_warning_AP = 0.1232`
- `covered_unacceptable_AP = 0.0737`

### 这组结果说明了什么

这轮最关键的现象不是“分数低”，而是：

1. conformal set 太大  
   平均 set size 已经到 `4.19`，而总 bin 才 `5` 个。
2. 大多数 set 会跨越多个 business side  
   `ambiguous_business_set_rate = 0.9225`
3. 一旦按“跨 business side 就 retest”的规则执行，几乎所有样本都会被打回  
   `retest_rate = 0.9225`

换句话说，这条路线的主旨没有错，但这第一版 APS 式 reject 规则太保守，导致：

- 它确实避免了边界上的强行硬切；
- 但代价是几乎不愿意做决定。

### 对边界行为的判断

严格说，边界 reject 确实略高于整体 reject：

- overall `retest_rate = 0.9225`
- boundary `retest_rate = 0.9268`

这说明 reject 至少没有明显跑偏到“只在容易样本上乱拒判”。

但问题也很明显：

- 非边界样本的决策覆盖率也只有 `0.0796`

所以这并不是“reject 成功聚焦到边界”的状态，而是“整体都拒得太多，只是边界更严重一点”。

### 逐折观察

- `fold 1 / 2 / 3`
  - `qhat` 基本贴近 `1.0`
  - set 很大
  - `retest` 接近全量
- `fold 4`
  - 稍微收缩了一点
  - `decision_coverage = 0.1256`
- `fold 5`
  - 是这轮里最敢做决定的一折
  - 但 `hard_bin_coverage` 又掉到了 `0.5749`

这说明当前第一版方法的主要张力已经很清楚：

- 想保 coverage，就会让 set 大到几乎全拒
- 想收 decision coverage，又会明显伤到 hard-bin coverage

### 当前结论

这轮不能得出“distributional + conformal 这条路不行”的结论。  
更准确的写法应该是：

1. 你指出的歧义是对的，我前一轮实现确实把重点又拉回了塌缩后三类。
2. 纠偏后的 cleanroom 主线已经建立完成。
3. 这条新主线的首轮结果说明：
   - **split conformal APS + unique-business-side-else-retest**
   - 在当前设定下过于保守
   - 暂时还不能作为可用的现场 reject policy

### 下一步应该怎么走

如果继续沿这条新主线推进，下一步不应该回去继续围绕 ordinal/cumulative，而应继续留在这个分支内，改的是：

- conformal / reject policy 本身

更具体地说，后续更值得尝试的是：

- 更窄的 business-aware set construction
- 不把所有跨 side 的 set 一律判成 `retest`
- 或做更局部、更分层的 conformal 校准

### 本轮新增产物

- 配置：
  - `configs/distributional_conformal_reject_current_head.yaml`
- 脚本：
  - `scripts/run_distributional_conformal_reject_current_head.py`
- 摘要：
  - `reports/distributional_conformal_reject_current_head/distributional_conformal_summary.md`
- JSON：
  - `artifacts/distributional_conformal_reject_current_head/distributional_conformal_summary.json`
- 逐折结果：
  - `artifacts/distributional_conformal_reject_current_head/distributional_conformal_per_fold.csv`
- alpha 搜索：
  - `artifacts/distributional_conformal_reject_current_head/distributional_conformal_alpha_search.csv`

---

## 2026-04-01 第二轮执行记录：class-conditional quantile calibrated set

### 这一轮为什么要做

首轮 APS 式 conformal 的主要问题已经很明确：

- set 太大
- `retest` 太多
- 几乎所有跨 business side 的样本都会被拒掉

所以第二轮我没有换模型，也没有换特征，而是只改一件事：

- 从“APS 累积概率大集合”
- 改成“按 bin 分别做 class-conditional quantile threshold”

这样做的目的，是把问题拆开：

> 如果第二轮明显收缩了 set size 和 `retest_rate`，就说明上一轮的主要瓶颈在 set construction，而不是 5-bin 模型本身。

### 本轮方法

#### 目标模型

本轮仍然使用：

- `5-bin multinomial logistic`

选择它的原因很简单：

1. 它已经是这条 cleanroom 分支里最可审计、最轻量的 5-bin 基础模型。
2. 这轮想验证的是“不确定性校准和 reject policy”，不是重新比较模型家族。
3. 如果连在这个最简单模型上都看不清趋势，就不应该急着上更复杂模型。

#### 校准方法

每个 outer fold 内：

1. 用前段数据做 `proper-train`
2. 用末段数据做 `calibration`
3. 在 calibration 上，针对每个 bin 分别收集：
   - 真正落在该 bin 的样本
   - 它们的预测概率
4. 对每个 bin 的“真 bin 概率”做 quantile threshold 校准
5. 测试时，只要某个 bin 的概率超过自己的阈值，就把它纳入 prediction set
6. 如果最后没有任何 bin 被选中，则用 `argmax` 兜底，避免空集

这轮仍然搜索：

- `alpha in {0.05, 0.10, 0.15, 0.20}`

#### reject 规则

和首轮保持同一个 business rule：

- 若 selected bin set 只落入一个 business side，则直接输出该 side
- 若跨越多个 business side，则输出 `retest`

### 结果

pooled outer-test：

- `hard_bin_coverage = 0.6890`
- `mean_set_size = 2.9264`
- `singleton_rate = 0.2493`
- `ambiguous_business_set_rate = 0.7374`
- `retest_rate = 0.7374`
- `decision_coverage = 0.2626`
- `boundary_retest_rate = 0.7873`
- `non_boundary_decision_coverage = 0.2859`

次级业务诊断：

- `covered_macro_f1 = 0.1698`
- `covered_balanced_accuracy = 0.2743`
- `covered_warning_AP = 0.1627`
- `covered_unacceptable_AP = 0.0699`

### 和首轮 APS 的关系

相比首轮 APS：

1. 这轮确实明显收缩了 set
   - `mean_set_size: 4.1855 -> 2.9264`
2. 也明显减少了“几乎全拒”的问题
   - `retest_rate: 0.9225 -> 0.7374`
   - `decision_coverage: 0.0775 -> 0.2626`
3. 边界仍然比整体更容易被拒
   - overall `retest_rate = 0.7374`
   - boundary `retest_rate = 0.7873`

所以，这轮确实回答了一个关键问题：

> 上一轮的主要问题不只是 5-bin 模型本身，更多来自 APS 式 set construction 太保守。

### 新暴露出来的代价

但这轮也带来了新的明显代价：

- `hard_bin_coverage` 从 `0.8656` 掉到了 `0.6890`

也就是说，第二轮不是“全面更好”，而是把问题从：

- “几乎全拒，太保守”

换成了：

- “更敢做决定了，但 formal coverage 掉得比较多”

这很重要，因为它说明当前真正要继续优化的已经非常明确：

- 不是 5-bin 目标本身
- 也不是要不要继续回去改 ordinal/cumulative
- 而是 `conformal / reject policy` 的结构设计

### 当前阶段判断

到这里，这条新分支已经可以形成一个更稳的阶段判断：

1. 你要求的纠偏是对的，主任务应该是：
   - `5-bin soft distribution + calibration + reject`
2. 首轮 APS 版本过于保守，不适合作为 operational policy。
3. 第二轮 class-conditional 版本证明了：
   - 通过改 calibration / set construction，确实可以把 `retest` 压下来
4. 但它也同时证明：
   - 当前这版 class-conditional threshold 还没有达到一个理想折中
   - 因为 coverage 掉得太多

### 本轮新增产物

- 配置：
  - `configs/distributional_conformal_reject_class_conditional_current_head.yaml`
- 脚本：
  - `scripts/run_distributional_conformal_reject_class_conditional_current_head.py`
- 摘要：
  - `reports/distributional_conformal_reject_class_conditional_current_head/distributional_conformal_summary.md`
- JSON：
  - `artifacts/distributional_conformal_reject_class_conditional_current_head/distributional_conformal_summary.json`
- 逐折结果：
  - `artifacts/distributional_conformal_reject_class_conditional_current_head/distributional_conformal_per_fold.csv`
- alpha 搜索：
  - `artifacts/distributional_conformal_reject_class_conditional_current_head/distributional_conformal_alpha_search.csv`

---

## 2026-04-01 第三轮执行记录：ordered contiguous interval conformal set

### 这一轮的实验方法

前两轮已经把问题拆清楚了：

- APS：
  - coverage 较高
  - 但几乎全拒
- class-conditional threshold：
  - `retest` 明显下降
  - 但 coverage 掉得太多

所以第三轮我继续不换模型，只换 prediction set 的几何形状。

目标模型仍然保持：

- `5-bin multinomial logistic`

原因不变：

- 这条 cleanroom 当前要分辨的是 reject policy，而不是模型家族差异。

这一轮的新方法是：

- `ordered contiguous interval conformal set`

具体做法：

1. 对每个样本，先找到概率最大的 bin
2. 从这个 bin 出发，沿 T90 顺序向左或向右连续扩张
3. 每次优先吸收相邻概率更大的那一侧
4. 直到累计质量达到 calibration 得出的 quantile 阈值 `qhat`
5. 得到的 set 必须是一个连续区间，而不是离散散点

然后业务层仍然保持同一个规则：

- 若整个区间只落在一个 business side，则直接输出该 side
- 若区间跨多个 business side，则输出 `retest`

### 为什么这轮值得做

这轮的直觉非常直接：

- T90 的模糊性主要是“局部边界模糊”
- 而不是“这边一点、那边一点的离散乱跳”

所以如果 prediction set 的结构本身就被限制为连续区间，它会更贴近这类问题的真实几何。

### 结果

pooled outer-test：

- `hard_bin_coverage = 0.7952`
- `mean_set_size = 3.6903`
- `singleton_rate = 0.2075`
- `ambiguous_business_set_rate = 0.7925`
- `retest_rate = 0.7925`
- `decision_coverage = 0.2075`
- `boundary_retest_rate = 0.8343`
- `non_boundary_decision_coverage = 0.2270`

次级业务诊断：

- `covered_macro_f1 = 0.1674`
- `covered_balanced_accuracy = 0.2729`
- `covered_warning_AP = 0.1594`
- `covered_unacceptable_AP = 0.0597`

### 和前两轮的关系

这轮最重要的价值，是把三种 policy 的相对位置排清楚了。

#### 相比首轮 APS

- `retest_rate: 0.9225 -> 0.7925`
- `mean_set_size: 4.1855 -> 3.6903`
- `hard_bin_coverage: 0.8656 -> 0.7952`

含义是：

- 它确实把 APS 的“过度保守”往回收了一些
- 但不是无代价，coverage 也随之下降

#### 相比第二轮 class-conditional threshold

- `hard_bin_coverage: 0.6890 -> 0.7952`
- `retest_rate: 0.7374 -> 0.7925`
- `mean_set_size: 2.9264 -> 3.6903`

含义是：

- 它比 class-conditional 更保 coverage
- 但又重新抬高了 `retest` 和 set size

### 当前判断

到这一步，这条分支的三轮关系已经比较稳定了：

1. APS
   - 太保守
   - 几乎全拒
2. class-conditional threshold
   - 更敢做决定
   - 但 coverage 损失过大
3. ordered interval
   - 落在中间
   - 目前是三者中最像“折中版”的方案

更准确地说，ordered interval 还不能叫“已经足够好”，但它已经比前两轮更清楚地说明：

- 这条线的关键不是换模型
- 而是找到一个既符合有序结构、又不把 coverage 或 reject 推到极端的 set policy

### 本轮新增产物

- 配置：
  - `configs/distributional_conformal_reject_ordered_interval_current_head.yaml`
- 脚本：
  - `scripts/run_distributional_conformal_reject_ordered_interval_current_head.py`
- 摘要：
  - `reports/distributional_conformal_reject_ordered_interval_current_head/distributional_conformal_summary.md`
- JSON：
  - `artifacts/distributional_conformal_reject_ordered_interval_current_head/distributional_conformal_summary.json`
- 逐折结果：
  - `artifacts/distributional_conformal_reject_ordered_interval_current_head/distributional_conformal_per_fold.csv`
- alpha 搜索：
  - `artifacts/distributional_conformal_reject_ordered_interval_current_head/distributional_conformal_alpha_search.csv`

---

## 2026-04-01 第四轮执行记录：ordered interval + boundary-aware alpha selector

### 这一轮怎么想的

第三轮 `ordered interval` 已经说明：

- set 的几何形状比 APS 和独立 threshold 更合理
- 但逐折看，alpha 的选择仍然有些偏保守

尤其是前几折，会明显偏向：

- 高 coverage
- 但很高的 `retest`

所以第四轮我继续不换模型，也不换 set 形状，只改：

- `alpha` 在 calibration fold 里的选择逻辑

### 方法

目标模型仍然保持：

- `5-bin multinomial logistic`

prediction set 仍然保持：

- `ordered contiguous interval`

唯一变化是 selector：

- 不再主要按 nominal coverage 选 alpha
- 而是对每个 alpha 候选算一个 boundary-aware score

这份 score 同时：

正向奖励：

- `boundary_focus`
- `decision_coverage`
- `hard_bin_coverage`

负向惩罚：

- `retest_rate`
- `mean_set_size`

这样做的意图很直接：

- 不希望继续为了 coverage 牺牲太多 decision coverage
- 也不希望为了 coverage 把 set 做得太大
- 同时仍然要求 reject 更集中在边界，而不是均匀扩散到所有样本

### 结果

pooled outer-test：

- `hard_bin_coverage = 0.7727`
- `mean_set_size = 3.3436`
- `singleton_rate = 0.2194`
- `retest_rate = 0.7806`
- `boundary_retest_rate = 0.8246`
- `non_boundary_decision_coverage = 0.2400`
- `covered_macro_f1 = 0.1710`
- `covered_balanced_accuracy = 0.2826`

### 和第三轮 ordered interval 的比较

第三轮基线是：

- `hard_bin_coverage = 0.7952`
- `mean_set_size = 3.6903`
- `retest_rate = 0.7925`
- `boundary_retest_rate = 0.8343`
- `non_boundary_decision_coverage = 0.2270`
- `covered_macro_f1 = 0.1674`
- `covered_balanced_accuracy = 0.2729`

所以第四轮相对第三轮的变化可以概括成：

1. 更敢做决定了一点
   - `retest_rate: 0.7925 -> 0.7806`
   - `non_boundary_decision_coverage: 0.2270 -> 0.2400`
2. set 也更紧了一点
   - `mean_set_size: 3.6903 -> 3.3436`
3. 覆盖样本上的业务指标也小幅改善
   - `covered_macro_f1: 0.1674 -> 0.1710`
   - `covered_balanced_accuracy: 0.2729 -> 0.2826`
4. 代价是：
   - `hard_bin_coverage` 略降
   - `boundary_retest_rate` 略降

### 当前判断

这轮不算“质变”，但它是一个真实的增量改进。

更重要的是，它把当前分支的方向进一步坐实了：

1. 不需要急着换模型
2. `5-bin multinomial logistic` 仍然足够支撑 cleanroom 中的下一步判断
3. 主要杠杆确实还在：
   - conformal set policy
   - alpha selector
   - reject policy 的业务映射

如果只看当前已经跑过的几版，`ordered interval + boundary-aware alpha selector` 可以暂时视为：

- 这条新分支里当前最稳的一个中间版本

但它仍然没有完全解决核心矛盾：

- coverage
- decision coverage
- boundary focus

三者之间还在拉扯

### 本轮新增产物

- 配置：
  - `configs/distributional_conformal_reject_ordered_interval_boundary_selector_current_head.yaml`
- 脚本：
  - `scripts/run_distributional_conformal_reject_ordered_interval_boundary_selector_current_head.py`
- 摘要：
  - `reports/distributional_conformal_reject_ordered_interval_boundary_selector_current_head/distributional_conformal_summary.md`
- JSON：
  - `artifacts/distributional_conformal_reject_ordered_interval_boundary_selector_current_head/distributional_conformal_summary.json`
- 逐折结果：
  - `artifacts/distributional_conformal_reject_ordered_interval_boundary_selector_current_head/distributional_conformal_per_fold.csv`
- alpha 搜索：
  - `artifacts/distributional_conformal_reject_ordered_interval_boundary_selector_current_head/distributional_conformal_alpha_search.csv`

---

## 2026-04-01 第五轮执行记录：warning guardrail business mapping

### 这一轮的方法

前一轮已经得到一个比较稳的中间版本：

- `ordered interval`
- `boundary-aware alpha selector`

所以第五轮我仍然不换模型，也不换 conformal set，只改最后一层业务映射。

目标模型保持：

- `5-bin multinomial logistic`

prediction set 保持：

- `ordered contiguous interval`

alpha selector 保持：

- `boundary-aware selector`

唯一新增规则是：

- 如果 interval 只跨一个业务侧，照常输出该侧
- 如果 interval 只跨相邻业务侧：
  - `acceptable + warning`
  - 或 `warning + unacceptable`
  - 则直接映射成 `warning`
- 只有当 interval 同时跨到两端极值时，才保留 `retest`

### 为什么要这样做

这轮的核心判断是：

- 不是所有跨 side 的不确定性都必须复检
- 有一部分跨 side 的模糊，本质上更像“谨慎 warning”

如果把这部分样本全都打成 `retest`，会让 policy 过于保守，也不一定符合现场操作习惯。

### 结果

pooled outer-test：

- `hard_bin_coverage = 0.7727`
- `mean_set_size = 3.3436`
- `retest_rate = 0.6489`
- `decision_coverage = 0.3511`
- `boundary_retest_rate = 0.6948`
- `non_boundary_decision_coverage = 0.3726`
- `covered_macro_f1 = 0.1648`
- `covered_balanced_accuracy = 0.3126`
- `covered_warning_AP = 0.1839`
- `covered_unacceptable_AP = 0.0750`

### 和上一轮的比较

上一轮 `ordered interval + boundary-aware selector` 是：

- `retest_rate = 0.7806`
- `decision_coverage = 0.2194`
- `boundary_retest_rate = 0.8246`
- `non_boundary_decision_coverage = 0.2400`
- `covered_macro_f1 = 0.1710`
- `covered_balanced_accuracy = 0.2826`
- `covered_warning_AP = 0.1583`
- `covered_unacceptable_AP = 0.0672`

所以第五轮相对第四轮最重要的变化是：

1. `retest` 大幅下降
   - `0.7806 -> 0.6489`
2. decision coverage 明显提升
   - `0.2194 -> 0.3511`
3. 非边界样本覆盖率也明显提升
   - `0.2400 -> 0.3726`
4. `warning_AP` 和 `unacceptable_AP` 都有提升
   - `warning_AP: 0.1583 -> 0.1839`
   - `unacceptable_AP: 0.0672 -> 0.0750`
5. `covered_balanced_accuracy` 也提升
   - `0.2826 -> 0.3126`

代价是：

- `boundary_retest_rate` 明显下降
  - `0.8246 -> 0.6948`
- `covered_macro_f1` 略降
  - `0.1710 -> 0.1648`

### 当前判断

这轮是目前这条分支里最像“可操作 policy 雏形”的一版。

原因不是它在所有指标上都最好，而是：

1. 它第一次把 `retest` 压到了一个更像现场可用的区间
2. 而且没有靠换模型实现
3. 它把大量“相邻业务侧的模糊样本”吸收到 `warning`
4. 同时又保留了对真正跨两端极值样本的 `retest`

所以如果只看当前这条新分支的几轮实验，当前最值得冻结观察的是：

- `ordered interval`
- `boundary-aware alpha selector`
- `adjacent-overlap -> warning`
- `extreme-span -> retest`

### 这轮意味着什么

它进一步说明：

- 这条分支的主要杠杆已经不在模型本身
- 而在“set -> business action”的 policy 设计

也就是说，下一步最值得继续深挖的，不是换模型，而是继续围绕：

- 哪些模糊该吸收到 `warning`
- 哪些模糊必须保留为 `retest`

### 本轮新增产物

- 配置：
  - `configs/distributional_conformal_reject_ordered_interval_boundary_selector_warning_guardrail_current_head.yaml`
- 脚本：
  - `scripts/run_distributional_conformal_reject_ordered_interval_boundary_selector_warning_guardrail_current_head.py`
- 摘要：
  - `reports/distributional_conformal_reject_ordered_interval_boundary_selector_warning_guardrail_current_head/distributional_conformal_summary.md`
- JSON：
  - `artifacts/distributional_conformal_reject_ordered_interval_boundary_selector_warning_guardrail_current_head/distributional_conformal_summary.json`
- 逐折结果：
  - `artifacts/distributional_conformal_reject_ordered_interval_boundary_selector_warning_guardrail_current_head/distributional_conformal_per_fold.csv`
- alpha 搜索：
  - `artifacts/distributional_conformal_reject_ordered_interval_boundary_selector_warning_guardrail_current_head/distributional_conformal_alpha_search.csv`
