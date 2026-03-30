# T90 Cleanroom Validation 中文执行记录

## 1. 归属与边界

- 目标线：`projects/T90/dev/cleanroom_validation/`
- 工作性质：cleanroom 研究验证，不属于稳定交付线，也不属于 `v3/` 孵化线
- 本记录的目的：在阅读 `cleanroom_validation_experiment_plan_ordinal_cumulative.md` 后，明确我准备实际执行的实验路线，而不是提前给出方法有效性的结论

本轮只回答一个窄问题：

> 在当前 T90 问题下，是否应该先把任务从“硬三分类”改写为“面向阈值的 ordinal / cumulative 概率建模”？

因此，本轮不把 EWMA、PH 增强、多阶段融合、future-head、深度模型当成主变量。

## 2. 我对原计划的理解

原计划的核心判断是对的：当前难点更像是边界不稳定、标签抖动、业务判断更接近告警/可接受性，而不是模型容量不够。因此第一阶段应该先验证“任务定义是否该改”，而不是同时去优化特征表示。

我会遵守以下 cleanroom 原则：

- 可以复用可信的数据读取、样本对齐、当前样品标签构造工具
- 不继承既有的传感器 shortlist、历史 top feature、PH 结论、窗口结论、调好参数
- 不把先前 `dev/` 里的实验结论当作既定真理
- 所有监督筛选都必须严格限制在训练折内部

## 3. 我准备实际执行的主路线

我准备按“先定主线，再做小范围稳健性检查”的方式执行，而不是一开始把实验轴铺得很宽。

### 3.1 Phase 0：数据与标签审计

先做一个不带模型偏好的数据审计，确认本轮 cleanroom 的统一输入表。

固定做法：

- 标签来源：当前样品的当前实验室 T90
- 时间定义：`decision_time = 当前实验室样品时间`
- 特征来源：广义 DCS 原始/近原始变量池
- 特征窗口：只使用因果窗口，不用未来信息
- 不引入 EWMA
- 不引入 PH 作为主线输入

这一步要先确认三件事：

1. 样本级对齐表是否能稳定产出。
2. 当前样本数是否足以支持 `TimeSeriesSplit(n_splits=5)`，否则降到 4 折，但不低于 3 折。
3. 边界邻域样本量是否足够支持单独诊断。

### 3.2 Phase 1：锁定第一版 cleanroom 主实验配置

为了先验证任务定义，而不是陷入窗口/筛选联合搜索，我会先锁定一套主配置，只跑最必要的主对比。

第一版主配置准备如下：

- 窗口：`60 min` 单一因果窗口
- 特征族：`mean/std/min/max/last/range/delta`
- 无监督预清洗：去常数列、近常数冻结列、重复列、极高缺失列
- 监督筛选：训练折内 top-k 传感器筛选
- 主 top-k：`40`
- 验证：`TimeSeriesSplit(5)`，若样本太少则 `TimeSeriesSplit(4)`

这里我故意把主窗口先定在 `60 min`：

- 它足够保守、简单、可审计
- 它接近现有开发线常用的短窗量级，但没有直接照搬既有 `50 min` 结论
- 它能让本轮重点继续放在“任务定义”而不是“窗口调参”

### 3.3 Phase 2：先只比较两个核心 formulation

第一轮正式比较只做下面两个：

- Baseline A：硬三分类
  - `below_spec`
  - `in_spec`
  - `above_spec`
- Treatment C：独立阈值 cumulative probability 建模
  - `P(y < 8.0)`
  - `P(y < 8.2)`
  - `P(y < 8.7)`
  - `P(y < 8.9)`

我暂时不把手工两阶段分类作为第一轮必跑项，原因是：

- 它更像诊断性 baseline，不是 cleanroom 第一判断点
- 先把 A 对 C 结论做干净，结果更容易解释
- 如果 A 与 C 差距不清楚，再补 Baseline B 更合适

### 3.4 Phase 3：模型家族安排

为了隔离“任务定义”的作用，我准备采用一主一辅两个简单模型家族。

主模型家族：

- 硬三分类：multinomial logistic
- cumulative：每个阈值一个 binary logistic

辅助模型家族：

- hard three-class：balanced random forest
- cumulative：每个阈值一个 balanced random forest

执行顺序上，我会先用 logistic 完成主判断，再用 random forest 做“结果是否依赖单一模型家族”的补充检查。

### 3.5 Phase 4：cumulative 输出与业务决策塌缩规则

为了让比较可执行，我会先固定 interval 概率和业务决策塌缩规则，然后全程不改。

先由 cumulative 概率导出 5 个区间概率：

- `p1 = P(y < 8.0)`
- `p2 = P(8.0 <= y < 8.2)`
- `p3 = P(8.2 <= y < 8.7)`
- `p4 = P(8.7 <= y < 8.9)`
- `p5 = P(y >= 8.9)`

再塌缩为 3 个业务决策概率：

- `unacceptable = p1 + p5`
- `warning = p2 + p4`
- `acceptable = p3`

最终业务决策先采用最简单固定规则：

- 取三者中概率最大的类别作为输出决策

这样做的原因是：

- 规则简单、可审计
- 不需要在 cleanroom 第一轮就做额外阈值调参
- 便于与硬三分类直接对比 macro F1 与 balanced accuracy

如果后续发现 cumulative 明显更好，再单独研究“warning / unacceptable”的运营阈值策略。

### 3.6 Phase 5：必须出的诊断

除了常规指标，我会把下面几项视为必出项：

- 每个阈值事件的 Brier score
- 每个阈值事件的 calibration 误差
- cumulative 单调性违例率
- 是否进行了单调性后处理修正
- 边界邻域子集表现

边界邻域我准备先固定两段：

- `7.9 <= y < 8.3`
- `8.6 <= y < 8.8`

这两段基本覆盖当前计划强调的低侧和高侧边界不稳定区域。

### 3.7 Phase 6：小范围稳健性检查，而不是大网格搜索

如果主实验跑完后 A 与 C 已经出现明确差异，我只做小范围稳健性检查，不展开大规模搜索。

准备保留的稳健性轴：

- `topk_sensors in {20, 40, 80}`
- 窗口敏感性补充：`120 min` 或 `240 min` 只选一个做复核
- 第二模型家族：balanced random forest

我不会在第一轮同时扫：

- 多窗口组合
- EWMA
- PH 滞后
- 阶段识别
- 深模型
- future-head

否则就无法回答“到底是任务定义变好，还是别的改动带来的改善”。

## 4. 我准备实际采用的执行顺序

我准备按下面顺序落地，而不是并行把所有分支都做出来。

1. 建立 cleanroom 样本级特征表与标签审计。
2. 固定第一版实验配置：`60 min + simple window stats + topk=40 + TimeSeriesSplit`。
3. 跑 logistic 的 A vs C 主对比。
4. 输出 cumulative 概率、区间概率、业务塌缩结果、边界邻域诊断。
5. 做 monotonicity audit，默认先不修正，只记录违例率。
6. 若 C 相对 A 有持续改进，再补 random forest 复核。
7. 若结论仍然成立，再补 `topk=20/80` 的小范围稳健性检查。
8. 只有当前两步结论不清楚，才补可选的手工两阶段 baseline B。

## 5. 暂不执行的内容

本记录明确排除以下内容作为第一轮 cleanroom 主线：

- paper-faithful EWMA 作为主 treatment
- 多尺度加权融合
- PH 增强
- 阶段化/分段建模
- future-head 预测
- fuzzy 方法
- 深度模型
- 复杂告警后处理

这些都可以在 ordinal/cumulative 先被支持之后，再作为后续 cleanroom 分支。

## 6. 成功与失败的判定口径

我准备采用和原计划一致、但更偏执行口径的判定方式。

支持 ordinal / cumulative 的最低要求：

- 在主配置下，C 对 A 的 `macro_f1` 不低于且最好明显优于 A
- `balanced_accuracy` 不低于且最好优于 A
- 边界邻域表现优于 A，或者至少更稳定
- calibration 指标更合理
- 改善不是只出现在单个幸运折
- logistic 与 random forest 至少有一个简单模型家族表现稳定

如果这些条件不满足，我会把结论写成：

> 在简单 cleanroom 设置下，暂时没有足够证据支持用 ordinal / cumulative 替换当前硬三分类。

而不会把它误写成“EWMA 无效”或“别的方法无效”。

## 7. 我准备留下的文件与记录

本轮执行时，我准备让 cleanroom 目录至少留下以下产物：

- `cleanroom_validation_zh.md`
- `plans/ordinal_cumulative_validation.md`
- `configs/ordinal_cumulative_current_head.yaml`
- `scripts/run_ordinal_cumulative_current_head.py`
- `artifacts/ordinal_cumulative_results.csv`
- `artifacts/ordinal_cumulative_summary.json`
- `artifacts/ordinal_cumulative_feature_rows.csv`
- `reports/ordinal_cumulative_audit.md`

其中审计文件必须明确写清：

- 使用了哪些阈值
- cumulative 是独立建模还是联合建模
- 是否检查单调性
- 是否做单调性修正
- 监督筛选是否严格训练折内
- 哪些历史结论被明确忽略

## 8. 当前结论

当前我准备实际执行的不是“大而全 cleanroom”，而是一条收敛主线：

> 先用 `60 min` 因果简单统计特征，在严格时间序列验证下，比对“硬三分类”与“独立阈值 cumulative logistic”，把任务定义是否值得改写先判断清楚；只有在这一步得到正结果后，才进入 EWMA 或其他更复杂表示的 cleanroom 第二阶段。

这条路线的优点是：

- 与原计划一致
- 能直接回答当前最关键的问题
- 可审计、可复现、易于解释
- 不会因为同时改太多东西而把结论搅混

## 9. 执行记录（2026-03-30，第一轮）

### 9.0 补充前提确认

用户补充的关键前提是：

- 真正想识别的是 `8.2-8.7` 之间的合格产品
- 但 `8.2` 与 `8.7` 附近不能被当成完全刚性的硬边界
- 边界附近存在大量模糊样本
- 这正是本轮 cleanroom 实验的主旨

对照原始实验计划后，我的判断是：

- 这个前提在原计划的“问题动机”层面其实已经写到了
- 原计划已经强调：
  - 许多样本位于 specification boundary 附近
  - 实验室检测有噪声
  - 硬边界会带来不稳定标签
  - 业务目标更接近告警/可接受性判断，而不是纯硬分类

所以这条前提不是新增方向，而是需要在后续实现和评估里更严格贯彻。

### 9.1 本轮已完成内容

本轮已经实际完成以下工作：

- 建立 cleanroom 子目录结构：
  - `plans/`
  - `configs/`
  - `scripts/`
  - `reports/`
  - `artifacts/`
- 固化第一版实验计划：
  - `plans/ordinal_cumulative_validation.md`
- 固化第一版实验配置：
  - `configs/ordinal_cumulative_current_head.yaml`
- 实现并运行第一版脚本：
  - `scripts/run_ordinal_cumulative_current_head.py`
- 生成第一轮实验产物：
  - `artifacts/ordinal_cumulative_feature_rows.csv`
  - `artifacts/ordinal_cumulative_results.csv`
  - `artifacts/ordinal_cumulative_summary.json`
  - `reports/ordinal_cumulative_current_head_audit.md`

### 9.2 第一轮实际设置

本轮使用的固定设置如下：

- 任务：current-head cleanroom validation
- 窗口：`60 min` 因果窗口
- 特征：`last/mean/std/min/max/range/delta`
- 监督筛选：训练折内 `topk_sensors = 40`
- 验证：`TimeSeriesSplit(n_splits=5)`
- Baseline A：hard three-class logistic
- Treatment C：4 个独立阈值 logistic
- 累积阈值：`8.0 / 8.2 / 8.7 / 8.9`
- 不使用 EWMA / PH / stage-aware / future-head

### 9.3 数据审计结果

第一轮样本与特征审计结果：

- LIMS 分组后样本时间点：`9653`
- `t90` 非空样本：`2908`
- 成功对齐出 cleanroom 特征行：`2726`
- DCS 原始变量数：`64`
- 平均窗口行数：`60.97`
- 原始 simple-window 特征数：`448`
- 预清洗后特征数：`441`
- 被删除的常数特征：`7`

标签分布如下：

- hard three-class：
  - `in_spec = 1983`
  - `above_spec = 643`
  - `below_spec = 100`
- business decision：
  - `acceptable = 1983`
  - `warning = 517`
  - `unacceptable = 226`

边界邻域样本数：

- `7.9 <= y < 8.3`：`284`
- `8.6 <= y < 8.8`：`630`
- 合并边界邻域：`914`

### 9.4 第一轮结果摘要

整体业务决策指标：

- hard baseline
  - `macro_f1 = 0.2054`
  - `balanced_accuracy = 0.3281`
  - `unacceptable AP = 0.0957`
- cumulative treatment
  - `macro_f1 = 0.2781`
  - `balanced_accuracy = 0.3301`
  - `unacceptable AP = 0.0850`

边界邻域诊断：

- 低侧边界 `7.9 <= y < 8.3`
  - hard baseline `macro_f1 = 0.1567`
  - cumulative `macro_f1 = 0.2168`
- 高侧边界 `8.6 <= y < 8.8`
  - hard baseline `macro_f1 = 0.1667`
  - cumulative `macro_f1 = 0.2921`
- 合并边界邻域
  - hard baseline `macro_f1 = 0.1632`
  - cumulative `macro_f1 = 0.2727`

### 9.5 当前判断

第一轮结果说明：

- ordinal / cumulative 的任务定义方向有一定支持
- 支持点主要体现在：
  - 整体 `macro_f1` 提升
  - 边界邻域表现明显更好
  - `balanced_accuracy` 小幅领先
- 但当前实现还不能直接视为可替换方案

主要保留意见有两个：

- `unacceptable` 的 average precision 没有优于 hard baseline
- 独立阈值建模的单调性问题很重：
  - `any_violation_rate = 0.5224`
  - 说明超过一半样本存在至少一处 cumulative 概率顺序违例

另外还有一个更关键的实现偏差：

- 第一轮虽然使用了 `acceptable / warning / unacceptable` 的结构
- 但最终主评估仍然过度依赖“塌缩后的离散标签预测”
- 这会把本来应被视为“边界模糊”的问题，又重新拉回偏硬分类口径

因此，第一轮结果应被解释为：

- 它是一个有价值的探索性预跑
- 说明 threshold-oriented formulation 在边界邻域可能更对路
- 但它还不能算“正式完成的第一步”

因此当前最合理的中间结论是：

> cleanroom 第一轮已经提供了“任务定义值得继续验证”的证据，但还没有提供“当前这版独立阈值实现可以直接替换 hard baseline”的证据；更重要的是，第一步本身还需要按“边界模糊而非硬切”的主旨重新明确。

### 9.5A 按补充前提重看的结果

在补充“边界不能硬切”的前提后，我又把第一步的评估口径改成了更贴近主旨的版本，重点增加了：

- 核心合格区 `8.2-8.7` 的识别能力
- 边界区 `8.0-8.2` 与 `8.7-8.9` 的 warning 概率能力
- 边界区是否被高置信度地误判成非 warning

重看的关键结果如下：

- hard baseline
  - `core_qualified_AP = 0.7293`
  - `boundary_warning_AP = 0.1890`
  - 边界区非 warning 输出比例：`1.0000`
  - 边界区高置信非 warning 比例：`0.7057`
- cumulative treatment
  - `core_qualified_AP = 0.7236`
  - `boundary_warning_AP = 0.1784`
  - 边界区非 warning 输出比例：`0.6455`
  - 边界区高置信非 warning 比例：`0.2856`

这组结果说明：

- 如果只看“核心合格区 AP”与“warning AP”，当前 cumulative 版本还没有超过 hard baseline
- 但如果看“是否尊重边界模糊性”，cumulative 明显更贴合主旨
- hard baseline 在边界样本上几乎总是强行给出非 warning 判定
- cumulative 至少显著降低了边界区的高置信硬判倾向

因此，在你补充的前提下，第一步现在可以被认为：

- 仍然有意义
- 而且意义主要在于验证“任务定义是否更尊重边界模糊性”
- 不再只是验证“谁的硬标签分类分数更高”

### 9.6 下一步准备

基于用户这次补充，我的判断是：

- “第一步重新明确”这件事现在已经完成
- 并且基于重定义后的口径，第一步仍然有继续价值
- 因此现在可以进入下一步，但要沿着新的主旨进入，而不是回到偏硬分类的思路

重新明确后的正式第一步应当是：

1. 把 hard three-class 继续保留为诊断性 baseline，而不是最终 truth。
2. 把 `8.2-8.7` 明确解释为“核心合格区”。
3. 把 `8.0-8.2` 与 `8.7-8.9` 明确解释为“边界模糊/预警区”，而不是简单错误区。
4. 评价重点从“谁更像硬标签”调整为：
   - 谁更能把核心合格区与明显不合格区分开；
   - 谁在边界区给出的概率更稳定、更可解释；
   - 谁更少把边界模糊样本误判成高确信度结论。
5. hard label accuracy 仍可保留，但只能作为辅指标，不能作为主判据。

因此，新的后续顺序应改为：

1. 先重写第一步的评估定义与成功标准。
2. 在同一数据与特征框架下重跑第一步。
3. 只有在这个版本的第一步仍然支持 cumulative 时，才进入单调性修正、top-k 稳健性和后续扩展。

上述 1 和 2 已完成，因此下一步正式进入：

3. 单调性修正 / 约束版本验证。

下一步实验我准备优先做的内容是：

- 在相同 cleanroom 数据与筛选框架下，加入 cumulative 概率的单调性修正版本
- 比较“修正前 vs 修正后”是否能：
  - 保持边界模糊区的更温和输出
  - 降低 `any_violation_rate`
  - 不明显破坏核心合格区与明显不合格区的区分能力
- 暂不进入 EWMA

原先打算直接做的两件事先后顺序要后移：

1. 先在同一 cleanroom 框架下加入单调性约束/修正版本，验证高违例率是否是当前 treatment 的主要短板。
2. 在不改变任务定义的前提下做小范围稳健性复核：
   - `topk = 20 / 40 / 80`
   - 视情况补一个 `120 min` 窗口复核

在单调性修正版本完成前，暂不进入 EWMA、PH、stage-aware 或 future-head。

## 10. 执行记录（2026-03-30，第二轮：单调性修正）

### 10.1 本轮目标

在不改变数据、窗口、特征、筛选与模型家族的前提下，只加入最简单、可审计的 cumulative 单调性后处理：

- 阈值序列按 `8.0 -> 8.2 -> 8.7 -> 8.9`
- 对每个样本做逐阈值累积最大值修正
- 目的不是“调优模型”，而是验证：
  - 单调性违例是不是当前 cumulative 方案的主要短板
  - 修正后是否还能保住“边界不硬切”的优势

### 10.2 本轮产物

本轮新增产物位于：

- `configs/ordinal_cumulative_current_head_monotonic.yaml`
- `artifacts/monotonic/ordinal_cumulative_feature_rows.csv`
- `artifacts/monotonic/ordinal_cumulative_results.csv`
- `artifacts/monotonic/ordinal_cumulative_summary.json`
- `reports/monotonic/ordinal_cumulative_current_head_audit.md`

### 10.3 与第一轮相比的关键结果

整体业务决策指标：

- hard baseline
  - `macro_f1 = 0.2054`
  - `balanced_accuracy = 0.3281`
- monotonic cumulative
  - `macro_f1 = 0.2840`
  - `balanced_accuracy = 0.3287`

与第一轮未修正 cumulative 相比：

- `macro_f1`：`0.2781 -> 0.2840`
- `balanced_accuracy`：`0.3301 -> 0.3287`
- 变化不大，但总体没有被明显破坏

### 10.4 单调性结果

单调性修正后的结果非常明确：

- `any_violation_rate = 0.0000`
- 三个相邻阈值对的违例率全部为 `0.0000`

这说明：

- 第一轮 cumulative 的一个主要技术短板确实就是独立阈值输出不单调
- 这个短板可以通过简单、透明的后处理被完全消除

### 10.5 按用户主旨重看的结果

重看“核心合格区 + 边界模糊区”的指标：

- hard baseline
  - `core_qualified_AP = 0.7293`
  - `boundary_warning_AP = 0.1890`
  - 边界区高置信非 warning 比例：`0.7057`
- monotonic cumulative
  - `core_qualified_AP = 0.7259`
  - `boundary_warning_AP = 0.1803`
  - 边界区高置信非 warning 比例：`0.2877`

这里最重要的不是 AP 的微小差异，而是：

- monotonic cumulative 仍然显著减少了边界区被高置信度硬判的比例
- 并且在加入单调性修正后，这个优势没有丢失

边界区整体分类表现也进一步改善：

- 合并边界邻域 `macro_f1`
  - hard baseline：`0.1632`
  - monotonic cumulative：`0.2799`
- 合并边界邻域 `balanced_accuracy`
  - hard baseline：`0.1331`
  - monotonic cumulative：`0.2575`

### 10.6 当前结论更新

到第二轮为止，我的判断可以更新为：

- 第一轮“任务定义值得继续”的信号是成立的
- 第二轮说明这个方向不是偶然的：
  - 加入简单单调性修正后，边界友好性仍然保留
  - 单调性违例被彻底消除
  - 边界邻域表现继续优于 hard baseline

因此，现在可以认为：

> 在当前 cleanroom 设置下，threshold-oriented ordinal / cumulative 路线已经有了继续推进的充分理由；下一步不需要回退到重新定义任务，而应进入小范围稳健性验证。

### 10.7 下一步

下一步我准备继续做既定的小范围稳健性检查：

1. 固定单调性修正版本不变。
2. 只扫描 `topk_sensors = 20 / 40 / 80`。
3. 检查结论是否稳定：
   - 边界区高置信硬判比例是否仍低于 hard baseline
   - 边界邻域指标是否仍优于 hard baseline
   - 核心合格区 AP 是否保持在可接受范围

在这一步完成前，仍然不进入 EWMA。

## 11. 执行记录（2026-03-30，第三轮：top-k 稳健性复核）

### 11.1 本轮目的

在保留以下设置不变的前提下，只检查 supervised sensor screening 的 `topk` 是否会改变主结论：

- `60 min` 因果窗口
- simple window stats
- monotonic cumulative 后处理开启
- `TimeSeriesSplit(5)`

本轮比较：

- `topk = 20`
- `topk = 40`
- `topk = 80`

### 11.2 本轮产物

本轮新增：

- `configs/ordinal_cumulative_current_head_monotonic_topk20.yaml`
- `configs/ordinal_cumulative_current_head_monotonic_topk80.yaml`
- `reports/ordinal_cumulative_topk_robustness_summary.md`
- `artifacts/monotonic_topk20/`
- `reports/monotonic_topk20/`
- `artifacts/monotonic_topk80/`
- `reports/monotonic_topk80/`

### 11.3 结果摘要

整体业务决策 `macro_f1`：

- `topk20`
  - hard：`0.2391`
  - cumulative：`0.2865`
- `topk40`
  - hard：`0.2054`
  - cumulative：`0.2840`
- `topk80`
  - hard：`0.2193`
  - cumulative：`0.2795`

边界区高置信非 warning 比例：

- `topk20`
  - hard：`0.5864`
  - cumulative：`0.2976`
- `topk40`
  - hard：`0.7057`
  - cumulative：`0.2877`
- `topk80`
  - hard：`0.7407`
  - cumulative：`0.3709`

合并边界邻域 `macro_f1`：

- `topk20`
  - hard：`0.1811`
  - cumulative：`0.2606`
- `topk40`
  - hard：`0.1632`
  - cumulative：`0.2799`
- `topk80`
  - hard：`0.1769`
  - cumulative：`0.2742`

核心合格区 AP：

- `topk20`
  - hard：`0.7357`
  - cumulative：`0.7228`
- `topk40`
  - hard：`0.7293`
  - cumulative：`0.7259`
- `topk80`
  - hard：`0.7301`
  - cumulative：`0.7148`

### 11.4 当前结论

这轮稳健性复核说明：

- 主结论对 `topk` 变化是稳定的
- 无论 `topk=20/40/80`，cumulative 路线都持续表现出：
  - 更低的边界区高置信硬判比例
  - 更好的边界邻域表现
  - 更高的整体 `macro_f1`
- 但 hard baseline 仍然保留一个稳定特点：
  - 对核心合格区 `8.2-8.7` 的 AP 略高

因此到目前为止，cleanroom 的结论已经比较明确：

> 如果目标是“识别合格区间，同时承认边界模糊”，那么 monotonic ordinal / cumulative 路线比 hard baseline 更符合这个任务定义；它的主要优势不是把核心合格区 AP 做到最高，而是显著降低对边界模糊样本的高置信硬切。

### 11.5 下一步建议

我建议下一步做一个很克制的窗口敏感性补充，而不是立刻引入 EWMA：

1. 只补一个 `120 min` 窗口。
2. 保持 monotonic cumulative 与 hard baseline 的对比框架不变。
3. 检查当前结论是否仍然稳定。

如果 `120 min` 下结论仍然成立，再进入下一阶段：

- `ordinal / cumulative + simple window stats`
  对比
- `ordinal / cumulative + paper-faithful EWMA`
