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

## 12. 数据源核对（2026-03-31）

用户补充说明当前 T90 相关数据源不止一个：

- `data/merge_data.csv`
- `data/merge_data_otr.csv`
- `data/B4-AI-C53001A.PV.F_CV.xlsx`（具有明显工艺滞后的 PH）
- `data/卤化点位.xlsx`（点位含义说明）

我已经回查当前 cleanroom 脚本与配置，结论如下：

### 12.1 之前实验实际用了什么

之前已经完成的 cleanroom 实验实际使用的是：

- `merge_data.csv`
- `t90-溴丁橡胶.xlsx`

### 12.2 之前实验没有用到什么

之前实验没有用到：

- `merge_data_otr.csv`
- `B4-AI-C53001A.PV.F_CV.xlsx`
- `卤化点位.xlsx`

### 12.3 这是否属于“遗漏”

需要区分两种情况：

- `merge_data_otr.csv`
  - 对之前几轮 cleanroom 结果来说，属于未使用
  - 这部分此前被我误写成更像快速迭代或小样本开发数据，这是不准确的
  - 重新核查后，应将其视为与主 DCS 时间轴对齐的补充 DCS 导出
- `B4-AI-C53001A.PV.F_CV.xlsx`
  - 对当前前三轮实验来说，属于有意未纳入
  - 因为这几轮的目标是先验证“任务定义 + 简单窗口统计 + 单调 cumulative”
  - 还没有进入 PH/EWMA 相关阶段
- `卤化点位.xlsx`
  - 对建模输入本身没有直接影响
  - 但对后续解释 top sensors、写工艺解释和审计注释有帮助
  - 因此应记为“当前未使用，但后续解释阶段应保留参考”

### 12.4 当前判断

所以更准确地说：

- 之前实验确实忽略了这些补充数据源
- 对 `merge_data_otr.csv`，这已经不是“可以忽略的小补充”，而是应当纳入核查的 DCS 补充源
- 但重新核查后还要再补一层说明：
  - `merge_data_otr.csv` 中至少当前核查到的 6 个位点，与 `merge_data.csv` 里的对应位点在共同时间上数值完全一致
  - 它们更像同一测点的另一套命名导出，而不是新的独立信号
  - 因此后续纳入时，正确做法是“作为补充源读取，但在特征层做去重”，而不是把它们当成全新的额外传感器直接叠加
- 对 PH 文件，不属于违背当前实验边界的错误
- 对 `卤化点位.xlsx`，当前不是建模缺陷，但属于解释层面尚未利用的参考资料

## 13. 执行记录（2026-03-31，第四轮：120min 窗口敏感性）

### 13.1 本轮目的

按上一轮建议，在保持以下内容不变的前提下，只补一个更长的因果窗口做敏感性检查：

- monotonic ordinal / cumulative
- `topk=40`
- simple window stats
- 不引入 EWMA
- 不引入 PH

这一步的目标是检查：

- 之前结论是否只对 `60 min` 成立
- 如果切到 `120 min`，边界友好性和整体优势是否仍然保留

### 13.2 本轮产物

本轮新增：

- `configs/ordinal_cumulative_current_head_monotonic_120min.yaml`
- `artifacts/monotonic_120min/`
- `reports/monotonic_120min/`
- `reports/ordinal_cumulative_window_sensitivity_summary.md`

### 13.3 关键结果

整体业务决策指标：

- `60 min`
  - hard：`macro_f1 = 0.2054`
  - cumulative：`macro_f1 = 0.2840`
  - hard：`balanced_accuracy = 0.3281`
  - cumulative：`balanced_accuracy = 0.3287`
- `120 min`
  - hard：`macro_f1 = 0.2232`
  - cumulative：`macro_f1 = 0.2949`
  - hard：`balanced_accuracy = 0.3395`
  - cumulative：`balanced_accuracy = 0.3498`

核心合格区与 warning AP：

- `60 min`
  - hard `core_AP = 0.7293`
  - cumulative `core_AP = 0.7259`
  - hard `warning_AP = 0.1890`
  - cumulative `warning_AP = 0.1803`
- `120 min`
  - hard `core_AP = 0.7192`
  - cumulative `core_AP = 0.7151`
  - hard `warning_AP = 0.1890`
  - cumulative `warning_AP = 0.1924`

边界区高置信非 warning 比例：

- `60 min`
  - hard：`0.7057`
  - cumulative：`0.2877`
- `120 min`
  - hard：`0.7287`
  - cumulative：`0.3917`

边界邻域 `macro_f1`：

- `60 min`
  - hard：`0.1632`
  - cumulative：`0.2799`
- `120 min`
  - hard：`0.1778`
  - cumulative：`0.2796`

### 13.4 当前判断

这轮窗口敏感性说明：

- 当前 cleanroom 结论不是 `60 min` 的偶然产物
- 切换到 `120 min` 后，monotonic cumulative 仍然优于 hard baseline
- 而且在整体 `macro_f1` 与 `balanced_accuracy` 上，`120 min` 版本反而更强

更细一点看：

- `120 min` 下 cumulative 的 warning AP 有小幅提升
- 核心合格区 AP 仍然略低于 hard baseline
- 边界高置信硬判比例在 `120 min` 下比 `60 min` 略高
- 但 cumulative 仍然显著低于 hard baseline

所以当前最稳妥的结论是：

> `120 min` 没有推翻我们前面的判断，反而进一步支持了 monotonic ordinal / cumulative 作为“边界模糊友好”的主路线；并且在 simple window stats 框架内，`120 min` 可以作为下一阶段对比 EWMA 的更强 simple baseline 候选。

### 13.5 与 PH / 点位说明文件的关系

本轮依然没有引入：

- `merge_data_otr.csv`
- `B4-AI-C53001A.PV.F_CV.xlsx`
- `卤化点位.xlsx`

这是有意保持实验边界：

- 当前这一步仍然是在验证 simple-window baseline 的窗口敏感性
- 还没有进入 PH / EWMA cleanroom 分支
- `卤化点位.xlsx` 仍可在下一阶段用于解释 top sensors 与工艺含义

### 13.6 下一步建议

现在可以进入此前计划中的下一阶段：

1. 保持任务定义不变：
   - 核心合格区 `8.2-8.7`
   - 边界模糊区 `8.0-8.2` 与 `8.7-8.9`
2. 以当前更强的 simple baseline 候选为参照：
   - `ordinal / cumulative + simple 120min stats`
3. 进入下一条 cleanroom 分支：
   - `ordinal / cumulative + paper-faithful EWMA`
4. 在那一阶段再正式评估 PH 是否值得纳入，而不是在此处混入

## 14. 执行记录（2026-03-31，第五轮：paper-faithful EWMA）

### 14.1 本轮修正后的前提

在正式进入 EWMA 分支前，先纠正一件关键理解：

- `merge_data_otr.csv` 不是快速迭代数据
- 它是补充 DCS 导出
- 但当前核查结果显示，其中至少多列与主 DCS 中对应测点在共同时间上完全一致
- 因此本轮采用的处理方式是：
  - 把它作为 supplemental DCS source 纳入读取
  - 再在特征层通过去重处理，避免同一测点被重复计入

### 14.2 本轮实验目标

本轮正式进入此前计划中的下一阶段，对比：

- baseline：
  - `ordinal / cumulative + simple 120min window stats`
- treatment：
  - `ordinal / cumulative + paper-faithful recursive EWMA`

本轮仍然不引入：

- PH
- future-head
- deep model
- stage-aware

### 14.3 本轮设置

本轮使用：

- DCS 主数据：`merge_data.csv`
- DCS 补充数据：`merge_data_otr.csv`
- LIMS：`t90-溴丁橡胶.xlsx`
- 点位说明：`卤化点位.xlsx` 作为解释参考保留

EWMA 搜索空间：

- `tau in {0, 120, 240}`
- `W in {120, 240}`
- `lambda in {0.97, 0.985}`
- `topk_sensors = 40`
- monotonic correction：开启

本轮产物：

- `configs/ordinal_cumulative_paper_faithful_ewma_current_head.yaml`
- `scripts/run_ordinal_cumulative_paper_faithful_ewma_current_head.py`
- `artifacts/ewma_current_head/ordinal_cumulative_ewma_results.csv`
- `artifacts/ewma_current_head/ordinal_cumulative_ewma_summary.json`
- `artifacts/ewma_current_head/ordinal_cumulative_ewma_best_feature_rows.csv`
- `reports/ewma_current_head/ordinal_cumulative_ewma_audit.md`
- `reports/ordinal_cumulative_ewma_summary.md`

### 14.4 数据核查结果

本轮合并后的 DCS 审计结果：

- main rows：`935398`
- supplemental rows：`933596`
- main sensors：`64`
- supplemental sensors：`7`
- combined sensors：`71`
- 共同时间点：`933596`

特征预清洗结果：

- 初始特征数：`497`
- 预清洗后特征数：`448`
- 被识别为重复特征：`35`

这进一步说明：

- supplemental DCS 确实被纳入了
- 但其中一部分只是已有 DCS 位点的重复导出
- cleanroom 去重是必要的

### 14.5 EWMA 结果

这轮最重要的结论是：

- 在当前搜索空间内
- 没有任何一个 EWMA 组合能够同时在：
  - `macro_f1`
  - `balanced_accuracy`
  上打赢 `simple 120min` baseline

扫描出的最佳 EWMA 组合是：

- `tau = 0`
- `W = 120`
- `lambda = 0.97`

其结果为：

- baseline
  - `macro_f1 = 0.2958`
  - `balanced_accuracy = 0.3394`
  - `warning_AP = 0.1915`
  - 边界区高置信非 warning 比例：`0.3665`
- EWMA
  - `macro_f1 = 0.2745`
  - `balanced_accuracy = 0.3432`
  - `warning_AP = 0.1801`
  - 边界区高置信非 warning 比例：`0.2943`

### 14.6 当前解释

这轮结果说明：

- EWMA 并不是完全没有优点
- 它依然保留了一定的边界友好性：
  - 对边界样本的高置信硬判比例更低
- 并且在最佳组合下，`balanced_accuracy` 略高

但更关键的是：

- 它没有稳定地提高整体 `macro_f1`
- 也没有提高 warning AP
- 在当前 current-head cleanroom 设置下，它没有显示出“可重复、可明确归因”的整体增益

因此，到本轮为止更稳妥的结论是：

> paper-faithful EWMA 在当前 current-head、ordinal/cumulative、CPU-only 的 cleanroom 设置下，还没有被支持为优于 simple 120min stats 的下一步表示改进。

### 14.7 当前总判断

到目前为止，cleanroom 的总判断可以整理为：

1. 任务定义改成 ordinal / cumulative 是值得的。
2. monotonic correction 是必要且有效的。
3. `simple 120min stats` 是当前更强、更稳的 baseline。
4. paper-faithful EWMA 在当前阶段还没有打赢这个 baseline。
5. PH 仍然不应被提前混入当前结论。

### 14.7A 关于位点数量与“负增益位点”的补充说明

以当前最强 simple baseline 候选为例：

- 窗口：`120 min`
- monotonic cumulative
- `topk_sensors = 40`
- `TimeSeriesSplit(5)`

在这个设计下：

- 每一折实际选取的 DCS 位点数都是 `40`
- 5 折合并后的位点并集一共有 `54` 个

跨 5 折都被选中的稳定位点有 `23` 个，包括：

- `TI_C54002_PV_F_CV`
- `TI_C54003_PV_F_CV`
- `LIC_C53002A_PV_F_CV`
- `FIC_C51801_PV_F_CV`
- `TI_C50604_PV_F_CV`
- `FIC_C51003_PV_F_CV`
- `TI_C51401_S_PV_CV`
- `FIC_C51001_PV_F_CV`
- `FIC_C51605_PV_F_CV`
- `FI_C51005_S_PV_CV`
- `AT_C50002A_BIIR_PV_CV`
- `FIC_C53003A_PV_F_CV`
- `PIC_C53002A_PV_F_CV`
- `II_CM514_PV_CV`
- `TI_CM53001_PV_F_CV`
- `TI_C51101B_S_PV_CV`
- `TI_CM53201_PV_F_CV`
- `FIC_C51401_PV_F_CV`
- `LIC_C54002_PV_F_CV`
- `TI_CM54001_PV_F_CV`
- `II_CM530A_PV_CV`
- `II_CM513_PV_CV`
- `TI_C51007B_S_PV_CV`

关于“是否存在负增益位点”，当前记录需要非常谨慎地解读：

- 现有实验并没有做逐位点 ablation
- 所以不能仅凭当前记录就点名说某一个具体位点已经被证明是负增益

但从方法上讲，可以合理怀疑“当前选中集合里并非所有位点都是真正正增益”，原因有三：

1. 当前筛选是训练折内单变量排序，不是逐位点边际增益优化。
2. 40 个位点在 5 折中的并集达到 54 个，说明边缘位点稳定性有限。
3. 在 EWMA 分支里，同样数量级的选点并没有转化成整体稳定提升，说明“被选中”不等于“在当前表示下稳定正贡献”。

因此更准确的结论应是：

> 目前我们知道“每折选了 40 个位点”，也知道“有一批稳定位点”，但还没有通过逐位点消融实验把负增益位点明确识别出来。

### 14.7B 关于是否过早否定 EWMA

如果把当前结果写成“EWMA 没用”或者“EWMA 被否定了”，这确实过于片面。

当前更严格、也更诚实的说法应该是：

> 在目前这一次 cleanroom 设计下，即：
> `current-head + ordinal/cumulative + monotonic correction + simple 120min baseline 对照 + 当前 tau/W/lambda 搜索范围`
> paper-faithful EWMA 没有显示出稳定且可重复的整体增益。

这不是对 EWMA 方法本身的普遍否定，原因是：

- 这只是 current-head 场景
- 还没有引入 PH
- EWMA 只是 paper-faithful 直译版，还没有做任务特异化改进
- 也还没有做“位点子集是否更适合 EWMA”的定向实验

因此当前阶段更合理的判断是：

- 不能宣布“EWMA 无效”
- 但也不能声称“EWMA 已经被证明有效”
- 目前只能说：在当前这版 cleanroom 对照下，EWMA 还没有打赢更强的 simple 120min baseline

### 14.8 下一步建议

基于当前结果，我不建议继续在 current-head simple-vs-EWMA 这条线上盲目扩搜索空间。

更合理的下一步有两个方向：

- 方向 A：
  - 回到当前最强的 `ordinal / cumulative + simple 120min stats`
  - 开始做变量解释、点位解释、工艺可读性整理
- 方向 B：
  - 如果还要继续方法验证，则改成更明确的新问题
  - 例如 future-head 或显式 PH 分支
  - 而不是继续在 current-head simple-vs-EWMA 上加更多组合
# 2026-03-31 逐位点消融补充记录

## 本轮目的

在当前最强 cleanroom 基线下，先回答一个比 EWMA 更基础的问题：

- 当前 `120 min + monotonic ordinal/cumulative + topk=40` 的入选位点里，是否存在“被选中了，但在当前表示下其实是负增益”的 DCS 位点？

本轮不改 task formulation，不改 EWMA，只做 fold-local 已选位点集合内的逐位点消融。

## 实验口径

- 参考基线：`artifacts/monotonic_120min/ordinal_cumulative_summary.json`
- 实际复现脚本：`scripts/run_ordinal_cumulative_current_head_monotonic_120min_ablation_v2.py`
- 配置：`configs/ordinal_cumulative_current_head_monotonic_120min_ablation_v2.yaml`
- 数据边界：与 `ordinal_cumulative_current_head_monotonic_120min.yaml` 保持一致，只使用 `merge_data.csv + t90-溴丁橡胶.xlsx`
- 评估方式：每一折先固定该折 baseline 已选出的 `40` 个位点，再逐个删去其中 1 个，比较删前删后的
  - `macro_f1`
  - `balanced_accuracy`
  - `core_qualified_average_precision`
  - `boundary_warning_average_precision`
  - `boundary_high_confidence_non_warning_rate`

## 本轮结论

- 当前窗口设计下，每折实际入选 DCS 位点数仍然是 `40`，5 折并集为 `54`
- 通过逐位点消融，可以确认：当前入选集合里确实存在一批“去掉后反而更好”的疑似负增益位点
- 在当前判据下，疑似负增益位点共有 `16` 个
- 如果只看 `selection_count >= 4` 的相对稳定候选，则有 `11` 个

其中更值得优先关注的一批是：

- `TI_C50604_PV_F_CV`
  - `selection_count = 5`
  - 去掉后 `mean_delta_macro_f1 = +0.0124`
  - 去掉后 `mean_delta_balanced_accuracy = +0.0045`
- `FIC_C51802_PV_F_CV`
  - `selection_count = 4`
  - 去掉后 `mean_delta_macro_f1 = +0.0121`
  - 去掉后 `mean_delta_balanced_accuracy = +0.0124`
- `PI_C51101B_S_PV_CV`
  - `selection_count = 4`
  - 去掉后 `mean_delta_macro_f1 = +0.0083`
  - 去掉后 `mean_delta_balanced_accuracy = +0.0048`
- `II_CM532_PV_CV`
  - `selection_count = 4`
  - 去掉后 `mean_delta_macro_f1 = +0.0056`
  - 去掉后 `mean_delta_balanced_accuracy = +0.0051`
- `FIC_C51003_PV_F_CV`
  - `selection_count = 5`
  - 去掉后 `mean_delta_macro_f1 = +0.0039`
  - 去掉后 `mean_delta_balanced_accuracy = +0.0147`

还有一类更像“边界友好性拖后腿”的位点，也值得单独记住：

- `TI_C51101B_S_PV_CV`
  - `selection_count = 5`
  - 去掉后 `mean_delta_macro_f1 = +0.0117`
  - 虽然 `balanced_accuracy` 均值略降，但边界区高置信非 warning 比例平均下降 `0.0665`
- `FI_C51005_S_PV_CV`
  - `selection_count = 5`
  - 去掉后 `mean_delta_macro_f1 = +0.0008`
  - 边界区高置信非 warning 比例平均下降 `0.0121`

这说明此前的判断需要更新为更明确的版本：

> 在当前 `120 min + monotonic ordinal/cumulative + fold-local topk=40` 设定下，不能再只说“可能存在负增益位点”，而应该说“已经通过逐位点消融确认存在疑似负增益位点，而且其中有一批在 4-5 折里重复出现”。

## 对后续第二类实验的意义

这轮结果直接支持一个更谨慎的后续原则：

- 如果要继续评估 EWMA，不应该默认继承当前 simple baseline 的全部选点结果
- 更合理的第二类实验，应当至少考虑“先去掉这批疑似负增益位点，再看 EWMA 是否更合适”

## 产物位置

- 明细：`artifacts/monotonic_120min_ablation/sensor_ablation_per_fold.csv`
- 汇总：`artifacts/monotonic_120min_ablation/sensor_ablation_summary.csv`
- JSON：`artifacts/monotonic_120min_ablation/sensor_ablation_summary.json`
- 报告：`reports/monotonic_120min_ablation/sensor_ablation_summary.md`

# 2026-03-31 EWMA 专用选点与 EWMA 位点消融补充记录

## 本轮目的

基于你的补充，当前更合理的问题不是“沿用 simple baseline 的位点子集时 EWMA 是否有效”，而是：

- 如果专门为 EWMA 重做一套位点子集，EWMA 会不会更像样？
- 如果这套 EWMA 专用子集里仍有拖后腿位点，那么它们是谁？

因此本轮拆成两步：

1. 先做 `EWMA representation-specific screening`
2. 再在最佳 EWMA 组合上做 `EWMA-basis ablation`

## 第一步：EWMA 自己选点

### 实验口径

- 脚本：`scripts/run_ordinal_cumulative_paper_faithful_ewma_rep_specific_current_head.py`
- 配置：`configs/ordinal_cumulative_paper_faithful_ewma_rep_specific_current_head.yaml`
- baseline 仍是 `simple 120min monotonic ordinal/cumulative`
- 关键变化只有一条：
  - baseline 位点子集继续由 baseline simple 特征在训练折内筛选
  - EWMA 位点子集改为由 EWMA recursive 特征在训练折内独立筛选

### 结果

- 搜索空间仍为：
  - `tau in {0, 120, 240}`
  - `W in {120, 240}`
  - `lambda in {0.97, 0.985}`
- 在这次“EWMA 自己选点”的设定下，`beat_both_count = 0`
- 没有任何一个 EWMA 组合同时打赢 baseline 的
  - `macro_f1`
  - `balanced_accuracy`

最佳 EWMA 组合仍然是：

- `tau = 0`
- `W = 120`
- `lambda = 0.97`

但它的结果是：

- baseline `macro_f1 = 0.2958`
- EWMA `macro_f1 = 0.2536`
- baseline `balanced_accuracy = 0.3394`
- EWMA `balanced_accuracy = 0.3309`
- baseline `warning_AP = 0.1915`
- EWMA `warning_AP = 0.1877`
- baseline 边界区高置信非 warning 比例 = `0.3665`
- EWMA 边界区高置信非 warning 比例 = `0.4147`

这比上一轮“共享 simple 选点”的 EWMA 结果更弱，而不是更强。

### 一个关键机制性现象

EWMA 专用选点并没有自然避开 supplemental DCS 的短名/长名重复别名，反而在多折里把它们同时选了进去，例如：

- `FI_C51004_S` 与 `FI_C51004_S_PV_CV` 同时入选 `4` 折
- `LI_C53003A_S` 与 `LI_C53003A_S_PV_CV` 同时入选 `3` 折
- `PI_C51203A_S` 与 `PI_C51203A_S_PV_CV` 同时入选 `3` 折
- `TICA_C52601` 与 `TICA_C52601_PV_F_CV` 同时入选 `3` 折

这说明当前 cleanroom 虽然做了特征列层面的去重，但在“传感器身份”层面还没有把这些别名彻底视作同一个位点；对 EWMA 来说，这一点已经足以影响子集质量。

## 第二步：在 EWMA 专用子集上继续做位点消融

### 实验口径

- 脚本：`scripts/run_ordinal_cumulative_paper_faithful_ewma_rep_specific_ablation.py`
- 配置：`configs/ordinal_cumulative_paper_faithful_ewma_rep_specific_ablation.yaml`
- 以最佳 EWMA 组合 `tau=0, W=120, lambda=0.97` 为基线
- 每折固定该折 EWMA 自己选出的 `40` 个位点，再逐个删去其中 1 个

### 结果

- 每折 EWMA 入选位点数：`40`
- 5 折并集：`62`
- 疑似负增益位点：`13`
- 若只看 `selection_count >= 4` 的相对稳定候选，则有 `6` 个：
  - `TI_C53202_PV_F_CV`
  - `FIC_C51802_PV_F_CV`
  - `TI_CM511A_PV_F_CV`
  - `PI_C51301_S_PV_CV`
  - `TI_C51007B_S_PV_CV`
  - `TI_C53205_PV_F_CV`

其中更值得优先关注的是：

- `TI_C53202_PV_F_CV`
  - `selection_count = 4`
  - 去掉后 `mean_delta_macro_f1 = +0.0098`
  - 去掉后 `mean_delta_balanced_accuracy = +0.0056`
- `FIC_C51802_PV_F_CV`
  - `selection_count = 4`
  - 去掉后 `mean_delta_macro_f1 = +0.0062`
  - 去掉后 `mean_delta_balanced_accuracy = +0.0022`
- `PI_C51301_S_PV_CV`
  - `selection_count = 4`
  - 去掉后 `mean_delta_macro_f1 = +0.0018`
  - 去掉后 `mean_delta_balanced_accuracy = +0.0032`

## 本轮总判断

到这里可以把判断进一步收紧为：

1. “专门为 EWMA 重做一套位点子集”这个想法是有必要验证的，而且现在已经验证过了。
2. 在当前 cleanroom 设定下，这一步并没有把 EWMA 救回来。
3. 相反，EWMA 专用选点暴露出两个更具体的问题：
   - 传感器身份级别的别名重复没有被充分处理
   - EWMA 子集内部仍存在一批疑似负增益位点

因此当前最稳妥的结论不是“EWMA 无效”，而是：

> 在 `current-head + ordinal/cumulative + monotonic correction + current search grid` 下，即使让 EWMA 自己选点，它也还没有显示出优于 simple 120min baseline 的整体增益；下一步更该做的是“按传感器身份做严格去重后，再做一轮面向 EWMA 的定向验证”，而不是直接继续扩大 tau/W/lambda 网格。

## 本轮产物

- 主实验 JSON：`artifacts/ewma_rep_specific_current_head/ordinal_cumulative_ewma_rep_specific_summary.json`
- 主实验结果表：`artifacts/ewma_rep_specific_current_head/ordinal_cumulative_ewma_rep_specific_results.csv`
- 主实验摘要：`reports/ordinal_cumulative_ewma_rep_specific_summary.md`
- 消融 JSON：`artifacts/ewma_rep_specific_ablation/ewma_sensor_ablation_summary.json`
- 消融结果表：`artifacts/ewma_rep_specific_ablation/ewma_sensor_ablation_summary.csv`
- 消融报告：`reports/ewma_rep_specific_ablation/ewma_sensor_ablation_summary.md`

# 2026-03-31 传感器身份严格去重后的 EWMA cleanroom 验证

## 本轮目的

基于上一轮结果，当前更关键的前置问题已经从“EWMA 要不要自己选点”进一步收紧为：

- 在 `merge_data.csv` 与 `merge_data_otr.csv` 之间，长短点是否本质上是同一个传感器？
- 如果答案是“是”，那么在位点筛选前必须先按传感器身份去重，再谈 EWMA 是否有效

因此本轮先不扩搜索空间，不先做新的位点消融，而是先做一版更干净的 EWMA 验证：

- 先做传感器身份级别去重
- 再在去重后的统一点位池上重新跑 baseline vs EWMA

## 实验口径

- 脚本：`scripts/run_ordinal_cumulative_paper_faithful_ewma_identity_dedup_current_head.py`
- 配置：`configs/ordinal_cumulative_paper_faithful_ewma_identity_dedup_current_head.yaml`
- 规则：
  - 先在主表与 supplemental 表的共享时间上逐列对齐
  - 只有当两列满足
    - `same_ratio = 1.0`
    - `max_abs_diff = 0.0`
    时，才视作同一个传感器身份
  - 这种情况下保留 main 的长名列，删除 supplemental 的短名别名列

## 身份去重结果

本轮共识别并剔除了 `7` 对长短点别名：

- `PI_C51203A_S` -> `PI_C51203A_S_PV_CV`
- `TICA_C52601` -> `TICA_C52601_PV_F_CV`
- `FI_C53001` -> `FI_C53001_PV_F_CV`
- `PIC_C53002A` -> `PIC_C53002A_PV_F_CV`
- `LI_C53003A_S` -> `LI_C53003A_S_PV_CV`
- `TICA_C53005A` -> `TICA_C53005A_PV_F_CV`
- `FI_C51004_S` -> `FI_C51004_S_PV_CV`

这一点带来的直接变化非常重要：

- 合并后 DCS 位点数从 `71` 回到 `64`
- `preclean` 阶段的 `dropped_duplicate_count` 从上一轮的 `35` 下降到 `0`

也就是说，传感器身份严格去重确实解决了上一轮 EWMA 选点里“短名/长名同时入选”的机制性污染。

## 去重后重新验证 EWMA

搜索空间仍为：

- `tau in {0, 120, 240}`
- `W in {120, 240}`
- `lambda in {0.97, 0.985}`

结果是：

- `beat_both_count = 0`
- 仍然没有任何一个 EWMA 组合同时打赢 baseline 的
  - `macro_f1`
  - `balanced_accuracy`

去重后最佳 EWMA 组合变成了：

- `tau = 120`
- `W = 120`
- `lambda = 0.97`

其表现为：

- baseline `macro_f1 = 0.2949`
- EWMA `macro_f1 = 0.2464`
- baseline `balanced_accuracy = 0.3498`
- EWMA `balanced_accuracy = 0.3223`
- baseline `warning_AP = 0.1924`
- EWMA `warning_AP = 0.1799`
- baseline 边界区高置信非 warning 比例 = `0.3917`
- EWMA 边界区高置信非 warning 比例 = `0.2954`

这里需要非常谨慎地解读：

- 传感器身份去重这一步是有效的，而且现在应该视为后续 EWMA 验证的强制前提
- 但即使在这个更干净的 cleanroom 设定下，EWMA 仍然没有成为整体更强的表示
- 它保留了一部分边界友好性，但还没有转化成整体指标优势

## 本轮总判断

到这里为止，可以把关于 EWMA 的 cleanroom 结论更新为：

1. 之前“EWMA 专用选点”实验里暴露出的长短点重复问题，已经被本轮严格确认并修正。
2. “按传感器身份严格去重”应当成为 T90 项目后续 EWMA 验证的默认前提。
3. 但在这个修正后的 cleanroom 下，EWMA 依然没有打赢当前 `simple 120min` baseline。

因此当前更稳的表述是：

> EWMA 还不能被判定为当前 current-head 路线上的下一步增益方法；不过今后如果还要继续验证 EWMA，必须以“身份去重后的点位池”为起点，而不是再回到未去重的选点设定。

## 本轮产物

- 主结果 JSON：`artifacts/ewma_identity_dedup_current_head/ordinal_cumulative_ewma_identity_dedup_summary.json`
- 结果表：`artifacts/ewma_identity_dedup_current_head/ordinal_cumulative_ewma_identity_dedup_results.csv`
- 别名映射：`artifacts/ewma_identity_dedup_current_head/sensor_identity_alias_pairs.csv`
- 摘要：`reports/ordinal_cumulative_ewma_identity_dedup_summary.md`
- 审计：`reports/ewma_identity_dedup_current_head/ordinal_cumulative_ewma_identity_dedup_audit.md`

# 2026-03-31 第一类实验：baseline 在决策层借鉴边界友好性

## 本轮目的

前面已经确认：

- 当前 strongest baseline 是 `simple 120min + monotonic ordinal/cumulative`
- EWMA 仍保留了一部分“边界更谨慎”的倾向

因此第一类实验先不改表示，不改位点，不改训练目标，只验证一件事：

> baseline 是否能只在“决策层”借鉴这种边界友好性，从而在不明显伤整体性能的前提下变得更稳？

## 实验设计

- 脚本：`scripts/run_ordinal_cumulative_current_head_monotonic_120min_boundary_policy.py`
- 配置：`configs/ordinal_cumulative_current_head_monotonic_120min_boundary_policy.yaml`
- 基础表示保持不变：
  - `simple 120min causal stats`
  - `fold-local topk=40`
  - `monotonic cumulative logistic`

本轮只新增一类 warning-favoring 决策规则：

- 若默认预测为 `acceptable` 或 `unacceptable`
- 且 `warning_prob >= warning_floor`
- 且 `top_non_warning_prob - warning_prob <= margin`
- 则将输出改判为 `warning`

参数选择方式：

- 在每个 outer train fold 内，再做 `TimeSeriesSplit(3)` 的 inner search
- 候选策略共 `21` 个：
  - `default`
  - `warning_floor in {0.15, 0.20, 0.25, 0.30}`
  - `margin in {0.03, 0.05, 0.08, 0.10, 0.15}`
- 先筛掉相对 default 在 inner 上明显掉性能的策略
- 再在剩余策略里优先选边界区高置信非 warning 比例更低的策略

## 结果

这轮结果没有支持“只改决策层就能把 EWMA 的边界友好性借过来”。

在 outer test 聚合后：

- default `macro_f1 = 0.2392`
- policy `macro_f1 = 0.2372`
- default `balanced_accuracy = 0.3566`
- policy `balanced_accuracy = 0.3530`
- default 边界区高置信非 warning 比例 = `0.4836`
- policy 边界区高置信非 warning 比例 = `0.4836`

也就是说：

- 整体指标略降
- 目标边界指标完全没有改善

按折看也能看出同样的方向：

- `fold 4` 有小幅正增益
- 但 `fold 2 / 3 / 5` 都是轻微退步
- `fold 1` 最终甚至仍然选回了 `default`

## 关键解释

这里最重要的不是“这类方法一定没用”，而是：

> 我这次测试的 warning-favoring margin 规则，虽然在部分折里确实改动了预测，但它没有真正改掉那些“边界上的高置信非 warning”样本，所以没有把我们想借的那部分友好性转化出来。

换句话说，这一类实验目前得到的是一个负结果，但这个负结果是有信息量的：

- baseline 的问题不只是“决策阈值太硬”
- 至少在当前这组规则下，边界友好性并不能靠一个简单的 warning override 规则直接借来

## 当前结论更新

第一类实验到目前为止可以先写成：

1. “只改决策层、不改表示”的方向值得验证，而且已经完成第一轮验证。
2. 本轮 tested policy family 没有带来整体性能提升。
3. 更重要的是，它没有降低边界区高置信非 warning 比例，说明它没有命中真正的问题样本。

因此当前最准确的说法是：

> baseline 借鉴 EWMA 边界友好性的想法本身仍然成立，但第一版“warning_floor + margin”的决策层策略没有验证成功。

## 本轮产物

- 汇总 JSON：`artifacts/monotonic_120min_boundary_policy/boundary_policy_summary.json`
- 外层逐折结果：`artifacts/monotonic_120min_boundary_policy/boundary_policy_per_fold.csv`
- 每折入选策略：`artifacts/monotonic_120min_boundary_policy/boundary_policy_selected_per_fold.csv`
- 内层搜索明细：`artifacts/monotonic_120min_boundary_policy/boundary_policy_inner_search.csv`
- 摘要：`reports/monotonic_120min_boundary_policy/boundary_policy_summary.md`

# 2026-03-31 第二类实验：baseline 在表示层借鉴稀疏 EWMA
## 本轮目的

前一轮“只改决策层”的第一类实验没有成功借到 EWMA 的边界友好性，因此这一轮转向第二类：

> 不让 EWMA 整体替代当前 strongest baseline，而是只给一小部分位点补充 EWMA 表示，验证 `simple baseline + sparse EWMA` 是否能在尽量少打乱原表示的前提下带来增益。

这轮仍然沿用前面已经确认的 cleanroom 前提：

- DCS 源使用 `merge_data.csv + merge_data_otr.csv`
- 先做“按传感器身份严格去重”，剔除长短点精确别名
- baseline 仍为 `simple 120min causal stats + monotonic ordinal/cumulative + fold-local topk=40`
- EWMA 参考参数固定为上一轮 identity-dedup EWMA cleanroom 下的最佳组合：
  - `tau = 120`
  - `W = 120`
  - `lambda = 0.97`

## 实验设计

- 脚本：`scripts/run_ordinal_cumulative_simple_plus_sparse_ewma_identity_dedup.py`
- 配置：`configs/ordinal_cumulative_simple_plus_sparse_ewma_identity_dedup.yaml`

这一轮的关键约束是：

1. 先在 simple baseline 特征上做训练折内 `topk=40` 位点筛选。
2. EWMA 不再单独从全体位点里重选，而是只允许在这 `40` 个 baseline 已选位点里再做一次稀疏筛选。
3. 稀疏 EWMA 子集大小搜索：
   - `topm in {5, 10, 15, 20}`
4. hybrid treatment 的输入是：
   - baseline 的全部 simple 特征
   - 再加 sparse EWMA 子集位点的 `__ewma_recursive` 特征

这相当于验证：
> 是否存在一种“只给少数已经被 baseline 认可的位点补一层 EWMA 表示”的方式，能够保留 baseline 主体稳定性，同时吸收一部分 EWMA 的时序形状信息。

## 计分口径修正说明

本轮脚本初版在总评估时，把“尚未进入 outer-test 的时间前缀样本”也一起带入了汇总，导致预测列存在缺口。  
已修正为：

- 只对实际收到 outer-test 预测的样本计分
- `common_samples_raw` 记录 simple/EWMA 合并后的总交集样本数
- `scored_samples` 记录真正参与最终评分的样本数

本轮对应为：

- `common_samples_raw = 2726`
- `scored_samples = 2270`

这个口径比之前更严格，也更符合 cleanroom 交叉验证的真实可评分边界。

## 结果

结果表明，这条“稀疏混合表示”路线有一定信息量，但仍未形成可直接宣称优于 baseline 的结论。

整体结果：

- `beat_both_count = 0`
- 没有任何一个 `sparse_topm` 组合同时打赢 baseline 的
  - `macro_f1`
  - `balanced_accuracy`

按当前排序规则选出的最佳 sparse 配置为：

- `topm = 10`

对应结果：

- baseline `macro_f1 = 0.2610`
- hybrid `macro_f1 = 0.2574`
- baseline `balanced_accuracy = 0.3549`
- hybrid `balanced_accuracy = 0.3581`
- baseline `core_AP = 0.7151`
- hybrid `core_AP = 0.7176`
- baseline `warning_AP = 0.1924`
- hybrid `warning_AP = 0.1918`
- baseline 边界区高置信非 `warning` 比例 = `0.4945`
- hybrid 边界区高置信非 `warning` 比例 = `0.5097`
- hybrid `any_violation_rate = 0.0`

也就是说，这个最佳 `topm=10` hybrid 版本呈现出一种很典型的“有得有失”：

- 它在
  - `balanced_accuracy`
  - `core_AP`
  - `unacceptable AP`
  上有轻微正增益
- 但在
  - `macro_f1`
  - `warning_AP`
  - 边界区高置信非 `warning` 比例
  上反而变差

从“8.2-8.7 合格区存在模糊边界”的任务目标看，这并不是我们想要的增益形态。

## 不同 sparse_topm 的对比

四个已测组合结果如下：

- `topm=5`
  - `macro_f1 = 0.2556`
  - `balanced_accuracy = 0.3499`
  - `core_AP = 0.7209`
  - 边界区高置信非 `warning` 比例 = `0.5235`
- `topm=10`
  - `macro_f1 = 0.2574`
  - `balanced_accuracy = 0.3581`
  - `core_AP = 0.7176`
  - 边界区高置信非 `warning` 比例 = `0.5097`
- `topm=15`
  - `macro_f1 = 0.2511`
  - `balanced_accuracy = 0.3468`
  - `core_AP = 0.7145`
  - 边界区高置信非 `warning` 比例 = `0.5138`
- `topm=20`
  - `macro_f1 = 0.2573`
  - `balanced_accuracy = 0.3492`
  - `core_AP = 0.7142`
  - 边界区高置信非 `warning` 比例 = `0.5069`

可以看出：

1. 稀疏 EWMA 补充并非完全无效，因为它确实能把部分 global discrimination 指标推高。
2. 但它目前的增益方向更像“让模型更愿意做非 warning 判断”，而不是让模型对边界更谨慎。
3. 所以它虽然比“整套 EWMA 替代 simple baseline”更有希望，但还没有实现我们真正要的那种边界友好型提升。

## 稀疏 EWMA 子集里反复出现的位点

在最佳 `topm=10` 配置下，5 折里最常重复进入 sparse EWMA 子集的位点是：

- `FIC_C51801_PV_F_CV`（`5/5`）
- `TI_C54002_PV_F_CV`（`5/5`）
- `LIC_C53002A_PV_F_CV`（`4/5`）
- `TI_C50604_PV_F_CV`（`4/5`）
- `TI_C54003_PV_F_CV`（`4/5`）
- `TI_CM53201_PV_F_CV`（`4/5`）
- `FIC_C53051A_PV_F_CV`（`3/5`）
- `TI_CM53001_PV_F_CV`（`3/5`）
- `TI_CM54001_PV_F_CV`（`3/5`）

这一批点位当前更适合被解释为：

> 在当前 cleanroom 设置下，和 sparse EWMA 表示“相容性较高”的候选位点。

它们还不能直接被解释成最终确认的关键工艺点位，也还不能直接推出“这些位点一定适合继续堆更多 EWMA 特征”。

## 当前判断更新

到这一步为止，关于第二类实验可以写成：

1. “simple baseline + sparse EWMA” 这条路线是值得保留的。
2. 它比“整套 EWMA 替代 simple baseline”更接近一个可用方向，因为至少能带来局部指标正增益。
3. 但当前版本还没有把这种正增益转化成我们更关心的“边界模糊处理更好”。
4. 因此当前最准确的结论不是“第二类成功了”，而是：

> 稀疏 EWMA 混合表示已经显示出一定潜力，但目前更像是在加强整体区分能力，而不是改善边界模糊区的谨慎判断；它还不能作为当前 strongest baseline 的直接替代版本。

## 本轮产物

- 汇总 JSON：`artifacts/simple_plus_sparse_ewma_identity_dedup/simple_plus_sparse_ewma_summary.json`
- 候选结果表：`artifacts/simple_plus_sparse_ewma_identity_dedup/simple_plus_sparse_ewma_results.csv`
- 最优配置逐样本结果：`artifacts/simple_plus_sparse_ewma_identity_dedup/simple_plus_sparse_ewma_best_feature_rows.csv`
- 别名映射：`artifacts/simple_plus_sparse_ewma_identity_dedup/sensor_identity_alias_pairs.csv`
- 审计摘要：`reports/simple_plus_sparse_ewma_identity_dedup/simple_plus_sparse_ewma_audit.md`
- 人类可读摘要：`reports/simple_plus_sparse_ewma_identity_dedup/simple_plus_sparse_ewma_summary.md`

# 2026-03-31 第三类实验：baseline 在训练目标层借鉴边界友好性
## 本轮目的

第三类实验的目标是：

> 不改表示、不改选点、不改决策层规则，只改训练目标本身，看 baseline 是否能通过“边界样本加权训练”学到更符合模糊边界任务的判断方式。

这是对前两类实验的顺承：

- 第一类只改决策层，没有成功
- 第二类稀疏 EWMA 混合表示有一定潜力，但还没有改善边界模糊处理

因此第三类先走一条最克制、最可归因的路线：

- 仍然使用当前 strongest baseline 的 simple 表示
- 不引入新传感器、不引入新特征
- 不做额外 decision policy
- 只在每个 cumulative threshold 的 logistic 训练里施加样本权重

## 实验设计

- 脚本：`scripts/run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_identity_dedup.py`
- 配置：`configs/ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_identity_dedup.yaml`

本轮 cleanroom 设定：

- DCS 源：`merge_data.csv + merge_data_otr.csv`
- 先做严格传感器身份去重
- 表示层保持：
  - `simple 120min causal stats`
  - `monotonic ordinal/cumulative`
  - `fold-local topk=40`
- 训练层新增：
  - `boundary_weight`
  - `warning_weight`

也就是：

- 若样本属于 `boundary_any_flag`
  - 训练时乘上 `boundary_weight`
- 若样本业务标签是 `warning`
  - 再乘上 `warning_weight`

候选网格为：

- `boundary_weight in {1.25, 1.50, 2.00, 3.00}`
- `warning_weight in {1.00, 1.25, 1.50, 2.00}`
- 再加 `default = (1.0, 1.0)`

总计 `17` 个候选。

每个 outer train fold 内，再做 `TimeSeriesSplit(3)` 的 inner search。  
选择规则和第一类类似：

1. 先筛出相对 default 在 inner 上没有明显掉 `macro_f1` / `balanced_accuracy` 的候选
2. 再优先选
   - 边界区高置信非 `warning` 比例更低
   - `warning_AP` 更高
   - `macro_f1` / `balanced_accuracy` 更高

## 结果

这一轮是到目前为止，第一版就出现了比较明确正信号的一类实验。

在 pooled outer-test 评分上：

- default `macro_f1 = 0.2610`
- weighted `macro_f1 = 0.2675`
- default `balanced_accuracy = 0.3549`
- weighted `balanced_accuracy = 0.3559`
- default `core_AP = 0.7151`
- weighted `core_AP = 0.7234`
- default `warning_AP = 0.1924`
- weighted `warning_AP = 0.1965`
- default `unacceptable_AP = 0.0863`
- weighted `unacceptable_AP = 0.0823`
- default 边界区高置信非 `warning` 比例 = `0.4945`
- weighted 边界区高置信非 `warning` 比例 = `0.4765`

对应增量：

- `macro_f1_delta = +0.0066`
- `balanced_accuracy_delta = +0.0010`
- `core_AP_delta = +0.0083`
- `warning_AP_delta = +0.0040`
- `boundary_high_conf_non_warning_rate_delta = -0.0180`
- `unacceptable_AP_delta = -0.0040`

也就是说：

1. 这轮不是只有边界指标改善，而是 `macro_f1`、`balanced_accuracy`、`core_AP`、`warning_AP` 和边界谨慎性都一起向正确方向移动。
2. 但它不是零代价的，`unacceptable_AP` 出现了小幅下降。

所以这轮最准确的结论不是“第三类已经完全成功”，而是：

> 第三类的第一版已经拿到了目前最有价值的正向结果：通过只改训练目标，就能让 baseline 同时变得更强一些、也更边界友好一些；但它仍伴随 clearly-unacceptable 识别能力的小幅回落。

## 折内选择情况

5 个 outer fold 最终选中的权重方案分别是：

- `fold 1`: `boundary_3.00_warning_2.00`
- `fold 2`: `boundary_1.50_warning_1.25`
- `fold 3`: `boundary_1.50_warning_2.00`
- `fold 4`: `boundary_1.50_warning_1.50`
- `fold 5`: `boundary_3.00_warning_2.00`

这里有两个重要观察：

1. `5` 折里没有一折最终选回 `default`，说明这条方向并不是“完全没有可学信号”。
2. 比较稳定的候选集中在：
   - `boundary_weight = 1.5 or 3.0`
   - `warning_weight = 1.25 / 1.50 / 2.00`

其中：

- `boundary_3.00_warning_2.00` 在 `fold 1` 和 `fold 5` 被重复选中
- 中等强度的 `boundary_1.50 + warning_upweight` 在 `fold 2/3/4` 更常见

这说明当前更像存在“两类可行解”：

- 一类是更激进地把边界和 warning 同时抬权
- 一类是更温和地把边界抬到 `1.5x`，再给 warning 少量附加权重

## 需要特别注意的 trade-off

虽然整体结果是正向的，但这一轮也暴露出一个不能忽略的 trade-off：

- `fold 5` 的边界高置信非 `warning` 比例明显改善
- 但同一折的 `macro_f1` 和 `balanced_accuracy` 是回落的

换句话说，这种样本加权并不是在所有时间段都稳定占优。  
它更像是在“把模型拉向更谨慎的边界判断”时，有时会顺带牺牲一部分 clearly-unacceptable 方向的区分。

因此后续如果继续深挖第三类，重点不该只是继续把权重开大，而应该去验证：

1. 是否能保留当前边界收益，同时减少 `unacceptable_AP` 的损失
2. 是否存在更稳的中等权重带，而不是让个别折走到偏激进方案

## 当前判断更新

到这一步，三类实验的阶段性判断可以更新为：

1. 第一类“只改决策层”当前不成立。
2. 第二类“稀疏 EWMA 混合表示”有潜力，但还没有形成边界友好型提升。
3. 第三类“训练目标层边界加权”是目前最有希望的一条线。

更准确地说：

> 如果目标是识别 `8.2-8.7` 合格区，同时承认边界不是硬切，那么到目前为止，最值得继续推进的不是 decision policy，也不是直接换成 EWMA，而是继续沿着“simple baseline + monotonic cumulative + 边界感知训练目标”这条线细化。

## 本轮产物

- 汇总 JSON：`artifacts/monotonic_120min_boundary_weighted_identity_dedup/boundary_weighted_summary.json`
- 外层逐折结果：`artifacts/monotonic_120min_boundary_weighted_identity_dedup/boundary_weighted_per_fold.csv`
- 每折选中的权重候选：`artifacts/monotonic_120min_boundary_weighted_identity_dedup/boundary_weighted_selected_candidate_per_fold.csv`
- 内层搜索明细：`artifacts/monotonic_120min_boundary_weighted_identity_dedup/boundary_weighted_inner_search.csv`
- 逐样本评分结果：`artifacts/monotonic_120min_boundary_weighted_identity_dedup/boundary_weighted_results.csv`
- 别名映射：`artifacts/monotonic_120min_boundary_weighted_identity_dedup/sensor_identity_alias_pairs.csv`
- 摘要：`reports/monotonic_120min_boundary_weighted_identity_dedup/boundary_weighted_summary.md`

# 2026-03-31 第三类细化：只对内侧阈值做边界加权
## 为什么要做这一步

第三类第一版虽然是目前最有价值的正向结果，但我在结果拆解后发现了一个很具体的机制问题：

- `unacceptable_AP` 的损失主要不是发生在边界模糊样本上
- 而是发生在高侧 clearly-unacceptable 一侧
- 被拉走的样本大多在 `8.9 / 9.0 / 9.2`
- 也就是已经超过当前“边界模糊区”定义的外侧区域

更直接地说：

- full weighting 版本里，`default -> weighted` 导致的 `unacceptable -> non-unacceptable` 损失共有 `12` 条
- 这 `12` 条全部不在当前 `boundary_any_flag` 内

这说明：

> 我们上一轮把边界/warning 加权同时施加到所有 cumulative threshold，实际上把“外侧 clearly-unacceptable 的边界”也一起推软了。

因此这一轮的目标不是继续盲目调权重，而是验证一个更有针对性的假设：

> 模糊边界主要存在于 `8.2-8.7` 合格区附近，因此训练目标层的边界加权应该优先只作用于内侧阈值 `8.2 / 8.7`，而不是连外侧 `8.0 / 8.9` 一起加权。

## 实验设计

- 脚本：`scripts/run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup.py`
- 配置：`configs/ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup.yaml`

与第三类第一版相比，唯一关键变化是：

- 样本加权只应用在 cumulative thresholds:
  - `8.2`
  - `8.7`
- 外侧 thresholds:
  - `8.0`
  - `8.9`
  保持默认训练

其它条件保持不变：

- `simple 120min causal stats`
- `monotonic ordinal/cumulative`
- `fold-local topk=40`
- `merge_data.csv + merge_data_otr.csv`
- 严格传感器身份去重
- inner `TimeSeriesSplit(3)` 选权重

## 结果

这轮结果非常关键，因为它验证了上面的机制判断大体是对的。

pooled outer-test 结果：

- default `macro_f1 = 0.2610`
- inner-threshold weighted `macro_f1 = 0.2674`
- default `balanced_accuracy = 0.3549`
- inner-threshold weighted `balanced_accuracy = 0.3580`
- default `core_AP = 0.7151`
- inner-threshold weighted `core_AP = 0.7215`
- default `warning_AP = 0.1924`
- inner-threshold weighted `warning_AP = 0.1963`
- default `unacceptable_AP = 0.0863`
- inner-threshold weighted `unacceptable_AP = 0.0850`
- default 边界区高置信非 `warning` 比例 = `0.4945`
- inner-threshold weighted 边界区高置信非 `warning` 比例 = `0.4848`

对应增量：

- `macro_f1_delta = +0.0064`
- `balanced_accuracy_delta = +0.0031`
- `core_AP_delta = +0.0064`
- `warning_AP_delta = +0.0038`
- `boundary_high_conf_non_warning_rate_delta = -0.0097`
- `unacceptable_AP_delta = -0.0013`

## 和第三类第一版的直接对比

把“full weighting”与“只对内侧阈值加权”并排看：

- full weighting:
  - `macro_f1 = 0.2675`
  - `balanced_accuracy = 0.3559`
  - `core_AP = 0.7234`
  - `warning_AP = 0.1965`
  - `unacceptable_AP = 0.0823`
  - 边界区高置信非 `warning` 比例 = `0.4765`
- inner-threshold-only weighting:
  - `macro_f1 = 0.2674`
  - `balanced_accuracy = 0.3580`
  - `core_AP = 0.7215`
  - `warning_AP = 0.1963`
  - `unacceptable_AP = 0.0850`
  - 边界区高置信非 `warning` 比例 = `0.4848`

可以看到：

1. 两者的 `macro_f1` 几乎相同。
2. inner-threshold-only 版本的 `balanced_accuracy` 更高。
3. inner-threshold-only 版本保住了更多 `unacceptable_AP`。
4. 它的边界友好性略弱于 full weighting，但仍然优于 default baseline。

更关键的是 clearly-unacceptable 方向的损失：

- full weighting 里，`unacceptable -> non-unacceptable` 的丢失样本是 `12`
- inner-threshold-only 里，这个数字降到 `4`

这说明：

> 只对内侧阈值做加权，确实把第三类的收益更好地局限在“模糊边界区域”，而没有像 full weighting 那样把 clearly-unacceptable 外边界也一起推软。

## 折内行为

这一轮 5 折的权重选择是：

- `fold 1`: `boundary_1.25_warning_1.25`
- `fold 2`: `boundary_3.00_warning_1.25`
- `fold 3`: `default`
- `fold 4`: `boundary_3.00_warning_1.25`
- `fold 5`: `boundary_3.00_warning_2.00`

和 full weighting 相比，这里也有两个有价值的变化：

1. 有一折直接回到了 `default`
   - 说明在“只加权内侧阈值”的约束下，模型不会像上一轮那样几乎总被推向更激进的加权方案
2. `warning_weight` 明显趋向更克制
   - `1.25` 更常见
   - 而不是上一轮普遍偏向 `1.5~2.0`

这进一步支持“内侧阈值定向加权更稳”的判断。

## 当前判断更新

到目前为止，第三类路线可以进一步收敛为：

1. 训练目标层边界加权是当前最有希望的方向。
2. 但“对所有阈值一视同仁地加权”并不是最优实现。
3. 当前最好的折中版本是：

> 只对内侧阈值 `8.2 / 8.7` 做边界感知加权，外侧 `8.0 / 8.9` 保持默认训练。

它不是当前所有指标里绝对最优的单点，但它是目前“边界收益、整体指标、clearly-unacceptable 保护”三者平衡最好的一版。

## 本轮产物

- 汇总 JSON：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup/boundary_weighted_inner_thresholds_summary.json`
- 外层逐折结果：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup/boundary_weighted_inner_thresholds_per_fold.csv`
- 每折选中的权重候选：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup/boundary_weighted_inner_thresholds_selected_candidate_per_fold.csv`
- 内层搜索明细：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup/boundary_weighted_inner_thresholds_inner_search.csv`
- 逐样本评分结果：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup/boundary_weighted_inner_thresholds_results.csv`
- 别名映射：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup/sensor_identity_alias_pairs.csv`
- 摘要：`reports/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup/boundary_weighted_inner_thresholds_summary.md`
- 第三类两版对比：`reports/monotonic_120min_boundary_weighting_comparison.md`

# 2026-03-31 第三类再细化：低侧/高侧边界非对称加权
## 本轮动机

在“只对内侧阈值 `8.2 / 8.7` 做加权”的版本里，我又额外检查了低侧和高侧边界的行为，发现：

- 高侧边界样本变化更多
- 低侧边界上的收益没有高侧那么明显

因此这轮进一步验证一个更细的假设：

> 也许低侧边界和高侧边界并不需要同样的训练目标强度，应该允许 `8.2` 阈值和 `8.7` 阈值拥有不同的边界加权强度。

## 实验设计

- 脚本：`scripts/run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_asymmetric_inner_thresholds_identity_dedup.py`
- 配置：`configs/ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_asymmetric_inner_thresholds_identity_dedup.yaml`

这轮与上一版第三类的不同点是：

- `8.2` 阈值只对 `boundary_low_flag` 样本加权
- `8.7` 阈值只对 `boundary_high_flag` 样本加权
- 低侧和高侧边界权重分别搜索

搜索网格：

- `low_boundary_weight in {1.0, 1.25, 1.5, 2.0}`
- `high_boundary_weight in {1.0, 1.25, 1.5, 2.0, 3.0}`

总计 `20` 个候选（含 default）。

## 结果

这轮结果有信息量，但没有成为新的主线最优版本。

pooled outer-test 结果：

- default `macro_f1 = 0.2610`
- asymmetric `macro_f1 = 0.2712`
- default `balanced_accuracy = 0.3549`
- asymmetric `balanced_accuracy = 0.3561`
- default `core_AP = 0.7151`
- asymmetric `core_AP = 0.7170`
- default `warning_AP = 0.1924`
- asymmetric `warning_AP = 0.1943`
- default `unacceptable_AP = 0.0863`
- asymmetric `unacceptable_AP = 0.0848`
- default 边界区高置信非 `warning` 比例 = `0.4945`
- asymmetric 边界区高置信非 `warning` 比例 = `0.5097`

从数值上看，它的一个吸引点是：

- `macro_f1` 提升到了当前第三类实验中的最高值

但更关键的坏消息是：

- 边界区高置信非 `warning` 比例不降反升
- 甚至重新高于 default baseline

这意味着：

> 虽然非对称加权能把一部分全局分类指标再往上推，但它在当前 cleanroom 下丢掉了我们最想保住的“边界谨慎性”。

## 折内行为

5 折选中的方案分别是：

- `fold 1`: `low_2.00_high_1.50`
- `fold 2`: `low_1.00_high_3.00`
- `fold 3`: `default`
- `fold 4`: `low_2.00_high_1.00`
- `fold 5`: `low_2.00_high_2.00`

这说明它确实学出了“低侧/高侧不同权重”的模式，但这个模式目前并不稳定：

- 有的折偏高侧更强
- 有的折偏低侧更强
- 也有一折直接回到 `default`

换句话说，非对称性本身不是没有信号，但当前这版实现更像是在放大折间异质性，而不是提升整体稳定性。

## 与当前主线版本的对比

和上一轮“对称的内侧阈值加权”相比：

- asymmetric:
  - `macro_f1 = 0.2712`
  - `balanced_accuracy = 0.3561`
  - `unacceptable_AP = 0.0848`
  - 边界区高置信非 `warning` 比例 = `0.5097`
- inner-threshold-only symmetric:
  - `macro_f1 = 0.2674`
  - `balanced_accuracy = 0.3580`
  - `unacceptable_AP = 0.0850`
  - 边界区高置信非 `warning` 比例 = `0.4848`

因此当前更合理的判断不是“非对称更强”，而是：

1. 它在 `macro_f1` 上更激进。
2. 但它把边界友好性又丢回去了。
3. 所以它还不能替代当前的第三类主线版本。

## 当前判断更新

到目前为止，第三类主线可以进一步收紧成：

1. “对所有阈值统一加权”过于激进。
2. “只对内侧阈值 `8.2 / 8.7` 做对称加权”是当前最稳的折中版本。
3. “再把低侧和高侧拆开做非对称加权”目前没有带来更好的整体平衡。

也就是：

> 第三类真正值得继续深挖的主线，仍然是“内侧阈值定向加权”，但下一步更应该围绕‘如何保持边界收益并减少副作用’继续细化，而不是继续放大低/高侧自由度。

## 本轮产物

- 汇总 JSON：`artifacts/monotonic_120min_boundary_weighted_asymmetric_inner_thresholds_identity_dedup/boundary_weighted_asymmetric_summary.json`
- 外层逐折结果：`artifacts/monotonic_120min_boundary_weighted_asymmetric_inner_thresholds_identity_dedup/boundary_weighted_asymmetric_per_fold.csv`
- 每折选中的权重候选：`artifacts/monotonic_120min_boundary_weighted_asymmetric_inner_thresholds_identity_dedup/boundary_weighted_asymmetric_selected_candidate_per_fold.csv`
- 内层搜索明细：`artifacts/monotonic_120min_boundary_weighted_asymmetric_inner_thresholds_identity_dedup/boundary_weighted_asymmetric_inner_search.csv`
- 逐样本评分结果：`artifacts/monotonic_120min_boundary_weighted_asymmetric_inner_thresholds_identity_dedup/boundary_weighted_asymmetric_results.csv`
- 别名映射：`artifacts/monotonic_120min_boundary_weighted_asymmetric_inner_thresholds_identity_dedup/sensor_identity_alias_pairs.csv`
- 摘要：`reports/monotonic_120min_boundary_weighted_asymmetric_inner_thresholds_identity_dedup/boundary_weighted_asymmetric_summary.md`

# 2026-03-31 第三类继续深挖：距离阈值的软加权
## 本轮动机

在第三类已经得到的几版结果里：

- “内侧阈值定向加权”是当前主线最稳的版本
- 但它仍然存在少量副作用
- 而“低/高侧非对称加权”又把边界谨慎性做坏了

因此这轮继续沿主线，但不再增加自由度，而是做一个更平滑的版本：

> 只对 `8.2 / 8.7` 两个内侧阈值加权，但不再把边界区样本一刀切地抬到同一权重，而是按照“距离阈值越近、权重越高”的方式做线性软加权。

## 实验设计

- 脚本：`scripts/run_ordinal_cumulative_current_head_monotonic_120min_soft_threshold_weighted_inner_identity_dedup.py`
- 配置：`configs/ordinal_cumulative_current_head_monotonic_120min_soft_threshold_weighted_inner_identity_dedup.yaml`

这一轮的 soft weighting 规则是：

- 只作用于 inner thresholds:
  - `8.2`
  - `8.7`
- 对任一样本，若它到目标阈值的距离为 `d`
- 若 `d <= radius`
  - 权重 = `1 + (peak_weight - 1) * (1 - d / radius)`
- 若 `d > radius`
  - 权重回到 `1.0`

也就是说：

- 阈值附近的样本权重最高
- 离阈值越远，额外权重线性衰减
- 超过半径后完全不再额外加权

搜索网格：

- `peak_weight in {1.25, 1.50, 2.00, 3.00}`
- `radius in {0.05, 0.10, 0.15, 0.20, 0.30}`
- 再加 `default`

总计 `21` 个候选。

## 结果

这轮结果说明：soft weighting 确实把第三类推进到了一个更“温和”的版本，但它也明显削弱了边界友好性提升。

pooled outer-test 结果：

- default `macro_f1 = 0.2610`
- soft `macro_f1 = 0.2687`
- default `balanced_accuracy = 0.3549`
- soft `balanced_accuracy = 0.3579`
- default `core_AP = 0.7151`
- soft `core_AP = 0.7163`
- default `warning_AP = 0.1924`
- soft `warning_AP = 0.1922`
- default `unacceptable_AP = 0.0863`
- soft `unacceptable_AP = 0.0861`
- default 边界区高置信非 `warning` 比例 = `0.4945`
- soft 边界区高置信非 `warning` 比例 = `0.4931`

对应增量：

- `macro_f1_delta = +0.0078`
- `balanced_accuracy_delta = +0.0031`
- `core_AP_delta = +0.0012`
- `warning_AP_delta = -0.0002`
- `unacceptable_AP_delta = -0.0002`
- `boundary_high_conf_non_warning_rate_delta = -0.0014`

## 关键解读

这轮 soft weighting 的最大优点是：

- clearly-unacceptable 侧的副作用已经非常小
- `unacceptable -> non-unacceptable` 的丢失样本只剩 `2` 条

从这个角度看，它是目前第三类里“最保守、最少副作用”的版本。

但它也有很明显的代价：

- 边界区高置信非 `warning` 比例虽然略降
- 但下降得非常有限
- `warning_AP` 也没有继续提升

也就是说：

> soft weighting 并没有把第三类主线继续推向“更边界友好”，它更像是在用更小的副作用，换取一个更温和的整体提升。

## 选中的权重带

5 折最终选中的候选是：

- `fold 1`: `peak_1.25_radius_0.15`
- `fold 2`: `peak_2.00_radius_0.10`
- `fold 3`: `default`
- `fold 4`: `peak_2.00_radius_0.10`
- `fold 5`: `peak_2.00_radius_0.15`

这说明当前更稳的 soft weighting 带大致集中在：

- `peak_weight = 2.0`
- `radius = 0.10 ~ 0.15`

也就是：

- 不需要非常大的峰值权重
- 也不适合把影响半径拉得过宽

## 与当前主线版本的比较

和上一轮“内侧阈值对称硬加权”相比：

- inner-threshold-only weighting:
  - `macro_f1 = 0.2674`
  - `balanced_accuracy = 0.3580`
  - `warning_AP = 0.1963`
  - `unacceptable_AP = 0.0850`
  - 边界区高置信非 `warning` 比例 = `0.4848`
- soft weighting:
  - `macro_f1 = 0.2687`
  - `balanced_accuracy = 0.3579`
  - `warning_AP = 0.1922`
  - `unacceptable_AP = 0.0861`
  - 边界区高置信非 `warning` 比例 = `0.4931`

因此可以把两者的分工理解成：

1. 如果更重视“边界友好性”，当前仍应优先保留内侧阈值的对称硬加权版本。
2. 如果更重视“尽量少副作用，同时保留一定总体收益”，soft weighting 更像一个安全版备选。

## 当前判断更新

到这一步，第三类的主线已经比较清楚了：

1. 主线仍然成立。
2. 当前最优平衡点仍是：
   - “只对 `8.2 / 8.7` 做内侧阈值定向加权”
3. soft weighting 没有把这条主线继续推强，但它给出了一个很重要的补充结论：

> 第三类的收益确实可以被做得更温和、更少副作用；只是当我们把它做得太温和时，边界友好性的提升也会一起变弱。

这对后续继续细化第三类很重要，因为它说明下一步更值得尝试的，不是继续增加结构自由度，而是：

- 在“硬加权的边界收益”与“软加权的副作用控制”之间寻找更好的折中

## 本轮产物

- 汇总 JSON：`artifacts/monotonic_120min_soft_threshold_weighted_inner_identity_dedup/soft_threshold_weighted_summary.json`
- 外层逐折结果：`artifacts/monotonic_120min_soft_threshold_weighted_inner_identity_dedup/soft_threshold_weighted_per_fold.csv`
- 每折选中的候选：`artifacts/monotonic_120min_soft_threshold_weighted_inner_identity_dedup/soft_threshold_weighted_selected_candidate_per_fold.csv`
- 内层搜索明细：`artifacts/monotonic_120min_soft_threshold_weighted_inner_identity_dedup/soft_threshold_weighted_inner_search.csv`
- 逐样本评分结果：`artifacts/monotonic_120min_soft_threshold_weighted_inner_identity_dedup/soft_threshold_weighted_results.csv`
- 别名映射：`artifacts/monotonic_120min_soft_threshold_weighted_inner_identity_dedup/sensor_identity_alias_pairs.csv`
- 摘要：`reports/monotonic_120min_soft_threshold_weighted_inner_identity_dedup/soft_threshold_weighted_summary.md`

# 2026-03-31 第三类继续细化：硬软混合的内侧阈值加权
## 本轮动机

前一轮 soft weighting 给出了一个很有价值的观察：

- 它把副作用压得更小
- 但也把边界友好性的提升做弱了

因此这一轮尝试一个很自然的折中思路：

> 在 `8.2 / 8.7` 两个内侧阈值上，先给边界样本一个较温和的硬权重下限，再对距离阈值更近的样本叠加一层局部 soft peak，看看能不能同时保留 hard 版的边界推动力和 soft 版的低副作用。

## 实验设计

- 脚本：`scripts/run_ordinal_cumulative_current_head_monotonic_120min_hard_soft_threshold_weighted_inner_identity_dedup.py`
- 配置：`configs/ordinal_cumulative_current_head_monotonic_120min_hard_soft_threshold_weighted_inner_identity_dedup.yaml`

规则是：

- 仍然只对 inner thresholds:
  - `8.2`
  - `8.7`
  生效
- 对 boundary 样本先给一个统一的 `hard_weight`
- 对离阈值距离在 `radius` 内的样本，再给一个局部 soft peak
- 最终样本权重取：
  - 硬下限
  - 与 soft 局部峰值
  中较大的那个

搜索网格：

- `hard_weight in {1.10, 1.25, 1.50}`
- `peak_weight in {1.50, 2.00, 2.50}`
- `radius in {0.05, 0.10, 0.15}`

## 结果

这轮结果说明，“硬软混合”并没有自动形成我们想要的甜点区。

pooled outer-test 结果：

- default `macro_f1 = 0.2610`
- mixed `macro_f1 = 0.2680`
- default `balanced_accuracy = 0.3549`
- mixed `balanced_accuracy = 0.3561`
- default `core_AP = 0.7151`
- mixed `core_AP = 0.7176`
- default `warning_AP = 0.1924`
- mixed `warning_AP = 0.1929`
- default `unacceptable_AP = 0.0863`
- mixed `unacceptable_AP = 0.0861`
- default 边界区高置信非 `warning` 比例 = `0.4945`
- mixed 边界区高置信非 `warning` 比例 = `0.4972`

对应增量：

- `macro_f1_delta = +0.0071`
- `balanced_accuracy_delta = +0.0013`
- `core_AP_delta = +0.0025`
- `warning_AP_delta = +0.0005`
- `unacceptable_AP_delta = -0.0002`
- `boundary_high_conf_non_warning_rate_delta = +0.0028`

最关键的是最后一项：

- 边界区高置信非 `warning` 比例没有下降
- 反而比 baseline 还略高

这说明：

> mixed 版虽然保住了整体指标和低副作用，但没有保住第三类主线里最重要的“边界谨慎性收益”。

## 与当前几版第三类的比较

把三种主线相关版本并排看：

- 对称硬加权（当前主线）：
  - `macro_f1 = 0.2674`
  - `balanced_accuracy = 0.3580`
  - `warning_AP = 0.1963`
  - `unacceptable_AP = 0.0850`
  - 边界区高置信非 `warning` 比例 = `0.4848`
- soft 加权：
  - `macro_f1 = 0.2687`
  - `balanced_accuracy = 0.3579`
  - `warning_AP = 0.1922`
  - `unacceptable_AP = 0.0861`
  - 边界区高置信非 `warning` 比例 = `0.4931`
- hard-soft 混合：
  - `macro_f1 = 0.2680`
  - `balanced_accuracy = 0.3561`
  - `warning_AP = 0.1929`
  - `unacceptable_AP = 0.0861`
  - 边界区高置信非 `warning` 比例 = `0.4972`

从这个对比看：

1. mixed 比 soft 没有明显更强。
2. mixed 明显不如当前主线 hard 版的边界友好性。
3. 它并没有形成真正优于 hard/soft 两边的折中点。

## 当前判断再收紧

到这里，第三类主线可以进一步收紧成：

1. 当前最优主线仍然是：
   - “只对 `8.2 / 8.7` 做内侧阈值对称硬加权”
2. soft 版是低副作用备选。
3. hard-soft mixed 版没有成为新的更优折中方案。

也就是说：

> 目前第三类真正被 cleanroom 支持的，不是“越复杂的加权结构越好”，而是“控制得当的内侧阈值对称硬加权”。

## 本轮产物

- 汇总 JSON：`artifacts/monotonic_120min_hard_soft_threshold_weighted_inner_identity_dedup/hard_soft_threshold_weighted_summary.json`
- 外层逐折结果：`artifacts/monotonic_120min_hard_soft_threshold_weighted_inner_identity_dedup/hard_soft_threshold_weighted_per_fold.csv`
- 每折选中的候选：`artifacts/monotonic_120min_hard_soft_threshold_weighted_inner_identity_dedup/hard_soft_threshold_weighted_selected_candidate_per_fold.csv`
- 内层搜索明细：`artifacts/monotonic_120min_hard_soft_threshold_weighted_inner_identity_dedup/hard_soft_threshold_weighted_inner_search.csv`
- 逐样本评分结果：`artifacts/monotonic_120min_hard_soft_threshold_weighted_inner_identity_dedup/hard_soft_threshold_weighted_results.csv`
- 别名映射：`artifacts/monotonic_120min_hard_soft_threshold_weighted_inner_identity_dedup/sensor_identity_alias_pairs.csv`
- 摘要：`reports/monotonic_120min_hard_soft_threshold_weighted_inner_identity_dedup/hard_soft_threshold_weighted_summary.md`

# 2026-03-31 第三类继续精修：内侧阈值对称硬加权的窄带搜索

## 本轮动机

到上一轮为止，第三类已经比较清楚：

- 当前主线是“只对 `8.2 / 8.7` 做内侧阈值对称硬加权”
- `soft` 版和 `hard-soft mixed` 版都没有成为更优主线
- 但当前主线的入选权重已经开始表现出明显集中：
  - `boundary_weight` 多次落在 `3.0`
  - `warning_weight` 多次落在 `1.25`

因此这一步不再增加结构复杂度，而是只做一件事：

> 保持第三类当前最强结构不变，只把权重搜索带收窄到已经反复显示优势的区域，看能不能在不牺牲边界友好性的前提下，再把主线往前推一步。

## 实验设计

- 脚本：`scripts/run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup.py`
- 新配置：`configs/ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_refined_band.yaml`

与当前第三类主线保持一致的部分：

- `simple 120min causal stats`
- `monotonic ordinal/cumulative`
- 只对 inner thresholds:
  - `8.2`
  - `8.7`
  加权
- `fold-local topk=40`
- `merge_data.csv + merge_data_otr.csv`
- strict sensor-identity de-dup
- outer `TimeSeriesSplit(5)` + inner `TimeSeriesSplit(3)`

本轮唯一变化是把搜索带收窄为：

- `boundary_weight in {2.50, 2.75, 3.00, 3.25, 3.50}`
- `warning_weight in {1.00, 1.10, 1.25, 1.40, 1.50}`
- 再加 `default`

总计 `26` 个候选。

## 结果

这轮“窄带精修”确实把整体分类指标又往上推了一点，但它没有取代当前主线版本。

pooled outer-test 结果：

- default `macro_f1 = 0.2610`
- refined-band `macro_f1 = 0.2681`
- default `balanced_accuracy = 0.3549`
- refined-band `balanced_accuracy = 0.3593`
- default `core_AP = 0.7151`
- refined-band `core_AP = 0.7213`
- default `warning_AP = 0.1924`
- refined-band `warning_AP = 0.1954`
- default `unacceptable_AP = 0.0863`
- refined-band `unacceptable_AP = 0.0847`
- default 边界区高置信非 `warning` 比例 = `0.4945`
- refined-band 边界区高置信非 `warning` 比例 = `0.4917`

对应增量：

- `macro_f1_delta = +0.0072`
- `balanced_accuracy_delta = +0.0045`
- `core_AP_delta = +0.0061`
- `warning_AP_delta = +0.0030`
- `unacceptable_AP_delta = -0.0016`
- `boundary_high_conf_non_warning_rate_delta = -0.0028`

## 与当前主线版本的直接比较

和上一轮“内侧阈值对称硬加权主线”相比：

- 当前主线 hard 版：
  - `macro_f1 = 0.2674`
  - `balanced_accuracy = 0.3580`
  - `core_AP = 0.7215`
  - `warning_AP = 0.1963`
  - `unacceptable_AP = 0.0850`
  - 边界区高置信非 `warning` 比例 = `0.4848`
- refined-band 版：
  - `macro_f1 = 0.2681`
  - `balanced_accuracy = 0.3593`
  - `core_AP = 0.7213`
  - `warning_AP = 0.1954`
  - `unacceptable_AP = 0.0847`
  - 边界区高置信非 `warning` 比例 = `0.4917`

这说明：

1. 窄带精修确实把 `macro_f1` 和 `balanced_accuracy` 稍微推高了。
2. 但它没有保住当前主线最重要的边界友好性优势。
3. 它也没有把 `unacceptable_AP` 保持在当前主线 hard 版的水平。

换句话说：

> refined-band 更像是“更激进一点的硬加权精修”，它在整体区分力上略有收益，但开始把 warning 侧与边界谨慎性的收益往回退，因此还不能替代当前主线版本。

## 每折入选结果

这轮最终每折选中的候选是：

- `fold 1`: `default`
- `fold 2`: `boundary_3.50_warning_1.10`
- `fold 3`: `default`
- `fold 4`: `boundary_3.00_warning_1.25`
- `fold 5`: `boundary_3.50_warning_1.40`

这给了一个很有用的新信息：

- 窄带搜索后，真正会被选中的权重开始向更强的 `boundary_weight = 3.0 ~ 3.5` 集中
- `warning_weight` 则集中在 `1.10 ~ 1.40`
- 但并不是所有折都需要加权：
  - `fold 1`
  - `fold 3`
  依然直接回到 `default`

因此更合理的理解是：

> 当前第三类并不是“所有时间段都应统一加更强的边界权重”，而是“某些折段确实受益于较强的 inner-threshold hard weighting，但另一些折段仍然更适合 default”。

## 风险侧核对

这轮我额外核对了 clearly-unacceptable 侧的损失：

- `unacceptable -> non-unacceptable` 的丢失样本数 = `4`
- 且 `4` 条都不在 `boundary_any` 内

这和当前主线 hard 版在“丢失样本数”上相同，但 refined-band 的 pooled `unacceptable_AP` 更低，说明它的副作用不只是来自丢失数量，也来自分数排序层面的轻微退化。

## 当前判断更新

到这一步，第三类主线可以进一步收紧成：

1. 当前最稳主线仍然是：
   - “只对 `8.2 / 8.7` 做内侧阈值对称硬加权”
2. 只做“窄带精修”并不能自动带来更好的总体折中。
3. 当前这轮 refined-band 的价值在于：
   - 它确认了更强的 `boundary_weight = 3.0 ~ 3.5` 确实能继续推高全局分类指标
   - 但也确认了这样做会开始吃掉当前主线 hard 版的边界谨慎性优势

所以现在 cleanroom 支持的不是：

> “继续把权重往更强区域精修，就会自然得到更优版本”

而更像是：

> “当前第三类已经接近一条有效前沿；如果更偏向边界友好性，保留原主线 hard 版更合适；如果更偏向整体区分力，refined-band 是一个可以保留的激进备选。”

## 本轮产物

- 汇总 JSON：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_refined_band/boundary_weighted_inner_thresholds_summary.json`
- 外层逐折结果：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_refined_band/boundary_weighted_inner_thresholds_per_fold.csv`
- 每折选中的候选：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_refined_band/boundary_weighted_inner_thresholds_selected_candidate_per_fold.csv`
- 内层搜索明细：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_refined_band/boundary_weighted_inner_thresholds_inner_search.csv`
- 逐样本评分结果：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_refined_band/boundary_weighted_inner_thresholds_results.csv`
- 别名映射：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_refined_band/sensor_identity_alias_pairs.csv`
- 摘要：`reports/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_refined_band/boundary_weighted_inner_thresholds_summary.md`

# 2026-03-31 第三类继续深挖：显式多目标 inner-CV 候选选择

## 本轮动机

上一轮 refined-band 给出的信息已经很明确：

- 更强的 `boundary_weight = 3.0 ~ 3.5` 确实能继续推高整体区分力
- 但它也会把 `warning_AP`、`unacceptable_AP` 和边界谨慎性往回拉

因此接下来的合理问题不是再改权重结构，而是：

> refined-band 的问题，到底是“候选权重带本身的问题”，还是“inner-CV 选候选规则太单一”的问题？

为此，这一轮专门验证我前一轮提出的思路：

> 保持 refined-band 的候选网格不变，只把 inner-CV 的候选选择改成显式多目标筛选，把 `boundary_high_confidence_non_warning_rate` 和 `unacceptable_AP` 一起纳入 guardrail。

## 实验设计

- 新脚本：`scripts/run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_multiobjective.py`
- 新配置：`configs/ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_multiobjective.yaml`

与上一轮 refined-band 保持一致的部分：

- `simple 120min causal stats`
- `monotonic ordinal/cumulative`
- 只对 inner thresholds:
  - `8.2`
  - `8.7`
  做对称硬加权
- 使用同一套 refined-band 网格：
  - `boundary_weight in {2.50, 2.75, 3.00, 3.25, 3.50}`
  - `warning_weight in {1.00, 1.10, 1.25, 1.40, 1.50}`
- `fold-local topk=40`
- `merge_data.csv + merge_data_otr.csv`
- strict sensor-identity de-dup
- outer `TimeSeriesSplit(5)` + inner `TimeSeriesSplit(3)`

本轮唯一变化在 inner-CV 候选选择规则：

1. 先要求候选满足基础性能 guardrail：
   - `macro_f1 >= default - 0.005`
   - `balanced_accuracy >= default - 0.005`
2. 再要求候选满足风险/边界 guardrail：
   - `unacceptable_AP >= default - 0.0015`
   - `boundary_high_confidence_non_warning_rate <= default + 0.01`
3. 在通过 guardrail 的候选里，再按下面顺序排序：
   - `boundary_high_confidence_non_warning_rate` 越低越优先
   - `unacceptable_AP` 越高越优先
   - `warning_AP` 越高越优先
   - `macro_f1` 越高越优先
   - `balanced_accuracy` 越高越优先

也就是说，这一步是在测试：

> 如果我们把“边界谨慎性”和“不可接受侧保护”显式写进 inner-CV 的选候选规则里，结果会不会比 plain refined-band 更稳？

## 结果

这轮结果非常有价值，但它是一条“强负结果”：

> 在当前 refined-band 候选集上，显式多目标 inner-CV 选择并没有改变最终结果。

pooled outer-test 结果与上一轮 refined-band 完全一致：

- default `macro_f1 = 0.2610`
- multiobjective `macro_f1 = 0.2681`
- default `balanced_accuracy = 0.3549`
- multiobjective `balanced_accuracy = 0.3593`
- default `core_AP = 0.7151`
- multiobjective `core_AP = 0.7213`
- default `warning_AP = 0.1924`
- multiobjective `warning_AP = 0.1954`
- default `unacceptable_AP = 0.0863`
- multiobjective `unacceptable_AP = 0.0847`
- default 边界区高置信非 `warning` 比例 = `0.4945`
- multiobjective 边界区高置信非 `warning` 比例 = `0.4917`

更关键的是，逐折选中的候选也与上一轮 refined-band 完全一致：

- `fold 1`: `default`
- `fold 2`: `boundary_3.50_warning_1.10`
- `fold 3`: `default`
- `fold 4`: `boundary_3.00_warning_1.25`
- `fold 5`: `boundary_3.50_warning_1.40`

## 关键解读

这轮结果说明了一件很重要的事：

1. 在当前 refined-band 候选网格里，plain selector 和显式多目标 selector 最终选到了完全相同的折内候选。
2. 因此，上一轮 refined-band 的 trade-off 不是因为“inner-CV 选候选规则太粗”，而是因为：
   - 候选集本身就把最佳点落在了这几个候选上。
3. 换句话说，当前瓶颈已经不在 selector，而在 candidate set 本身。

这比单纯说“多目标选择没提升”更重要，因为它把问题定位清楚了：

> 在当前第三类主线上，继续调 inner-CV 选候选规则，已经很难再带来新的 cleanroom 增益；如果还想继续突破，下一步更可能需要改候选集合本身，而不是继续改 selector。

## 与上一轮 refined-band 的关系

从实验判断上，这一轮其实相当于给 refined-band 做了一次“机制核验”：

- 上一轮怀疑：
  - 也许 refined-band 的不足来自 selector 偏向整体指标
- 这轮验证结果：
  - 不是
- 因为就算显式把：
  - 边界高置信非 `warning`
  - `unacceptable_AP`
  都写进选候选规则
  最终还是选回了同样的候选

因此现在可以更有把握地说：

> refined-band 的问题不是“选错了候选”，而是“候选带本身的有效前沿就落在这里”。

## 当前判断再收紧

到这一步，第三类主线的判断可以进一步收紧为：

1. 当前最稳主线仍然是：
   - “只对 `8.2 / 8.7` 做内侧阈值对称硬加权”
2. refined-band 是一个偏向整体区分力的激进备选。
3. 显式多目标 inner-CV selector 没有改变 refined-band 的结果。
4. 因此下一步如果继续深挖，不应再优先投入在 selector 上，而应转向：
   - 候选网格/候选结构本身
   - 或更上游的表示与目标设计

## 本轮产物

- 汇总 JSON：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_multiobjective/boundary_weighted_inner_thresholds_summary.json`
- 外层逐折结果：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_multiobjective/boundary_weighted_inner_thresholds_per_fold.csv`
- 每折选中的候选：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_multiobjective/boundary_weighted_inner_thresholds_selected_candidate_per_fold.csv`
- 内层搜索明细：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_multiobjective/boundary_weighted_inner_thresholds_inner_search.csv`
- 逐样本评分结果：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_multiobjective/boundary_weighted_inner_thresholds_results.csv`
- 别名映射：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_multiobjective/sensor_identity_alias_pairs.csv`
- 摘要：`reports/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_multiobjective/boundary_weighted_inner_thresholds_summary.md`

# 2026-03-31 第三类继续深挖：专门设计“保边界、保 unacceptable 侧”的 candidate set

## 先做代价与收益判断

在开始这一轮之前，我先对“是否值得继续深挖”做了一个比较务实的判断。

继续深挖的代价：

1. 工程代价低到中等。
   - 不需要改 stable line
   - 不需要改数据处理主干
   - 只需要新增一套 config，复用已有 cleanroom 脚本
2. 计算代价可接受。
   - 单轮 outer `TimeSeriesSplit(5)` + inner `TimeSeriesSplit(3)` 的运行时间大约几分钟量级
   - 可以承受再做一轮精修验证
3. 认知代价相对可控。
   - 当前第三类主线已经非常清楚
   - 因此这时继续做一轮，不会让实验树失控

继续深挖的可能收益：

1. 高概率不会出现“跃迁式改进”，更可能是增量收益。
2. 但仍然值得做，因为上一轮已经定位出：
   - 瓶颈不在 selector
   - 那么 candidate set 本身就成为最自然、最值得验证的下一杠杆
3. 最理想的收益不是大幅抬升所有指标，而是：
   - 保住当前 hard 主线的大部分边界友好性
   - 同时吸收 refined-band 在整体区分力上的一部分收益

因此我的判断是：

> 这一步是值得做的，但要带着“增量优化而非期待翻盘”的预期来做。

## 实验设计

基于上一轮分析，我专门设计了一套更偏“保边界、保 unacceptable 侧”的新候选网格。

- 脚本：`scripts/run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_multiobjective.py`
- 新配置：`configs/ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_guardrail_candidate_set.yaml`

设计思路是：

1. 不再使用 refined-band 里最激进的 `boundary_weight = 3.5`
2. 把边界权重限制在中强区间：
   - `boundary_weight in {2.00, 2.25, 2.50, 2.75, 3.00}`
3. 同时保留较强的 warning 侧权重：
   - `warning_weight in {1.00, 1.25, 1.50, 1.75, 2.00}`
4. 继续配合显式多目标 selector：
   - `macro_f1`
   - `balanced_accuracy`
   - `unacceptable_AP`
   - `boundary_high_confidence_non_warning_rate`
   一起纳入 guardrail

也就是说，这轮不是再改 selector，而是：

> 直接把候选集合本身往“更保边界、更保 clearly-unacceptable 侧”的方向收束。

## 结果

这轮结果说明：candidate set 本身确实是一个真实杠杆，但它带来的仍然是“增量位移”，不是新的最优前沿。

pooled outer-test 结果：

- default `macro_f1 = 0.2610`
- guardrail-candidate-set `macro_f1 = 0.2671`
- default `balanced_accuracy = 0.3549`
- guardrail-candidate-set `balanced_accuracy = 0.3578`
- default `core_AP = 0.7151`
- guardrail-candidate-set `core_AP = 0.7202`
- default `warning_AP = 0.1924`
- guardrail-candidate-set `warning_AP = 0.1951`
- default `unacceptable_AP = 0.0863`
- guardrail-candidate-set `unacceptable_AP = 0.0848`
- default 边界区高置信非 `warning` 比例 = `0.4945`
- guardrail-candidate-set 边界区高置信非 `warning` 比例 = `0.4862`

对应增量：

- `macro_f1_delta = +0.0062`
- `balanced_accuracy_delta = +0.0029`
- `core_AP_delta = +0.0050`
- `warning_AP_delta = +0.0027`
- `unacceptable_AP_delta = -0.0015`
- `boundary_high_conf_non_warning_rate_delta = -0.0083`

## 与当前两条关键版本的比较

把它和“当前 hard 主线”和“refined-band 激进备选”放在一起看，会更清楚：

- 当前 hard 主线：
  - `macro_f1 = 0.2674`
  - `balanced_accuracy = 0.3580`
  - `warning_AP = 0.1963`
  - `unacceptable_AP = 0.0850`
  - 边界区高置信非 `warning` 比例 = `0.4848`
- refined-band：
  - `macro_f1 = 0.2681`
  - `balanced_accuracy = 0.3593`
  - `warning_AP = 0.1954`
  - `unacceptable_AP = 0.0847`
  - 边界区高置信非 `warning` 比例 = `0.4917`
- guardrail candidate set：
  - `macro_f1 = 0.2671`
  - `balanced_accuracy = 0.3578`
  - `warning_AP = 0.1951`
  - `unacceptable_AP = 0.0848`
  - 边界区高置信非 `warning` 比例 = `0.4862`

这说明它的定位非常明确：

1. 它比 refined-band 更保边界，也略微更保 `unacceptable_AP`。
2. 但它没有超过当前 hard 主线。
3. 它也没有保住 refined-band 那部分更高的整体区分力。

换句话说：

> 这轮 candidate-set redesign 把结果推到了 hard 主线和 refined-band 之间的一个中间点，但没有形成新的最优版本。

## 逐折入选变化

这轮最终每折选中的候选是：

- `fold 1`: `default`
- `fold 2`: `boundary_3.00_warning_1.25`
- `fold 3`: `default`
- `fold 4`: `boundary_3.00_warning_1.25`
- `fold 5`: `boundary_2.50_warning_1.75`

这个结果很关键，因为它证明了：

1. 改 candidate set 本身，确实会改逐折入选。
2. 这次与 refined-band 的最大差异在 `fold 5`：
   - refined-band 选的是 `boundary_3.50_warning_1.40`
   - 这一轮变成了 `boundary_2.50_warning_1.75`
3. 这正符合设计初衷：
   - 减少过强边界权重
   - 增加 warning 侧支撑

也就是说：

> 与上一轮“selector 不再是杠杆”的判断不同，这一轮清楚地说明了 candidate set 本身仍然是一个真实杠杆。

## 风险侧核对

这轮 clearly-unacceptable 侧的核对结果：

- `unacceptable -> non-unacceptable` 的丢失样本数 = `4`
- 且 `4` 条都不在 `boundary_any` 内

这和当前 hard 主线以及 refined-band 一样，说明新 candidate set 没有引入更糟的 clearly-unacceptable 丢失数量，但 pooled `unacceptable_AP` 仍略低，说明排序层面依然存在轻微退化。

## 当前判断更新

到这一步，关于“是否值得继续深挖”的判断可以更清楚了：

1. 候选集设计是值得做的。
   - 因为它确实改变了结果
   - 不是无效动作
2. 但它带来的收益已经明显进入“增量优化区”。
3. 当前最稳主线仍然没有被替换：
   - 依然是“只对 `8.2 / 8.7` 做内侧阈值对称硬加权”的当前 hard 主线
4. 这轮 guardrail candidate set 的价值在于：
   - 它证明 candidate set 是真实杠杆
   - 但也证明当前第三类已经非常接近一条有效前沿

因此下一步如果还要继续深挖，我会建议：

> 不要再做大面积盲搜，而是只针对 `fold 2` 这类最难折段，定向设计少量候选族，看能不能专门修掉“边界收益与 warning/unacceptable 侧冲突”的那一段。

## 本轮产物

- 汇总 JSON：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_guardrail_candidate_set/boundary_weighted_inner_thresholds_summary.json`
- 外层逐折结果：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_guardrail_candidate_set/boundary_weighted_inner_thresholds_per_fold.csv`
- 每折选中的候选：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_guardrail_candidate_set/boundary_weighted_inner_thresholds_selected_candidate_per_fold.csv`
- 内层搜索明细：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_guardrail_candidate_set/boundary_weighted_inner_thresholds_inner_search.csv`
- 逐样本评分结果：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_guardrail_candidate_set/boundary_weighted_inner_thresholds_results.csv`
- 别名映射：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_guardrail_candidate_set/sensor_identity_alias_pairs.csv`
- 摘要：`reports/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_guardrail_candidate_set/boundary_weighted_inner_thresholds_summary.md`

# 2026-03-31 阶段性冻结与回退盘点

## 1. 当前冻结主线

在本轮 cleanroom 到目前为止，正式冻结的主线是：

> `simple 120min causal stats + monotonic ordinal/cumulative + fold-local topk=40 + sensor-identity de-dup + 只对 8.2/8.7 做内侧阈值对称硬加权`

对应实现：

- 配置：`configs/ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup.yaml`
- 脚本：`scripts/run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup.py`

冻结理由：

1. 它仍然是当前“总体折中”最稳的版本。
2. 后续 refined-band、multiobjective selector、guardrail candidate set 都没有把它整体替代。
3. 如果当前目标是先收敛一个可靠主线，而不是继续深挖第三类细枝末节，那么这条线最适合作为阶段性基准。

当前冻结主线的关键指标：

- `macro_f1 = 0.2674`
- `balanced_accuracy = 0.3580`
- `warning_AP = 0.1963`
- `unacceptable_AP = 0.0850`
- 边界区高置信非 `warning` 比例 = `0.4848`

## 2. 回退后仍值得看的方向

如果我们从第三类深挖里退出来，回到之前已经讨论过的方向重新盘点，我认为还值得继续优化的方向主要有下面几类：

### 2.1 PH 滞后分支

状态：

- 已明确知道 `data/B4-AI-C53001A.PV.F_CV.xlsx` 具有明显工艺滞后
- 但到目前为止还没有正式纳入 cleanroom 主比较

为什么值得回退后优先考虑：

1. 它是一个尚未真正验证的新信息源，而不是在当前第三类主线上继续挤牙膏。
2. 它与当前“边界模糊处理”主旨并不冲突，反而可能帮助解释某些边界样本。
3. 它的收益潜力大于继续微调第三类候选网格。

当前判断：

> 如果从当前第三类深挖中退出，PH lag-aware cleanroom 是最值得优先回到的方向之一。

### 2.2 稀疏 EWMA 混合表示

状态：

- 已经验证过
- 目前没有打赢冻结主线
- 但确实表现出一部分整体区分力信号

为什么它还值得保留：

1. 这条线不是完全无效，而是“有局部增益，但还没变成边界友好型版本”。
2. 它比继续深挖当前第三类更像一条结构上不同的备选路线。

当前判断：

> 这条线值得保留，但优先级低于 PH，因为当前证据更像“局部有信号”，不是“下一条最可能的主线”。

### 2.3 点位含义与工艺分组

状态：

- `data/卤化点位.xlsx` 还没有真正用到 cleanroom 解释层

为什么它仍值得做：

1. 工程代价很低
2. 虽然它未必直接提升指标，但可能帮助：
   - 解释 top sensors
   - 识别工艺上重复/近重复点位
   - 为后续 PH、阶段化或分组筛选提供更稳的先验

当前判断：

> 这不是“最直接抬指标”的方向，但它是一个低成本、对后续实验决策有帮助的准备工作。

### 2.4 stage-aware / future-head

状态：

- 两者都还没有正式展开 cleanroom

为什么现在不建议立刻回去：

1. 结构复杂度高
2. 更容易把当前 cleanroom 的归因边界再次打散
3. 在 PH 和点位解释都还没系统纳入前，先做这两条线性价比不高

当前判断：

> 这两条线先保留，不作为回退后的第一优先级。

## 3. 当前优先级建议

基于以上回退盘点，如果我们现在先冻结第三类最佳主线，再重新选下一方向，我建议的优先级是：

1. `PH lag-aware cleanroom`
2. `点位含义 / 工艺分组梳理`
3. `稀疏 EWMA 混合表示` 的后续精修
4. `stage-aware / future-head`

## 4. 当前阶段结论

因此，到这一时点可以把策略收紧成：

1. 第三类最佳主线已经冻结，不再继续往更深的局部权重搜索里走。
2. 如果要继续优化，应该回退到更早讨论过、但还没有充分展开的新方向。
3. 其中最值得优先重新打开的是：
   - `PH lag-aware cleanroom`

# 2026-03-31 回退后第一优先线结果更新：PH lag-aware cleanroom

## 1. 实验目的

这一步的目标不是替换当前冻结主线，而是回答一个更具体的问题：

> 在保持当前最强 cleanroom 主线不变的前提下，把具有明显工艺滞后的 `PH` 信号以 lag-aware 的方式追加进去，是否能带来边界处理或整体判别上的稳定增益？

## 2. 实验设计

本轮严格建立在当前冻结主线之上：

- 冻结 baseline：
  - `simple 120min causal stats`
  - `monotonic ordinal/cumulative`
  - `fold-local topk=40`
  - `sensor-identity de-dup`
  - `只对 8.2 / 8.7 做内侧阈值对称硬加权`
- DCS 数据源：
  - `data/merge_data.csv`
  - `data/merge_data_otr.csv`
  - 并继续执行严格的点位身份去重
- PH 数据源：
  - `data/B4-AI-C53001A.PV.F_CV.xlsx`
- PH 接入方式：
  - 将 PH 视为额外单信号增强，不参与 DCS `topk=40` 位点竞争
  - 只在冻结主线的 DCS 特征集合之外，追加一组 PH 窗口统计特征
- lag 处理方式：
  - 搜索 `lag_minutes in {0, 30, 60, 90, 120, 150, 180, 240}`
  - 每个 outer fold 内用 inner `TimeSeriesSplit(3)` 选择 lag
  - 候选选择继续沿用“尽量保边界”的判据：
    - 对 default 允许很小的 `macro_f1 / balanced_accuracy` 容忍带
    - 在 eligible 候选中优先更低的 `boundary high-confidence non-warning rate`

换句话说，这轮并没有改动冻结主线的筛选逻辑，只是在其上回答：

> “PH 在考虑滞后后，能不能作为额外补充信号稳定帮助当前最优主线？”

## 3. 数据审计

PH 文件经清洗后：

- 行数：`939054`
- 时间范围：`2024-01-20 03:37:00` 到 `2025-11-05 09:00:00`
- 本轮可用 PH 统计特征数：`7`
  - `last / mean / std / min / max / range / delta`

本轮与冻结主线一致的 DCS 审计：

- 身份去重后 DCS 位点数：`64`
- 明确长短点别名对：`7`
- 与 LIMS 对齐后的样本数：`2726`
- outer test 实际计分样本：`2270`

## 4. 结果

pooled outer-test 结果如下：

- 冻结 baseline `macro_f1 = 0.2674`
- PH lag-aware `macro_f1 = 0.2648`
- 冻结 baseline `balanced_accuracy = 0.3580`
- PH lag-aware `balanced_accuracy = 0.3476`
- 冻结 baseline `core_AP = 0.7215`
- PH lag-aware `core_AP = 0.7210`
- 冻结 baseline `warning_AP = 0.1963`
- PH lag-aware `warning_AP = 0.1947`
- 冻结 baseline `unacceptable_AP = 0.0850`
- PH lag-aware `unacceptable_AP = 0.0805`
- 冻结 baseline 边界区高置信非 `warning` 比例 = `0.4848`
- PH lag-aware 边界区高置信非 `warning` 比例 = `0.4848`

对应增量为：

- `macro_f1_delta = -0.0026`
- `balanced_accuracy_delta = -0.0104`
- `core_AP_delta = -0.0006`
- `warning_AP_delta = -0.0016`
- `unacceptable_AP_delta = -0.0045`
- `boundary_high_conf_non_warning_rate_delta = 0.0000`

## 5. 各折 lag 选择

每个 outer fold 最终选中的 PH lag 是：

- `fold 1`: `ph_lag_060`
- `fold 2`: `ph_lag_060`
- `fold 3`: `ph_lag_120`
- `fold 4`: `ph_lag_060`
- `fold 5`: `ph_lag_240`

这个现象本身很重要，因为它说明：

1. inner-CV 并不是总回退到 `default`
2. PH 的确存在“训练内可分辨的滞后候选”
3. 但这些候选没有在 pooled outer-test 上稳定转化为收益

也就是说，当前更准确的判断不是“PH 完全没信号”，而是：

> `PH lag` 在训练内有可选结构，但以“单信号追加 + 简单窗口统计”这条实现方式，还没有稳定打赢冻结主线。

## 6. 逐折观察

逐折看，PH 并非所有场景都退化：

- `fold 1`
  - `macro_f1` 与 `balanced_accuracy` 都略有提升
  - 边界区高置信非 `warning` 比例也下降
- `fold 4`
  - `macro_f1` 小幅提升
  - `balanced_accuracy` 基本持平略降
- `fold 2 / 3 / 5`
  - 整体上退化更明显
  - 其中 `fold 5` 的退化最大

这说明当前 PH 分支更像：

- 在局部折段有帮助
- 但泛化不稳定
- 还不足以作为当前主线的稳健增强项

## 7. 当前判断更新

到这一步，可以把对 PH 分支的判断收紧成：

1. `PH lag-aware cleanroom` 这一步是值得做的，因为它回答了一个之前没有被真正验证的问题。
2. 结果不支持把 PH 直接并入当前冻结主线。
3. 但结果也不支持把 PH 简单判定为“无用信号”，因为：
   - inner 选择并未总回退 default
   - 局部折段存在正向迹象
4. 当前更像是：
   - `PH 有潜在信息`
   - 但“单点位 + 简单窗口统计 + 直接追加到冻结主线”不是足够强的接入方式

因此，当前最稳妥的结论是：

> 冻结主线保持不变；PH 分支记录为“有局部信号，但当前实现未形成稳健增益”。

## 8. 对回退优先级的影响

这轮之后，回退方向的优先级应更新为：

1. `点位含义 / 工艺分组梳理`
2. `PH` 的更细化接入方式（如果继续做，应避免继续沿“单点位简单追加”小步微调）
3. `稀疏 EWMA 混合表示` 的后续精修
4. `stage-aware / future-head`

原因是：

- `PH lag-aware` 已经完成了第一轮最自然、最克制的 cleanroom 验证
- 它没有赢冻结主线
- 因此下一步更值得先补“点位含义与工艺结构”这层先验，再决定是否回到 PH 的更细化实现

## 9. 本轮产物

- 配置：`configs/ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_ph_lag.yaml`
- 脚本：`scripts/run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_ph_lag.py`
- 摘要 JSON：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_ph_lag/ph_lag_summary.json`
- 逐折结果：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_ph_lag/ph_lag_per_fold.csv`
- 每折选中 lag：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_ph_lag/ph_lag_selected_candidate_per_fold.csv`
- inner 搜索明细：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_ph_lag/ph_lag_inner_search.csv`
- 逐样本结果：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_ph_lag/ph_lag_results.csv`
- 点位身份别名表：`artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_ph_lag/sensor_identity_alias_pairs.csv`
- 摘要报告：`reports/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_ph_lag/ph_lag_summary.md`

# 2026-04-01 冻结主线可视化评价与在线测试判断

## 1. 目标

在不新增实验的前提下，基于当前冻结主线已有产物，生成一份便于现场沟通的可视化评价图，并明确回答：

> 这条主线现在是否已经适合到现场进行在线测试？

## 2. 可视化内容

本轮生成了一张综合评价图，放在：

- `reports/frozen_mainline_visual_review/frozen_mainline_visual_review.png`

图中包含四部分：

1. pooled 指标：对比 default 与当前冻结主线
2. 逐折增益：看相对 default 的稳定性
3. 冻结主线 confusion matrix：看三类样本误判结构
4. 边界样本预测构成：看边界区是否仍存在高置信非 `warning`

同时生成了配套判断报告：

- `reports/frozen_mainline_visual_review/frozen_mainline_online_readiness.md`

以及可复现脚本：

- `scripts/plot_frozen_mainline_visual_review.py`

## 3. 图表给出的核心读数

冻结主线相对 default 的 pooled 改善仍然成立：

- `macro_f1: 0.2610 -> 0.2674`
- `balanced_accuracy: 0.3549 -> 0.3580`
- `core_AP: 0.7151 -> 0.7215`
- `warning_AP: 0.1924 -> 0.1963`
- 边界区高置信非 `warning` 比例：`0.4945 -> 0.4848`

但 confusion matrix 也清楚暴露了当前方法的上限：

- acceptable recall：`0.2563`
- warning recall：`0.4219`
- unacceptable recall：`0.3957`

同时，边界区仍有：

- `351 / 724` 个样本被高置信地判成非 `warning`
- 对应比例 `0.4848`

也就是说，这条主线虽然比 default 更稳妥，但离“可直接替代人工判别”的可靠性还有明显距离。

## 4. 当前判断

到这一步，可以把是否适合现场在线测试的结论明确收紧成：

> 可以去现场做旁路在线测试，但不适合直接进入无人值守自动在线判级或联动控制。

更具体地说：

1. `Go` for shadow / human-in-the-loop online test
2. `No-Go` for autonomous online deployment

## 5. 判断依据

支持去现场做旁路在线测试的原因：

1. 当前主线已经冻结，口径清晰，实验边界干净。
2. identity de-dup、内侧阈值加权、monotonic cumulative 这些关键环节都已经固定。
3. 相对 default，当前主线确实带来了小幅但一致的边界友好性改进。

不支持直接自动上线的原因：

1. 绝对指标仍偏低，尤其 `warning_AP` 和 `unacceptable_AP` 仍不足以承担高风险自动判定。
2. 边界样本里仍有接近一半被高置信地判成非 `warning`。
3. 折间表现仍不够稳定，说明对时段变化的鲁棒性还不足。
4. 当前 cleanroom 只验证了离线时间切分效果，还没有完成真正在线链路上的缺失、漂移、延迟与可解释性校验。

## 6. 对后续工作的含义

因此，这次图表评估的价值不在于“宣布可以上线”，而在于把当前阶段定位得更清楚：

1. 冻结主线已经足以支撑一次有控制的现场旁路试运行。
2. 现场试运行的目标应是验证：
   - 边界批次的实时表现
   - 明显不合格批次的识别可靠性
   - 点位缺失/漂移时的输出稳定性
   - 预测与工艺员理解之间是否可解释
3. 在这些问题没有补齐前，不应把当前方法直接作为自动判级器投入生产。

# 2026-04-01 新分支切换：distributional interval prediction + reject option

## 1. 分支切换原因

在冻结当前最佳 ordinal / cumulative 主线之后，我重新阅读了你补充的：

- `cleanroom_validation_experiment_plan_distributional_reject_option.md`

这次切换的核心，不是把前面的工作推翻，而是承认一个越来越清楚的事实：

> 当前剩余困难，可能更像“边界样本本身存在不可消除的模糊性”，而不只是“模型还不够强”。

也就是说，问题未必是还要继续在：

- EWMA
- PH
- 更多权重微调

这些方向上挤局部增益；更可能是要正面验证：

- 是否应该从“必须给出最终类别”转向“先给区间分布，再允许拒判 / 复检”

我认为这个方向判断是成立的，因此正式打开新的 cleanroom 分支。

## 2. 我对新计划的理解

这份新计划最重要的地方，不是“改个模型”，而是“改决策真相观”：

1. ordinal / cumulative 并没有被判错
2. 它只是说明：
   - 单纯把问题从硬三分类改成阈值概率，还不够
   - 因为最后仍然被迫塌缩成一个最终标签
3. 如果样本在 `8.2 / 8.7` 附近本来就带有真实的不确定性，那么更合适的 cleanroom 问题就应当是：
   - 先预测 ordered interval distribution
   - 再在业务层允许 `retest`

换句话说，这条新分支不是在否定前线，而是在问：

> “显式拒判”是不是当前这类边界模糊问题里缺失的那一层。

## 3. 当前冻结比较锚点

这条新分支的比较锚点保持不变，仍然是当前冻结主线：

- `simple 120min causal stats`
- `fold-local topk=40`
- `sensor-identity de-dup`
- `monotonic ordinal / cumulative`
- `只对 8.2 / 8.7 做内侧阈值对称硬加权`

当前锚点 pooled 指标：

- `macro_f1 = 0.2674`
- `balanced_accuracy = 0.3580`
- `warning_AP = 0.1963`
- `unacceptable_AP = 0.0850`
- 边界区高置信非 `warning` 比例 = `0.4848`

这意味着：

- 新分支不是“从零开始”
- 而是必须正面对比一个已经冻结、可解释、可旁路试运行的参考线

## 4. 我决定实际采用的首轮实验路线

在仔细读完新计划后，我没有直接把它展开成很多变体，而是收敛成一条很克制的首轮路线。

### 4.1 数据与特征边界

首轮不改特征系：

- 仍使用 `merge_data.csv + merge_data_otr.csv`
- 仍使用严格 identity de-dup
- 仍使用 `simple 120min causal window stats`
- 仍使用 fold-local `topk=40`
- 仍使用 `TimeSeriesSplit(5)`

原因很明确：

- 这条分支要先验证的是“输出 formulation + reject layer”
- 不是“特征是不是还要再换一轮”

### 4.2 首轮区间定义

固定 5 个 ordered bins：

- `bin_1: y < 8.0`
- `bin_2: 8.0 <= y < 8.2`
- `bin_3: 8.2 <= y < 8.7`
- `bin_4: 8.7 <= y < 8.9`
- `bin_5: y >= 8.9`

### 4.3 首轮 soft-label 规则

我决定首轮不做“全阈值都模糊化”，而是采用更克制的规则：

- 外侧 `8.0 / 8.9` 先保持硬边界
- 只对内侧 `8.2 / 8.7` 做局部 soft labeling
- 固定 `soft_radius = 0.05`
- 在 `8.2 ± 0.05` 内，把质量线性分配给 `bin_2 / bin_3`
- 在 `8.7 ± 0.05` 内，把质量线性分配给 `bin_3 / bin_4`
- 其余位置保持所在 bin 的硬归属

我之所以这样定，不是随意简化，而是为了延续我们前面已经学到的东西：

1. 当前业务最关心的模糊性确实集中在 `8.2 / 8.7`
2. 之前 outer threshold 一起软化，往往更容易伤到 clearly-unacceptable 侧
3. 所以 first run 应先验证“只软化真正关心的内侧边界”是否已经足够

### 4.4 首轮模型

首轮模型先走最简单可审计路线：

- 5-bin multinomial logistic

考虑到标准实现对 soft targets 支持不直接，首轮实现准备采用：

- 基于 non-zero soft-label bins 的加权样本复制

这样做的原因：

- 轻量
- 容易 audit
- 不会把 cleanroom 一上来带进复杂自定义损失

### 4.5 首轮比较结构

首轮只比较三件事：

1. Frozen Baseline A：
   - 当前冻结 ordinal / cumulative 主线
2. Baseline B：
   - distributional prediction
   - 不允许 reject
   - 直接塌缩为 `acceptable / warning / unacceptable`
3. Treatment C：
   - 同样的 distributional prediction
   - 允许 `retest`

### 4.6 首轮 reject 规则

reject 层我也决定先固定为最简单、最可解释的一版：

- 先得到 5-bin 分布
- 再得到：
  - `acceptable_prob = P(bin_3)`
  - `warning_prob = P(bin_2) + P(bin_4)`
  - `unacceptable_prob = P(bin_1) + P(bin_5)`
- 计算：
  - `max_business_prob`
  - 5-bin normalized entropy
- 若满足其一则输出 `retest`：
  - `max_business_prob < tau_conf`
  - 或 `normalized_entropy > tau_entropy`

首轮阈值搜索网格先固定为：

- `tau_conf in {0.45, 0.50, 0.55, 0.60}`
- `tau_entropy in {0.80, 0.85, 0.90, 0.95}`

这些阈值只允许在 outer-train 内通过 inner `TimeSeriesSplit(3)` 选，不得看 outer-test。

### 4.7 首轮 guardrail

为了避免 reject 被滥用，首轮还额外冻结一个 operational guardrail：

- `reject_rate <= 0.25`

如果一个 reject 规则只能靠大面积拒判来换指标，那这轮不算支持它。

## 5. 首轮成功标准

这条新分支不会因为“看起来更高级”就被判成功。  
我准备按下面的标准来判断：

1. reject 集合是否真的集中在模糊样本，而不是简单样本
2. covered-sample 的 macro_f1 是否不劣于冻结主线太多
3. non-rejected 样本上的 unacceptable miss 风险是否没有变坏
4. 边界区高置信非 `warning` 是否下降
5. Brier / log loss / entropy 这些分布质量指标是否合理
6. 收益是否不是只出现在一两个幸运折里

如果这些条件达不到，我会明确记为：

- “distributional + reject 在当前 simple baseline 下尚未证明优于冻结主线”

而不会模糊写成“方向大概有希望”。

## 6. 当前阶段记录状态

到这一步，我已经完成的是：

1. 仔细阅读新实验路径文档
2. 明确它与当前冻结主线的关系
3. 将其收敛为一个具体、可执行、可审计的首轮路线
4. 落下新的计划文档与 audit 锚点

当前还没有开始的是：

- 新分支的实际模型实现
- distributional baseline B 的首轮运行
- reject option treatment C 的首轮运行

也就是说，当前状态是：

> 新分支规划已冻结，实验尚未启动。

## 7. 本轮新增记录

本轮新增的正式记录文件：

- `plans/distributional_reject_option_validation.md`
- `reports/distributional_reject_option_current_head_audit.md`

这两个文件的分工是：

1. `plan`：
   - 记录这条新分支的首轮执行路线
2. `audit`：
   - 作为这条分支后续实验记录的锚点
   - 当前状态明确标注为 `pre-run planning frozen`

## 8. 下一步建议

下一步不应再继续讨论抽象方向，而是进入这条新分支的第一个真实 cleanroom run：

1. 先实现 Baseline B：
   - distributional prediction without reject
2. 再实现 Treatment C：
   - same distributional prediction + reject / retest
3. 首轮完成后，再判断这条新路径是否真的值得继续

这一步应该成为接下来的主线，而不是再回去继续微调旧分支。

## 9. 全局 lag cleanroom：冻结主线下的 L0 / L60 / L120 / L180 / L240 验证

### 9.1 本轮目的

- 驻厂反馈提示可能存在约 `3h` 的工艺滞后。
- 本轮先不切换到 centered-quality，也不切换到 distributional / reject。
- 只回答一个更窄的问题：
  - 在当前已经冻结的 strongest threshold-oriented cleanroom 下，
  - DCS 特征窗口是否应该相对样品时间整体后移，
  - 以及 `L180` 是否有稳定信号。

### 9.2 本轮 baseline 与控变量方式

- baseline 继续固定为当前冻结主线：
  - `simple 120min causal stats`
  - `merge_data.csv + merge_data_otr.csv`
  - strict sensor-identity de-dup
  - fold-local `topk=40`
  - monotonic ordinal / cumulative
  - inner-threshold symmetric hard weighting on `8.2 / 8.7`
- 本轮只改 DCS 窗口锚点，不改：
  - 标签语义
  - 模型家族
  - 训练目标
  - split 方式
- 为了把 lag 本身和“重筛位点”区分开，本轮额外加了一个更严格的控制：
  - 每个 outer fold 的传感器筛选与权重候选，统一在 `L0` train 上选出
  - `L60 / L120 / L180 / L240` 只复用这套 `L0` 选择结果
  - 不允许各 lag 臂自己重新筛点

### 9.3 lag 定义

- `L0`: `[t-120, t]`
- `L60`: `[t-180, t-60]`
- `L120`: `[t-240, t-120]`
- `L180`: `[t-300, t-180]`
- `L240`: `[t-360, t-240]`

### 9.4 样本对齐说明

- 为了保证各 lag 臂完全可比，本轮使用了所有 lag 臂的共同样本交集。
- 结果是：
  - `L0 / L60 / L120 / L180` 原始都可对齐出 `2726` 条
  - `L240` 原始为 `2725` 条
  - 最终共同 scored rows = `2725`
- 因此，本轮 lag cleanroom 里的 `L0` 数值，不应与之前 frozen 主线在 `2726` 条样本上的旧数值直接逐位对照；
  - 它是“在共同样本集 + 共同选点控制下”的新对照 `L0`

### 9.5 pooled 结果

- `L0`:
  - `macro_f1 = 0.2860`
  - `balanced_accuracy = 0.3501`
  - `warning_AP = 0.1979`
  - `unacceptable_AP = 0.0870`
  - boundary high-confidence non-warning `= 0.5152`
- `L60`:
  - `macro_f1 = 0.3053`
  - `balanced_accuracy = 0.3586`
  - `warning_AP = 0.1795`
  - `unacceptable_AP = 0.1011`
  - boundary high-confidence non-warning `= 0.4848`
- `L120`:
  - `macro_f1 = 0.2776`
  - `balanced_accuracy = 0.3442`
  - `warning_AP = 0.1763`
  - `unacceptable_AP = 0.0851`
  - boundary high-confidence non-warning `= 0.4613`
- `L180`:
  - `macro_f1 = 0.2886`
  - `balanced_accuracy = 0.3510`
  - `warning_AP = 0.2047`
  - `unacceptable_AP = 0.0860`
  - boundary high-confidence non-warning `= 0.4254`
- `L240`:
  - `macro_f1 = 0.2812`
  - `balanced_accuracy = 0.3379`
  - `warning_AP = 0.1740`
  - `unacceptable_AP = 0.0914`
  - boundary high-confidence non-warning `= 0.4945`

### 9.6 对 `L180` 的具体判断

- `L180` 并不是这轮里的整体最优臂。
- 如果看整体判别与 unacceptable 侧：
  - `L60` 更强
  - 它同时提高了 `macro_f1 / balanced_accuracy / unacceptable_AP`
- 但 `L180` 也不是没有信号。
- 它体现出的优势很集中：
  - `warning_AP` 是五个 lag 臂里最高的
  - boundary high-confidence non-warning 也是五个 lag 臂里最低的
- 也就是说，`L180` 最像“边界更谨慎”的 lag 版本，而不是“整体最强”的 lag 版本。

### 9.7 fold-level 信号

- `L180` 相对 `L0`：
  - `macro_f1` 更好：`3 / 5` folds
  - `balanced_accuracy` 更好：`3 / 5` folds
  - `warning_AP` 更好：`3 / 5` folds
  - `unacceptable_AP` 不差于 `L0`：`2 / 5` folds
  - boundary overconfidence 更好：`5 / 5` folds
- 这说明：
  - `L180` 的“边界谨慎性改善”是稳定的
  - 但它的“整体最优”并不稳定

### 9.8 当前结论

- 可以支持“存在 lag sensitivity”
- 但当前证据还不能支持“约 `3h` 已被确认”
- 更准确的写法应是：
  - 在冻结主线下，DCS 窗口整体后移确实会改变结果
  - `1h` 后移更像当前任务下的整体最优方向
  - `3h` 后移更像边界谨慎性更强的方向
  - 因而当前观察到的是“宽范围 lag effect”，而不是“`3h` 特异最优”

### 9.9 对下一步的意义

- 这轮结果已经足够说明：
  - 后续不应该再把 lag 当成可忽略因素
- 但下一步不该立刻写成“固定采用 `3h` lag”
- 更合理的推进顺序是：
  1. 先把 `L60` 和 `L180` 视为两个有代表性的 lag 候选
  2. 再进入更细的模型 / 特征工程阶段
  3. 检查哪些特征族或哪些点位更偏向 `L60`，哪些更偏向 `L180`

### 9.10 本轮产物

- 配置：
  - `configs/ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_global_lag.yaml`
- 脚本：
  - `scripts/run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_global_lag.py`
- 摘要：
  - `reports/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_global_lag/global_lag_summary.md`
- JSON：
  - `artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_global_lag/global_lag_summary.json`

## 10. L60 vs L180 点位偏好 cleanroom：固定 L0 选点下的逐传感器消融

### 10.1 本轮目的

- 上一轮已经说明：
  - `L60` 更像整体最优 lag
  - `L180` 更像边界谨慎性更强的 lag
- 因此本轮不再扩 lag 网格，而是直接追问：
  - 这两个 lag 的差异，是否能下沉到点位层
  - 哪些传感器更支持 `L60` 的整体判别
  - 哪些传感器更支持 `L180` 的边界谨慎性

### 10.2 本轮协议

- 仍然固定使用上一轮 lag cleanroom 的共同样本集
- 仍然固定使用每个 outer fold 在 `L0` 上选出的：
  - `selected_sensors`
  - `chosen boundary / warning weight candidate`
- 本轮只比较两个 lag：
  - `L60`
  - `L180`
- 方法：
  - 对 fold 内已选的每个传感器做一次 leave-one-sensor-out ablation
  - 分别观察在 `L60` 和 `L180` 下，去掉该传感器后：
    - `macro_f1`
    - `balanced_accuracy`
    - `warning_AP`
    - `unacceptable_AP`
    - boundary high-confidence non-warning
    的变化

### 10.3 解释规则

- 对某个 lag 来说：
  - 如果去掉传感器后 `macro_f1 / balanced_accuracy` 下降，
    - 说明这个传感器在支持该 lag 的整体判别
  - 如果去掉传感器后 `warning_AP` 下降，
    - 说明它在支持边界 warning 捕捉
  - 如果去掉传感器后 boundary high-confidence non-warning 上升，
    - 说明它在帮助抑制边界高置信硬切

### 10.4 主要结果

- 支持 `L60` 整体判别更明显的一批点位是：
  - `II_CM511A_PV_CV`
  - `TICA_C52601_PV_F_CV`
  - `II_CM510B_PV_CV`
  - `TI_C51003_PV_F_CV`
  - `TI_C51101A_S_PV_CV`
  - `TI_C51401_S_PV_CV`
- 支持 `L180` 边界谨慎性更明显的一批点位是：
  - `FI_C53001_PV_F_CV`
  - `PI_C51203A_S_PV_CV`
  - `II_CM530A_PV_CV`
  - `FI_C51004_S_PV_CV`
  - `TI_CM511A_PV_F_CV`
  - `TI_C51101B_S_PV_CV`
  - `TI_C50604_PV_F_CV`
  - `FIC_C51801_PV_F_CV`

### 10.5 一个重要现象

- 这两组点位并不相同。
- 也就是说：
  - `L60` 的收益并不是简单来自“所有点位一起整体后移都更好”
  - `L180` 的边界谨慎性也不是所有点位都共同支持
- 当前更像是：
  - 有一批点位更接近短 lag 响应
  - 也有一批点位更接近长 lag / 慢反应

### 10.6 当前判断

- 这轮结果已经让“统一单一 lag”变得不再那么自然。
- 但还不能直接跳到“按点位分层 lag 模型已经成立”。
- 更准确的判断应是：
  - `L60` 和 `L180` 的差异有点位层支持
  - 因而后续值得进入“按特征族 / 按点位分层 lag”的阶段
  - 但这仍然需要新的 cleanroom 验证，不能直接把这一轮分析当成最终建模规则

### 10.7 对下一步的指向

- 现在更值得做的，不是继续扩 `L300 / L360` 之类的新 lag 网格
- 而是开一个更小、更干净的下一轮：
  - 以当前 strongest baseline 为骨架
  - 只对一小批 `L180` 偏好的候选点位尝试 longer-lag
  - 其余大部分点位仍保留 `L60` 或 `L0/L60` 路线
- 也就是说，下一步应从“单 lag cleanroom”推进到“分层 lag cleanroom”

### 10.8 本轮产物

- 配置：
  - `configs/ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_global_lag_sensor_preference.yaml`
- 脚本：
  - `scripts/run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_global_lag_sensor_preference.py`
- 摘要：
  - `reports/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_global_lag_sensor_preference/global_lag_sensor_preference_summary.md`
- JSON：
  - `artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_global_lag_sensor_preference/global_lag_sensor_preference_summary.json`

## 11. 分层 lag cleanroom：少数 L180 偏好点位走长 lag，其余点位保持 L60

### 11.1 本轮目的

- 上一轮已经把问题压到了点位层：
  - `L60` 更像整体最优
  - `L180` 更像边界谨慎性更强
  - 而且两者不是同一批点位在起作用
- 因此本轮正式尝试一个更具体的工作假设：
  - 不是所有点位都用同一个 lag
  - 只让少数 `L180` 偏好点位走长 lag
  - 其余大部分点位仍保留 `L60`

### 11.2 本轮候选组

- base short lag:
  - `L60`
- slow lag only on selected sensors:
  - `hybrid_l180_top3`
    - `FI_C53001_PV_F_CV`
    - `TI_CM511A_PV_F_CV`
    - `TI_CM53201_PV_F_CV`
  - `hybrid_l180_top5`
    - 上述 `top3`
    - `FI_C51005_S_PV_CV`
    - `FIC_C51801_PV_F_CV`
  - `hybrid_l180_top8`
    - 上述 `top5`
    - `TICA_C52601_PV_F_CV`
    - `TI_C51202B_S_PV_CV`
    - `FIC_C51802_PV_F_CV`

### 11.3 控变量方式

- 仍然复用 global lag cleanroom 的共同样本集
- 仍然复用：
  - 每个 outer fold 在 `L0` 上选出的 `selected_sensors`
  - 每个 outer fold 在 `L0` 上选出的 `boundary / warning weight candidate`
- 本轮不重新筛点
- 本轮不改模型
- 本轮只改：
  - 某一小批点位到底取 `L60` 还是 `L180`

### 11.4 pooled 结果

- `L60`:
  - `macro_f1 = 0.3053`
  - `balanced_accuracy = 0.3586`
  - `warning_AP = 0.1795`
  - `unacceptable_AP = 0.1011`
  - boundary high-confidence non-warning `= 0.4848`
- `L180`:
  - `macro_f1 = 0.2886`
  - `balanced_accuracy = 0.3510`
  - `warning_AP = 0.2047`
  - `unacceptable_AP = 0.0860`
  - boundary high-confidence non-warning `= 0.4254`
- `hybrid_l180_top3`:
  - `macro_f1 = 0.3033`
  - `balanced_accuracy = 0.3683`
  - `warning_AP = 0.1860`
  - `unacceptable_AP = 0.0997`
  - boundary high-confidence non-warning `= 0.4696`
- `hybrid_l180_top5`:
  - `macro_f1 = 0.3049`
  - `balanced_accuracy = 0.3670`
  - `warning_AP = 0.1902`
  - `unacceptable_AP = 0.1025`
  - boundary high-confidence non-warning `= 0.4378`
- `hybrid_l180_top8`:
  - `macro_f1 = 0.2881`
  - `balanced_accuracy = 0.3478`
  - `warning_AP = 0.1796`
  - `unacceptable_AP = 0.1042`
  - boundary high-confidence non-warning `= 0.4406`

### 11.5 本轮最重要的结果

- `hybrid_l180_top5` 是当前最有信号的版本。
- 相比纯 `L60`：
  - `macro_f1` 基本持平：
    - `0.3053 -> 0.3049`
  - `balanced_accuracy` 提升：
    - `0.3586 -> 0.3670`
  - `core_AP` 提升：
    - `0.7302 -> 0.7368`
  - `warning_AP` 提升：
    - `0.1795 -> 0.1902`
  - `unacceptable_AP` 也略升：
    - `0.1011 -> 0.1025`
  - boundary high-confidence non-warning 明显下降：
    - `0.4848 -> 0.4378`

### 11.6 如何理解这轮结果

- 这轮第一次出现了一个很像“同时借到两边优点”的版本：
  - 保住了 `L60` 的整体判别
  - 又吸收了一部分 `L180` 的边界谨慎性
- 同时它没有像纯 `L180` 那样明显牺牲 unacceptable 侧。
- 这说明：
  - “统一单 lag”很可能不是当前任务下的最优结构
  - “少数慢点位走长 lag，其余点位走短 lag”开始出现真实信号

### 11.7 需要保留的克制判断

- 这轮还不能直接升级成新的冻结主线。
- 原因是：
  - `hybrid_l180_top5` 的候选点位来自上一轮分析筛选
  - 它仍然属于 hypothesis-driven cleanroom，而不是最终稳定标准
- 但它已经足够强，值得进入下一轮更正式的确认：
  - 现在不该回去继续扩单 lag 网格
  - 而应该把“分层 lag”视为新的主工作假设

### 11.8 当前最合理的下一步

- 下一步不建议立刻再扩更多点位到 `L180`
- 因为 `top8` 已经说明：
  - 长 lag 点位加得过多，会把整体性能重新拉坏
- 更合理的是：
  1. 先围绕 `hybrid_l180_top5` 做确认性实验
  2. 检查这 5 个点位的工艺含义是否一致
  3. 再决定是继续微调其中 1 到 2 个点位，还是进入更正式的分层 lag 建模

### 11.9 本轮产物

- 配置：
  - `configs/ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_stratified_lag.yaml`
- 脚本：
  - `scripts/run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_stratified_lag.py`
- 摘要：
  - `reports/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_stratified_lag/stratified_lag_summary.md`
- JSON：
  - `artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_stratified_lag/stratified_lag_summary.json`

## 12. hybrid_l180_top5 的工艺含义复核与名单敏感性分析

### 12.1 先看 top5 的工艺含义

- `FI_C53001_PV_F_CV`
  - `闪蒸罐胶液进料量`
- `TI_CM511A_PV_F_CV`
  - `R511A搅拌温度`
- `TI_CM53201_PV_F_CV`
  - `V532搅拌温度`
- `FI_C51005_S_PV_CV`
  - `卤化工段胶液总量1`
- `FIC_C51801_PV_F_CV`
  - `ESBO加注量`

### 12.2 对这 5 个点位的当前理解

- 它们并不是完全同一类变量。
- 但有一个共同特征开始变清楚：
  - 这批点位更像“物料通过 / 混合 / 添加 / 停留时间”相关信号
  - 而不是单纯的瞬时快变量
- 换句话说，它们确实有理由比一般点位更容易表现出较长 lag 效应。

### 12.3 top5 名单敏感性实验

- 为了确认 `hybrid_l180_top5` 不是偶然凑出来的组合，本轮又做了一组 leave-one-from-top5 的确认性实验。
- 比较的候选包括：
  - `hybrid_l180_top5`
  - `drop_FI_C53001`
  - `drop_TI_CM511A`
  - `drop_TI_CM53201`
  - `drop_FI_C51005`
  - `drop_FIC_C51801`

### 12.4 敏感性结果

- `hybrid_l180_top5` 仍然是一个很强的平衡方案：
  - `macro_f1 = 0.3049`
  - `balanced_accuracy = 0.3670`
  - `warning_AP = 0.1902`
  - `unacceptable_AP = 0.1025`
  - boundary high-confidence non-warning `= 0.4378`

- 但如果目标偏向“整体最优”，当前更强的是：
  - `hybrid_l180_drop_FI_C53001`
  - 相对 `L60`：
    - `macro_f1 +0.0072`
    - `balanced_accuracy +0.0107`
    - `warning_AP +0.0103`
    - `unacceptable_AP +0.0081`
  - 代价是边界高置信硬切改善幅度没有 `top5` 那么大

- 如果目标偏向“边界收益更稳”，当前更像最佳折中的是：
  - `hybrid_l180_drop_TI_CM53201`
  - 相对 `L60`：
    - `balanced_accuracy +0.0041`
    - `core_AP +0.0095`
    - `warning_AP +0.0129`
    - boundary high-confidence non-warning `-0.0401`
    - `unacceptable_AP` 仅小幅 `-0.0005`

- 相反，`drop_FIC_C51801` 明显把整体与边界一起拉弱，说明：
  - `FIC_C51801_PV_F_CV`
  - 在当前分层 lag 方案里更像是保留项，而不是优先删除项

### 12.5 这轮最重要的判断

- `hybrid_l180_top5` 不是最终名单。
- 当前更合理的理解是：
  - 这 5 个点位里，至少有 `FI_C53001_PV_F_CV` 和 `TI_CM53201_PV_F_CV` 还值得继续裁剪验证
  - 而 `FIC_C51801_PV_F_CV` 更像当前应优先保留的慢点位
- 也就是说，分层 lag 的方向已经得到进一步支持；
  - 但下一步应该从 `top5` 继续往 `top4` 精修，而不是再扩回更大的慢点位集合

### 12.6 当前最合理的下一步

- 下一步建议不再继续做宽搜索。
- 更合适的是围绕两个版本做确认性实验：
  1. `hybrid_l180_drop_FI_C53001`
     - 偏整体最优
  2. `hybrid_l180_drop_TI_CM53201`
     - 偏边界收益与整体折中
- 然后再决定：
  - 是把分层 lag 主线朝“整体更强”推进
  - 还是朝“边界更稳”推进

### 12.7 本轮产物

- 配置：
  - `configs/ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_stratified_lag_top5_sensitivity.yaml`
- 脚本：
  - 继续复用
  - `scripts/run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_stratified_lag.py`
- 摘要：
  - `reports/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_stratified_lag_top5_sensitivity/stratified_lag_summary.md`
- JSON：
  - `artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_stratified_lag_top5_sensitivity/stratified_lag_summary.json`

## 13. 最后一轮确认：两条 top4 分层 lag 候选的正面复核

### 13.1 本轮目的

- 上一轮已经把分层 lag 主线收敛成两个候选：
  1. `hybrid_l180_drop_FI_C53001`
     - 偏整体最优
  2. `hybrid_l180_drop_TI_CM53201`
     - 偏边界收益与整体折中
- 本轮不再扩其他名单
- 只做这两条候选和 `L60 / L180` 的最终同口径确认

### 13.2 pooled 结果

- `L60`:
  - `macro_f1 = 0.3053`
  - `balanced_accuracy = 0.3586`
  - `warning_AP = 0.1795`
  - `unacceptable_AP = 0.1011`
  - boundary high-confidence non-warning `= 0.4848`
- `L180`:
  - `macro_f1 = 0.2886`
  - `balanced_accuracy = 0.3510`
  - `warning_AP = 0.2047`
  - `unacceptable_AP = 0.0860`
  - boundary high-confidence non-warning `= 0.4254`
- `hybrid_l180_drop_FI_C53001`:
  - `macro_f1 = 0.3125`
  - `balanced_accuracy = 0.3693`
  - `warning_AP = 0.1897`
  - `unacceptable_AP = 0.1092`
  - boundary high-confidence non-warning `= 0.4669`
- `hybrid_l180_drop_TI_CM53201`:
  - `macro_f1 = 0.3024`
  - `balanced_accuracy = 0.3627`
  - `warning_AP = 0.1924`
  - `unacceptable_AP = 0.1005`
  - boundary high-confidence non-warning `= 0.4448`

### 13.3 本轮确认结论

- 两条 `top4` 候选都比纯 `L180` 更合理。
- 而且两条都对纯 `L60` 给出了正向增益，但增益方向不同：

- `hybrid_l180_drop_FI_C53001`
  - 是当前“整体最强”的版本
  - 相对 `L60`：
    - `macro_f1 +0.0072`
    - `balanced_accuracy +0.0107`
    - `warning_AP +0.0103`
    - `unacceptable_AP +0.0081`
  - 但边界高置信硬切只改善了一小截：
    - `0.4848 -> 0.4669`

- `hybrid_l180_drop_TI_CM53201`
  - 是当前“边界与整体最均衡”的版本
  - 相对 `L60`：
    - `balanced_accuracy +0.0041`
    - `core_AP +0.0095`
    - `warning_AP +0.0129`
    - boundary high-confidence non-warning `0.4848 -> 0.4448`
    - `unacceptable_AP` 基本持平，仅 `-0.0005`
  - 代价是 `macro_f1` 略低于 `L60`

### 13.4 当前最清楚的判断

- 分层 lag 这条线已经被确认有价值。
- 而且现在已经不只是“方向上有信号”，而是：
  - 明确出现了比纯 `L60` 更好的混合版本
- 当前可以把分层 lag 主线收成两个口径：
  1. 如果偏向总体判别最强：
     - `hybrid_l180_drop_FI_C53001`
  2. 如果偏向边界更稳、同时尽量不伤 unacceptable：
     - `hybrid_l180_drop_TI_CM53201`

### 13.5 当前建议

- 如果要继续往现场或后续建模推进，我建议优先把：
  - `hybrid_l180_drop_TI_CM53201`
  作为下一步主工作版本
- 原因是它更符合这一路实验的原始目标：
  - 不只是追求整体分数
  - 还要更诚实地处理边界模糊
- 同时它没有像纯 `L180` 那样明显牺牲 unacceptable 侧。

### 13.6 本轮产物

- 配置：
  - `configs/ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_stratified_lag_final_confirmation.yaml`
- 脚本：
  - 继续复用
  - `scripts/run_ordinal_cumulative_current_head_monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_stratified_lag.py`
- 摘要：
  - `reports/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_stratified_lag_final_confirmation/stratified_lag_summary.md`
- JSON：
  - `artifacts/monotonic_120min_boundary_weighted_inner_thresholds_identity_dedup_stratified_lag_final_confirmation/stratified_lag_summary.json`
