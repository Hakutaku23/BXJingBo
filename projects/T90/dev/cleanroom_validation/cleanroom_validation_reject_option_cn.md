# Cleanroom 实验计划——分布式区间预测 + 拒判 / 复检决策验证

## 目的

本 cleanroom 实验只回答一个窄问题：

在当前 T90 问题设定下，相比当前的阈值式 ordinal / cumulative 主线，**分布式区间预测 + 显式拒判 / 复检** 是否更适合 current-head 的边界敏感决策？

这份计划将 **预测不确定性与拒判机制** 作为下一阶段的主验证目标。  
它默认项目已经从前一阶段 cleanroom 中学到：

- hard 三分类在边界附近过于脆弱；
- ordinal / cumulative 方向正确，但效果仍不够理想；
- 只要模型最终仍被迫输出一个确定类别，就会重新把模糊边界样本硬切开；
- 当前最强且最稳的表示层，仍然是 simple window stats。

本阶段不把 EWMA、PH 增强、future-head 作为主验证对象。

---

## 为什么这一步应当接在 ordinal / cumulative 之后

上一阶段已经说明：任务定义确实重要。  
但它也暴露了一个重复出现的限制：

- 边界友好型 formulation 确实改善了部分决策行为；
- 但模型最后仍然要塌缩成一个离散类；
- 这会再次把真正模糊的边界样本硬切掉。

在当前项目条件下：

- 大量样本聚集在 8.2 与 8.7 附近；
- 化验结果存在检测误差；
- 靠近阈值的相邻样本不应得到完全不同的硬标签；
- 实际业务更接近“告警 / 可接受性判断”，而不是纯分类。

因此下一步更合理的问题应当是：

> 是否应当把问题从“阈值分类”进一步推进到“区间分布预测”，并在决策层正式允许模型对不确定样本输出拒判 / 复检”？

所以本阶段会把目标从：
- hard class prediction  
或
- cumulative threshold probabilities

推进到：
- 对有序 T90 区间的概率分布预测，
- 再叠加显式的 reject / retest 决策层。

---

## 实验边界

本阶段只验证一个方法家族：

- 分布式区间预测 + reject option

本阶段不包括：

- paper-faithful EWMA 作为主 treatment
- multiscale weighted fusion
- PH lag-aware augmentation 作为主 treatment
- future-head
- 深度模型
- 复杂序列网络
- stage-aware 分支建模作为主线

这些都可以成为后续 cleanroom 分支，但不属于当前第一优先。

---

## 核心假设

如果当前剩余的主要困难更像是**边界不可约模糊性**，而不是模型能力不足，那么分布式建模 + reject option 应当能够：

- 降低边界附近的高置信硬错误；
- 在不恶化核心合格区判别的前提下，更诚实地表达 8.2 / 8.7 附近的不确定性；
- 让业务上更自然地区分：
  - 可直接判合格；
  - 可直接判不合格；
  - 需要预警 / 复检。

---

## 任务定义

- 任务：current-head 决策建模
- 标签来源：当前化验样本
- 目标：验证“区间分布预测 + 拒判”是否比当前阈值式 cumulative 更合理

### 需要保留的参考 baseline
上一阶段冻结的参考主线是：

- simple 120min causal stats
- fold-local topk sensor screening
- sensor-identity de-dup
- monotonic ordinal / cumulative
- 仅对 8.2 / 8.7 做内侧阈值对称硬加权

这条线是对照锚点，不是本阶段要重写的对象。

### 新 cleanroom treatment 任务
新 treatment 直接预测有序 T90 区间的概率分布。

推荐第一版区间定义：

- `bin_1: y < 8.0`
- `bin_2: 8.0 <= y < 8.2`
- `bin_3: 8.2 <= y < 8.7`
- `bin_4: 8.7 <= y < 8.9`
- `bin_5: y >= 8.9`

这些区间在一次 cleanroom run 内必须固定不变。

---

## 标签设计

本阶段不应再以 one-hot 作为主要监督目标。  
应当使用 **soft distribution label**。

### 原则
如果样本靠近阈值，它的监督信号不应全部压在单个 bin 上。

例如：
- 靠近 8.2 的样本，可同时在 `bin_2` 与 `bin_3` 上分配概率质量；
- 靠近 8.7 的样本，可同时在 `bin_3` 与 `bin_4` 上分配概率质量。

### 推荐第一版实现
使用一个简单、固定的局部分布规则，把观测到的 T90 值映射到 5 个 bin 的软标签。

可接受的第一版实现包括：
- 邻近 bin 上的三角形 soft label
- 基于 bin center 的 Gaussian-like kernel
- 在阈值附近做局部线性概率重分配

具体规则必须在实验开始前固定，并写入审计说明。

### 硬约束
soft-label 规则只能依赖：
- 当前样本的 T90 值
- 以及固定的 bin 定义

不能根据 outer test 结果去调。

---

## 要比较的方法家族

本阶段只比较以下三类 formulation。

### Baseline A —— 冻结版 ordinal / cumulative 参考线
直接沿用上一阶段冻结的 strongest cleanroom 主线。

### Baseline B —— 不带 reject 的分布式区间预测
模型输出 5-bin 分布，但最终仍强制塌缩成：
- acceptable
- warning
- unacceptable

这一步用来隔离“仅仅改成 distribution target 是否已经有帮助”。

### Treatment C —— 带 reject / retest 的分布式区间预测
模型输出同样的 5-bin 分布，但决策层允许第四类：
- acceptable
- warning
- unacceptable
- retest

这是本阶段的主 treatment。

---

## 推荐实现方式

优先级是可审计，不是复杂度。

### 方案 1：直接对 5-bin 做概率预测
用一个简单概率模型直接预测 5 个区间的概率分布。

允许：
- multinomial logistic
- balanced random forest probabilities

如果模型不直接支持 soft target，可用轻量可审计的近似方式：
- 基于样本复制和权重的近似 soft supervision
- 轻量级 custom cross-entropy
- 固定目标向量的 label smoothing 式训练

### 方案 2：通过 ordered 概率重构区间分布
如果代码基础里已经有简单可用的 ordered reconstruction，也可采用。

但第一版 cleanroom 应优先选择更直接、可审计的方案。

---

## Reject / Retest 决策层

这是本阶段最关键的新组件。

模型必须允许对不确定样本“拒判”。

### 必须输出的中间结果
至少输出：
- `P(bin_1)`
- `P(bin_2)`
- `P(bin_3)`
- `P(bin_4)`
- `P(bin_5)`

然后导出：
- `acceptable_prob = P(bin_3)`
- `warning_prob = P(bin_2) + P(bin_4)`
- `unacceptable_prob = P(bin_1) + P(bin_5)`

### 新增业务状态
增加：
- `retest`

### 推荐第一版 reject 规则
用一个简单、固定、可解释的规则决定是否拒判。

例如满足以下任一条件则输出 `retest`：
- top class probability 低于固定阈值；
- 概率质量跨越关键边界两侧；
- entropy 高于固定阈值；
- acceptable 与 unacceptable 概率同时不小；
- warning 概率较高但又不足以直接判 warning。

具体规则必须在比较前固定。

### 硬约束
reject rule 的参数不能在 outer test 上调。  
它只能：
- 在 train fold 内决定；
- 或预先固定。

---

## 特征策略

这一阶段仍然优先验证输出形式，而不是高级特征工程。

### 第一轮允许的特征族
直接使用当前 strongest simple baseline：

- simple 120min causal window stats

例如：
- mean
- std
- min
- max
- last
- range
- delta

本阶段不引入 EWMA。

### 为什么先不加 EWMA
如果同时改输出形式和特征表示，就无法归因。  
这一阶段必须先回答：**仅靠“分布预测 + reject”是否已经比 threshold classification 更合适。**

---

## 特征筛选策略

沿用当前 cleanroom 规则。

### 阶段 1：无监督预清洗
允许在全量 raw sensor pool 上做：
- 删除常值列
- 删除近常值冻结列
- 删除重复列
- 删除极高缺失列

不能用标签信息。

### 阶段 2：监督筛选
只能在每个训练折内部做：
- 单变量打分
- fixed top-k
- 或 fixed score-threshold

要求：
- 训练折内拟合
- 测试折不能参与
- 所有被比较的方法在同一折内必须共享同一套传感器集合

### 推荐 top-k
先保持当前强 reference：
- `topk_sensors = 40`

第一轮不把 top-k 扫描作为主实验轴。

---

## 模型

保持模型家族简单，以隔离 formulation 本身的作用。

允许：
- multinomial logistic
- balanced random forest

推荐第一轮：
- 用 multinomial logistic 做 5-bin 分布预测主实验
- balanced random forest 作为后续稳健性复核

本阶段不引入新的复杂模型家族。

---

## 决策时刻与因果性

- `decision_time` 仍定义为当前化验时间
- 所有特征必须因果
- 不得使用未来信息
- 继续使用当前 strongest reference 的 120min causal window

这一阶段不是用来优化 lag/window 的。  
它是用来验证：在当前最强 simple 表示之上，输出形式能否进一步改进。

---

## 验证方式

统一采用时间序列验证：

- 默认 `TimeSeriesSplit(n_splits=5)`
- 若样本不足可降到 4，但不得低于 3

不能随机打乱。

若能识别工况切换：
- 记录样本窗口是否跨越 regime boundary
- 尽可能报告非跨工况子集表现

---

## 指标体系

本阶段必须同时评价：
- 决策质量
- 不确定性是否有用

### 主要决策指标
针对非拒判样本：
- macro_f1
- balanced_accuracy
- warning average precision
- unacceptable average precision

### Coverage / Abstention 指标
必须输出：
- reject rate
- decision coverage
- covered-sample macro_f1
- non-rejected unacceptable miss rate
- boundary-neighborhood reject rate

### 分布质量指标
必须输出：
- 5-bin multiclass Brier score
- calibration error（若实现可行）
- multiclass log loss（若实现可行）
- entropy 统计

### 边界诊断
至少在以下区间做子集诊断：
- `7.9–8.3`
- `8.6–8.8`

每个边界子集至少报告：
- reject rate
- high-confidence non-warning rate
- unacceptable false-clear rate
- acceptable confidence behavior

---

## 业务决策映射

业务层规则必须在比较前固定。

### 推荐第一版映射
由 5-bin 分布先得到：
- acceptable = `P(bin_3)`
- warning = `P(bin_2) + P(bin_4)`
- unacceptable = `P(bin_1) + P(bin_5)`

然后先应用 reject rule：
- 若满足 reject 条件 -> `retest`
- 否则在 acceptable / warning / unacceptable 中取最大概率类

### 为什么要这样做
因为这可以让决策层：
- 简单
- 可解释
- 并能直接和上一阶段 frozen reference 对照

---

## 成功标准

若大部分条件同时满足，则可认为“distributional + reject”得到支持：

- 边界区的高置信硬错误下降；
- reject 集合主要集中在真正模糊样本，而不是容易样本；
- covered-sample macro_f1 不低于当前冻结主线；
- non-rejected unacceptable miss risk 不劣化，最好还能下降；
- calibration / Brier 行为改善或至少可接受；
- 收益不是只出现在单一幸运折。

若不满足，则结论应写为：

- 当前 simple baseline 下，加入 distributional + reject 还不足以替代冻结版 ordinal / cumulative 参考线。

---

## 本阶段不能得出的结论

本阶段不能宣称：
- EWMA 无用
- PH 无用
- future-head 无用
- ordinal / cumulative 是错误方向
- fuzzy 思想不需要

它只能回答：

- interval-distribution prediction 是否比 threshold-only classification 更合适；
- 显式 reject / retest 是否能改善当前 simple baseline 下的决策质量。

---

## 必须产出的文件

- `distributional_reject_results.csv`
- `distributional_reject_summary.json`
- `distributional_reject_feature_rows.csv`
- `distributional_reject_audit.md`

### 审计说明必须写清
- 5-bin 定义是什么
- soft-label 规则是什么
- 分布目标是直接建模还是重构得到
- reject rule 是什么
- reject 阈值如何选择
- 监督筛选是否仅在训练折内完成
- 明确忽略了哪些历史假设

---

## 推荐在 cleanroom 目录中的第一批产物

- `plans/distributional_reject_option_validation.md`
- `configs/distributional_reject_option_current_head.yaml`
- `scripts/run_distributional_reject_option_current_head.py`
- `reports/distributional_reject_option_current_head_audit.md`

---

## 下一阶段（仅当本阶段成功时）

如果本阶段成功，下一轮 cleanroom 才比较：

- distributional + reject + simple 120min stats
vs
- distributional + reject + paper-faithful EWMA
或
- distributional + reject + PH lag-aware augmentation

那时才能真正隔离“表示增强是否在更合理输出形式下仍有增益”。

---

## 目录建议

继续使用：

`projects/T90/dev/cleanroom_validation/`

因为 cleanroom 原则本身没有变，只是当前主验证对象变了。

---

# 2026-04-01 首轮执行记录：Baseline B（distributional prediction without reject）

## 本轮执行内容

按这条新分支冻结的首轮路线，我先没有直接上 reject / retest，而是先完成了：

- Frozen Baseline A：
  - 当前冻结 ordinal / cumulative 主线
- Baseline B：
  - 5-bin distributional interval prediction
  - 不带 reject
  - 最终仍塌缩为 `acceptable / warning / unacceptable`

本轮保持与冻结主线一致的比较边界：

- `merge_data.csv + merge_data_otr.csv`
- strict sensor-identity de-dup
- `simple 120min causal window stats`
- fold-local `topk=40`
- `TimeSeriesSplit(5)`

## 首轮实现细节

### 1. 5-bin 定义

- `bin_1: y < 8.0`
- `bin_2: 8.0 <= y < 8.2`
- `bin_3: 8.2 <= y < 8.7`
- `bin_4: 8.7 <= y < 8.9`
- `bin_5: y >= 8.9`

### 2. soft-label 规则

首轮采用了计划中冻结的“只软化内侧阈值”：

- 外侧 `8.0 / 8.9` 保持硬边界
- 仅对 `8.2 / 8.7` 做局部线性 soft label
- `soft_radius = 0.05`

因此：

- 落在 `8.2 ± 0.05` 的样本，在 `bin_2 / bin_3` 之间分配质量
- 落在 `8.7 ± 0.05` 的样本，在 `bin_3 / bin_4` 之间分配质量
- 其余样本仍是所在 bin 的 hard assignment

本轮统计到：

- `soft_8_2` 样本：`201`
- `soft_8_7` 样本：`235`
- 其余 hard 样本：`2290`

### 3. 模型实现

首轮使用：

- `5-bin multinomial logistic`
- 通过 non-zero soft bins 的加权样本复制，近似实现 soft supervision

### 4. 对照线实现

Frozen Baseline A 不是直接引用历史结果，而是在同一轮 cleanroom 中重跑：

- 相同 outer `TimeSeriesSplit(5)`
- 相同 fold-local sensor screening
- 相同每折的 DCS 特征集合
- 相同 frozen reference 的权重候选选择逻辑

这样 A/B 的比较是干净的。

## 总体结果

pooled outer-test 结果如下：

- Frozen Baseline A `macro_f1 = 0.2674`
- Baseline B `macro_f1 = 0.2269`
- Frozen Baseline A `balanced_accuracy = 0.3580`
- Baseline B `balanced_accuracy = 0.3107`
- Frozen Baseline A `core_AP = 0.7215`
- Baseline B `core_AP = 0.6981`
- Frozen Baseline A `warning_AP = 0.1963`
- Baseline B `warning_AP = 0.1784`
- Frozen Baseline A `unacceptable_AP = 0.0850`
- Baseline B `unacceptable_AP = 0.0777`

这说明一件事非常清楚：

> “只把监督目标改成 distributional，但仍然强制塌缩成最终业务类”，在当前 simple baseline 下并没有打赢冻结主线。

## 分布质量结果

虽然决策指标明显退化，但分布质量指标反而有改善：

- Frozen A multiclass Brier = `1.1375`
- Baseline B multiclass Brier = `1.1137`
- Frozen A multiclass log loss = `13.1879`
- Baseline B multiclass log loss = `5.9524`

同时，Baseline B 的 entropy 更高：

- Frozen A entropy mean = `0.3255`
- Baseline B entropy mean = `0.3978`

这组现象的含义是：

1. Baseline B 的输出更“分布化”
2. 它对边界样本更不愿意做极端集中判断
3. 但这种更柔和的分布，在“仍然被迫塌缩成最终业务类”时，并没有转化成更好的 business decision

换句话说：

> Baseline B 失败的方式不是“完全没学到东西”，而是“分布质量有信号，但塌缩决策质量不够”。

## 边界行为

这一轮最值得注意的地方在边界行为。

整体边界区高置信非 `warning` 比例：

- Frozen A：`0.4848`
- Baseline B：`0.4061`

两个边界子区间也都更谨慎：

### `7.9–8.3`

- Frozen A high-conf non-warning = `0.5142`
- Baseline B high-conf non-warning = `0.3962`

### `8.6–8.8`

- Frozen A high-conf non-warning = `0.4727`
- Baseline B high-conf non-warning = `0.4102`

也就是说，Baseline B 的确吸收了一部分“别在边界上太硬判”的性质。

但这份收益是有代价的：

- 它不是在“保持主指标不掉”的前提下拿到的
- 而是靠明显牺牲整体 business discrimination 换来的

## 逐折观察

按 5 折看，Baseline B 并不是每折都同样失败：

1. `fold 2`
   - `balanced_accuracy` 略高于 Frozen A
   - 边界高置信非 `warning` 显著下降
2. `fold 3 / 4`
   - 边界谨慎性改善明显
   - 但主指标仍偏弱
3. `fold 1 / 5`
   - 决策指标退化明显
   - 其中 `fold 5` 的退化最重

所以当前更精确的判断不是“这条路完全错误”，而是：

- distributional supervision 本身确实改变了输出风格
- 但如果没有 reject layer，它在 current-head 下还不足以成为更好的业务决策器

## 当前阶段结论

这轮 `Baseline B` 给出的结论可以明确写成：

1. 当前 simple baseline 下，**distributional prediction without reject 不被支持为冻结主线的替代方案**。
2. 但它提供了一个重要的新证据：
   - 分布化监督确实能降低边界上的高置信硬判
   - 只是这些收益无法在“必须强制输出最终业务类”时保留下来
3. 因此，这轮结果反而更支持新计划里的主旨：

> 如果这条线还有价值，那么关键不在 Baseline B 本身，而在 **reject / retest 层是否能把这部分边界不确定性真正转化成有用决策**。

## 对下一步的影响

根据这条分支一开始冻结的执行顺序，`Baseline B` 已经完成。  
当前最自然的下一步就是：

- 进入 `Treatment C`
- 即：同样的 5-bin distributional prediction
- 但加入显式 `retest`

不过这轮结果也提醒了一个边界：

- 不应把 `Treatment C` 展开成大规模复杂调参
- 应当先跑一版最简单、最可审计的 reject 规则

## 本轮新增产物

- 配置：`configs/distributional_reject_option_current_head.yaml`
- 脚本：`scripts/run_distributional_reject_option_current_head.py`
- 逐样本结果：`artifacts/distributional_reject_option_current_head_baseline_b/distributional_reject_results.csv`
- 特征行：`artifacts/distributional_reject_option_current_head_baseline_b/distributional_reject_feature_rows.csv`
- 汇总 JSON：`artifacts/distributional_reject_option_current_head_baseline_b/distributional_reject_summary.json`
- 逐折结果：`artifacts/distributional_reject_option_current_head_baseline_b/distributional_reject_per_fold.csv`
- 参考线每折权重候选：`artifacts/distributional_reject_option_current_head_baseline_b/distributional_reject_reference_candidate_per_fold.csv`
- 摘要报告：`reports/distributional_reject_option_current_head_baseline_b/distributional_reject_summary.md`

---

# 2026-04-01 第二轮执行记录：Treatment C（distributional + simple reject / retest）

## 本轮执行内容

在 `Baseline B` 完成后，我继续按冻结路线做了最简单、最可审计的一版 `Treatment C`：

- 基础分布模型与 `Baseline B` 完全相同
- 只在决策层增加：
  - `max_business_prob`
  - 5-bin normalized entropy
- 若满足以下任一条件则输出 `retest`：
  - `max_business_prob < tau_conf`
  - `normalized_entropy > tau_entropy`

首轮阈值网格：

- `tau_conf in {0.45, 0.50, 0.55, 0.60}`
- `tau_entropy in {0.80, 0.85, 0.90, 0.95}`

并保持了最初冻结的 guardrail：

- `reject_rate <= 0.25`

所有阈值都只允许在 outer-train 内通过 inner `TimeSeriesSplit(3)` 选择。

## 最关键结果

这轮最关键的结果不是某个指标，而是：

> 5 个 outer fold 最终全部选择了 `default_no_reject`。

也就是说：

- `fold 1`: `default_no_reject`
- `fold 2`: `default_no_reject`
- `fold 3`: `default_no_reject`
- `fold 4`: `default_no_reject`
- `fold 5`: `default_no_reject`

对应 pooled outer-test 结果也因此完全退化成了 `Baseline B` 本身：

- `reject_rate = 0.0000`
- `decision_coverage = 1.0000`
- `covered_macro_f1 = 0.2269`
- `covered_balanced_accuracy = 0.3107`
- `boundary_reject_rate = 0.0000`
- 边界区高置信非 `warning` 比例 = `0.4061`

换句话说：

> 在当前这版最简单 reject 规则下，训练内并没有找到任何一组阈值，能够在 guardrail 下比“不拒判”更优。

## 这说明了什么

这轮结果的含义比“Treatment C 没提升”更具体：

1. `Baseline B` 身上的那部分“边界更谨慎”的性质，确实存在。
2. 但当前这版 reject 规则并没有把这种谨慎性转化成有用的 operational abstention。
3. 在训练内看来：
   - 一旦开始拒判，就不值得
   - 最优动作反而总是“不要拒判”

所以当前不应把结论写成“reject option 没意义”，而应写成：

> `distributional + simple confidence/entropy reject` 在当前 simple baseline 下未被支持。

## 与 Baseline B 的关系

到这一步，这条新分支的首轮判断已经很清楚：

### Baseline B

- 改善了边界谨慎性
- 改善了部分 distribution-quality 指标
- 但 business decision 明显退化

### Treatment C

- 并没有把这些“更柔和的分布输出”转化成有效的拒判决策
- 训练内每折都自动回退到 `default_no_reject`

也就是说：

> 首轮分支并没有支持“换成 distributional + reject 就比当前冻结 ordinal / cumulative 更好”。

## 当前阶段结论

到现在为止，这条新分支的 first run 可以收紧成：

1. 仅做 distributional supervision，不足以替代冻结主线。
2. 在当前最简单 reject 规则下，reject 层也没有被支持。
3. 因此，这一整轮 `distributional + reject option` 的 first pass，当前应记为：
   - **not yet supported under the current simple implementation**

## 为什么这个结论仍然有价值

虽然结果没有支持新分支，但它并不是“白做”：

1. 它帮助排除了一个很自然但可能误导人的乐观判断：
   - “只要把目标改成 distributional，再加个简单 reject，大概就会更好”
2. 它说明当前问题并不是只缺一个简单 abstain rule。
3. 它也说明：
   - 这条分支若要继续，必须换 reject 设计本身
   - 不能只在当前 `confidence + entropy` 网格上继续小修小补

## 当前建议

基于首轮完整结果，我不建议立刻在这条分支上继续做大量阈值微调。  
更合适的判断应当是：

1. 先把这一轮作为 first-pass cleanroom 完整收档。
2. 当前分支状态记为：
   - `not yet supported`
3. 若后续还要继续这条线，应当重新设计 reject policy 本身，而不是继续扩当前阈值网格。

## 本轮新增产物

- 配置：`configs/distributional_reject_option_current_head_treatment_c.yaml`
- 脚本：`scripts/run_distributional_reject_option_current_head_treatment_c.py`
- 逐样本结果：`artifacts/distributional_reject_option_current_head_treatment_c/distributional_reject_results.csv`
- 特征行：`artifacts/distributional_reject_option_current_head_treatment_c/distributional_reject_feature_rows.csv`
- 汇总 JSON：`artifacts/distributional_reject_option_current_head_treatment_c/distributional_reject_summary.json`
- 逐折结果：`artifacts/distributional_reject_option_current_head_treatment_c/distributional_reject_per_fold.csv`
- 每折选中的 reject 候选：`artifacts/distributional_reject_option_current_head_treatment_c/distributional_reject_selected_candidate_per_fold.csv`
- inner 搜索明细：`artifacts/distributional_reject_option_current_head_treatment_c/distributional_reject_inner_search.csv`
- 摘要报告：`reports/distributional_reject_option_current_head_treatment_c/distributional_reject_summary.md`
