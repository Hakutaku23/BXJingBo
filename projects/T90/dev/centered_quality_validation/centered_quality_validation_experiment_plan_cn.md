# Cleanroom 实验计划——风险约束下的中心质量目标验证

## 目的

本实验用于验证 T90 的一种新任务定义：

> 不再把“只要在规格内都一样好”当作默认前提，
> 而是把质量目标拆成：
> 1. 规格风险约束
> 2. 围绕目标中心值的质量优选程度

核心问题是：

在当前 T90 场景下，**带风险约束的中心质量目标**，是否比当前基于阈值的 current-head 分类更符合真实工艺目标？

这条线应作为一个新的 cleanroom 分支。  
它不是 ordinal/cumulative 分支的简单延长，也不是 distributional/reject 分支的直接续写。

---

## 为什么值得单开一个分支

前面的 cleanroom 已经说明：

- hard 三分类在边界附近过于脆弱；
- ordinal / cumulative 方向上更合理，但仍然不够理想；
- distributional + conformal + reject 更诚实地表达了不确定性，但 operational trade-off 仍然难处理。

现在更深一层的问题是：

- 真实工艺目标并不是“落在 8.2–8.7 内都一样好”；
- 实际控制上，更靠近目标中心（例如 8.45）通常优于只是落在允许区间内；
- 因此，现有任务定义可能仍然把一个连续质量梯度，压成了粗糙的可行域判断。

这就需要一个新分支，显式建模：
- 是否越界 / 是否有风险
- 以及是否足够靠近最优中心值

---

## 建议新目录

建议新开目录：

`projects/T90/dev/centered_quality_validation/`

原因：
- 这条思路改变的是任务目标本身；
- 不宜太早和现有 ordinal/cumulative 或 conformal/reject 分支混在一起；
- 后续很可能会形成独立的计划、配置、脚本和报告。

---

## 核心假设

如果真实工艺确实更偏好目标中心，而不仅仅是“在规格区间内”，那么中心质量目标应当能够：

- 区分“只是合格”与“真正优质”；
- 避免把所有 in-spec 样本压成一个扁平类别；
- 更符合工艺人员对好坏的真实理解；
- 为后续告警和优化决策提供更自然的基础。

---

## 实验范围

本阶段只验证一个方法家族：

- 风险约束下的中心质量建模

本阶段不把以下内容作为主 treatment：

- paper-faithful EWMA
- PH augmentation
- future-head
- multiscale weighted fusion
- 深度模型
- 复杂端到端策略优化

这些都可以放到后续，只在任务定义先验证通过后再考虑。

---

## 任务定义

### 当前业务背景
当前工艺目标可写为：

- 目标中心值：`8.45`
- 容差半宽：`0.25`
- 允许区间：`8.20–8.70`

需要验证的是，过程真实目标更接近：
- 单纯阈值约束
还是
- 带中心偏好的 desirability 目标。

### 新 formulation
建议把任务拆成两个耦合目标：

#### 目标 A —— 风险 / 可行性
预测样本是否越界，或是否逼近规格边界。

推荐输出：
- `P(y < 8.2)`
- `P(y > 8.7)`
或等价的风险量。

#### 目标 B —— 中心质量分数
刻画当前样本相对目标中心的“优质程度”。

推荐中心值：
- `target_center = 8.45`

推荐第一版 desirability：
- 在 `8.45` 附近最高
- 随着 `|y - 8.45|` 增大而下降
- 在规格边界附近已经明显变差
- 规格外可裁剪为低值或 0

---

## 推荐目标设计

本分支至少比较以下三种目标设计。

### Baseline A —— 冻结版阈值参考线
沿用当前冻结的 threshold-oriented strongest cleanroom 作为对照锚点。

### Treatment B —— 仅做中心质量分数
只建一个连续的 centered-quality score，例如：

#### 线性 desirability
`q(y) = max(0, 1 - |y - 8.45| / 0.25)`

含义：
- `q(8.45) = 1`
- 越靠近边界分数越低
- 越界后可直接为 0

也允许使用更平滑的形式，但必须写入审计说明：
- Gaussian-like desirability
- 二次型 utility
- 钟形 desirability

### Treatment C —— 风险 + 中心质量联合目标
这是本阶段的主 treatment。

同时建：
- 风险输出
- 中心质量分数

最终由这两类输出共同决定业务结论。

---

## 为什么不建议把“优质率”当成唯一主目标

可以定义一个“优质 / 非优质”的二分类作为辅助诊断，但不建议把它当成唯一目标。

原因：
- 它仍然会丢掉距离信息；
- 很可能引入一个新的稀有分类问题；
- 表达能力不如连续的 desirability 分数。

因此，若做 premium rate，也应当是辅助对照，不应替代主目标。

---

## 中心质量分数设计

这一部分必须在实验开始前冻结。

### 推荐第一版
使用线性分数：

`q(y) = max(0, 1 - |y - center| / tolerance_half_width)`

其中：
- `center = 8.45`
- `tolerance_half_width = 0.25`

### 可选第二版
使用平滑分数：

`q(y) = exp(- (y - center)^2 / (2 * sigma^2))`

若使用该形式，`sigma` 必须在实验开始前固定，不允许在 outer test 上调参。

---

## 风险目标设计

第一版保持简单、可审计。

### 推荐第一版
使用两个二元风险头：
- 低侧越界风险
- 高侧越界风险

### 为什么这样做
低侧和高侧在真实工艺中可能风险不对称，拆成两个方向更容易解释，也比单一 out-of-spec 标签更有信息量。

---

## 业务决策映射

比较前必须固定业务映射规则。

### 推荐第一版业务状态
输出四类：

- `unacceptable`
- `acceptable`
- `premium`
- `retest`

### 推荐第一版映射逻辑
1. 若低侧或高侧越界风险超过硬阈值 -> `unacceptable`
2. 否则若中心质量分数很高 -> `premium`
3. 否则若风险较低但质量分数一般 -> `acceptable`
4. 否则若不确定性过高或风险与质量信号冲突 -> `retest`

这里只是推荐结构。  
具体阈值必须在实验前固定并写进审计说明。

### 说明
本分支不是在替代现场控制逻辑。  
它是在验证：相对于阈值式建模，中心质量目标是否更接近真实工艺目标。

---

## 特征策略

本阶段仍然优先验证“目标定义”，而不是高级特征工程。

### 第一轮允许的特征族
继续使用当前最强 simple baseline 表示：

- simple 120min causal window stats
- sensor-identity de-dup
- fold-local screening

例如：
- mean
- std
- min
- max
- last
- range
- delta

本阶段不引入 EWMA。

### 为什么先不引入 EWMA
如果同时改目标和特征表示，就无法归因。  
这一阶段必须先回答：**中心质量目标本身有没有价值。**

---

## 特征筛选策略

沿用现有 cleanroom 规则。

### 阶段 1：无监督预清洗
允许在全量 sensor pool 上：
- 删除常值列
- 删除近常值冻结列
- 删除重复列
- 删除极高缺失列

不能使用标签信息。

### 阶段 2：监督筛选
只能在每个训练折内部做：
- 单变量打分
- fixed top-k
- 或 fixed score-threshold

要求：
- 训练折内拟合
- 测试折不能参与
- 同一折内所有被比较方法共享同一套传感器集合

### 推荐第一轮 top-k
- `topk_sensors = 40`

---

## 模型

保持模型简单，以隔离目标设计本身的作用。

允许的第一轮模型：
- ridge / linear regression 用于中心质量分数
- logistic 用于风险头
- random forest 作为稳健性复核

如果实现简单且可审计，也可使用轻量多输出包装。  
否则，第一版使用多个独立简单头即可。

本阶段不引入复杂模型家族。

---

## 决策时刻与因果性

- `decision_time` 仍定义为当前化验时间
- 所有特征必须因果
- 不允许使用未来信息
- 继续使用当前 strongest simple baseline 的 120min causal window

这一阶段不是为了优化 lag/window。  
它是为了验证任务定义。

---

## 验证方式

统一采用时间顺序验证：

- 默认 `TimeSeriesSplit(n_splits=5)`
- 若样本不足可降到 4，但不得低于 3

不能随机打乱。

若能识别工况切换：
- 记录样本窗口是否跨越 regime boundary
- 尽可能报告非跨工况子集表现

---

## 指标体系

本阶段必须同时评价：
- 风险行为
- 中心质量目标是否有用

### 风险指标
- lower-risk AP
- upper-risk AP
- unacceptable recall
- unacceptable miss rate

### 中心质量指标
- desirability score 的 MAE / RMSE
- 预测 desirability 与真实 desirability 的排序相关性
- 若定义 premium 诊断阈值，则报告 premium precision / recall

### 决策指标
若启用四类业务状态，报告：
- `unacceptable / acceptable / premium / retest` 的 macro_f1
- 若存在 retest，则 covered decision 的 balanced_accuracy
- premium precision
- unacceptable false-clear rate

### 诊断指标
- in-spec 区间内的预测 desirability 分布
- 模型是否真的偏好中心而不是边缘
- 接近 8.45 的样本是否确实得到比接近 8.2 / 8.7 更高的预测质量

---

## 成功标准

当大部分条件成立时，可认为本分支得到支持：

- 中心质量目标能稳定地区分“中心优质”与“边缘合格”；
- 风险 + desirability 的联合 formulation 比单纯阈值分类更符合工艺直觉；
- premium-like 样本能被识别，而 unacceptable false-clear 风险不显著变坏；
- 决策层仍然简单、可解释、可审计；
- 收益不是只出现在单一幸运折。

若不满足，则结论应写为：

- 当前数据条件下，中心质量目标还不足以替代现有冻结的 threshold-oriented 参考线。

---

## 本阶段不能得出的结论

本分支不能宣称：
- ordinal / cumulative 无用
- conformal / reject 无用
- EWMA 无用
- future-head 不需要

它只能回答：

- 中心质量目标是否更符合真实工艺目标；
- 联合风险 + desirability 是否比单纯阈值逻辑更有意义。

---

## 必须产出的文件

- `centered_quality_results.csv`
- `centered_quality_summary.json`
- `centered_quality_feature_rows.csv`
- `centered_quality_audit.md`

### 审计说明必须写清
- 使用的 target center 是多少
- desirability 公式是什么
- 是否包含 premium 诊断
- 风险头如何定义
- 决策映射是什么
- 监督筛选是否仅在训练折内完成
- 明确忽略了哪些历史假设

---

## 建议新目录中的第一批产物

- `plans/centered_quality_validation.md`
- `configs/centered_quality_current_head.yaml`
- `scripts/run_centered_quality_current_head.py`
- `reports/centered_quality_current_head_audit.md`

---

## 下一阶段（仅当本阶段成功时）

如果中心质量分支成功，下一步才比较：

- centered-quality + simple 120min stats
vs
- centered-quality + paper-faithful EWMA
或
- centered-quality + PH lag-aware augmentation

那时再去比较表示层增强，才有意义。

---

## 目录建议

建议使用独立目录：

`projects/T90/dev/centered_quality_validation/`

因为这条分支改变的是任务目标本身，应该先与现有 cleanroom 主线隔离，直到价值清晰。
