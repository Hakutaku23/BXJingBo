# 标签-X 受控矩阵实验主记录

## 记录约定

- 本文件作为“标签语义 vs X 侧特征工程”验证线的统一主记录。
- 后续与这条验证线相关的实验，统一追加到本文件。
- 文件编码使用 UTF-8。

---

## 实验 1：第一轮受控矩阵

### 目的

回答两个问题：

1. `soft target` 与 `centered_desirability` 的差异，是否主要来自标签语义？
2. 还是主要来自两条线对 `X` 的处理方式不同？

### 设计原则

本轮尽量做“受控”而不是“各自 strongest 直接对打”。

固定项：

- 模型家族：AutoGluon 回归
- 时间切分：`TimeSeriesSplit(5)`
- 特征选择预算：`top_k = 220`
- 共同数据源：`uncleaned_source`
- 共同主诊断指标：
  - `hard_out_ap_diagnostic`
  - `hard_out_auc_diagnostic`

变化项只有两个：

- 标签
  - `soft_target`
  - `centered_desirability`
- X 配方
  - `soft_x_only`
  - `centered_x_only`
  - `union_x`

### 本轮三种 X 的定义

`soft_x_only`

- 取 `soft target` 分支当前最优的 `whole_window_range_position`
- 本质：窗口内位置几何增强

`centered_x_only`

- 取 `centered_desirability` 分支当前最优的 `lag120_win60 + flow_balance + combined_quality`
- 本质：lag-scale + 工艺交互 + 质量状态增强

`union_x`

- 直接把 `soft_x_only` 和 `centered_x_only` 对齐后拼接
- 仍然用同一个 `top_k=220` 做训练折内筛选

### 实现文件

- 脚本：
  - `projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_label_x_controlled_matrix_round1.py`
- 配置：
  - `projects/T90/dev/tabular_framework_validation/configs/autogluon_label_x_controlled_matrix_round1.yaml`
- 产物：
  - `projects/T90/dev/tabular_framework_validation/artifacts/label_x_controlled_matrix_round1/label_x_controlled_matrix_round1_summary.json`
  - `projects/T90/dev/tabular_framework_validation/artifacts/label_x_controlled_matrix_round1/label_x_controlled_matrix_round1_results.csv`
  - `projects/T90/dev/tabular_framework_validation/reports/label_x_controlled_matrix_round1/label_x_controlled_matrix_round1_summary.md`

### 核心结果

| 标签 | X | 主指标 | hard_out AP | hard_out AUC |
|---|---|---:|---:|---:|
| `soft_target` | `soft_x_only` | `soft_brier = 0.089262` | `0.263113` | `0.557914` |
| `soft_target` | `centered_x_only` | `soft_brier = 0.085936` | `0.317011` | `0.633363` |
| `soft_target` | `union_x` | `soft_brier = 0.086905` | `0.300860` | `0.620777` |
| `centered_desirability` | `soft_x_only` | `mae = 0.310983` | `0.205043` | `0.532339` |
| `centered_desirability` | `centered_x_only` | `mae = 0.307957` | `0.301822` | `0.618104` |
| `centered_desirability` | `union_x` | `mae = 0.312145` | `0.203349` | `0.513981` |

说明：

- `soft_target` 的主指标是 `soft_brier`，越低越好。
- `centered_desirability` 的主指标是 `mae`，越低越好。
- `hard_out AP/AUC` 是两条线对齐后的共同风险诊断指标。

### 第一轮结论

#### 结论 1：`centered_x_only` 是当前更强的 X 方案

它在两条标签线上都赢了 `soft_x_only`：

- 对 `soft_target`
  - `soft_brier: 0.089262 -> 0.085936`
  - `hard_out AP: 0.263113 -> 0.317011`
  - `hard_out AUC: 0.557914 -> 0.633363`
- 对 `centered_desirability`
  - `mae: 0.310983 -> 0.307957`
  - `hard_out AP: 0.205043 -> 0.301822`
  - `hard_out AUC: 0.532339 -> 0.618104`

这说明当前项目里，X 侧成熟度差异是真实存在的，而且影响很大。

#### 结论 2：`union_x` 第一轮没有证明“直接全量整合更好”

`union_x` 在两条标签线上都没有打赢 `centered_x_only`：

- 对 `soft_target`
  - `centered_x_only` 仍优于 `union_x`
- 对 `centered_desirability`
  - `union_x` 反而退化得更明显

所以当前不支持“把两边所有优点直接拼起来就会更强”。

#### 结论 3：如果目标是“预测不合格概率”，标签语义目前更偏向 `soft_target`

在同一个 X 上看风险诊断：

`soft_x_only` 上：

- `soft_target`: `AP 0.263113 / AUC 0.557914`
- `centered_desirability`: `AP 0.205043 / AUC 0.532339`

`centered_x_only` 上：

- `soft_target`: `AP 0.317011 / AUC 0.633363`
- `centered_desirability`: `AP 0.301822 / AUC 0.618104`

也就是说，在“预测 out-of-spec 风险”这个目标下：

- `soft_target` 在两套 X 上都更占优
- `centered_desirability` 虽然也能做风险诊断，但它不是最直接的风险标签

#### 结论 4：当前 strongest centered 并不意味着 centered 标签本身更适合风险任务

这一轮已经说明，之前看到 centered 线更强，不能直接解释成“centered 标签更好”。

更准确的判断是：

- 过去 centered 线更强，有相当一部分原因来自它的 X 工程更成熟
- 当我们把更强的 `centered_x_only` 喂给 `soft_target` 后，`soft_target` 反而成了当前最强风险方案

### 当前阶段判断

如果目标是：

- “某个工况下产品不合格概率有多大”

那么当前更值得继续深挖的是：

- 标签：`soft_target`
- X：`centered_x_only`

也就是：

- `soft label semantics`
- 配 `centered branch` 更成熟的 X 处理

### 当前不建议直接做的事

1. 不建议现在就把 `union_x` 当成新主线。
2. 不建议再把“centered stronger”直接写成“标签更优”。
3. 不建议把 `centered_desirability` 强行解释成失败概率本体。

### 更值得继续的下一步

1. 在 `soft_target + centered_x_only` 这条线上做第二轮确认性实验。
2. 检查 `centered_x_only` 里到底是哪一类特征带来了迁移收益：
   - `flow_balance`
   - `combined_quality`
   - 还是 `lag120_win60` 本身
3. 进一步做更细的消融，而不是直接继续堆 union。

---

## 当前阶段小结

第一轮受控矩阵已经给出一个比较清楚的答案：

- **X 的差距是真实存在的，而且当前影响不小。**
- **如果任务目标是风险概率，`soft_target` 标签语义更合适。**
- **最值得深挖的组合不是两条 strongest 原样对打，而是 `soft_target + centered_x_only`。**

---

## 实验 2：`soft_target + centered_x_only` 确认性消融

### 目的

在第一轮受控矩阵中，当前最强组合是：

- `soft_target + centered_x_only`

这一轮不再扩标签，也不再引入新的 X。
只回答一个更窄的问题：

- `centered_x_only` 的收益，主要来自哪一部分？

### 设计

固定项：

- 标签：`soft_target`
- 模型家族：AutoGluon 回归
- 时间切分：`TimeSeriesSplit(5)`
- 训练折内特征选择：`top_k = 220`

消融项：

1. `lag_only`
2. `lag_plus_interaction`
3. `lag_plus_quality`
4. `lag_plus_interaction_plus_quality`

其中：

- `lag` 指 `lag120_win60`
- `interaction` 指 `flow_balance`
- `quality` 指 `combined_quality`

### 实现文件

- 脚本：
  - `projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_soft_target_centered_x_ablation_round2.py`
- 配置：
  - `projects/T90/dev/tabular_framework_validation/configs/autogluon_soft_target_centered_x_ablation_round2.yaml`
- 产物：
  - `projects/T90/dev/tabular_framework_validation/artifacts/soft_target_centered_x_ablation_round2/soft_target_centered_x_ablation_round2_summary.json`
  - `projects/T90/dev/tabular_framework_validation/artifacts/soft_target_centered_x_ablation_round2/soft_target_centered_x_ablation_round2_results.csv`
  - `projects/T90/dev/tabular_framework_validation/reports/soft_target_centered_x_ablation_round2/soft_target_centered_x_ablation_round2_summary.md`

### 核心结果

| 方案 | soft_brier | hard_out AP | hard_out AUC | soft_mae |
|---|---:|---:|---:|---:|
| `lag_only` | `0.090288` | `0.222359` | `0.566401` | `0.261535` |
| `lag_plus_interaction` | `0.089374` | `0.275569` | `0.592619` | `0.256337` |
| `lag_plus_quality` | `0.089137` | `0.205349` | `0.548372` | `0.259148` |
| `lag_plus_interaction_plus_quality` | `0.086020` | `0.317108` | `0.628811` | `0.249909` |

### 第二轮结论

#### 结论 1：完整包不是“冗余堆料”，而是真正最优

`lag_plus_interaction_plus_quality` 在四个方案里最好：

- `soft_brier` 最低
- `hard_out AP` 最高
- `hard_out AUC` 最高
- `soft_mae` 也最低

所以它不是简单靠特征更多取巧，而是当前确实形成了最好的组合。

#### 结论 2：`interaction` 比 `quality` 更像主要增益源

从 `lag_only` 往上加时：

- 加 `interaction`
  - `soft_brier: 0.090288 -> 0.089374`
  - `AP: 0.222359 -> 0.275569`
  - `AUC: 0.566401 -> 0.592619`
- 加 `quality`
  - `soft_brier: 0.090288 -> 0.089137`
  - `AP: 0.222359 -> 0.205349`
  - `AUC: 0.566401 -> 0.548372`

也就是说：

- `quality` 单独加入时，对 `soft_brier` 有一点帮助
- 但对真正更关键的 `hard_out AP/AUC` 反而不稳
- `interaction` 单独加入时，对风险诊断提升更直接

#### 结论 3：`quality` 的价值更像协同项，而不是单独主项

虽然 `quality` 单独上阵并不强，但它和 `interaction` 叠在一起后，完整包显著优于 `lag_plus_interaction`：

- `soft_brier: 0.089374 -> 0.086020`
- `AP: 0.275569 -> 0.317108`
- `AUC: 0.592619 -> 0.628811`

这说明：

- `quality` 不是没用
- 它更像是在已有工艺交互骨架上做稳固和修正
- 它的价值主要体现在“协同增强”

### 当前阶段判断更新

到这一轮为止，可以把判断进一步收紧成：

1. 如果目标是风险概率预测，当前最值得继续的组合仍然是：
   - `soft_target + lag120_win60 + flow_balance + combined_quality`
2. 这条线的收益不是单纯来自“标签更好”。
3. 更准确地说，是：
   - `soft_target` 提供了更合适的风险语义
   - `centered` 分支发展出来的 `interaction + quality` 提供了更成熟的 X 工程
4. 其中：
   - `interaction` 是更强的主增益源
   - `quality` 是重要的协同增强项

### 下一步建议

如果继续深挖，这条线最值得做的不是再回到 `union_x`，而是二选一：

1. 对 `interaction` 包进一步拆小，确认究竟是哪几个 flow-balance 关系最关键。
2. 对 `quality` 包做受控收缩，确认哪些质量特征是真协同，哪些只是陪跑。
