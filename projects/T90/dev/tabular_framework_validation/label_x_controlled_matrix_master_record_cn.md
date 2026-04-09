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

---

## 实验 3：`flow_balance` 内部 pair 级别消融

### 目的

第二轮已经说明：

- `interaction` 是更强的主增益源
- `quality` 是协同增强项

所以这轮继续向下拆 `interaction`，只回答一个问题：

- `flow_balance` 里到底是哪几个 pair 最关键？

### 设计

固定项：

- 标签：`soft_target`
- 底座：`lag120_win60 + combined_quality`
- 模型家族：AutoGluon 回归
- 时间切分：`TimeSeriesSplit(5)`
- 训练折内特征选择：`top_k = 220`

变动项只在 `flow_balance` 内部：

1. `full_all_pairs`
2. `single_FIC_C51001_PV_F_CV__FIC_C51003_PV_F_CV`
3. `single_FIC_C51401_PV_F_CV__FIC_C30501_PV_F_CV`
4. `single_FIC_C51401_PV_F_CV__FI_C51005_S_PV_CV`
5. `single_FIC_C51401_PV_F_CV__FIC_C51003_PV_F_CV`
6. 四个对应的 `drop_*`

### 实现文件

- 脚本：
  - `projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_soft_target_flow_balance_pair_ablation_round3.py`
- 配置：
  - `projects/T90/dev/tabular_framework_validation/configs/autogluon_soft_target_flow_balance_pair_ablation_round3.yaml`
- 产物：
  - `projects/T90/dev/tabular_framework_validation/artifacts/soft_target_flow_balance_pair_ablation_round3/soft_target_flow_balance_pair_ablation_round3_summary.json`
  - `projects/T90/dev/tabular_framework_validation/artifacts/soft_target_flow_balance_pair_ablation_round3/soft_target_flow_balance_pair_ablation_round3_results.csv`
  - `projects/T90/dev/tabular_framework_validation/reports/soft_target_flow_balance_pair_ablation_round3/soft_target_flow_balance_pair_ablation_round3_summary.md`

### 核心结果

| 方案 | soft_brier | hard_out AP | hard_out AUC | soft_mae |
|---|---:|---:|---:|---:|
| `full_all_pairs` | `0.086020` | `0.317108` | `0.628811` | `0.249909` |
| `single_FIC_C51401_PV_F_CV__FI_C51005_S_PV_CV` | `0.086375` | `0.262550` | `0.583450` | `0.251363` |
| `single_FIC_C51401_PV_F_CV__FIC_C30501_PV_F_CV` | `0.086917` | `0.243871` | `0.582146` | `0.252575` |
| `single_FIC_C51401_PV_F_CV__FIC_C51003_PV_F_CV` | `0.088339` | `0.289111` | `0.621840` | `0.254521` |
| `single_FIC_C51001_PV_F_CV__FIC_C51003_PV_F_CV` | `0.090991` | `0.212012` | `0.555513` | `0.261122` |
| `drop_FIC_C51001_PV_F_CV__FIC_C51003_PV_F_CV` | `0.088200` | `0.271740` | `0.612255` | `0.255992` |
| `drop_FIC_C51401_PV_F_CV__FIC_C30501_PV_F_CV` | `0.088221` | `0.290310` | `0.623413` | `0.255696` |
| `drop_FIC_C51401_PV_F_CV__FIC_C51003_PV_F_CV` | `0.088642` | `0.250794` | `0.585960` | `0.251038` |
| `drop_FIC_C51401_PV_F_CV__FI_C51005_S_PV_CV` | `0.089140` | `0.253873` | `0.590004` | `0.256306` |

### 第三轮结论

#### 结论 1：完整 `flow_balance` 包仍然最强

没有任何一个 `single_*` 或 `drop_*` 版本打赢 `full_all_pairs`。

这说明：

- 不是某一个 pair 独自撑起了全部收益
- 当前更像是多 pair 协同结构

#### 结论 2：单 pair 里最强的是 `FIC_C51401` 对 `FI_C51005`

在所有 `single_*` 版本里，它最好：

- `soft_brier = 0.086375`
- `hard_out AP = 0.262550`
- `hard_out AUC = 0.583450`

这说明它是一个很强的单独关系信号。

#### 结论 3：最弱的单 pair 是 `FIC_C51001` 对 `FIC_C51003`

它的单独表现最差：

- `soft_brier = 0.090991`
- `hard_out AP = 0.212012`
- `hard_out AUC = 0.555513`

所以它不太像“单独就能成立的主力 pair”。

#### 结论 4：删 pair 时，最伤的是两条 `FIC_C51401` 相关边

如果看 `drop_*` 版本：

- `drop_FIC_C51401_PV_F_CV__FI_C51005_S_PV_CV`
  - `soft_brier` 退化最大
- `drop_FIC_C51401_PV_F_CV__FIC_C51003_PV_F_CV`
  - `AP/AUC` 退化更明显

这说明当前最关键的不是某一个孤立 pair，而是：

- 以 `FIC_C51401_PV_F_CV` 为中心的多条平衡关系

#### 结论 5：`FIC_C51401` 是 interaction 包里的结构中心

从单 pair 和 drop-one 两侧一起看：

- 单独最能站住的 pair 里，主角是 `FIC_C51401`
- 删掉后最伤的 pair 里，主角也还是 `FIC_C51401`

所以当前最合理的结构判断是：

- `flow_balance` 的有效信息并不是平均分布的
- 而是明显围绕 `FIC_C51401_PV_F_CV` 展开

### 当前阶段判断更新

到这一轮为止，可以把主结论进一步压实为：

1. 当前风险主线仍然是：
   - `soft_target + lag120_win60 + flow_balance + combined_quality`
2. `interaction` 的核心不是“任意 flow pair 都有用”。
3. 更像是：
   - 以 `FIC_C51401_PV_F_CV` 为中心的多 pair 工艺平衡网络在起作用
4. `quality` 的作用仍然是协同稳定，而不是替代这些工艺关系。

### 下一步建议

如果继续，我建议优先做这两类之一：

1. 以 `FIC_C51401_PV_F_CV` 为中心，继续做更细的 interaction 收缩，验证能否在不伤性能的前提下压缩 interaction 包。
2. 转去压缩 `quality` 包，因为 interaction 这条主干已经比较清楚了。

---

## 实验 4：`quality` 包压缩确认 + `RADI` 离线部署评价

### 目的

- 固定当前最强软风险主线：
  - `soft_target + lag120_win60 + full flow_balance interaction`
- 不再改 `lag` 和 `interaction`。
- 只压缩 `quality` 包，并把新增规范 [cqdi_radi_offline_evaluation_spec.md](/D:/PSE/博兴京博/BXJingBo/projects/T90/cqdi_radi_offline_evaluation_spec.md) 里的 `RADI` 一起接进来。

### 设计

- 模型家族：AutoGluon 回归
- 外层切分：`TimeSeriesSplit(5)`
- 内层阈值选择：`TimeSeriesSplit(3)`
- 共享 `top_k = 220`
- `RADI` 的 `ap_floor` 取上一轮当前工程基线 `full_all_pairs` 的
  - `hard_out_ap_diagnostic * 0.95`
  - 实际值：`0.317108 * 0.95 = 0.301253`

本轮比较 4 个 `quality` 版本：

1. `full_quality`
   - `missing_ratio + valid_count + freeze_length + freeze_ratio + max_jump + time_since_last_jump + update_irregularity`
2. `freeze_missing`
   - `missing_ratio + valid_count + freeze_length + freeze_ratio`
3. `jump_irregularity`
   - `max_jump + time_since_last_jump + update_irregularity`
4. `compact_core4`
   - `missing_ratio + freeze_ratio + max_jump + time_since_last_jump`

### 实现与产物

- 脚本：
  - [run_autogluon_soft_target_quality_ablation_round4.py](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_soft_target_quality_ablation_round4.py)
- 配置：
  - [autogluon_soft_target_quality_ablation_round4.yaml](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/configs/autogluon_soft_target_quality_ablation_round4.yaml)
- 常规摘要：
  - [soft_target_quality_ablation_round4_summary.json](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/soft_target_quality_ablation_round4/soft_target_quality_ablation_round4_summary.json)
  - [soft_target_quality_ablation_round4_summary.md](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/reports/soft_target_quality_ablation_round4/soft_target_quality_ablation_round4_summary.md)
- `RADI` 相关离线表：
  - [offline_eval_scored_rows.csv](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/soft_target_quality_ablation_round4/offline_eval_scored_rows.csv)
  - [offline_soft_threshold_candidates.csv](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/soft_target_quality_ablation_round4/offline_soft_threshold_candidates.csv)
  - [offline_soft_fold_summary.csv](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/soft_target_quality_ablation_round4/offline_soft_fold_summary.csv)
  - [offline_soft_deployability_summary.csv](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/soft_target_quality_ablation_round4/offline_soft_deployability_summary.csv)

### 核心结果

| 方案 | soft_brier | hard_out AP | hard_out AUC | worst_case RADI | status |
|---|---:|---:|---:|---:|---|
| `full_quality` | `0.086020` | `0.317108` | `0.628811` | `0.0` | `FAIL` |
| `freeze_missing` | `0.086255` | `0.278916` | `0.611078` | `0.0` | `FAIL` |
| `compact_core4` | `0.088370` | `0.281862` | `0.616377` | `0.0` | `FAIL` |
| `jump_irregularity` | `0.089353` | `0.300555` | `0.638499` | `0.0` | `FAIL` |

### 这轮的直接结论

#### 结论 1：从常规风险指标看，`full_quality` 仍然最强

- `soft_brier` 最好：`0.086020`
- `hard_out AP` 最高：`0.317108`
- `hard_out AUC` 虽不是最高，但整体仍最均衡

所以当前不支持把 `combined_quality` 直接压成更小包。

#### 结论 2：如果只看排序能力，`jump_irregularity` 有局部价值

- 它的 `hard_out AUC = 0.638499` 是四个版本里最高
- 但 `soft_brier` 和 `hard_out AP` 都不如 `full_quality`

这说明跳变/更新不规则性更像一组“风险排序增强项”，但还不足以单独顶替完整 `quality` 包。

#### 结论 3：`RADI` 当前没有支持任何一个版本进入可部署区

四个版本的 `worst_case RADI` 全部都是 `0.0`，`gate_pass_rate` 也都是 `0.0`。

更具体地说，拖住 `RADI` 的不是 `AP floor`，而是动作区门槛本身：

- 内层最优阈值几乎全部退化到同一个默认角点：
  - `clear_threshold = 0.10`
  - `alert_threshold = 0.60`
- 在 inner OOF 上，`clear` 区通常几乎不存在，所以
  - `train_clear_outspec_prob_mean` 反复落到回退值 `1.0`
- 外层 test 上虽然能切出一点 `clear`，但仍远远不够安全：
  - `full_quality`
    - `test_clear_outspec_prob_mean = 0.2477 / 0.4462 / 0.4612`（对应误差场景 `0.10 / 0.15 / 0.20`）
  - `freeze_missing`
    - `0.3671 / 0.3939 / 0.4068`
  - `jump_irregularity`
    - `0.3653 / 0.3880 / 0.4005`
  - `compact_core4`
    - `0.6880 / 0.7051 / 0.5929`

这些值都明显高于 `RADI` 的 `clear_outspec_prob_mean <= 0.08` gate。

#### 结论 4：当前问题已经不再是“quality 该不该压缩”，而是“soft risk 的动作阈值语义还不成立”

也就是说：

- 从拟合与诊断角度，`soft_target + full_quality` 仍然是当前最强风险主线
- 但从 `RADI` 口径看，这条线还不能被直接解释成可靠的 `clear / retest / alert` 动作器
- 当前失败主要来自：
  - 低风险清放区不够可分
  - alert 区召回也偏低
  - 阈值搜索只能退回默认角点

### 当前阶段判断更新

到这一轮为止，可以把判断进一步收紧成：

1. 如果目标是“当前最强的 soft risk 拟合器”，主线仍然是：
   - `soft_target + lag120_win60 + flow_balance + full_quality`
2. `quality` 包的压缩版本暂时都没有整体替代 `full_quality`
3. 新增的 `RADI` 指标很有价值，因为它暴露出了一个此前常规指标没完全显化的问题：
   - 我们的风险分数可以用于排序和诊断
   - 但还不能直接当成稳定的三段式动作规则

### 下一步建议

如果继续，我建议优先走下面这条，而不是继续压 `quality`：

1. 固定当前最强 `soft_target + full_quality` 底座。
2. 专门做一轮 `RADI-oriented policy / label` 校准实验：
   - 不先动 `X`
   - 先验证如何让 `clear` 区真正形成可放行的低风险样本群

---

## 实验 5：`RADI-oriented` 标签校准（固定最强 `X`）

### 目的

- 承接实验 4 的结论：
  - 当前最大问题已经不是 `quality` 包压不压缩
  - 而是 `soft risk` 还不能稳定形成可放行的 `clear` 区
- 因此这轮不改 `X`，只改 `soft target` 的标签映射形状
- 看是否能把 `RADI` 从全失败里拉出来

### 设计

固定不变项：

- `X`：`lag120_win60 + full flow_balance interaction + full_quality`
- AutoGluon 回归
- 外层：`TimeSeriesSplit(5)`
- 内层：`TimeSeriesSplit(3)`
- `RADI` 口径与实验 4 完全相同

只改标签映射：

1. `logistic_s0p03`
2. `logistic_s0p05`
3. `logistic_s0p08`
4. `tanh_s0p05`
5. `smoothstep_s0p05`

### 实现与产物

- 脚本：
  - [run_autogluon_soft_target_radi_label_calibration_round5.py](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/scripts/run_autogluon_soft_target_radi_label_calibration_round5.py)
- 配置：
  - [autogluon_soft_target_radi_label_calibration_round5.yaml](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/configs/autogluon_soft_target_radi_label_calibration_round5.yaml)
- 结果：
  - [soft_target_radi_label_calibration_round5_summary.json](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/soft_target_radi_label_calibration_round5/soft_target_radi_label_calibration_round5_summary.json)
  - [soft_target_radi_label_calibration_round5_summary.md](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/reports/soft_target_radi_label_calibration_round5/soft_target_radi_label_calibration_round5_summary.md)
  - [offline_eval_scored_rows.csv](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/soft_target_radi_label_calibration_round5/offline_eval_scored_rows.csv)
  - [offline_soft_threshold_candidates.csv](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/soft_target_radi_label_calibration_round5/offline_soft_threshold_candidates.csv)
  - [offline_soft_fold_summary.csv](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/soft_target_radi_label_calibration_round5/offline_soft_fold_summary.csv)
  - [offline_soft_deployability_summary.csv](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/soft_target_radi_label_calibration_round5/offline_soft_deployability_summary.csv)

### 核心结果

| 方案 | soft_brier | hard_out AP | hard_out AUC | worst_case RADI | status |
|---|---:|---:|---:|---:|---|
| `logistic_s0p08` | `0.086020` | `0.317108` | `0.628811` | `0.0` | `FAIL` |
| `logistic_s0p05` | `0.124460` | `0.263507` | `0.599539` | `0.0` | `FAIL` |
| `logistic_s0p03` | `0.150549` | `0.269724` | `0.605324` | `0.0` | `FAIL` |
| `tanh_s0p05` | `0.151485` | `0.313527` | `0.624097` | `0.0` | `FAIL` |
| `smoothstep_s0p05` | `0.162205` | `0.280352` | `0.586061` | `0.0` | `FAIL` |

### 这轮的直接结论

#### 结论 1：当前最优标签映射没有变，仍然是 `logistic_s0p08`

它依然是这组里最好的：

- `soft_brier` 最优
- `hard_out AP` 最优
- `hard_out AUC` 也在高位

也就是说，实验 4 的当前主标签并不是“偶然选错了形状”。

#### 结论 2：把标签变硬，并没有把 `clear` 区真正救出来

更硬的 `logistic_s0p05 / s0p03` 确实改变了风险分布，但没有让 `RADI` 过门槛：

- `logistic_s0p05`
  - `test_clear_outspec_prob_mean = 0.2158 / 0.4139 / 0.4260`
- `logistic_s0p03`
  - `0.4183 / 0.4362 / 0.4451`

相比 baseline `logistic_s0p08` 的

- `0.2477 / 0.4462 / 0.4612`

局部有改善，但远远达不到 `RADI` 要求的 `<= 0.08`。

同时，它们还明显伤到了拟合质量：

- `soft_brier` 明显恶化
- `hard_out AP` 也回落

#### 结论 3：换 label shape 也没有解决问题

- `tanh_s0p05` 的 `hard_out AP = 0.313527`，接近 baseline
- 但 `soft_brier` 已明显变差，而且 `RADI` 仍然是 `0`
- `smoothstep_s0p05` 在 `clear_outspec_prob_mean` 上也没有跨过关键门槛

所以当前不支持“只靠换 soft label 形状，就能把动作语义做出来”。

#### 结论 4：`RADI` 的主要瓶颈已经更明确了

这一轮所有方案依旧在 inner 里退回同一个默认角点：

- `clear_threshold = 0.10`
- `alert_threshold = 0.60`

并且 `train_gate_pass` 全部仍为 `0`。

这说明问题更深一层：

- 当前风险头确实能做排序
- 但它还没有形成“足够安全的 clear 尾部”
- 这不是简单的 label smoothness 问题

### 当前阶段判断更新

到这一轮为止，可以把结论再压实一步：

1. 当前最强风险拟合底座仍然是：
   - `soft_target(logistic, softness=0.08) + lag120_win60 + flow_balance + full_quality`
2. `RADI` 暂时没有被“quality 压缩”或“label 形状校准”救回来
3. 因此下一步更该转向：
   - `policy / decision semantics`
   - 或更明确的 `clear-zone-aware` 目标定义

### 下一步建议

如果继续，我建议不再围绕 `soft label` 形状打转，而是进入下面这条：

1. 固定 `logistic_s0p08 + strongest X` 不动。
2. 直接做一轮 `clear-zone-aware` 的目标重定义或双头策略：
   - 让一个头专门学习“可安全放行”
   - 而不是继续指望单一 soft risk 分数自然长出 `clear / alert / retest`

---

## 实验 5b：提高 `boundary_softness` 的趋势验证

### 目的

- 回答一个更具体的问题：
  - 如果在当前最强 `soft_target` 里，把 `boundary_softness` 从 `0.08` 继续提高，会不会让 `RADI` 更接近可用
- 这轮仍然固定最强 `X` 底座不动
- 只测 `logistic` 家族的 `softness` 上调趋势

### 设计

固定不变项：

- `X`：`lag120_win60 + full flow_balance interaction + full_quality`
- 模型：AutoGluon 回归
- `RADI` 口径与实验 5 完全相同

比较的 `softness`：

1. `0.08`
2. `0.10`
3. `0.12`
4. `0.15`

### 实现与产物

- 配置：
  - [autogluon_soft_target_radi_label_calibration_round5_trend.yaml](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/configs/autogluon_soft_target_radi_label_calibration_round5_trend.yaml)
- 结果：
  - [soft_target_radi_label_calibration_round5_summary.json](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/soft_target_radi_label_calibration_round5_trend/soft_target_radi_label_calibration_round5_summary.json)
  - [soft_target_radi_label_calibration_round5_summary.md](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/reports/soft_target_radi_label_calibration_round5_trend/soft_target_radi_label_calibration_round5_summary.md)
  - [offline_soft_fold_summary.csv](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/soft_target_radi_label_calibration_round5_trend/offline_soft_fold_summary.csv)
  - [offline_soft_threshold_candidates.csv](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/tabular_framework_validation/artifacts/soft_target_radi_label_calibration_round5_trend/offline_soft_threshold_candidates.csv)

### 核心结果

| softness | soft_brier | hard_out AP | hard_out AUC | worst_case RADI |
|---|---:|---:|---:|---:|
| `0.08` | `0.085901` | `0.320042` | `0.632321` | `0.0` |
| `0.10` | `0.069990` | `0.216634` | `0.542426` | `0.0` |
| `0.12` | `0.054706` | `0.264266` | `0.580267` | `0.0` |
| `0.15` | `0.040647` | `0.254303` | `0.584359` | `0.0` |

### 这轮的直接结论

#### 结论 1：提高 `softness` 会持续降低 `soft_brier`

这是很明确的单调趋势：

- `0.08 -> 0.10 -> 0.12 -> 0.15`
- `soft_brier` 从 `0.085901` 一路降到 `0.040647`

说明分数确实被拉得更软、更像“概率回归”。

#### 结论 2：但这不是我们当前想要的工程收益

随着 `softness` 增大，`hard_out AP/AUC` 明显回落：

- `AP`
  - `0.320042 -> 0.216634 -> 0.264266 -> 0.254303`
- `AUC`
  - `0.632321 -> 0.542426 -> 0.580267 -> 0.584359`

也就是说，分数更平滑了，但 out-of-spec 区分能力被削弱了。

#### 结论 3：没有证据表明“把 `0.08` 往上调，会把 `clear` 区救出来”

更关键的是，`RADI` 仍然全部为 `0.0`。

而且 `clear` 区安全性并没有出现我们需要的下降到 `<= 0.08`：

- `0.08`
  - `test_clear_outspec_prob_mean = 0.4079 / 0.5902 / 0.6014`
- `0.10`
  - `0.7458 / 0.7480 / 0.7522`
- `0.12`
  - `0.5477 / 0.5623 / 0.5695`
- `0.15`
  - `0.5514 / 0.3954 / 0.4069`

这说明“继续增大 `softness`”不会自然形成足够安全的 `clear` 尾部。

#### 结论 4：趋势只部分成立

最初的直觉是：

- `softness` 更大
- 分数更温和
- 可能 `retest` 更高
- `clear` 更难形成

现在看，前两条是成立的：

- 分数确实更温和
- 拟合误差确实更小

但“`retest` 单调增加”并没有稳定出现；更准确的判断是：

- 提高 `softness` 主要带来的是“风险排序变钝”
- 而不是“把动作语义修好”

### 当前阶段判断再更新

到这一步为止，可以把判断收紧成：

1. `boundary_softness = 0.08` 不是偶然值，它仍是当前这条主线里最合理的折中点
2. 继续往上调 `softness`，更像是在优化平滑拟合，不是在提升 `RADI`
3. 所以当前不建议再沿 `softness > 0.08` 继续搜索
4. 下一步仍应转去：
   - `clear-zone-aware` 目标定义
   - 或双头 / 分层动作策略
