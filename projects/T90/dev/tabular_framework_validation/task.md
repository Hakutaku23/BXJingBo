# AutoGluon 阶段化特征工程任务表

## 1. 目标

本任务表用于在当前 T90 项目中，为 AutoGluon 构建一套分阶段、可审计、可回滚的特征工程流程。

核心原则：

- 先做低风险、高确定性的特征增强
- 再逐步引入更复杂的时序与工艺状态特征
- 每一阶段都必须和上一阶段做可对照比较
- 所有监督式筛选必须严格限制在训练折内
- 不允许未来信息泄漏

---

## 2. 统一实验约束

| 项目 | 要求 |
|---|---|
| 验证方式 | `TimeSeriesSplit(n_splits=5)`，样本不足时可降到 4，但不得低于 3 |
| 基础窗口 | 第一轮固定 `120min causal window` |
| 基础统计 | `mean / std / min / max / last / range / delta` |
| 统一基线 | 使用当前 simplest causal snapshot baseline |
| 模型平台 | `AutoGluon TabularPredictor` |
| 任务优先级 | 风险二头 > centered desirability 回归 > 可选 5-bin 分类 |
| 特征筛选 | 无监督清洗可全量执行；监督筛选只能训练折内执行 |
| 数据说明 | 数据源起点为未经系统清洗的数据源 |
| 输出要求 | 每阶段必须输出结果表、审计说明、特征清单 |

---

## 3. 阶段总览

| 阶段 | 名称 | 目标 | 预期产出 |
|---|---|---|---|
| S0 | 基线表构建 | 建立统一 snapshot 表 | baseline feature table |
| S1 | 多滞后/多尺度统计 | 验证窗口与滞后扩展是否有效 | lag-scale feature pack |
| S2 | 动态形态特征 | 引入趋势、波动、局部形态信息 | dynamic feature pack |
| S3 | 状态/工况特征 | 弥补 snapshot 状态不足 | regime/state feature pack |
| S4 | 工艺交互特征 | 引入差值、比值、偏差等工艺表达 | interaction feature pack |
| S5 | 数据质量特征 | 显式表达缺失、冻结、异常等可靠性 | quality feature pack |
| S6 | centered-quality 专用特征 | 服务于中心质量目标 | centered-quality pack |
| S7 | 组合筛选与收敛 | 合并有效特征包并做最终筛选 | final engineered table |

---

## 4. 详细任务表

## S0. 基线表构建

### 目标
构建统一的 AutoGluon 输入表，作为后续所有阶段的对照基线。

### 任务
- [ ] 统一 `decision_time`
- [ ] 按 `120min causal window` 构造基础 snapshot
- [ ] 计算基础统计特征：
  - [ ] mean
  - [ ] std
  - [ ] min
  - [ ] max
  - [ ] last
  - [ ] range
  - [ ] delta
- [ ] 生成以下任务标签：
  - [ ] 风险二头：`y < 8.2`、`y > 8.7`
  - [ ] centered desirability：`q(y) = max(0, 1 - |y - 8.45| / 0.25)`
  - [ ] 可选 5-bin 分类标签
- [ ] 输出 baseline 表

### 交付物
- `artifacts/stage0_baseline_features.csv`
- `reports/stage0_baseline_audit.md`

### 验收标准
- [ ] 所有特征均为因果特征
- [ ] 标签生成逻辑固定且有审计说明
- [ ] 可直接喂给 AutoGluon

---

## S1. 多滞后 / 多尺度统计特征

### 目标
验证不同滞后与窗口长度是否能改善当前 simple baseline。

### 推荐实验轴
- `tau ∈ {0, 120, 240, 360, 480} min`
- `W ∈ {60, 120, 240, 480} min`

### 任务
- [ ] 为每个 `(tau, W)` 构造基础统计特征
- [ ] 特征命名规范化，如：
  - `{sensor}__lag120_win240_mean`
  - `{sensor}__lag240_win120_std`
- [ ] 与 S0 基线做逐轮比较
- [ ] 记录每个窗口包的增益与负贡献

### 交付物
- `artifacts/stage1_lag_scale_features.csv`
- `reports/stage1_lag_scale_summary.md`

### 验收标准
- [ ] 至少找到 1 组优于 baseline 的窗口包，或明确证明无增益
- [ ] 不允许使用未来窗口
- [ ] 结果可追溯到具体 `(tau, W)` 组合

---

## S2. 动态形态特征

### 目标
补足“只看水平，不看变化形态”的不足。

### 候选特征
- [ ] slope
- [ ] diff_mean
- [ ] diff_std
- [ ] diff_max_abs
- [ ] sign_change_count
- [ ] upward_ratio
- [ ] downward_ratio
- [ ] local_autocorr
- [ ] volatility_ratio
- [ ] catch22（可选）
- [ ] tsfresh 小规模试验（可选）

### 任务
- [ ] 在当前最优窗口基础上构造动态特征
- [ ] 先做手工小特征包
- [ ] 再做 `catch22` 轻量试验
- [ ] 若需要，再做 `tsfresh` 小规模 relevant extraction

### 交付物
- `artifacts/stage2_dynamic_features.csv`
- `reports/stage2_dynamic_summary.md`

### 验收标准
- [ ] 动态特征增益必须独立于窗口长度变化来评估
- [ ] 特征数量不能失控膨胀
- [ ] 若引入自动特征提取，必须有筛选记录

---

## S3. 状态 / 工况特征

### 目标
缓解“同一 snapshot 对应多个隐状态”的问题。

### 候选特征
- [ ] 负荷区间
- [ ] 班次/时段
- [ ] 关键设定值区间
- [ ] 启停工标记
- [ ] 聚类状态 ID
- [ ] regime probability / cluster distance

### 任务
- [ ] 整理已有元数据
- [ ] 若无显式工况标签，则在训练集上做无监督状态聚类
- [ ] 将状态 ID / 状态得分作为附加特征
- [ ] 比较“无状态特征”与“有状态特征”性能差异

### 交付物
- `artifacts/stage3_regime_features.csv`
- `reports/stage3_regime_summary.md`

### 验收标准
- [ ] 状态特征不得跨折泄漏
- [ ] 聚类器必须仅用训练折拟合
- [ ] 必须明确状态特征是人工定义还是无监督生成

---

## S4. 工艺交互特征

### 目标
显式表达工艺逻辑中的差值、比值、偏差和平衡关系。

### 候选特征
- [ ] 温差
- [ ] 压差
- [ ] 流量比
- [ ] `PV - SP`
- [ ] 前后段 spread
- [ ] 长短窗口均值差
- [ ] 长短窗口均值比

### 任务
- [ ] 基于工艺知识先列出有限候选交互清单
- [ ] 先做 20–50 个高价值交互
- [ ] 用 AutoGluon `feature_importance()` 评估正负贡献
- [ ] 删除明显负增益交互

### 交付物
- `artifacts/stage4_interaction_features.csv`
- `reports/stage4_interaction_summary.md`

### 验收标准
- [ ] 不允许做全组合爆炸
- [ ] 每个交互特征必须能解释其工艺意义
- [ ] 负贡献交互必须记录

---

## S5. 数据质量 / 传感器健康特征

### 目标
把“数据可靠性”本身显式变成模型输入。

### 候选特征
- [ ] missing_ratio
- [ ] valid_count
- [ ] freeze_length
- [ ] freeze_ratio
- [ ] max_jump
- [ ] time_since_last_jump
- [ ] alias_disagreement
- [ ] update_irregularity

### 任务
- [ ] 统计每个窗口内的数据质量特征
- [ ] 将数据质量特征与主特征表合并
- [ ] 分析哪些质量特征能解释风险与不稳定预测

### 交付物
- `artifacts/stage5_quality_features.csv`
- `reports/stage5_quality_summary.md`

### 验收标准
- [ ] 数据质量特征必须来源于当前窗口本身
- [ ] 不允许借用未来质量信息
- [ ] 必须记录哪些变量存在明显健康问题

---

## S6. centered-quality 专用特征

### 目标
服务于“8.45 最优”这一新任务定义。

### 候选特征
- [ ] 中心稳定度
- [ ] 偏离中心的趋势
- [ ] 近短窗与长窗偏离中心的一致性
- [ ] 向中心收敛/发散速度
- [ ] 稳定保持中心的时长
- [ ] 低侧风险代理特征
- [ ] 高侧风险代理特征

### 任务
- [ ] 围绕 centered desirability 重新整理一组专用特征
- [ ] 比较：
  - [ ] 只用通用特征
  - [ ] 通用特征 + centered-quality 特征
- [ ] 重点看是否更能区分“中心优质”与“边缘合格”

### 交付物
- `artifacts/stage6_centered_quality_features.csv`
- `reports/stage6_centered_quality_summary.md`

### 验收标准
- [ ] centered-quality 特征必须体现“越靠近 8.45 越好”
- [ ] 不能直接泄漏 y
- [ ] 必须有单独增益对比

---

## S7. 组合筛选与最终收敛

### 目标
把前几阶段中有效的特征包组合起来，形成最终 AutoGluon 工程化输入表。

### 任务
- [ ] 汇总各阶段有效特征包
- [ ] 进行无监督预清洗：
  - [ ] 常值列
  - [ ] 极高缺失列
  - [ ] 重复列
- [ ] 在训练折内做监督式筛选
- [ ] 比较不同特征包组合：
  - [ ] baseline + lag-scale
  - [ ] + dynamic
  - [ ] + regime
  - [ ] + interaction
  - [ ] + quality
  - [ ] + centered-quality
- [ ] 形成最终推荐输入表

### 交付物
- `artifacts/stage7_final_feature_table.csv`
- `reports/stage7_final_summary.md`

### 验收标准
- [ ] 最终组合优于 S0 baseline
- [ ] 特征包组合关系清晰
- [ ] 所有监督式筛选都只在训练折内执行

---

## 5. AutoGluon 执行要求

| 项目 | 要求 |
|---|---|
| 平台 | `TabularPredictor` |
| 任务 | binary / regression / optional multiclass |
| leaderboard | 每阶段都输出 |
| feature importance | 每阶段都输出一次 |
| 时间切分 | 固定不变 |
| 对比方式 | 只比较特征包差异，不随意改任务定义 |

---

## 6. 统一产物要求

每一阶段至少产出：

- `features_stageX.csv`
- `results_stageX.csv`
- `summary_stageX.json`
- `audit_stageX.md`

审计说明至少包括：

- 本阶段新增了哪些特征
- 是否使用自动特征提取
- 是否进行了无监督预清洗
- 是否进行了训练折内监督筛选
- 是否发现明显负增益特征
- 与前一阶段相比的增益或退化

---

## 7. 最终目标

本任务表的最终目标不是“堆最多特征”，而是形成一套：

- 适合 AutoGluon
- 适合当前工业场景
- 可解释
- 可审计
- 可持续扩展

的阶段化特征工程体系。
"""

agents = """# AGENTS.md — Tabular Framework Validation Rules for T90

This file applies to work inside `projects/T90/dev/tabular_framework_validation/` and refines the parent `projects/T90/AGENTS.md`.

## Scope
- This directory is reserved for validating strong tabular frameworks such as AutoGluon and TabPFN on the current project.
- The workflow here is explicitly two-stage:
  1. quick validation from uncleaned-source data with minimal necessary preprocessing
  2. framework-specific feature engineering and data cleaning, only if stage 1 is promising
- The default purpose is framework validation, not direct productionization.

## Required dataset statement
Every experiment and report in this directory must explicitly state:

- the starting data source is currently **uncleaned**
- “uncleaned” refers to the source state, not to a ban on all preprocessing
- stage 1 may use only minimal necessary preprocessing
- stage 2 may introduce dedicated feature engineering only if stage 1 has demonstrated value

## Primary objective
- Determine whether AutoGluon and TabPFN are worth keeping as project-level tabular benchmarks.
- If they are, determine how to design a framework-specific feature-engineering workflow for them.
- Keep the workflow auditable and leakage-controlled.

## Allowed reuse
You may reuse:
- raw / near-raw data loaders
- current current-head label utilities
- simple causal snapshot feature builders
- offline reference metadata such as `projects/T90/data/卤化位点.xlsx` for signal interpretation

You should avoid reusing as fixed truth:
- prior branch-specific selected sensor lists
- prior handcrafted feature bundles
- prior best-threshold logic
- prior best-window logic

## Directory expectations
Recommended contents under this directory:
- `plans/`
- `configs/`
- `scripts/`
- `reports/`
- `artifacts/`

## Experiment discipline
- Preserve causality and avoid future leakage.
- Use time-ordered validation only.
- Clearly separate:
  - stage 1 quick validation
  - stage 2 dedicated feature engineering
- Record what was deliberately left uncleaned in stage 1.
- Record what was newly engineered or cleaned in stage 2.

## Baseline rules
- Every benchmark must define a simple baseline.
- Baseline and framework runs must use the same data split.
- Stage 2 is not allowed unless stage 1 has shown a credible positive signal.

## Output rules
Each experiment should leave behind:
- a clear plan,
- runnable configs or scripts,
- a machine-readable result table,
- and an audit note explicitly describing:
  - starting source condition,
  - stage 1 processing,
  - stage 2 processing if applicable.

## Promotion rule
Nothing from this directory should be treated as delivery code by default.
Promotion into `core/`, `interface.py`, `example.py`, or `README.md` requires an explicit user decision.
"""

parent_append = """## Additional directory responsibility for tabular framework validation

### `dev/tabular_framework_validation/`
- Use this directory for validating AutoGluon, TabPFN, and similar tabular frameworks in a two-stage workflow.
- The starting assumption must be explicit: the available source data is currently **uncleaned**, but stage 1 may still apply minimal necessary preprocessing.
- Stage 1 is for quick signal validation only.
- Stage 2 is allowed only if stage 1 shows that the framework is worth further investment, and may then introduce dedicated feature engineering and data cleaning.
- Reuse of offline reference metadata is allowed when it improves signal naming and interpretation. In particular, `projects/T90/data/卤化位点.xlsx` remains the authoritative offline reference for DCS tag interpretation and feature naming.
- Outputs here are development support only, not delivery-boundary artifacts, unless the user explicitly promotes them.

## Additional tabular framework validation rules
- Do not silently turn stage 1 into a heavily engineered benchmark.
- Do not skip directly to stage 2.
- Keep time-ordered validation throughout.
- If cleaned-data comparisons are introduced, they must be clearly labeled as stage-2 conditions.