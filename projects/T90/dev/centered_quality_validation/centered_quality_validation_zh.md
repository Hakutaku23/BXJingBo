# Centered Quality Validation 实验记录

## 记录范围

- 本记录仅对应 `projects/T90/dev/centered_quality_validation/`
- 这是一条与 `cleanroom_validation/` 平行的 cleanroom 分支
- 当前 threshold-oriented strongest cleanroom 只作为 baseline 锚点保留，不再作为本分支继续优化的主线

## 首轮冻结路线

- Baseline A: 保留现有 strongest threshold-oriented cleanroom 作为对照
- Treatment B: centered desirability only
- Treatment C: lower-risk + upper-risk + centered desirability

## 首轮固定输入边界

- DCS: `merge_data.csv + merge_data_otr.csv`
- 先做 strict sensor-identity de-dup
- 特征表示: simple causal `120 min` window stats
- fold-local `topk=40`
- `TimeSeriesSplit(5)`

## 首轮固定 centered-quality 定义

- 目标中心: `8.45`
- 半宽: `0.25`
- desirability:

`q(y) = max(0, 1 - |y - 8.45| / 0.25)`

## 首轮固定决策映射

1. `max_risk >= 0.50` -> `unacceptable`
2. `max_risk < 0.20` 且 `q_hat >= 0.80` -> `premium`
3. `max_risk < 0.35` 且 `q_hat >= 0.30` -> `acceptable`
4. 否则 -> `retest`

## 当前状态

- 新目录已确认建立
- 首轮实验方案已冻结
- 后续脚本、配置、审计与结果文件均在本目录内维护
- 首轮实验待执行
