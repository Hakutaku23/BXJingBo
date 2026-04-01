# Center-Band 语义实验记录

## 这轮实验的前提

- `t90` 来自人工测定
- 人工测定精度只能到 `0.1`
- 因此像 `8.45` 这样的数值并不是当前化验数据里会直接出现的观测值

这意味着 centered-quality 不能再继续按“围绕 `8.45` 的连续回归目标”去理解，而应该改成更贴近真实观测支持的 `center-band` 语义。

## 这轮冻结路线

- Baseline A: 继续保留 frozen threshold-oriented strongest cleanroom
- Treatment D: `lower-risk + upper-risk + center-band semantics`

## 这轮 center-band 定义

- center band 按当前人工测定支持写成 `8.4 / 8.5`
- 实现上用区间 `[8.35, 8.55)` 对应这一观测带
- 在 centered truth 中：
  - out-of-spec -> `unacceptable`
  - center-band -> `premium`
  - 其余 in-spec -> `acceptable`

## 这轮业务语义

1. clear risk -> `unacceptable`
2. moderate risk -> `retest`
3. low risk 且 center-band 概率高 -> `premium`
4. 其余 -> `acceptable`

## 固定不变的边界

- DCS: `merge_data.csv + merge_data_otr.csv`
- strict sensor-identity de-dup
- simple causal `120 min` window stats
- fold-local `topk=40`
- `TimeSeriesSplit(5)`

## 结果摘要

### Baseline A

- threshold macro_f1: `0.2674`
- threshold balanced_accuracy: `0.3580`
- boundary warning AP: `0.1963`

### Treatment D

- center-band truth share: `0.3938`
- center-band AP: `0.3962`
- premium precision / recall: `0.3732 / 0.0593`
- centered decision macro_f1_vs_truth: `0.2006`
- decision coverage: `0.8009`
- retest rate: `0.1991`
- unacceptable recall: `0.4652`
- unacceptable false-clear rate: `0.0238`

## 和上一轮 centered-quality 相比

这轮不是简单小修，而是一次真正的语义修正。

主要改进有：

- centered decision macro_f1_vs_truth 从 `0.1439` 提到 `0.2006`
- premium precision 从 `0.1028` 提到 `0.3732`
- 不再像上一轮那样把大量样本直接压成 `unacceptable`

但代价也很明确：

- unacceptable recall 从 `0.7219` 降到 `0.4652`
- unacceptable false-clear 从 `0.0150` 升到 `0.0238`

## 当前判断

- “按人工测定分辨率重写 centered-quality 语义” 这条方向是对的
- `center-band` 比上一轮伪连续 desirability 更符合真实标签支持
- 但当前这一版仍不能替代 frozen baseline，因为 unacceptable 侧的 trade-off 还偏大

## 当前产物位置

- 配置: `configs/centered_quality_center_band_current_head.yaml`
- 脚本: `scripts/run_centered_quality_center_band_current_head.py`
- 审计: `reports/centered_quality_center_band_current_head_audit.md`
- 摘要: `reports/centered_quality_center_band_current_head/centered_quality_center_band_summary.md`
- 结果表: `artifacts/centered_quality_center_band_current_head/centered_quality_center_band_results.csv`
- 汇总 JSON: `artifacts/centered_quality_center_band_current_head/centered_quality_center_band_summary.json`
