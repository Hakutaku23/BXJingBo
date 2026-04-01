# Centered Quality Validation 总记录

## 说明

- 本文件作为 `projects/T90/dev/centered_quality_validation/` 下的统一实验记录文件
- 从当前轮开始，centered-quality 相关实验记录统一追加在本文件中
- 其他 `reports/` 文件继续保留为审计或摘要产物，但不再承担主实验记录职责

## 分支前提

- 保留当前 threshold-oriented strongest cleanroom 作为 baseline
- centered-quality 分支不再继续围绕 ordinal/cumulative 本身打转
- 当前阶段重点是验证 centered-quality 的目标定义与业务语义，而不是切换复杂模型族

## 实验 1: point-centered desirability 首轮

### 固定设定

- DCS: `merge_data.csv + merge_data_otr.csv`
- strict sensor-identity de-dup
- simple causal `120 min` window stats
- fold-local `topk=40`
- `TimeSeriesSplit(5)`

### 首轮目标定义

- `q(y) = max(0, 1 - |y - 8.45| / 0.25)`
- Treatment B: centered desirability only
- Treatment C: lower-risk + upper-risk + centered desirability

### cleanroom 修正

- 第一版首跑曾误把 `true_desirability`、risk truth、center flags 留在候选特征池中
- 该版结果因标签泄漏废弃
- accepted run 已显式剔除这些列后重跑

### accepted run 结果

- Baseline A: `macro_f1 = 0.2674`，`balanced_accuracy = 0.3580`
- Treatment B:
  - desirability `MAE = 0.3744`
  - desirability `RMSE = 0.4532`
  - desirability `Spearman = -0.0434`
- Treatment C:
  - lower-risk `AP = 0.0157`
  - upper-risk `AP = 0.1690`
  - unacceptable recall `= 0.7219`
  - unacceptable false-clear rate `= 0.0150`
  - centered decision `macro_f1_vs_truth = 0.1439`

### 当时结论

- strongest threshold-oriented baseline 明显强于首轮 centered-quality treatment
- 这版 centered-quality 更像“伪连续目标 + 过激 risk 语义”，尚不成立

## 语义复盘

### 新增关键前提

- `t90` 来自人工测定
- 人工测定精度只能达到 `0.1`
- 因此像 `8.45` 这样的点不是当前化验数据会直接给出的观测值

### 复盘结论

- 当前 `8.45` 中心点回归目标并不适合现有观测支持
- 现有 desirability 在当前数据上实际上退化成少数离散档
- `premium` truth 更接近“是否落在中心带”而不是“是否逼近一个连续中心点”
- 当前 decision map 把过多 moderate risk 样本直接压成了 `unacceptable`

## 实验 2: center-band semantics 重定义

### 固定不变的边界

- baseline 仍是 frozen threshold-oriented strongest cleanroom
- 不改 DCS 数据边界
- 不改 simple `120 min` 表示
- 不改 fold-local `topk=40`

### 重定义后的目标

- 不再用 point-centered regression 作为主头
- 改用 observed-support `center-band`
- 当前 center-band 定义为 `8.4 / 8.5`

### 重定义后的业务语义

1. clear risk -> `unacceptable`
2. moderate risk -> `retest`
3. low risk 且 center-band 概率高 -> `premium`
4. 其余 -> `acceptable`

### 第二轮结果

- center-band truth share `= 0.3938`
- center-band `AP = 0.3962`
- premium precision / recall `= 0.3732 / 0.0593`
- centered decision `macro_f1_vs_truth = 0.2006`
- decision coverage `= 0.8009`
- retest rate `= 0.1991`
- unacceptable recall `= 0.4652`
- unacceptable false-clear rate `= 0.0238`

### 相比首轮的变化

- centered decision `macro_f1_vs_truth`: `0.1439 -> 0.2006`
- premium precision: `0.1028 -> 0.3732`
- unacceptable 过度使用明显收敛

### 当前 trade-off

- unacceptable recall: `0.7219 -> 0.4652`
- unacceptable false-clear: `0.0150 -> 0.0238`

## 当前总判断

- 方向修正是对的
- `center-band` 比 point-centered desirability 更符合人工测定精度与真实标签支持
- 但当前 policy 还不够稳，还不能取代 frozen threshold-oriented strongest cleanroom

## 当前优先建议

- 下一步优先处理 unacceptable 侧 guardrail，而不是继续增加 centered 结构复杂度
- 最值得先做的是:
  - 重新校准双侧风险输出
  - 把 `clear risk / ambiguous risk / low risk` 的语义边界做得更稳
  - 在此基础上再细化 premium / acceptable 的 center-band 语义

## 实验 3: risk guardrail policy

### 本轮目的

- 不改 center-band 结构
- 不换模型
- 只在动作层加入 `center_prob` 保护条件
- 目标是让 `unacceptable` 只落给“高风险且明显不居中”的样本

### 本轮 policy

1. 如果 `max_risk >= clear_risk_threshold` 且 `center_prob <= unacceptable_center_ceiling` -> `unacceptable`
2. 否则如果 `max_risk >= retest_risk_threshold` -> `retest`
3. 否则如果 `center_prob >= premium_center_threshold` -> `premium`
4. 否则 -> `acceptable`

### 本轮结果

- unacceptable recall `= 0.4492`
- unacceptable false-clear rate `= 0.0207`
- centered decision `macro_f1_vs_truth = 0.1886`
- premium precision / recall `= 0.3500 / 0.0391`
- retest rate `= 0.3251`

### 和实验 2 相比

- false-clear 略有改善: `0.0238 -> 0.0207`
- 但 recall 略降: `0.4652 -> 0.4492`
- centered macro_f1 下降: `0.2006 -> 0.1886`
- premium precision 也略降: `0.3732 -> 0.3500`
- retest rate 明显升高: `0.1991 -> 0.3251`

### 本轮结论

- 这说明“在动作层给 unacceptable 增加 center_prob guardrail”这个方向本身是合理的，因为它确实压低了 false-clear。
- 但仅靠 policy guardrail 还不够，当前双侧风险头本身的排序质量仍然偏弱，所以 guardrail 会以较高的 `retest` 代价换来较小的 unacceptable 侧收益。

## 当前最新判断

- centered-quality 这条线里，`center-band` 方向仍然比 point-centered desirability 更可信
- 但单纯继续打磨动作层 policy，收益已经开始变小
- 下一步更值得做的，不是继续扩 guardrail 网格，而是回到风险头本身，重做更适合当前语义的 risk target 或 risk calibration

## 实验 4: two-tier risk

### 本轮目的

- 不再只修动作层
- 直接重写风险头语义
- 将风险拆成两层:
  - clear unacceptable risk
  - retest risk

### 本轮 target 定义

- clear low risk 目标: `y < 8.0`
- clear high risk 目标: `y >= 8.9`
- retest low risk 目标: `y < 8.2`
- retest high risk 目标: `y > 8.7`
- center-band 仍保持 `8.4 / 8.5`

### 本轮 policy

1. clear risk 高 -> `unacceptable`
2. 否则 retest risk 高 -> `retest`
3. 否则 center-band 概率高 -> `premium`
4. 否则 -> `acceptable`

### cleanroom 修正说明

- 第一版 two-tier risk 首跑时，`target_ge_8_9` 被错误带进候选特征池，构成标签泄漏
- 该版结果已废弃
- 接受版本已经显式剔除所有 `target_ge_*` 列后重跑

### accepted run 结果

- unacceptable recall `= 0.4278`
- unacceptable false-clear rate `= 0.0159`
- centered decision `macro_f1_vs_truth = 0.1659`
- premium precision / recall `= 0.3253 / 0.0302`
- retest rate `= 0.4115`

### 和实验 3 相比

- false-clear 继续改善: `0.0207 -> 0.0159`
- 但 unacceptable recall 继续下降: `0.4492 -> 0.4278`
- centered macro_f1 也下降: `0.1886 -> 0.1659`
- retest rate 进一步升高: `0.3251 -> 0.4115`

### 本轮结论

- 两层风险这条思路有一个明确优点: unacceptable 侧更保守，false-clear 压得更低
- 但当前代价过大，系统把太多样本推入 `retest`
- 说明当前阶段真正的瓶颈已经不是“policy 写法不够细”，而是风险头本身对当前 centered-quality 任务的可分性仍然不够强

## 当前最新总判断

- centered-quality 分支里，真正被支持的是:
  - `point-centered regression` 不适合当前人工测定分辨率
  - `center-band` 语义比伪连续 desirability 更合理
- 但不管是单层 risk guardrail，还是两层 risk 语义，到目前都还没拿到足够稳的 `unacceptable / retest / premium` 平衡
- 因此下一步如果继续深挖，不应再只靠动作层或风险分层结构，而应开始考虑:
  - 风险头的概率校准
  - 或直接改成更贴近现场语义的弱监督 / 排序式 centered target

## 实验 5: two-tier risk probability calibration

### 本轮目的

- 保持实验 4 的 `two-tier risk + center-band` 语义不变
- 不更换模型家族，仍使用当前线性的 binary heads
- 只验证一个问题:
  - 当前 unacceptable / retest 侧的不稳定，是否主要来自风险头概率未校准

### 本轮方法

- 在每个 outer-train 内再切出按时间顺序靠后的 calibration slice
- 对四个风险头分别做 `isotonic` 概率校准:
  - clear low risk: `y < 8.0`
  - clear high risk: `y >= 8.9`
  - retest low risk: `y < 8.2`
  - retest high risk: `y > 8.7`
- `center-band` 头保持不校准，避免这轮同时混入 centered 头变化
- calibration 参数:
  - `calibration_fraction = 0.20`
  - `min_calibration_rows = 80`

### 本轮结果

- unacceptable recall `= 0.0749`
- unacceptable false-clear rate `= 0.0590`
- unacceptable retest rate `= 0.0172`
- centered decision `macro_f1_vs_truth = 0.2651`
- decision coverage `= 0.7947`
- retest rate `= 0.2053`
- premium precision / recall `= 0.4641 / 0.0794`

### 和实验 4 相比

- centered decision `macro_f1_vs_truth`: `0.1659 -> 0.2651`
- decision coverage: `0.5885 -> 0.7947`
- retest rate: `0.4115 -> 0.2053`
- premium precision: `0.3253 -> 0.4641`
- 但 unacceptable recall 大幅下滑: `0.4278 -> 0.0749`
- unacceptable false-clear 明显恶化: `0.0159 -> 0.0590`

### 本轮结论

- 这轮说明“风险头概率未校准”并不是当前主要矛盾，至少这版 `isotonic` 校准没有把 `unacceptable / retest` 侧校得更稳，反而把风险头压得过软了。
- 它确实改善了 centered / premium 侧的动作输出，让系统更少进入 `retest`，也更容易给出 `premium`。
- 但代价是 unacceptable 侧几乎失守，这在当前任务语义下不可接受。
- 因此，这版 risk-head probability calibration 不能作为当前 centered-quality 主线的默认增强项。

## 当前最新总判断

- centered-quality 这条线里，`center-band` 语义仍然是被支持的方向。
- 但“直接给 two-tier risk 头做 isotonic 概率校准”没有修复核心矛盾，反而把系统推向了“centered 更好看、风险更失真”的一侧。
- 如果继续做风险头校准，下一步不能再泛化地做全头同权校准，而更适合考虑:
  - 只对 `retest risk` 做校准，保留 `clear unacceptable risk` 的原始保守性
  - 或在校准后加 unacceptable-side guardrail，避免 tail positives 被过度压扁

## 实验 6: retest-risk only calibration

### 本轮目的

- 延续实验 5 的校准思路，但不再对全部风险头同权处理
- 只对 `retest risk` 做 `isotonic` 概率校准
- `clear unacceptable risk` 保持实验 4 的原始保守输出
- 检查当前真正需要被“拉软”的是否只是 `retest` 侧

### 本轮方法

- `clear low risk: y < 8.0` 和 `clear high risk: y >= 8.9`
  - 直接使用原始 binary head 概率
- `retest low risk: y < 8.2` 和 `retest high risk: y > 8.7`
  - 在 outer-train 尾部 calibration slice 上做 `isotonic` 校准
- `center-band` 头保持不校准
- 其余 split、特征、点位筛选、baseline 均保持不变

### 本轮结果

- unacceptable recall `= 0.4278`
- unacceptable false-clear rate `= 0.0344`
- unacceptable retest rate `= 0.0128`
- centered decision `macro_f1_vs_truth = 0.2434`
- decision coverage `= 0.8185`
- retest rate `= 0.1815`
- premium precision / recall `= 0.4737 / 0.0604`

### 和实验 4 相比

- centered decision `macro_f1_vs_truth`: `0.1659 -> 0.2434`
- decision coverage: `0.5885 -> 0.8185`
- retest rate: `0.4115 -> 0.1815`
- premium precision: `0.3253 -> 0.4737`
- unacceptable recall 基本持平: `0.4278 -> 0.4278`
- 但 unacceptable false-clear 明显恶化: `0.0159 -> 0.0344`

### 和实验 5 相比

- 保住了 clear unacceptable 侧:
  - recall `0.0749 -> 0.4278`
  - false-clear `0.0590 -> 0.0344`
- 说明实验 5 的主要问题确实来自 clear risk 被一起校准后被压软
- 但本轮仍没有把 false-clear 拉回实验 4 的水平

### 本轮结论

- “只校准 retest risk”比“全风险头一起校准”合理得多。
- 它明显改善了 centered / premium / retest 侧的动作分布，而且没有再像实验 5 那样让 unacceptable recall 直接塌掉。
- 但 unacceptable false-clear 仍然从 `0.0159` 升到了 `0.0344`，说明即便只动 `retest` 侧，当前校准方式仍会改变 unacceptable 与 retest 的分界，代价还偏大。
- 因此，这轮可以视为一个比实验 5 更可信的中间版本，但仍不能取代实验 4 的风险保守性。

## 当前最新总判断

- centered-quality 里，“只校准 retest risk”是比“全头同权校准”更正确的方向。
- 当前主要矛盾已经进一步收紧为:
  - 我们可以明显改善 centered 侧动作输出和覆盖率
  - 但一旦对 retest 边界做概率修正，unacceptable false-clear 就会回升
- 如果继续沿风险头校准这条线深挖，下一步更适合做的不是再扩大校准范围，而是:
  - 给 retest-calibrated 版本加 unacceptable-side guardrail
  - 或只在靠近 `8.2 / 8.7` 的局部候选上校准，而不是整段概率映射一起拉动

## 实验 7: retest-risk calibration + unacceptable-side guardrail

### 本轮目的

- 以实验 6 的 `retest-only calibrated` 版本为底座
- 不扩大校准范围
- 只在动作层补一层 `unacceptable-side guardrail`
- 核心目标:
  - 把实验 6 回升的 `unacceptable false-clear` 压回去
  - 同时尽量保住实验 6 拿到的 centered / coverage 收益

### 本轮方法

- `clear unacceptable risk`
  - 继续使用原始 binary head 概率，不做校准
- `retest risk`
  - 继续使用实验 6 的 `isotonic` 校准
- 动作层新增:
  - `unacceptable_center_ceiling`
  - 只有当 clear risk 高，且 `center_prob <= unacceptable_center_ceiling` 时，才允许直接落入 `unacceptable`
- 其余样本仍按:
  - `retest risk`
  - `premium center threshold`
  - `acceptable`
  顺序决策

### 本轮结果

- unacceptable recall `= 0.3316`
- unacceptable false-clear rate `= 0.0273`
- unacceptable retest rate `= 0.0278`
- centered decision `macro_f1_vs_truth = 0.2318`
- decision coverage `= 0.6930`
- retest rate `= 0.3070`
- premium precision / recall `= 0.3917 / 0.0526`

### 和实验 6 相比

- false-clear 有所回落: `0.0344 -> 0.0273`
- 但 unacceptable recall 明显下降: `0.4278 -> 0.3316`
- retest rate 明显回升: `0.1815 -> 0.3070`
- centered decision `macro_f1_vs_truth` 也下降: `0.2434 -> 0.2318`
- premium precision 下降: `0.4737 -> 0.3917`

### 和实验 4 相比

- false-clear 仍高于原始 two-tier risk: `0.0159 -> 0.0273`
- unacceptable recall 也更低: `0.4278 -> 0.3316`
- 但 decision coverage 更高: `0.5885 -> 0.6930`

### 本轮结论

- 这轮说明 `unacceptable-side guardrail` 的方向是对的，它确实能把实验 6 的 false-clear 往回压。
- 但当前这层 guardrail 还不够“便宜”，因为它主要是靠把样本重新推回 `retest` 来换安全性。
- 换句话说，实验 7 比实验 6 更安全，但没有更均衡；比实验 4 更可覆盖，但 unacceptable 侧仍不够稳。

## 当前最新总判断

- 在当前 centered-quality 线里，风险头校准相关实验已经把问题范围压得比较清楚了:
  - 全风险头同权校准: 不可取
  - 只校准 retest risk: 方向正确，但 false-clear 回升
  - 再加 unacceptable-side guardrail: 能压回 false-clear，但代价是 recall 和 retest trade-off
- 因此，下一步如果继续沿这条线深挖，更值得做的不是再加更复杂的 guardrail 结构，而是:
  - 缩小 retest 校准的作用范围，只在靠近 `8.2 / 8.7` 的局部边界生效
  - 或把 calibration target 从全段概率映射，改成更局部的 boundary-aware calibration
