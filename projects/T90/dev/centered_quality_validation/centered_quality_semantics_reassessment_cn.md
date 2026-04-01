# Centered Quality 语义复盘

## 这次复盘想回答什么

- 首轮 centered-quality 为什么没有成立
- 问题更像出在目标定义，还是出在决策语义
- 下一轮应该怎样改，而不是继续在当前错误语义上微调

## 复盘结论 1

当前这版 centered target 在现有化验分辨率下，并不是真正连续的目标。

这轮数据里的 `t90` 基本落在 `0.1` 刻度上，而且没有真实样本落在 `8.45`。因此一开始定义的

`q(y) = max(0, 1 - |y - 8.45| / 0.25)`

在现有样本上实际上退化成了一个很粗的分档目标，而不是平滑连续目标。

在当前 in-spec 样本里，它只出现了 3 个实际取值：

- `0.0`
- `0.4`
- `0.8`

所以首轮并不是“连续 desirability 学不好”，而更像是“把一个分档目标伪装成了回归目标”。

## 复盘结论 2

当前 premium truth 本质上已经退化成了 “中心带成员资格” 问题。

因为 `8.45` 没有被观测到，当前最大 observed desirability 其实只有 `0.8`。又因为 `premium_true_threshold = 0.80`，所以当前 premium truth 实际上基本等价于：

- 样本是否落在 `8.4 / 8.5` 这个最靠近中心的观测带上

这说明下一轮更诚实的做法，不是继续假设“围绕 8.45 的连续效用”，而是直接把 centered-quality head 改写成 “center-band probability”。

## 复盘结论 3

当前 decision map 对 risk 的解释过于激进，导致大量样本被误压成 `unacceptable`。

在 accepted run 里：

- `71.76%` 的样本被判成了 `unacceptable`
- `premium` 只有 `2.60%`
- `acceptable` 只有 `15.20%`
- `retest` 只有 `10.44%`

这说明当前语义实际上是在做：

- “只要 risk 稍高，就直接判 unacceptable”

而不是：

- “clear risk -> unacceptable”
- “ambiguous risk -> retest”

所以现在更该修的是业务语义，不是继续死调当前阈值。

## 复盘结论 4

当前 absolute risk probability 还不够适合直接拿来做硬业务语义。

accepted run 里，`acceptable / premium / unacceptable` 三组样本的 risk score 并没有形成足够清晰的分层。也就是说，当前 risk head 还不足以支撑：

- `0.50` 以上就直接 unacceptable

这种过硬的映射。

这不等于风险头无效，而是说明：

- moderate risk 更应该先落到 `retest`
- `unacceptable` 应该只留给 clear risk

## 因此，下一轮应该怎么改

### 目标定义

把 centered-quality 主头从 point-centered regression 改成 observed-support center band。

建议第一版：

- `center_band = 1` 对应 `8.4 / 8.5`
- `center_band = 0` 对应其他样本

### 决策语义

建议语义改成：

1. clear risk -> `unacceptable`
2. ambiguous risk -> `retest`
3. low risk 且 center-band 概率高 -> `premium`
4. low risk 但不是 center-band -> `acceptable`

### 为什么这更合理

- 它尊重了当前化验分辨率的真实支持
- 它不再假设一个没有被观测到的 `8.45` 点可以直接当作连续回归中心
- 它把 `retest` 真正用作模糊和冲突的缓冲层
- 它保留了 centered-quality 的核心思想，但把表达方式改得更贴近现场数据

## 冻结的下一轮路线

- Baseline A 继续保持 frozen threshold-oriented strongest cleanroom
- 新 treatment 改为：
  - `lower-risk + upper-risk + center-band semantics`
- 不换特征
- 不换数据边界
- 不切回旧分支
- 只在 outer-train 内选择语义阈值

## 建议的下一轮阈值搜索范围

- clear unacceptable threshold: `{0.80, 0.90}`
- retest risk threshold: `{0.50, 0.60, 0.70}`
- premium center-band threshold: `{0.55, 0.65, 0.75}`

## 当前判断

这轮 centered-quality 不成立，不是因为 centered-quality 方向一定错了；更像是因为第一版把“观察上离散的中心偏好”错误地写成了“连续点中心效用”，同时又把 moderate risk 过早压成了 `unacceptable`。
