# Centered Desirability 标签修正方案（工程版）

## 1. 目的

本文给出一版更适合工程部署前评估与离线验证的 `centered_desirability` 标签规范，统一纳入以下三种构建方法：

1. **当前点式线性法**
2. **误差感知线性法**
3. **高斯核函数法**

本文目标不是立刻替换现有实现，而是把三种方法的定义、语义差异、适用场景和推荐用法统一整理，便于后续做同条件对照实验与部署前验收。

---

## 2. 问题背景

当前项目的 centered 目标语义可以概括为：

- 目标中心：`c = 8.45`
- 目标容忍半宽：`t = 0.25`
- 真实测量误差量级：约 `±0.10 ~ ±0.20`

这意味着：

- `8.45` 更像一个**业务语义中心**
- 而不是一个可以被高精度观测和稳定复现的“精确真值点”

因此，`centered_desirability` 的标签构建方式，不能只考虑“是否单调合理”，还必须考虑：

- 是否保留中心质量语义
- 是否与 `8.45 ± 0.25` 的业务容忍带一致
- 是否对测量误差足够稳健
- 是否适合作为工程部署前的离线验收参考

---

## 3. 三种 centered_desirability 构建方法

# 3.1 方法 A：当前点式线性法

这是当前项目中已经使用的 centered_desirability 定义。

设：

- 观测值为 $y$
- 中心为 $c$
- 容忍半宽为 $t$

则标签定义为：

$$
q_{\text{linear-point}}(y;c,t)
=
\max\left(0,\ 1-\frac{|y-c|}{t}\right)
$$

在当前项目中通常取：

- $c = 8.45$
- $t = 0.25$

即：

$$
q_{\text{linear-point}}(y)
=
\max\left(0,\ 1-\frac{|y-8.45|}{0.25}\right)
$$

## 3.1.1 语义解释

该方法的语义非常直接：

- 距离中心越近，标签越接近 1
- 距离中心越远，标签线性下降
- 一旦满足 $|y-c| \ge t$，标签直接归 0

因此它本质上是一个：

> **带硬容忍边界的线性工程效用函数**

## 3.1.2 优点

- 业务解释最直接
- 与当前 `8.45 ± 0.25` 的规则高度一致
- 与现有实验完全兼容
- 参数只有两个，容易理解和维护

## 3.1.3 局限

- 对测量误差敏感
- 在边界处存在硬截断
- 把所有带外样本都压成 0，无法区分“略偏”和“严重偏离”
- 中心附近标签斜率固定，容易把观测误差放大成标签波动

## 3.1.4 推荐定位

建议保留为：

- **开发基线标签**
- **历史结果对照标签**

不建议继续作为部署前唯一参考真值。

---

# 3.2 方法 B：误差感知线性法（推荐正式版本）

该方法保留方法 A 的业务语义，但显式考虑测量误差。

## 3.2.1 核心思路

先定义与方法 A 相同的点式线性核函数：

$$
q_0(y;c,t)
=
\max\left(0,\ 1-\frac{|y-c|}{t}\right)
$$

然后不再直接用观测值 $y_{\text{obs}}$ 计算最终标签，而是在测量误差分布下取期望：

设观测值为 $y_{\text{obs}}$，误差为 $\delta$，误差分布为 $p(\delta)$，则：

$$
q_{\text{linear-uncertain}}(y_{\text{obs}})
=
\mathbb{E}_{\delta \sim p(\delta)}
\left[
q_0(y_{\text{obs}}+\delta;c,t)
\right]
$$

即：

$$
q_{\text{linear-uncertain}}(y_{\text{obs}})
=
\int q_0(y_{\text{obs}}+\delta;c,t)\,p(\delta)\,d\delta
$$

## 3.2.2 推荐默认误差模型

若暂无更精确的测量误差统计证据，建议使用：

$$
\delta \sim \text{Uniform}[-e,\ e]
$$

默认参数建议：

- $e = 0.15$

同时做敏感性分析：

- $e = 0.10$
- $e = 0.15$
- $e = 0.20$

## 3.2.3 离散近似实现

工程实现中推荐使用离散平均。

给定：

- 误差网格点数：$K = 21$
- 误差网格：

$$
\delta_k \in \text{linspace}(-e,\ e,\ K)
$$

则：

$$
q_{\text{linear-uncertain}}(y_{\text{obs}})
=
\frac{1}{K}
\sum_{k=1}^{K}
\max\left(
0,\ 1-\frac{|(y_{\text{obs}}+\delta_k)-c|}{t}
\right)
$$

## 3.2.4 语义解释

该方法本质上是在问：

> 如果当前观测值附近存在合理测量误差，那么该样本平均意义下有多像“中心优质样本”？

它保留了线性法的容忍带语义，但降低了标签对单点测量误差的敏感度。

## 3.2.5 优点

- 保留 `8.45 ± 0.25` 的业务解释
- 比当前点式线性法更抗噪声
- 适合用于工程部署前验收
- 不会像高斯法那样完全改变标签语义

## 3.2.6 局限

- 仍然继承了线性法的“带外标签最终趋零”的结构
- 仍然有容忍带语义，不适合表达“带外继续细分严重程度”的需求
- 相比点式标签，计算略复杂

## 3.2.7 推荐定位

建议作为：

- **部署前离线验收的 centered 真值标签**
- **`CQDI` 计算时的主要参考标签**

---

# 3.3 方法 C：高斯核函数法

这是另一种常见的中心质量相似度构造方式。

## 3.3.1 定义

设：

- 观测值为 $y$
- 中心为 $c$
- 衰减尺度为 $\sigma$

则高斯型 centered_desirability 定义为：

$$
q_{\text{gauss}}(y;c,\sigma)
=
\exp\left(
-\frac{(y-c)^2}{2\sigma^2}
\right)
$$

在当前场景下可写为：

$$
q_{\text{gauss}}(y)
=
\exp\left(
-\frac{(y-8.45)^2}{2\sigma^2}
\right)
$$

## 3.3.2 语义解释

高斯法的语义与线性法不同：

- 越接近中心越好
- 但这种“好”是平滑衰减的
- 没有硬边界
- 离中心再远，标签也不会严格等于 0，只会越来越小

因此它本质上是：

> **无硬边界的中心相似度函数**

## 3.3.3 优点

- 数学形式平滑
- 更抗测量误差
- 对中心排序更自然
- 不会把带外样本全部压成同一个零值

## 3.3.4 局限

- 与 `8.45 ± 0.25` 的“硬容忍带”语义不完全一致
- 参数 $\sigma$ 不如线性法的容忍半宽直观
- 如果不额外约束，很容易失去工程规则的解释性

## 3.3.5 如何从容忍带反推 $\sigma$

高斯法没有直接的“容忍半宽”参数，因此必须人为约定：

> 在边界 $|y-c|=t$ 处，希望 desirability 还剩多少？

设在边界 $|y-c|=t$ 处：

$$
q_{\text{gauss}}(c \pm t)=\alpha
$$

则有：

$$
\alpha
=
\exp\left(
-\frac{t^2}{2\sigma^2}
\right)
$$

从而可解得：

$$
\sigma
=
\frac{t}{\sqrt{2\ln(1/\alpha)}}
$$

在当前项目中，$t=0.25$，于是：

$$
\sigma
=
\frac{0.25}{\sqrt{2\ln(1/\alpha)}}
$$

### 常见取值示例

若希望在边界处：

#### 情形 1：边界 desirability 为 0.50

$$
\alpha = 0.50
$$

则：

$$
\sigma
\approx
\frac{0.25}{\sqrt{2\ln 2}}
\approx 0.212
$$

#### 情形 2：边界 desirability 为 0.20

$$
\alpha = 0.20
$$

则：

$$
\sigma
\approx
\frac{0.25}{\sqrt{2\ln 5}}
\approx 0.139
$$

#### 情形 3：边界 desirability 为 0.10

$$
\alpha = 0.10
$$

则：

$$
\sigma
\approx
\frac{0.25}{\sqrt{2\ln 10}}
\approx 0.116
$$

## 3.3.6 推荐定位

高斯法适合作为：

- **centered 排序的备选标签**
- **对抗测量误差的平滑标签**
- **与线性语义标签做同条件对照实验的候选方法**

不建议在没有对照实验的情况下直接替换当前 centered 主线。

---

## 4. 三种方法的本质差异

| 方法 | 公式核心 | 是否有硬边界 | 是否考虑测量误差 | 主要语义 | 更适合的用途 |
|---|---|---:|---:|---|---|
| 点式线性法 | $1-|y-c|/t$ 截断 | 是 | 否 | 工程容忍带内效用 | 开发基线、历史对照 |
| 误差感知线性法 | 线性核的误差期望 | 是 | 是 | 稳健的工程容忍带效用 | 部署前验收、工程主评估 |
| 高斯核函数法 | $\exp(-(y-c)^2/(2\sigma^2))$ | 否 | 否（可扩展） | 中心相似度 | 排序、平滑建模、对照实验 |

---

## 5. 是否还可以做“误差感知高斯法”

可以。

如果需要，也可以把高斯法和误差建模结合：

$$
q_{\text{gauss-uncertain}}(y_{\text{obs}})
=
\mathbb{E}_{\delta\sim p(\delta)}
\left[
\exp\left(
-\frac{(y_{\text{obs}}+\delta-c)^2}{2\sigma^2}
\right)
\right]
$$

离散近似为：

$$
q_{\text{gauss-uncertain}}(y_{\text{obs}})
=
\frac{1}{K}
\sum_{k=1}^{K}
\exp\left(
-\frac{((y_{\text{obs}}+\delta_k)-c)^2}{2\sigma^2}
\right)
$$

但这已经属于更进一步的扩展版本。  
在当前阶段，不建议一次性同时改：

- 标签族
- 误差模型
- 特征工程
- 模型框架

否则无法解释提升究竟来自哪一部分。

---

## 6. 推荐字段命名

建议统一保留以下字段：

```yaml
centered_desirability_labels:
  target_centered_desirability_point:
    description: 当前点式线性法标签
  target_centered_desirability_uncertain:
    description: 误差感知线性法标签
  target_centered_desirability_gaussian:
    description: 高斯核函数法标签
```

如果后续扩展高斯误差版，可再增加：

```yaml
  target_centered_desirability_gaussian_uncertain:
    description: 误差感知高斯标签
```

---

## 7. 推荐默认参数

```yaml
centered_desirability_label:
  center: 8.45
  tolerance_half_width: 0.25

  linear_uncertainty:
    uncertainty_model: uniform
    uncertainty_half_width_default: 0.15
    uncertainty_sensitivity_grid:
      - 0.10
      - 0.15
      - 0.20
    integration_points: 21

  gaussian:
    sigma_selection_rule: match_boundary_value
    boundary_value_alpha_default: 0.20
    sigma_default_from_alpha_0p20: 0.139
    candidate_sigma:
      - 0.116
      - 0.139
      - 0.212

  center_band:
    low: 8.35
    high: 8.55

  spec_band:
    low: 8.20
    high: 8.70
```

---

## 8. 推荐采用策略

## 8.1 开发阶段
保留当前点式标签：

- `target_centered_desirability_point`

用于：

- 历史可比
- 快速实验
- 与现有脚本兼容

## 8.2 工程评估阶段
推荐新增并重点使用：

- `target_centered_desirability_uncertain`

用于：

- 部署前离线验收
- `CQDI` 评估
- 测量误差敏感性分析

## 8.3 对照实验阶段
建议把高斯法作为单独实验线：

- `target_centered_desirability_gaussian`

用于回答：

- 平滑中心相似度标签是否优于线性容忍带标签
- 改进来自标签语义，还是仅来自平滑化本身

---

## 9. 最小可执行对照方案

若只做一轮最小必要验证，建议在**相同特征工程、相同时间切分、相同模型框架**下，直接比较以下三种标签：

1. `target_centered_desirability_point`
2. `target_centered_desirability_uncertain`
3. `target_centered_desirability_gaussian`

统一比较：

- centered 主指标误差
- `CQDI`
- hard out-of-spec diagnostic
- 多测量误差场景下的稳定性

---

## 10. Python 伪代码

```python
import numpy as np

def centered_desirability_point(y_obs, center=8.45, tol=0.25):
    return np.maximum(0.0, 1.0 - np.abs(y_obs - center) / tol)

def centered_desirability_uncertain(
    y_obs,
    center=8.45,
    tol=0.25,
    err_half_width=0.15,
    n_grid=21,
):
    delta_grid = np.linspace(-err_half_width, err_half_width, n_grid)
    vals = [
        max(0.0, 1.0 - abs((y_obs + d) - center) / tol)
        for d in delta_grid
    ]
    return float(np.mean(vals))

def centered_desirability_gaussian(
    y_obs,
    center=8.45,
    sigma=0.139,
):
    return float(np.exp(-((y_obs - center) ** 2) / (2 * sigma ** 2)))
```

---

## 11. 最终建议

### 当前最推荐的工程口径
**误差感知线性法**。

原因：

- 它保留了当前 centered 主线最核心的业务语义
- 比当前点式标签更稳健
- 不会像高斯法那样显著改变“容忍带”解释

### 当前最推荐的实验口径
三者并行对照：

- 点式线性法：开发基线
- 误差感知线性法：工程主评估
- 高斯核函数法：平滑中心相似度备选线

如果只能做一个最小改动，优先顺序是：

1. 保留 `point`
2. 新增 `uncertain`
3. 把 `gaussian` 作为同条件对照实验线
