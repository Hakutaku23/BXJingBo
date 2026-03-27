# T90 不合格告警阶段说明

## 1. 当前阶段结论

本阶段已经单独验证了一个“`T90 是否不落入 8.45 +/- 0.25`”的独立告警模块。  
在当前离线数据和 `50min DCS` 窗口条件下，**最稳的方案不是阶段化推荐器的间接概率，也不是带 PH 的阶段化告警器，而是一个全局 `DCS-only` 的直接分类器**。

当前推荐保留的研发路线是：

- 主模型：`global_dcs_ensemble`
- 输入：当前 `50min` 的 DCS 窗口
- 输出：`T90_out_of_spec` 概率
- PH：不是这个告警模块的必要输入
- 阶段识别：不是这个告警模块的必要前置步骤

## 2. 为什么单独做独立告警模块

此前的推荐系统更偏向“给出钙/溴建议”，它的概率值本质上是推荐点局部可行性的概率，**不是专门为不合格告警训练的**。  
因此，如果直接把推荐器里的概率拿来当坏样本告警分数，表现会偏弱。

本轮实验把目标直接定义成：

- `1`：T90 不在 `8.45 +/- 0.25` 范围内
- `0`：T90 落在 `8.45 +/- 0.25` 范围内

这样训练出来的模块，目标和现场告警任务是一致的，所以更适合作为单独的“坏样本预警器”。

## 3. 已验证的三种策略

实验脚本：

```powershell
& 'D:\miniconda3\envs\Learn\python.exe' projects\T90\dev\oos_alert_experiment.py
```

比较过的策略有：

- `global_dcs_ensemble`
  - 直接用全量历史样本训练一个全局 `DCS-only` 二分类器
- `stage_dcs_ensemble`
  - 先按阶段拆分，再在阶段内训练 `DCS-only` 告警器
- `stage_hybrid_optional_ph`
  - 阶段内优先使用带 PH 的告警器，PH 不可用时回退到 `DCS-only`

## 4. 当前结果

本轮离线回放样本数：

- 全部样本：`1838`
- 有可评分 OOF 结果的样本：`1530`
- 不合格比例：`0.3660`

关键指标如下：

### 4.1 全局 DCS-only 直接分类器

- `ROC AUC = 0.5638`
- `Average Precision = 0.4073`
- 当前综合最优，已被选为推荐方案

### 4.2 阶段化 DCS-only 分类器

- `ROC AUC = 0.5145`
- `Average Precision = 0.3574`

### 4.3 阶段化可选 PH 分类器

- `ROC AUC = 0.5546`
- `Average Precision = 0.3957`

### 4.4 和当前推荐器的间接告警基线相比

当前推荐器间接告警基线定义为：

- `1 - best_point_probability`

它的结果是：

- `ROC AUC = 0.4824`
- `Average Precision = 0.1914`

所以当前可以明确认为：

- **独立告警模块明显优于“从推荐器概率间接推出告警”的方式**
- **当前最稳的独立告警路线是全局 `DCS-only` 直接分类**

## 5. 阈值应该怎么定

阈值扫描脚本：

```powershell
& 'D:\miniconda3\envs\Learn\python.exe' projects\T90\dev\oos_alert_threshold_sweep.py
```

这个脚本会单独给出三类阈值：

- `balanced_f1`
  - 偏整体平衡
- `low_miss`
  - 偏“漏报少”
- `low_false_alarm`
  - 偏“误报少”

建议现场优先把 `low_miss` 作为默认模式，因为产品质量场景里，**漏掉坏样本通常比多报几个人工复核更贵**。

更具体地说：

- 如果目标是“宁可多提示人工确认，也尽量不要漏掉坏样本”，用 `low_miss`
- 如果目标是“只在风险非常明显时才报警，减少无效告警”，用 `low_false_alarm`
- 如果是开发阶段做整体平衡对比，用 `balanced_f1`

当前这轮实际扫出来的结果是：

- `balanced_f1 = 0.13`
  - `precision = 0.3686`
  - `recall = 0.9946`
  - `false_positive_rate = 0.9835`
- `low_miss = 0.19`
  - `precision = 0.3753`
  - `recall = 0.9321`
  - `false_positive_rate = 0.8959`
- `low_false_alarm = 0.45`
  - `precision = 0.4519`
  - `recall = 0.4446`
  - `false_positive_rate = 0.3113`

这组结果说明：

- `low_miss` 更适合作为“早筛风险分数”或“人工复核门槛”
- `low_false_alarm` 更适合作为“现场显式告警门槛”
- `balanced_f1` 在当前任务上并不适合作为正式现场阈值，因为它几乎会把所有样本都判成风险样本

因此，本阶段更推荐的运行理解是：

- 默认研发模式：`low_miss`
- 默认现场显式告警模式：优先考虑 `low_false_alarm`

## 6. 更贴近未来正式接口的 dev 入口

本阶段已经新增了一个更接近未来正式接口的 dev 原型：

- `projects/T90/dev/oos_alert_interface_prototype.py`

函数入口：

- `predict_t90_oos_alert_v_next_dev(input_data)`

它的特点是：

- 只要求当前 `DCS` 窗口
- 不依赖 PH
- 不依赖阶段识别
- 支持通过 `alert_mode` 切换阈值策略
- 支持直接传自定义阈值

当前支持的 `alert_mode`：

- `low_miss`
- `low_false_alarm`
- `balanced_f1`

运行示例：

```powershell
& 'D:\miniconda3\envs\Learn\python.exe' projects\T90\dev\oos_alert_interface_prototype.py --use-private-example --alert-mode low_miss
```

## 7. 是否值得继续推进到正式第二版

当前判断是：**值得继续推进，但应先作为“独立告警子模块”进入，而不是立即替换现有推荐主链路。**

更稳的推进方式是：

- 推荐系统继续负责：
  - 钙/溴建议
- 独立告警模块负责：
  - 当前样本是否可能不合格

这样做的优点是：

- 目标分离更清楚
- 现场解释更自然
- 阈值更容易单独维护
- 后续可以独立升级告警器，而不必同时改推荐器

## 8. 当前限制

- 当前 `AUC` 和 `AP` 已经能支持“预警/筛查”，但还谈不上强鲁棒的精确判废器
- 现在的最佳方案仍然是 `DCS-only`，说明 PH 在坏样本告警任务上的增益还不够稳定
- 不同阶段表现确实不均衡，但在这个任务上，阶段化模型并没有打赢全局直接模型
- 所以当前最稳的策略应该是：**先把全局直接告警做扎实，再考虑复杂化**

## 9. 当前阶段建议

如果后续继续迭代，建议顺序如下：

1. 固化 `global_dcs_ensemble` 的阈值策略
2. 将 dev 接口原型继续打磨成可迁移到 `core/ + interface.py` 的正式接口
3. 在更多新批次数据上复验阈值稳定性
4. 只有当全局模型稳定后，再重新评估是否值得引入阶段化或 PH 增强
