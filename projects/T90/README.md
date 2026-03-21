# T90 推荐控制接口

该子项目不再把重点放在“精确预测单点 T90”，而是提供一个**推荐式控制方法**：  
在当前 DCS/LIMS 上下文相近的历史样本中，寻找“哪些钙/溴组合更容易让 T90 落入合格区间”，并优先给出**钙优先、溴辅助**的控制建议。

这更贴合目前问题背景：

- T90 难以建立高精度单点预测模型；
- 现场真正需要的是“当前工况下如何调节更容易合格”；
- DCS 实时状态更适合作为**上下文状态**，而不是强行作为高精度回归目标。

## 方法概述

交付接口采用一套 CPU-only 的推荐逻辑：

1. 读取样本数据，至少需要：
   - `sample_time`
   - `t90`
   - `calcium`
   - `bromine`
2. 把 `7.5 <= t90 <= 8.5` 视为当前默认合格区间。
3. 将除钙/溴外的数值变量视作**工况上下文特征**，用随机森林筛选最重要的上下文变量。
4. 对目标样本，在这些上下文特征空间中寻找历史近邻工况。
5. 在局部近邻样本内，仅用 `calcium` 和 `bromine` 建立一个轻量局部分类模型，估计哪些组合更可能进入合格区间。
6. 输出：
   - 当前样本信息；
   - 当前工况最重要的上下文特征；
   - 钙优先的推荐调整方向；
   - 在“当前溴固定”时可尝试的钙区间；
   - 在“当前钙固定”时可尝试的溴区间。

> 这是一种**推荐算法 / case-based control recommendation**，不是因果证明，也不是 APC 闭环控制器本身。

## 交付边界

正式交付文件：

- `core/`
- `interface.py`
- `example.py`
- `README.md`

`dev/` 下脚本继续保留为研究分析脚本，但正式接口不依赖 `dev/`。

## 运行环境

CPU-only 即可，不需要 CUDA / TensorRT / GPU。

运行时依赖见仓库根目录 `requirements/runtime.txt`：

- `numpy`
- `pandas`
- `scikit-learn`

## 对外接口

主入口：`interface.py`

```python
from interface import recommend_t90_controls

result = recommend_t90_controls({
    "use_example_data": True
})
```

### 支持的输入参数

`recommend_t90_controls(input_data: dict | None) -> dict`

常用参数如下：

- `data`：`pandas.DataFrame`
- `data_path`：`.parquet` 或 `.csv` 路径
- `use_example_data`：不传数据时，是否直接使用示例数据
- `sample_time`：指定目标样本时间；默认选择最近一个不合格样本
- `target_range`：例如 `[7.5, 8.5]`
- `top_context_features`：上下文特征筛选数量
- `neighbor_count`：上下文近邻样本数
- `local_neighbor_count`：局部控制模型使用的近邻数
- `probability_threshold`：建议点进入合格区间的最低概率阈值
- `grid_points`：钙/溴搜索网格密度
- `include_columns`：若希望只使用指定上下文列，可传列名列表

### 返回结果说明

返回字典主要包含：

- `target_range`：当前合格区间
- `target_sample`：当前目标样本的 T90、钙、溴状态
- `method`：推荐方法说明、钙优先控制策略、上下文重要特征
- `neighborhood`：当前工况匹配到的历史邻域统计
- `recommendation`
  - `best_point`：推荐尝试的最优钙/溴点
  - `recommended_adjustment`：相对当前点的建议增减量
  - `calcium_range_given_current_bromine`：溴不动时的钙推荐区间
  - `bromine_range_given_current_calcium`：钙不动时的溴推荐区间

## 示例

最小示例：

```bash
python projects/T90/example.py
```

也可以直接指定自己的数据：

```python
from interface import recommend_t90_controls

result = recommend_t90_controls({
    "data_path": "projects/T90/example_data/LIMS_data_example.parquet",
    "target_range": [7.5, 8.5],
    "neighbor_count": 120,
    "local_neighbor_count": 80,
    "probability_threshold": 0.60,
})
```

## 关于 DCS 实时状态的使用建议

你提到的核心想法是：  
**利用 DCS 实时状态，或者一个合适时间窗口内的趋势状态，动态推荐钙/溴调节，从而把 T90 控制到合格区间。**

这个方向是合理的，建议按下面思路落地：

1. **DCS 不直接承担高精度 T90 预测任务**  
   更适合作为“当前工况上下文”输入，例如温度、压力、流量、停留时间、挥发分趋势等。

2. **钙作为主控制手柄，溴作为辅助/约束变量**  
   当前接口默认输出钙优先建议，这与现场操作偏好一致。

3. **实时推荐按滚动窗口更新**  
   上线时可将最近 30~120 分钟的 DCS 聚合统计量（均值、末值、波动、斜率）作为上下文输入，再调用推荐接口。

4. **输出“可行区间”而不是单一硬设定值**  
   这样更便于工艺师结合经验、切换牌号、上游扰动和安全边界做最终选择。

5. **和工艺师共同确认合格区间**  
   当前默认区间为 `7.5~8.5`，但接口支持在上线时改为工艺认可的最终范围。

## 注意事项

- 该方法输出的是**推荐结果**，不是因果结论。
- 若示例数据中只有少量合格/不合格样本，局部模型稳定性会下降。
- 若未来把 DCS 窗口特征正式纳入交付数据表，建议将这些成熟逻辑继续沉淀到 `core/`，而不是留在 `dev/`。
