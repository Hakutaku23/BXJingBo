# T90 推荐控制接口

本子项目面向 CPU-only 部署，目标不是直接精确预测 `T90`，而是基于最近一段 DCS 工况窗口，从本地历史案例库中检索相似状态，并推荐更可能使 `T90` 落入目标区间的 `钙量范围` 与 `溴量范围`。

当前默认目标区间为 `8.45 ± 0.25`，等价于 `8.20 ~ 8.70`，可在运行时通过配置或接口参数调整。

## 交付内容

正式交付建议携带以下文件和目录：

- `config/`
- `assets/`
- `core/`
- `interface.py`
- `example.py`
- `README.md`

其中：

- `assets/t90_casebase.csv` 是默认随包携带的本地案例库；
- `config/t90_runtime.yaml` 用于约束目标区间、窗口长度、位点清单和案例库路径；
- `core/` 中保留推荐算法和窗口编码逻辑；
- `interface.py` 是正式对外入口。

`dev/`、`data/`、`test.py` 不属于厂区正式交付物。

### 建议带给厂方的最终文件清单

建议最终交付包只包含以下内容：

- `config/t90_runtime.yaml`
- `assets/t90_casebase.csv`
- `assets/t90_casebase.parquet`（可选，仅在目标环境确认支持 parquet engine 时携带）
- `assets/README.md`
- `core/__init__.py`
- `core/window_encoder.py`
- `core/casebase.py`
- `core/runtime_config.py`
- `core/online_recommender.py`
- `core/setup.py`
- `core/build_linux.sh`
- `interface.py`
- `example.py`
- `README.md`

不建议带入正式交付包的内容：

- `data/`
- `dev/`
- `test.py`
- `__pycache__/`
- 本机构建时产生的 `.c`、`.html`、`build/` 等中间文件

若希望在 Linux 上自动整理出这份最终交付包，可直接使用 [package_delivery.sh](/D:/PSE/博兴京博/BXJingBo/projects/T90/package_delivery.sh)。

## 方法说明

在线推荐分为两层：

1. 用最近固定时长的 DCS 窗口刻画当前工况状态。
2. 在历史案例库中检索相似工况，并给出目标 `钙量/溴量` 推荐区间。

这套方法不要求在线已知当前真实 `钙含量`、`溴含量` 或 `T90`。  
如果现场后续能提供最近一次化验值、设定值或人工参考值，也可以作为可选参考输入，但不是在线调用前提。

## 时间戳说明

系统对时间戳的依赖分成离线和在线两部分：

1. 离线建库阶段依赖时间戳。
   - 需要用 LIMS 的采样时间去对齐 DCS 历史窗口。
   - 这一步是为了把“某次化验结果”对应到“化验前一段时间的真实工况”。

2. 在线推荐阶段不依赖某个固定历史绝对日期。
   - 系统不关心当前是不是某个历史日期本身。
   - 系统关心的是“当前输入的 15 分钟窗口是否按时间顺序排列，以及窗口内点位的相对变化趋势”。

3. 运行时推荐更准确地说依赖的是窗口时序，而不是历史绝对时间。
   - 如果 `dcs_window` 里带有 `time`、`timestamp` 或 `sample_time` 列，系统会用它计算斜率等趋势特征。
   - 如果没有时间列，系统也能运行，但会退化为按行号估计趋势。
   - 因此现场部署时，建议始终传入真实在线时间戳列，并保证窗口按时间升序排列。

4. `runtime_time` 只是结果记录字段。
   - 它主要用于日志和结果追踪。
   - 它不会直接参与推荐计算。

结论：

- 在线系统不依赖“历史绝对日期一致”；
- 但建议依赖“真实在线时间顺序一致”。

## 运行依赖

运行时依赖见仓库根目录 [runtime.txt](/D:/PSE/博兴京博/BXJingBo/requirements/runtime.txt)：

- `numpy`
- `pandas`
- `scikit-learn`
- `PyYAML`
- `onnxruntime`

开发验证额外使用：

- `matplotlib`
- `openpyxl`

## 接口说明

主入口为 [interface.py](/D:/PSE/博兴京博/BXJingBo/projects/T90/interface.py)：

```python
from interface import recommend_t90_controls

result = recommend_t90_controls({
    "dcs_window": current_window_df,
    "runtime_time": "2025-10-19 11:01:00",
})
```

### 输入参数

`recommend_t90_controls(input_data: dict | None) -> dict`

常用输入字段如下：

- `dcs_window`
  最新 DCS 窗口，`pandas.DataFrame`，默认按配置使用最近 `15min`、`1min` 间隔的数据。
- `dcs_window_path`
  当 DCS 窗口以 `csv/parquet` 文件方式传入时可使用该参数。
- `casebase`
  已加载的历史案例库，`pandas.DataFrame`。
- `casebase_path`
  历史案例库路径；未传时默认使用 `assets/t90_casebase.csv`。
- `config_path`
  YAML 配置路径；未传时默认使用 `config/t90_runtime.yaml`。
- `runtime_time`
  当前调用时刻，仅用于结果记录。
- `reference_calcium`
  可选参考钙量，例如最近一次化验值、设定值或人工目标值。
- `reference_bromine`
  可选参考溴量，例如最近一次化验值、设定值或人工目标值。
- `target_spec`
  目标 T90 规格，形式如 `{"center": 8.45, "tolerance": 0.25}`。
- `target_range`
  目标 T90 区间，例如 `[8.2, 8.7]`。
- `include_sensors`
  需要使用的 DCS 位点列表。
- `include_columns`
  需要使用的已编码上下文特征列表。
- `skip_feature_ranking`
  若为 `True`，则跳过每次调用时的上下文特征排序，直接使用配置位点编码后的全部特征。
- `use_example_data`
  若为 `True`，则使用 `example_data/` 中的内置样例。

兼容性说明：

- 仍兼容旧字段 `current_calcium` / `current_bromine`；
- 但在线部署不应再把它们当作必填输入。

### 输出结构

返回结果包含 4 个主要部分：

- `target_range`
  当前目标 T90 区间。
- `runtime_context`
  当前窗口行数、调用时刻，以及可选参考值。
- `method`
  推荐方法说明、主控优先级、使用的上下文特征。
- `recommendation`
  推荐结果主体。

`recommendation` 中重点关注：

- `recommended_calcium_range`
- `recommended_bromine_range`
- `best_point`
- `empirical_good_ranges`
- `best_point_selection_anchor`

若传入了参考钙/溴，还会额外返回：

- `recommended_adjustment`
- `calcium_range_given_reference_bromine`
- `bromine_range_given_reference_calcium`

## 示例

最小示例见 [example.py](/D:/PSE/博兴京博/BXJingBo/projects/T90/example.py)：

```bash
python projects/T90/example.py
```

`example.py` 会构造一段模拟在线 DCS 窗口，然后直接调用接口。  
它不再传入当前真实 `钙/溴`，更接近厂区上线时的实际调用形式。

## 配置文件

默认配置在 [t90_runtime.yaml](/D:/PSE/博兴京博/BXJingBo/projects/T90/config/t90_runtime.yaml)。

当前配置包括：

- `target_spec.center`
- `target_spec.tolerance`
- `target_range`（可选兼容写法）
- `window.minutes`
- `window.step_minutes`
- `controls.primary`
- `controls.secondary`
- `artifacts.casebase_path`
- `data_sources`
- `context_sensors`

当前已经将在线推荐所使用的代表性位点维护在 `context_sensors` 中。后续如果需要增删位点，建议继续只修改 YAML 配置，而不要把位点名单重新硬编码到 Python 代码里。

## 案例库说明

默认案例库为 [t90_casebase.csv](/D:/PSE/博兴京博/BXJingBo/projects/T90/assets/t90_casebase.csv)。

案例库中每一行对应一条历史样本，包含：

- `sample_time`
- `t90`
- `calcium`
- `bromine`
- `is_in_spec`
- 一组由 DCS 窗口编码得到的上下文特征

在线调用时，本地部署包默认只需读取这个 `csv` 文件即可完成推荐，不依赖额外的 parquet 引擎。若目标环境已经确认提供 `pyarrow` 或 `fastparquet`，也可以改用 `parquet` 版本。

## Linux 构建 core

若交付前需要在 Linux 环境中将 `core/` 编译为扩展模块，可直接使用 [build_linux.sh](/D:/PSE/博兴京博/BXJingBo/projects/T90/core/build_linux.sh) 和 [setup.py](/D:/PSE/博兴京博/BXJingBo/projects/T90/core/setup.py)。

推荐环境：

- Ubuntu 22.04
- Python 3.10+
- CPU-only

执行方式：

```bash
cd projects/T90/core
chmod +x build_linux.sh
./build_linux.sh
```

该脚本会：

1. 安装 `setuptools`、`wheel`、`Cython`
2. 使用 Cython 原地编译 `core/` 中的主要模块
3. 清理中间构建缓存

当前参与构建的模块包括：

- `window_encoder.py`
- `casebase.py`
- `runtime_config.py`
- `online_recommender.py`

说明：

- 这一步属于交付前构建流程，不是运行时必需步骤；
- `interface.py` 和 `example.py` 无需改动，仍然通过 `from core import ...` 调用；
- 当前采用的是“先保持 Python 实现，再用 Cython 编译”的方式，便于维护和后续继续优化。

## Linux 打包交付

若需要在 Linux 上直接生成可交付压缩包，可使用 [package_delivery.sh](/D:/PSE/博兴京博/BXJingBo/projects/T90/package_delivery.sh)：

```bash
cd projects/T90
chmod +x package_delivery.sh
./package_delivery.sh
```

默认行为：

1. 调用 `core/build_linux.sh` 编译 `core/`
2. 只收集正式交付内容
3. 对 staging 包做一次 `import` 自检
4. 生成 `dist/T90_delivery_时间戳.tar.gz`

常用参数：

```bash
./package_delivery.sh --output-dir /tmp/t90_dist --package-name T90_BXJB
./package_delivery.sh --skip-build
./package_delivery.sh --python /usr/bin/python3.10
```

说明：

- 默认会检查 `core/` 下是否已经为每个主要模块生成 `.so` 或 `.pyd` 扩展模块；
- 若使用 `--skip-build`，则要求这些编译产物已经存在；
- 打包时会同时携带 `core/` 的 Python 源文件和编译产物，运行时优先使用可导入的模块；
- 打包阶段会对 staging 目录执行一次 `from interface import recommend_t90_controls` 自检；
- 默认打包完成后会删除临时 staging 目录，只保留最终 `tar.gz`；
- 若需要同时保留 staging 目录，可追加 `--keep-stage`。

## 离线验证

开发阶段可使用 [test.py](/D:/PSE/博兴京博/BXJingBo/projects/T90/test.py) 对 `data/` 中的全量私有数据做回放验证：

```bash
python projects/T90/test.py
```

该脚本会：

1. 读取 `data/` 下的 DCS 和 LIMS 数据。
2. 按配置位点构建历史案例库。
3. 逐条回放历史样本，只输入对应时刻前的 DCS 窗口。
4. 输出推荐钙/溴，与 LIMS 真实钙/溴做离线对比。
5. 生成 `csv/json/png` 验证产物。

默认输出位置：

- `dev/artifacts/test_recommendation_results.csv`
- `dev/artifacts/test_recommendation_summary.json`
- `dev/artifacts/recommendation_vs_actual.png`
- `dev/artifacts/recommendation_error_distribution.png`

## 钙量精度评估

若需要单独评估主操作量“钙量”的推荐精度，可使用 [evaluate_calcium_accuracy.py](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/evaluate_calcium_accuracy.py)：

```bash
python projects/T90/dev/evaluate_calcium_accuracy.py
```

该脚本会读取 `test.py` 生成的回放结果，并输出：

- 钙量平均绝对误差 `MAE`
- 钙量中位绝对误差
- 钙量 `P90` 绝对误差
- 钙量最大绝对误差
- 真实钙量落入推荐钙区间比例

默认输出文件：

- `dev/artifacts/calcium_accuracy_summary.json`

说明：

- `平均误差` 适合作为整体稳定性指标；
- `最大误差` 能反映最坏情况，但对异常点较敏感；
- 推荐同时关注 `MAE + P90误差 + 区间覆盖率`，不要只看单一指标。

## 达标判定

若需要自动判断当前推荐系统是否达到交付前验收标准，可使用 [judge_calcium_acceptance.py](/D:/PSE/博兴京博/BXJingBo/projects/T90/dev/judge_calcium_acceptance.py)：

```bash
python projects/T90/dev/judge_calcium_acceptance.py
```

默认判定阈值为：

- 成功推荐率 `>= 1.00`
- 钙量 `MAE <= 0.05`
- 钙量 `P90误差 <= 0.10`
- 钙量 `最大误差 <= 0.25`
- 合格样本落入推荐钙区间比例 `>= 0.55`

这些阈值均可通过命令行覆盖，例如：

```bash
python projects/T90/dev/judge_calcium_acceptance.py --mae-threshold 0.04 --p90-threshold 0.08
```

默认输出文件：

- `dev/artifacts/calcium_acceptance_judgement.json`

## 推荐的交付前验证顺序

建议每次调整配置、案例库或算法后，按以下顺序重新验证：

1. 运行 `python projects/T90/test.py`
2. 运行 `python projects/T90/dev/evaluate_calcium_accuracy.py`
3. 运行 `python projects/T90/dev/judge_calcium_acceptance.py`

如果目标规格从 `8.45 ± 0.25` 改成其他区间，也建议完整重跑这三步，因为样本的合格/不合格标签会一起变化。

## 后续如何继续提升准确性

若后续需要继续更新推荐系统，优先建议更新以下部分：

1. `config/t90_runtime.yaml`
   继续压缩或替换 `context_sensors`，让上下文位点更贴近真实可解释工况。
2. `assets/t90_casebase.csv`
   用更多历史样本、更新批次数据重新构建案例库。
3. `core/window_encoder.py`
   增强窗口特征，例如更稳健的趋势、波动、滞后特征。
4. `core/online_recommender.py`
   调整邻域检索、局部模型、可行域筛选和拒答逻辑。
5. `test.py`
   增加更多离线评估指标与可视化，用于比较新旧版本效果。

如果后续现场能够提供更多在线可观测代理量，例如投料累计量、设定值变化、在线挥发分趋势等，也建议优先把这些量并入 DCS 上下文，而不是重新把在线系统改回依赖当前真实钙/溴。
