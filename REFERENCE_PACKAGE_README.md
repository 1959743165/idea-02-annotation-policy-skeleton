# Idea 02｜把真实专家分歧变成研究对象，学习“该修什么标注”

**对应第一轮 Idea 2：「把真实专家分歧变成研究对象，学习‘该修什么标注’」。** 本包只研究这一方向：在相同专家人分钟预算下，从新增图像标注、修边界、补漏标、重判类别、额外专家复核和补充成像中，选择病例与动作，使独立测试集风险下降得更快。

这是一套**详细执行方案＋实际可运行的 CPU 参考实现＋明确的真实实验接入接口**。真实焊缝影像、真实专家分歧、工业检测器权重及真实人时数据未在本次工作中取得；包内没有真实训练结果。`runs/` 中的数据、动作时长、图表和模型均标为合成软件检查。参考实现是一小型 NumPy patch 分类器及岭回归动作策略，不是 Weld-LLaVA、YOLO、RT-DETR、ActiveLab 或 DARL 的复现。

## 1. 先读什么

| 文件 | 用途 |
|---|---|
| [中文完整执行手册](docs/实验方案_执行手册.md) | 假设、创新边界、样本计划、所有实验、失败判据、预算与周计划 |
| [专家分歧与动作标注规范](docs/专家分歧与动作标注规范.md) | 三专家独立读片、重复读取、六动作定义、计时与失败保留 |
| [数据获取与防泄漏](docs/数据获取与防泄漏.md) | 试样／原图／焊缝／设备连通分组、数据准入和测试封存 |
| [统计分析计划](docs/统计分析计划.md) | 主要终点、功效估计、试样与采购运行两层不确定性、OPE 边界 |
| [CPU 参考实现](docs/CPU参考实现.md) | 真正执行的模型训练、标签更新、模拟专家及局限 |
| [真实模型与 RapidX 接入](docs/真实模型与RapidX接入.md) | 数据、检测器、标注软件的实施步骤与未取得资源 |
| [评估接口说明](docs/评估接口说明.md) | 评估、nAUC、置信区间、OPE、可视化 API |
| [代码与真实实验边界](docs/代码与真实实验边界.md) | 已实现与待接入的严格对应关系、命令协议及人工台账 |
| [数据与 schema 说明](data/README.md) | 字段、标签编码、示例和真实数据转换规则 |

参考文献在 [可核验参考文献](references/可核验参考文献.md)，来源与核验范围在 [verification_registry.json](references/verification_registry.json)。没有实际核验的资源另列于 [待核验与接入清单](references/待核验与接入清单.md)，不拿它们支持已完成实验的结论。

本次交付文件名为 `idea_02_annotation_revision.zip`。当前运行与交付检查集中在 [本次交付检查记录](checks/本次交付检查记录.md)。本次重新执行完整 smoke 与 83 项单测；包内保留同一实现已运行的 17 项合成基线/消融矩阵，不把它们说成本次重新训练或真实实验。历史日志中的旧目录是执行溯源信息；在自己的机器上使用下方命令生成新运行目录。

## 2. 环境与一条命令跑通

实际验证环境为 **Python 3.12.14、CPU、无需 GPU、无需下载模型**。建议使用 Python 3.12；固定依赖要求 Python ≥3.11。Linux／macOS 示例：

```bash
cd idea_02_annotation_revision
python -m venv ../idea02_venv
source ../idea02_venv/bin/activate
python -m pip install -r requirements.txt
python scripts/smoke_test.py --output-dir /tmp/idea02_fresh_smoke
```

Windows 可在 PowerShell 中激活 `../idea02_venv/Scripts/Activate.ps1`，把 `/tmp/idea02_fresh_smoke` 换成自己的可写输出路径。若固定版本未适配本地 Python/平台，不要静默更换后声称复现；建立新的依赖快照并重跑检查。

smoke 会生成 **180 个模拟试样**、运行 **32 个 policy_train 和 16 个 policy_val 随机付费动作**，对每个动作做前后配对重训，拟合动作收益和正值成本点预测；随后执行四种策略的 20 分钟模拟预算，锁定全部轨迹后才评估 test，并另评 device_test、OPE、功效规划和图表，最后跑单元测试。

交付时这条流程已真实运行并通过；详细结果在 [smoke_results.json](runs/smoke/smoke_results.json)，原始命令输出在 `runs/smoke/logs/`。**验收不要求学习策略优于基线。** 当前合成示例中它未优于不确定性基线；这不证明研究失败或成功，只证明程序没有为了制造优势预设动作收益。

## 3. 逐步执行入口

以下命令在包根目录执行；默认配置里的相对输出路径按当前工作目录解释。

```bash
# 生成与校验公共影像、模拟专家读片；私有真值独立置于 oracle/
python scripts/make_synthetic.py --config configs/smoke.yaml
python scripts/validate_dataset.py --manifest runs/synthetic_data/manifest.jsonl --check-hashes

# 完整的随机动作 pilot → 策略学习 → 预算采购 → 锁定后的独立评价
python scripts/run_experiment.py --config configs/smoke.yaml

# 17 项可运行基线与消融；共享同一份已测 pilot，全部方法锁定后才评价
python scripts/run_matrix.py --manifest runs/smoke/data/manifest.jsonl --pilot-dir runs/smoke/experiment/pilot --output-dir /tmp/idea02_matrix

# 无需重训，重新画已有的独立评价轨迹
python scripts/visualize.py --timelines runs/smoke/experiment/random/test/evaluated_timeline.json runs/smoke/experiment/uncertainty_per_minute/test/evaluated_timeline.json runs/smoke/experiment/learned_gain_per_minute/test/evaluated_timeline.json --horizon-minutes 20 --output-dir /tmp/idea02_figures

# 单步、病例内动作 OPE：使用独立固定 ridge=1 模型，不能复用在 policy_val 调参的主策略
python scripts/offpolicy_evaluate.py --train-logs runs/smoke/experiment/pilot/policy_train_logs.jsonl --logs runs/smoke/experiment/pilot/policy_val_logs.jsonl --n-bootstrap 2000 --output /tmp/idea02_ope.json

# 假设驱动的样本量规划，不是已观察到的效果
python scripts/estimate_power.py --n-specimens 100 150 240 300 --effect 0.02 --paired-sd 0.08 --output /tmp/idea02_power.json
```

独立训练、策略训练和预算入口分别是 `train_detector.py`、`predict_detector.py`、`collect_pilot.py`、`train_policy.py`、`run_budget.py`、`evaluate.py`。每个入口都支持 `--help`；字段和参数以 [API_CONTRACT.md](API_CONTRACT.md) 为准。无需安装本包即可从这些脚本调用；在自己的 Python 中调用时，可先 `python -m pip install -e .`。

## 4. 主实验与可运行矩阵

正式规划的主预算为 **60 个专家人分钟**；20 分钟仅为 smoke 的模拟预算。主比较为 `learned_gain_per_minute` 对 `uncertainty_per_minute`，两者共用相同的成本模型、可行动作集合和初始训练数据。后者用冻结的分歧规则选择动作，按不确定性／预测成本排序，是有成本约束的强基线。

[matrix.yaml](configs/matrix.yaml) 实现 17 项：初始模型不追加标注、随机病例—动作、最低预测成本、不确定性／成本、分歧规则、六个固定动作、完整学习策略，以及去分歧特征、仅动作特征、去倾向权重、去成本排序、去 log-cost 残差校正五项消融。全部 17 项已实际运行，结果在 [matrix_summary.json](runs/matrix/matrix_summary.json)。[完整基线与消融登记](configs/baseline_matrix.csv) 还列出正式论文所需的外部复现和附加实验，其状态明确区分为已实现或待接入。

## 5. 研究上必须遵守的边界

- 多专家分歧不会自动等于错误，训练用相对多数标签也不等于独立真值；重复读片不能增加独立专家票数。
- `new_image` 在 CPU 示例中表示给**已有、尚未标注的影像**取得标注。它没有拍摄新物理试样。`reimage` 只是模拟付费新视图，也没有控制采集设备。
- 模拟 oracle 仅作为付费环境和独立评估参考；策略特征从白名单公开状态提取，不能接触隐藏噪声类型、未来动作效果、未来真实时间或测试风险。
- 奖励是开发 reward 试样上真实的重训前后风险差；policy_train 与 policy_val 另用互不重叠的 reward_fit / reward_tune 面板。
- 失败、无变化、负收益和真实超时都应保留。原始专家读片不覆盖；动作后的当前监督另存版本。现实中多人并行一分钟要按总人分钟收费。
- nAUC 在共同预算区间上按已完成动作作阶梯积分；超出预算才完成的动作，不能提前获得质量收益。固定轨迹的试样 bootstrap 不包含采购运行与专家面板变化。
- OPE 只评价预先指定病例内的单步动作政策，不能据此声称评估了完整的自适应病例选择与预算采购策略。
- 偏向权重只是监督拟合损失的权重。log-cost 回归加训练残差校正仍是成本点预测，不保证精确条件均值；真实实验须另报告成本误差与校准。

## 6. 真实数据与模型如何替换

`configs/real_protocol.yaml` 是未运行的预注册规划，不能直接当作已训练模型配置。先完成数据与专家 pilot；随后使用 `adapters.py` 中的命令协议接本地检测器，用 `prepare_pilot_request.py` 保存已知概率的随机动作请求，用 `import_human_action.py` 导入已完成的实际人时与版本，用 `attach_reward.py` 连接真正的前后重训指标。

这些脚本只处理本地文件。它们不会联系专家、下载授权未明的数据，也不会启动成像设备。`reference_adapter_worker.py` 是可运行的 CPU 协议示例；需要真实检测器时，把 fit/predict 两个操作替换为本地核验过的训练栈，并记录权重、代码版本、原始图预处理、训练试样集合和结果哈希。

RapidX 的公开仓库许可状态、部分模型缺失与导出/计时适配限制已写入接入文档；本包没有打包其源码或权重。

## 7. 检查与复现

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/check_package.py --verify-manifest --report /tmp/idea02_integrity.json
```

[checks/执行检查记录.md](checks/执行检查记录.md) 记录实际非空、语法、smoke、基线矩阵、图片查看及解压复跑结果；[checks/FILE_LIST.txt](checks/FILE_LIST.txt) 是完整文件清单；[checks/file_manifest.json](checks/file_manifest.json) 包含每个覆盖文件的字节数及 SHA256。清单本身和指定的完整性报告按规则排除自引用。修改源代码或在包内生成新结果会使冻结清单变化，这是正常的可追溯提示；把新运行输出放在包外即可保留原始交付快照。

真实论文结果请写到 [results_templates](results_templates/README.md) 对应模板，保持 `NOT_RUN/NA` 直至实验完成，不得把 `runs/smoke` 的数字移入真实主结果表。
