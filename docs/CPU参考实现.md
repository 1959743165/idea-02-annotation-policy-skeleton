# CPU 参考实现：数据、检测器与付费标注动作

本文件解释包中已经实现、可以执行的数值后端；它是 Idea 2 的软件验证组件。真实论文仍需使用独立专家、真实时间记录和适配后的工业检测模型完成实验。这里没有完成真实专家招募、真实焊缝训练、真实成本测量，也没有把模拟标注称为自然专家分歧。

## 1. 先运行什么

在包根目录执行：

```bash
python -m pip install -r requirements.txt
python scripts/make_synthetic.py --output-dir runs/synthetic_data
python scripts/train_detector.py --manifest runs/synthetic_data/manifest.jsonl --output-dir runs/reference_detector
python scripts/predict_detector.py --manifest runs/synthetic_data/manifest.jsonl --checkpoint runs/reference_detector/detector.npz --split test --output runs/reference_detector/test_predictions.jsonl
python -m unittest discover -s tests -p 'test_core.py' -v
```

完整动作策略、预算循环、评估和可视化入口见 README。仅运行 `train_detector.py` 不能检验“选什么标注动作更有效”的研究假设；必须运行带真实再训练的预算实验。

默认后端为 NumPy 全批量多类逻辑回归，依赖 NumPy、SciPy、Pillow、PyYAML；不下载权重，不要求 CUDA。这样可以在普通 CPU 上重复执行大量候选动作的前后重训，检查策略训练信号是否来自真实模型变化，而不是预设动作奖励。

## 2. 数据范围与防泄漏边界

### 2.1 三个数据空间

| 空间 | 可读取内容 | 不可读取内容 |
|---|---|---|
| 检测器与策略 | 当前图像、目前已付费得到的独立/重复标注、结构性可选动作、公开最长时长报价 | 私有像素真值、模拟错误类型、隐藏难度、未来动作成本与效果、测试集指标 |
| 付费动作模拟环境 | 仅执行动作时读取相应候选的私有真值，用于生成不完美的新标注 | 不得返回潜在奖励或其他未执行动作的结果给策略 |
| 独立评估器 | 指定 reward / validation / test 角色的封存参考标签 | 不得将测试指标回流到策略选择、超参或阈值搜索 |

`load_public_cases()` 只打开 `manifest.jsonl` 指向的公开图像和标注；拒绝绝对路径、`..` 路径、`oracle/` 路径以及解析后越界或进入 oracle 的符号链接。`SyntheticOracle` 与独立评估器是显式的私有读取边界。这是透明的软件边界，不是对恶意 Python 代码的操作系统隔离；正式实验应使用独立服务账号、目录权限或分离进程实施访问控制。

### 2.2 180 个独立模拟试样

| 角色 | 默认数量 | 可用于拟合检测器 | 可执行付费动作 | 对策略可见标注 |
|---|---:|---|---|---|
| seed_train | 32 | 是；共同初始集 | 否 | 每图至少三位模拟独立标注者 |
| policy_train | 32 | 仅策略开发的相应重训回合 | 是 | 24 张已有标注，8 张未标注 |
| policy_val | 16 | 仅策略开发的相应重训回合 | 是 | 12 张已有标注，4 张未标注 |
| reward | 20 | 否 | 否 | 公共视图无标签；独立评估器保管参考 |
| acquisition_pool | 32 | 主预算实验中当前已激活部分 | 是 | 24 张已有标注，8 张未标注 |
| detector_val | 16 | 否 | 否 | 无训练标注；仅固定阈值/验证用途 |
| test | 16 | 否 | 否 | 无训练标注；测试封存 |
| device_test | 16 | 否 | 否 | 无训练标注；设备外测试封存 |

模拟中每张图对应独立 specimen_id，因此“试样分组”检查可以直接验证。真实应用必须将同一试样的多个切片、多个曝光、裁剪、增强、近重复图统一分组；本模拟数据不会替研究者自动解决这些关系。`SYNTH_DEVICE_3` 仅出现在 device_test；其余角色共享另外三台模拟设备，并非每个角色都隔离设备。

图像为 48×48、8 位灰度、两种几何指示加背景：0 背景；1 圆形模拟指示；2 线状模拟指示。圆形/线状只是数值夹具的类别，不是对气孔、裂纹或安全等级的真实等同。设备外域移由噪声、模糊和灰度变换合成，不能据此声称跨设备泛化。

### 2.3 不完美标注如何产生

每个已有标注图像从六位模拟标注者中分配三位。隐藏的错误模式包括漏标、类混淆、边界扰动、混合错误及高不确定情形；不同标注者既有相关偏差，也有独立随机差异。误差参数是明确编写的实验假设，不是从专家人群估计出的分布。部分图像带同一模拟标注者的重复读取，用于验证重复读片的统计处理。

所有原始读取都保持不变。额外复核增加独立读取，修订动作写入 `effective_override`，并追加 `revision_history`；不覆盖原始三人标注。因此可以追溯“改变了什么、花了多少、是否完成”，也可以重新分析分歧。

## 3. 当前训练标签的聚合

`independent_reads(case)` 每位标注者仅保留一个最新非重复读取。重复读取不获得额外投票权，否则同一人的多次标注会伪造“多数意见”。重复读取的差异另算 `repeat_disagreement`。

`consensus_labelmap(case)` 对 0/1/2 类逐像素取相对多数。最高票并列或完全未知时写入 255。255 表示未知，不是背景，损失中应忽略。动作修订存在时使用 `effective_override`；原始读取仍完整保存。

相对多数仅是本 CPU 检测器的可复现训练聚合方法，不是最终金标准。真实试验应把独立裁决/复核参考与训练标签隔离，并报告其他聚合方法的敏感性分析。

将 48×48 标签投影为 8×8 个 patch 时，每个 patch 有 6×6 像素。目标为已观察像素中三类面积比例，权重为该 patch 已观察面积占比。一个只有 1 个像素的指示贡献 1/36 的面积目标，不会因最近邻缩放而随机消失；含 255 的 patch 不会把未知区域计为背景。该设计用于数值测试，不代表其定位分辨率足以支撑真实微小缺陷检测。

## 4. 检测器是真实优化的模型

### 4.1 仅从图像构造 15 维特征

每个 patch 提取灰度均值、标准差、极值、相对局部背景的暗度差、阈值暗区域比例、两个方向梯度能量、梯度各向异性、图像阈值连通区域的细长度与面积、邻域暗区域比例与对比度、归一化空间坐标。

连通区域来自当前图像阈值，不读取真值实例。所有特征函数仅接收图像数组；模型推理入口也只访问 `case.image`。这些特征对模拟几何具有明确归纳偏置，所以模型性能和动作收益不能外推到真实焊缝。

特征缓存键包含图像内容 SHA-256、图像尺寸和网格尺寸。重新成像后像素发生变化，缓存键也变化；同一图像标注变化时图像特征可以安全复用。缓存返回副本，调用方不能污染后续训练。

### 4.2 损失与优化

令第 j 个 patch 标准化图像特征为 x_j，面积目标为 y_jc，已观察比例为 v_j，类别权重为 w_c，预测为 p_j = softmax(x_j W + b)。优化：

\[
L=-\frac{\sum_j v_j\sum_c w_cy_{jc}\log p_{jc}}
{\sum_jv_j\sum_cw_cy_{jc}}+\frac{\lambda}{2}\lVert W\rVert_F^2.
\]

标准化均值和标准差仅由本次活跃训练集的图像特征计算。默认类别权重 `[1,3,4]`、L2=0.001、Adam 学习率 0.035、60 轮、初始化标准差 0.015、种子 17。Adam 一阶/二阶系数 0.9/0.999，epsilon=1e−8。参数、梯度、优化器均为实际数值实现。类加权 CE 的输出未经真实分布校准，不能直接称为工业缺陷概率。

每次模型保存 `initial_loss`、`final_loss`、参数变化范数、完整损失轨迹和实际耗时。有限差分测试逐一检查一个随机小问题中 W 和 b 的 15 个参数，含软目标、缺失像素权重、类别权重与 L2。测试还验证修改公开监督会实际改变参数、相同初始化和数据重训完全一致。

### 4.3 为什么同一初始化仍会产生不同结果

对候选动作的前后模型使用相同初始化、相同轮数和相同超参，目的是减少与动作无关的训练随机性。动作确实可能改变训练标签、激活一张新图或改变图像质量，随后优化得到的新参数和 reward 集预测都可能变化。动作可能无效、改善或变差。代码不赋予任一动作预定奖励，测试也不要求 proposed 策略获胜。

## 5. 策略状态的固定接口

`describe_case(case, model)` 返回下列数值字段。所有字段只使用当前公共状态。

| 字段 | 含义 |
|---|---|
| model_uncertainty | 模型各 patch 的归一化熵中最高 8 个的均值 |
| entropy | 全图 patch 归一化熵均值 |
| existence_disagreement | 独立标注者是否认为有任意指示的分歧，4p(1−p) |
| class_disagreement | 类 1/2 是否存在的分歧均值 |
| boundary_disagreement | 双方均报告前景时，前景掩膜 1−IoU 的成对均值 |
| boundary_pair_fraction | 满足边界比较条件的标注者对占比，避免缺省 0 被误读为一致 |
| repeat_disagreement | 同一标注者重复读取在前景并集中的标签差异比例 |
| image_quality | 公开图像的对比度/(对比度+4×高频噪声)；不是经临床/工业校准的质量分 |
| image_contrast | 第 95 与第 5 百分位灰度差 |
| n_annotators / repeat_read_count | 独立标注者数/重复读取数 |
| active | 是否已有可用于训练的当前标签 |
| annotation_foreground_fraction | 当前公开聚合标签的前景面积比例 |
| model_annotation_gap | 模型平均前景分数与公开前景面积比例之差的绝对值 |

该简化状态按整图聚合。多个实例的类别/边界分歧不能被完全解耦，真实实现须扩展实例匹配、专家置信度、缺陷属性和成像可读性记录，并在开发集验证。包中实现在文档与接口层面明确保留这一限制。

禁止把 `simulation_error_mode`、`latent_difficulty`、真实缺陷类别、待执行动作的真实时间、潜在收益或测试集结果追加为策略输入。测试会注入这些假 metadata，确认 context 数值不变；这是一条代码行为检查，不替代独立的数据访问审计。

## 6. 六类动作：效果与费用

| 动作 | 结构性前提 | 合成环境中的具体变更 | 刻意保留的限制 |
|---|---|---|---|
| new_image | 已采集但尚未标注的候选图 | 付费生成三位模拟标注者的首次读取；成功后激活训练 | 不代表免费新增物理采集；仍可能漏标、错类、边界误差 |
| boundary | 已有标注 | 匹配现有报告区域后修订其边界，保留此前报告类别 | 不主动补充未报告缺陷，不顺带纠正类别 |
| missing | 已有标注 | 对未被公共标签充分覆盖的指示执行搜索，可能添加新区域 | 搜索可能无收获；新增区域仍可能有边界/类别残差 |
| class | 已有标注 | 对已有区域重新判断类别，保持原报告形状 | 不补漏、不修边界 |
| review | 已有标注 | 新增第四位独立模拟复核者的读取，重新聚合 | 第四票可能引入平票、无变化或更差的聚合结果 |
| reimage | 已有标注且公开物流状态允许 | 用相同几何的较低噪声再成像替换当前图像 | 不免费修正标签；真实重新曝光的配准、缺陷变化和人工核对需另实现 |

公开最长报价按图像面积预先计算，在默认 48×48 时分别为 4.5、2.0、3.2、1.4、2.5、5.5 分钟。这些数字仅为可执行预算夹具，不是人力成本估计。实际费用在执行时生成，始终为正且不超过报价；未完成动作同样收费。正式实验必须改用任务系统实测总工时，记录加载、阅片、交互、纠错、复核和等待口径。

动作可能未完成，完成也可能没有改变标签，更不能把 `success=True` 解释成风险下降。每个候选在主实验中最多执行一次付费动作，失败也消耗这一次机会；这是明确的短时域动作分配问题，尚未实现对同一试样重复多步修订的长期控制。

随机潜在结果按 `(环境种子, image_id, action)` 决定，不依赖策略的执行顺序或 event_id。因而同种子下不同策略执行同一个动作会得到相同像素/标签变化和费用。event_id 只用于审计与命名。代码无“方法标签→优待概率”分支。

## 7. 可直接调用的接口

```python
from annotation_policy.dataset import generate_synthetic, load_public_cases
from annotation_policy.detector import train_detector, Detector
from annotation_policy.features import describe_case
from annotation_policy.oracle import SyntheticOracle
from annotation_policy.annotation_state import Case

manifest = generate_synthetic("runs/example", seed=17, n_specimens=180)
public = load_public_cases(manifest)
seed_cases = [c for c in public.values() if c.split == "seed_train"]
candidate = next(c for c in public.values()
                 if c.split == "policy_train" and "review" in c.available_actions)
model = train_detector(seed_cases + [candidate], {"epochs": 60}, seed=17)
context = describe_case(candidate, model)

# 仅环境执行器持有 oracle；选择函数只接收 context 和公开报价。
outcome = SyntheticOracle(manifest, seed=17).apply(candidate, "review", "example_001")
changed = outcome.case
changed.save_snapshot("runs/example/paid_case_snapshot.npz")
restored = Case.load_snapshot("runs/example/paid_case_snapshot.npz")
after = train_detector(seed_cases + [restored], {"epochs": 60}, seed=17)
after.save("runs/example/after_detector.npz")
loaded = Detector.load("runs/example/after_detector.npz")
scores = loaded.predict(restored)  # [8,8,3]；仅访问 image
```

快照使用 `allow_pickle=False` 的 NPZ，保存当前像素、全部原始及新增读取、修订覆盖、公开元数据、活跃状态和已发生的费用历史。恢复快照不重新读取原始标注路径，也不接触 oracle。检测器 checkpoint 同样不需要 pickle。

## 8. 更换真实数据与模型时必须完成的工程工作

1. 导出真实图像、试样/焊口/设备关系与至少三人原始标注，按 schema 保留所有读取；不要将专家姓名、私人联系方式或外部路径塞入模型特征。实例 mask/框需投影到统一坐标系，保留未确定区域和标注工具版本。
2. 按设备外测试与试样组锁定角色，先执行 split/重复检测审计，再打开参考标注。将独立裁决参考单独授权给评价进程。
3. 替换 NumPy fixture backend。真实模型适配器需实现 `train(cases, config, seed)`、`predict(case)`、`save/load`，并与评价器协商输出类别、空间尺度、实例匹配及校准协议。现有代码对 `is_synthetic=False` 的训练直接拒绝，不会静默冒充工业后端。
4. 用真实任务分发系统替换 `SyntheticOracle.apply`。系统接收 image_id、action、已展示状态版本和事件 ID；返回新读取或修订版本、是否完成、实测总工时及操作日志。不得提前向选择器发送其他动作的未来效果。
5. 重新采集随机化 pilot，按真实 reward 集上的模型前后重训测得奖励。不能复用模拟 gain/cost 训练权重，也不能把合成数据的动作成功率/报价当先验实测结论。
6. 对正式模型检查输入坐标、类别映射、无标签推理、相同初始化配对训练和真实总时间计费。完成小规模专家/工程双验收后再进入预注册的预算比较。

本包没有未运行的 PyTorch 检测器伪实现，也没有声称已集成 Weld-LLaVA。该 Idea 的主对象是标注动作选择；检测器应为受控下游任务，不需要把解释生成混入此方向。

## 9. 这些测试能证明什么

核心单元测试覆盖解析/训练、解析梯度、标签改变确实影响参数、推理仅访问图像、重复读取不重复投票、未知像素不等于背景、公共加载器不读私有路径、固定潜在结果、失败动作仍收费、六动作原始标注保持、检查点/公共状态快照往返一致。

它们证明本代码的数值通路与关键数据边界按设计工作。它们不能证明真实专家分歧的规律、真实工资工时分布、真实工业安全性能，也不能证明学习策略优于任何基线。报告实验结果时必须把 smoke test 和真实研究证据分别陈述，保留失败与负收益结果。
