# 数据 schema 与示例

## 示例的身份

`examples/public_manifest.jsonl` 从本包实际生成的 180 个模拟试样中各取一个角色，连同其引用的 PNG 和模拟专家标签一起提供。它是 **8 条合成格式样例**，不是样本量足够的研究数据。`examples/measured_action_logs.jsonl` 是本次合成 pilot 的三个实际重训动作日志，仍不是人工实验。完整可运行数据与日志在 `runs/smoke/`，也可重新运行 `make_synthetic.py` 生成。

`expert_read_format.json` 是明确标注为 synthetic 的采集字段示例，没有伪造专家人时；空 active_intervals 表示尚无人工记录。`completion_format_NOT_REAL_WORK.json` 的时间仅用于展示 ISO8601 格式，不能按真实工作导入或计费。实际请求必须换用当前真实 event_id、实测 interval、实际版本与产物哈希。

## 主表和单位

| 文件/字段 | 语义和约束 |
|---|---|
| public_manifest JSONL | 一行一个影像；原图、裁块、不同视图之间的物理来源必须可连接 |
| image_id | 当前影像唯一 ID；不按其字符串推断类别或错误类型 |
| specimen_id / weld_id / source_image_id | 物理试样、焊缝和源图分组；所有相关版本不得跨角色 |
| device_id / site_id / acquisition_session | 设备、机构和采集会话；device_test 设备必须完全留出 |
| initial_labeled | 初始版本是否有公开标注；付费后不会改写历史 |
| annotations | 原始独立专家读取；唯一键是 image_id＋annotator_id＋read_id |
| is_repeat | 同一专家复读；只能研究复现性，不能冒充新专家投票 |
| available_actions | 在随机化或策略选择前冻结的可行动作集合 |
| action_quotes_minutes | 选择前可见的人分钟报价；不是未来的实测用时 |
| active / current_labelmap_path | 当前训练状态导出专用；允许 new_image 从初始 false 变为当前 active true |
| cost_minutes | 所有人员主动工作区间的总和；并行两人各一分钟计两人分钟 |
| reward | 风险 before−after；真实动作只有取得配对重训结果后才能填写 |

## 编码与坐标

CPU 参考标签 PNG 与图像严格同尺寸：0=背景，1=模拟圆形显示，2=模拟线形显示，255=未知／未达相对多数。1/2 不是焊缝工程类别的完整分类，也不能解释为验收等级。推理输出是 grid×grid×3 概率，不含 255 通道；255 只表示监督和独立参考的未知像素。

真实标注保留原始灰度位深、读片窗宽窗位、像素尺度与所有坐标变换。若任务只有 bbox，不把矩形填充当作精确分割真值；在真实检测器与 evaluator 中采用框级 IoU、实例匹配、图像/试样风险，另注册权重。bbox 采用像素坐标 `[x_min,y_min,x_max,y_max)`，右/下边界不含；polygon 保存原图坐标并说明边界解释。原始文件不覆盖，修订以版本对象记录。

## 角色与使用权

角色为 `seed_train / policy_train / policy_val / reward / acquisition_pool / detector_val / test / device_test`。policy_train 和 policy_val 的动作 reward 另外分别来自 reward_fit / reward_tune 物理试样。detector_val 用于真实检测阈值校准；CPU smoke 的阈值在配置中预先固定，并未事后按 test 调参。

真实 seed_train 可以复用已授权历史单人标签，但必须显式记录 `annotation_provenance=authorized_historical_single_reader_training_only`、`is_synthetic=false`，并独立列账。此例外仅适用于初始训练，不适用于 policy_train、policy_val 或 acquisition_pool 的自然分歧研究。schema 检查会给出警告，不能把这种 seed 数据计入三专家分歧样本数；其他已有标签的研究角色仍要求至少三位独立读者。

`oracle/truth.jsonl` 是合成环境的私有参考。真实数据中应由独立参考面板及封存权限代替，不能从训练专家多数票自动生成“独立真值”。即使当前环境同一进程可以读所有文件，也不得声称这里实现了操作系统级权限隔离。

## 机器可读约束

`schemas/` 提供 Draft 2020-12 JSON Schema：public_manifest、expert_read、measured_action_log、action_request、action_completion、independent_reference、detector_wrapper_response。它们描述记录结构；`validate_dataset.py`、policy 的学习日志检查和 adapter 的人工导入检查还执行跨记录约束、哈希、形状、时间、可行性与分组检查。单独通过 JSON Schema 不能证明无泄漏或专家标注正确。

真实数据转换步骤：先建立原图 provenance 表→完成连通分组并封存 test→将已授权图像及三专家原始读取写入公共 manifest→用相同 image_id 保存独立参考面板→运行 schema/group 检查→用 pilot 校准成本和样本量→随机执行动作、保存当前监督→重训真实检测器→在开发参考上测 reward。正式 test 在所有策略轨迹冻结后一次性评价。
