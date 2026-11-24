# 附录 D：真实模型、RapidX 与专家动作接入手册

这份文件区分已实现接口和待研究生接入的真实后端。包中 NumPy 检测器、命令桥接、请求/完成事件验证与成本账本可运行；真实焊缝数据、真实专家、真实 YOLO/分割权重和重复成像没有随包提供，也没有完成训练。任何“完成”状态只指对应本地文件流程，不能转写为真实研究完成。

## D1. 第一天先做的核验

确认所持数据及模型的科研使用范围；取得原图、物理分组、类别词典和初始标注版本；选择固定检测任务；在独立环境安装所选真实训练框架并锁定版本、权重hash、CUDA/驱动和许可证。模型不是越新越好，应优先选择团队已经能稳定训练的中小型检测/分割基线。本文贡献不依赖具体厂商API。

RapidX 软件论文 [R03] 的题名、作者、SoftwareX31:102328 与 DOI 已由用户提供原文和作者仓库核验。作者当前仓库公开说明 XML/JSON 标注导出及操作日志、研究源码发布状态、部分模型实现与训练权重未附。其 README 同时指出 LICENSE.txt 当前为空、复用条款不清；这与论文软件表中的 GNU AGPLv3 记载不一致。执行者需要锁定具体commit并核实授权，不能仅按论文表格把第三方源码复制进本项目。当前核验日期为2026-11-07。[R03a]

该仓库的Windows/pywin32与历史requirements需要单独环境审计；本文ZIP不捆绑它的源码、模型、数据或许可证。操作日志入口存在，不等于已验证“逐动作、逐专家、活动时间段”的研究API。先拿10个授权测试例实际导出，检查字段后再写转换器。若软件暂不能运行，可用另一合法标注工具+本包sidecar账本，研究问题不受工具名称绑定。

## D2. 已实现接口总表

| 接口/入口 | 当前作用 | 真实接入缺口 |
|---|---|---|
| DetectorCommandAdapter | 启动研究者指定的本地命令，传request JSON，验response和文件hash | 编写适配实际检测框架的worker |
| scripts/reference_adapter_worker.py | 用包内NumPy后端演示fit/predict协议 | 它不是YOLO、不是工业模型 |
| scripts/run_detector_adapter.py | fit/predict命令桥 | 提供训练/推理manifest及真实命令 |
| scripts/prepare_pilot_request.py | 对一个预先确定pilot病例随机选可用动作并记概率 | 病例入组随机化与专家实际执行由研究协议管理 |
| scripts/suggest_action.py | 从公开上下文选下一任务并写本地请求 | 不自动联系专家、不自动采集图像 |
| scripts/import_human_action.py | 导入已完成/失败动作、汇总多专家人时、保留超时 | 真实标签导出和人工版本核验 |
| scripts/attach_reward.py | 把实际前后重训风险差附到已付费pilot事件 | 必须先真实训练、独立reward评估 |
| scripts/train_policy.py | 在验证后的train/val日志拟合成本敏感策略 | 真实公开特征与样本支持需自行审计 |

代码拒绝用内置SyntheticOracle处理真实病例。真实路径使用已执行的人员事件，没有“找不到真实数据就偷偷回退模拟”的行为。命令桥不下载模型、不访问外部服务、不控制成像设备。

## D3. 真实数据和当前监督导出

用真实图像与标注生成八角色manifest，字段见schema和数据附录。每个对象的原始RapidX导出独立保存；每个专家/读片使用不同文件，避免同名覆盖。转换器保留原始坐标、图像尺寸、类别字典版本、原始位深和变换，输出统一的训练监督。图像重拍后必须更新当前输入，同时继承specimen/weld/source关联。

注意 initial_labeled 只是最初状态，首次标注后应有 active=true 和当前监督路径。训练导出必须包含已购买new_image的新标签，也必须用已支付boundary/missing/class等最新版本。不能只按initial_labeled筛选，或每次训练都重新读取原始共识而忽略修订。

包内 export_current_training_manifest 为NumPy/示例把可见状态导成PNG和current_labelmap_path，另保留原始专家读取、revision_history与hash。current_labelmap_path 是有效训练标签，不是新增专家意见，也不是独立真值。它将示例图转换到8位；真实16位射线数据应由真实worker直接读取授权原始位深并实现经审计预处理，不能无声沿用示例导出而丢失灰度信息。

真实检测框应导出为后端所需的对象目标；若真实任务只有框，不应将框内填满当作精确缺陷掩膜评价。实例分割/像素风险需要专家多边形或掩膜参考。在图像尺度很大时，先按试样分组再切块，保存原图坐标和全图重建规则。

## D4. 训练命令协议

研究者编写 local_detector_worker.py，接收 --request 和 --response 两个参数。request.operation 为fit时读取training_manifest、seed、config、output_dir、training_specimen_ids、training_manifest_sha256。worker只能从该清单和允许的验证清单加载当前标签，初始化固定的同一权重，执行锁定训练步数，保存模型与训练配置/数据hash。

fit响应最少包括 request_id、status="completed"、model_path、model_sha256、training_specimen_ids。训练实际失败必须以非零退出码/失败状态返回，不生成假权重文件。适配器核对响应与请求ID、文件存在且非空、hash和准确训练试样集合；这些程序检查不等于操作系统隔离，worker代码仍须人工审计不能读取test或oracle。

下面命令仅说明已实现桥接的使用形式；真实的 real/current_training_manifest.jsonl 和 local_detector_worker.py 必须先由研究者准备，当前包未包含真实资料：

    python scripts/run_detector_adapter.py --operation fit --command-json '["python","local_detector_worker.py"]' --manifest real/current_training_manifest.jsonl --output-dir runs/real_fit_001 --seed 17 --config-json '{"epochs":50,"image_size":640}'

config-json的字段由真实worker解释，不保证任意框架直接接受。不要把演示epochs和size当成已优化真实超参。先用200优化步基准确认显存与吞吐，再冻结。微调前后用相同初始化/seed、同一训练流程和固定停止规则，避免给某动作更多训练时间。

## D5. 推理与评估协议

operation=predict时request给inference_manifest、model_path和output_dir。命令桥会物化image-only清单：拷贝请求图像，去掉专家标签/私有oracle路径；worker应只读该推理输入。响应包含 request_id、status、predictions_path、predictions_sha256。预测JSONL须覆盖每张image_id恰好一次，记录概率图或框/多边形、分数、类别、原始尺寸、模型hash和必要预处理。

包内评估函数可对三类栅格预测评估；真实多类别/框任务需提供输出转换和相应对象级指标适配。不要把类1圆形、类2线形自动映成工艺缺陷。软maxpatch概率与检测框置信度含义不同，阈值只能在detector_val按真实任务锁定。推理时不要输入annotation或truth；ground truth只交独立评估器。

实际拟合前后评估在同一reward试样上，用相同参考版本。评估JSON至少保存risk、split="reward"、specimen_ids、model_sha256、seed、training_specimen_ids。包内 attach_measured_reward 检查参考试样不在训练、前后参考相同、seed匹配、风险有效、动作病例不充当reward。仅匹配seed并不能证明深度训练完全确定，需记录实际训练随机性并用预定重复种子检查。

test/device_test 的统一评估在所有策略时间线锁定之后再做。真实工程可以把预测与参考留在不同目录/权限域，发布模型和时间线hash给评估人员；研究者不根据test曲线回改已冻结策略。

## D6. RapidX导出映射

先手工标两个已知几何小例：一个边界接触图像边缘，一个含两个对象及unknown。导出XML/JSON，保存原件；查看工具实际字段名和坐标约定，禁止猜测其API。建立mapping.yaml，明确原class字符串→内部class_id、图像文件匹配、box坐标含端点规则、polygon点顺序、旋转/翻转恢复、未知类表示。

转换器的最小验证：
- 同一图像标注导入再导出后，对象数、类别、位置与原文件一致；
- 右下边缘与单像素目标不丢失，不发生width/height颠倒；
- 原图、显示增强图和缩略图坐标不混用；
- 空标注只有在明确完成阴性读取时才是阴性，否则是缺失/未完成；
- 每位专家的read_id与独立文件完整保留；
- 多边形栅格化记录软件/规则，对自交多边形报错或人工修正；
- 重复读取不被新增annotator_id掩盖。

本包没有声称存在可直接调用的 RapidX API。若实际export结构与论文描述不同，以锁定版本实测为准，并把变化写入references/待核验与接入清单.md。

## D7. Sidecar计时和动作请求

无需改标注软件即可执行：准备一个包含image_id、specimen_id、split、features、available_actions、action_quotes_minutes的公开context JSON。pilot病例已经按计划确定后调用随机请求入口，保存分配概率和状态版本：

    python scripts/prepare_pilot_request.py --context real/pilot_case_context.json --output real/requests/event_request.json --protocol-id IDEA02_REAL_V1 --state-version case_v0 --seed 17001

真实执行者读取该本地任务单，不会由脚本自动收到消息。为每位参与者记录start_utc/end_utc有效活动段；暂停以分段表达，不把未参与者的墙钟时间算人时。同一专家重叠段被拒绝，不同专家同时工作的时间相加。完整输出completion含event_id、status、active_intervals、after_state_version、outcome_artifact_path、outcome_artifact_sha256、reason。

status允许completed、failed、cancelled_after_work。已开始后取消仍计费用；完成必须存在非空真实产物和hash。失败的产物可为空但原因必填。实际成本超过报价保留quote_overrun_minutes，不事后改报价。账本追加保留前一ledger的hash；并行多操作员时应改为事务数据库/单写者队列，当前JSONL不是并发系统。

导入命令形式：

    python scripts/import_human_action.py --request real/requests/event_request.json --completion real/completions/event_completion.json --ledger real/human_ledger.jsonl

该步骤只确认实际人时与产物，不发明reward。后续完成配对检测重训和reward评估，再调用：

    python scripts/attach_reward.py --ledger real/human_ledger.jsonl --event-id REPLACE_WITH_ACTUAL_EVENT_ID --before-metrics real/metrics/before.json --after-metrics real/metrics/after.json --reward-panel reward_fit --output real/learning_events/event_reward.jsonl

policy_train对应reward_fit，policy_val对应reward_tune，两个参考试样面板不重叠。每个event输出用新文件，汇总时检测重复ID。训练日志中的reward是前risk减后risk，可为负；失败动作也应在对应不变状态下测量/核对，不应从日志删除。

## D8. 真实预算调度

train_policy学习完并冻结后，准备acquisition_pool当前公共context JSONL，运行suggest_action生成一个待执行请求；remaining-minutes来自已经付费的真实账本，不能使用猜测已花时间。报价用于可行性，预测成本用于排序，真实未来成本不在context。

    python scripts/suggest_action.py --contexts real/current_contexts.jsonl --policy runs/real_policy.npz --remaining-minutes 60 --output real/requests/next_action.json --strategy learned_gain_per_minute --protocol-id IDEA02_REAL_V1 --state-version pool_v0

执行完成后导入账本、更新当前数据版本、训练检测器、重新计算公共context，再选择下一例；主研究同一case仅一次。每步锁定模型hash、总人时、完成状态、当前标签版本、策略配置。不存在可行任务则停止采购并保留未花预算。当前 built-in run_budget 是模拟闭环，真实闭环需由这组文件接口和研究调度程序连接，不能直接把真实manifest送入SyntheticOracle。

超时任务可能导致60分钟尚未交付，不能将之后的模型当作60分钟模型；保留实际费用并在正确完成点进入曲线。新任务是否能并行、GPU更新是否批量，必须所有策略相同且提前冻结。主实现总是继续选择可行且有训练支持的动作，即便预测收益为负；“不操作/提前停止”是可增加的扩展，不能静默只给Full使用。

## D9. 真机接入验收

接口完成条件不是“import成功”。至少做以下小规模干跑：已知标签修订被当前训练清单实际使用；new_image付费后进入训练；失败动作无产物仍扣人时；两专家同时2分钟合计4人分钟；同人重叠段拒绝；超报价保留；不允许重复event；predict清单没有参考标签；训练角色不含reward/test；reward前后试样一致且与训练不交；原始16位信息没有被无声截断；模型/产物hash回查一致。

再用10—20个真实授权例做流程可行性，核对专家看到的任务与日志一致；这些例子不进入锁定test。随后才扩大随机pilot。包内smoke通过只说明合成和接口检查通过，不代表RapidX已经启动、真实模型可训练、专家协议可用或论文假设成立。

