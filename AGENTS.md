# 项目说明

## 研究目标

在相同专家人分钟预算内，联合选择焊缝影像病例与标注改进动作，检验学习策略能否比按检测不确定性与预计成本排序的基线更快降低独立测试风险。主终点为 60 专家人分钟内的归一化风险曲线面积（nAUC），越低越好。真实实验开始前固定任务类别、独立参考、预算和比较方法。

## 数据路径

- `data/raw/`：获授权的原始影像，仅在受控环境保存，不提交真实数据到 Git。
- `data/metadata/`：匿名试样、焊缝、原图、设备、机构与采集会话关联表。
- `data/annotations/`：各专家独立初读及修订版本。
- `data/manifests/`：数据角色清单和哈希，角色包括 seed_train、policy_train、policy_val、reward、acquisition_pool、detector_val、test、device_test。
- `data/action_logs/`：随机动作概率、完成状态、专家活动时间和实际成本。
- `data/references/`：独立评估参考，访问权限须与训练及策略选择隔离。

## 代码入口

- `scripts/prepare_data.py`：数据入库、谱系分组与角色划分，待实现。
- `scripts/collect_pilot.py`：随机动作预实验，待实现。
- `scripts/train_detector.py`：检测器训练与预测，待实现。
- `scripts/train_policy.py`：动作收益及成本模型训练，待实现。
- `scripts/run_budget.py`：固定预算下执行策略，待实现。
- `scripts/evaluate.py`：轨迹锁定后的独立评估，待实现。

当前 `scripts/` 中的文件为空占位，不可运行。已有 CPU 参考实现位于原实验包 `D:\D02\idea_02_annotation_revision`，迁入时需保留数据角色与测试隔离约束。

## 论文模板路径

- `paper/main.tex`：论文正文占位。
- `paper/references.bib`：参考文献占位。
- `paper/tables/`：主结果、成本和消融表格占位。
- `paper/figures/`：风险曲线等图占位。

真实结果须与合成软件检查结果分开标记。不能把专家多数票直接当成独立真值，或把同一试样的关联影像分到不同数据角色。
