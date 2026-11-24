# 项目说明

## 研究目标

研究在固定专家人分钟预算下，联合选择焊缝影像病例与标注改进动作，检验学习策略能否比按检测不确定性与预计成本排序的基线更快降低独立测试风险。主终点是60专家人分钟内的归一化风险曲线面积（nAUC），越低越好。真实实验开始前冻结任务类别、参考标准、预算和统计比较。

## 数据路径与状态

- `data/examples/` 为合成格式样例，`runs/smoke/data/` 为已执行的合成数据。
- `data/raw/` 预留授权真实原图，不提交真实原图到公开 GitHub。
- `data/metadata/` 预留匿名试样、焊缝、原图、设备、机构与采集会话关系表。
- `data/annotations/` 预留专家独立初读与版本，`data/action_logs/` 预留随机动作及实测人时。
- `data/manifests/` 预留八个数据角色的清单，`data/references/` 预留独立评估参考。
- 字段和访问边界见 `data/README.md` 与 `docs/数据获取与防泄漏.md`。

## 代码入口

- `scripts/smoke_test.py` 运行完整合成流程和单元测试。
- `scripts/run_experiment.py` 运行合成随机 pilot、策略学习、预算实验和独立评估。
- `scripts/run_matrix.py` 运行17项合成基线与消融。
- `scripts/train_detector.py`、`scripts/predict_detector.py`、`scripts/train_policy.py`、`scripts/run_budget.py`、`scripts/evaluate.py` 是分步入口。
- `scripts/prepare_pilot_request.py`、`scripts/import_human_action.py`、`scripts/attach_reward.py` 提供真实人工动作接入协议。
- `scripts/prepare_data.py` 目前为空占位。代码实现位于 `src/annotation_policy/`，测试位于 `tests/`，依赖见 `requirements.txt`。

## 实验结果与论文

- `runs/smoke/` 和 `runs/matrix/` 包含已完成的合成软件检查、模型、日志、指标与图表。
- `results_templates/` 是真实实验的空结果表，状态保持 `NOT_RUN_REAL_EXPERIMENT`。
- `paper/main.tex` 是论文初稿骨架，`paper/references.bib` 为参考文献，`paper/figures/` 和 `paper/tables/` 放论文选用的图表。
- `docs/四周初稿计划.md` 列出一个月的工作安排与各阶段交付物。

提交论文结果前，核对数据授权和来源分组。专家多数票不能直接充当独立真值，同一试样的关联影像不能跨训练与测试角色。合成图表应始终标记为合成演示。
