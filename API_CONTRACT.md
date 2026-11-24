# 公开接口与职责

## 数据、图像模型与动作环境

```python
generate_synthetic(output_dir, seed=17, n_specimens=180, image_size=48) -> Path
load_public_cases(manifest_path, splits=None) -> dict[str, Case]
train_detector(cases, config=None, seed=17) -> Detector
Detector.predict(case) -> numpy.ndarray  # [grid_h, grid_w, 3]
Detector.save(path) -> Path
Detector.load(path) -> Detector
describe_case(case, detector) -> dict[str, float]
SyntheticOracle(manifest_path, seed=17).apply(case, action, event_id=...) -> ActionOutcome
```

`Case` 为像素、独立原始读片、当前 active 状态、可行动作、报价和付费修订的公开容器。`Case.clone()` 深复制；`save_snapshot()` / `load_snapshot()` 保存和读取真实发生的可见状态。`ActionOutcome` 的 `case` 为新状态，另有 `cost_minutes`、`success`、`action`、`event_id`、`details`。success 是任务完成而非收益保证。所有潜在动作结果按 seed／image_id／action 确定，不依赖策略执行顺序。

私有模拟参考位于数据根目录 `oracle/truth.jsonl`；公开 loader 拒绝 oracle 路径、越界路径及符号链接越界。真正工业数据的原始 16-bit 图像处理须由真实适配器定义，不能未经核验直接交给只接受 8-bit 的参考 loader。

## 策略与实际重训

```python
GainCostPolicy.fit(policy_train_records, config=None, ridge=1.0) -> GainCostPolicy
fit_policy(policy_train_records, policy_val_records, config=None) -> GainCostPolicy
policy.predict(features, actions, quotes=None) -> dict
choose_case_action(contexts, policy, strategy, remaining_minutes, rng) -> dict | None
collect_development_logs(manifest_path, output_dir, config) -> dict
train_policy_from_logs(train_path, val_path, output_path, config=None) -> GainCostPolicy
run_budget(manifest_path, policy, output_dir, config, strategy=..., seed=None) -> dict
evaluate_locked_timeline(manifest_path, timeline, output_dir, evaluation_config=None, split="test") -> dict
```

`contexts` 每项包含 `image_id`、公开 `features`、`available_actions`、`action_quotes_minutes`。提交人工请求时额外要求 `specimen_id`、`split`、`is_synthetic`。默认 feature allowlist 为 `features.CONTEXT_FEATURE_NAMES`；扩展必须在配置中声明 `allowed_feature_names` 并审计其来源。

模型输入设计为常数、标准化状态、动作 one-hot、状态×动作交互；输出训练 reward 和 log-cost。成本再施加训练残差的乘法校正，真实日志默认不按报价截断成本点预测。ridge 仅在 policy_val 选，标准化只在 policy_train 学习。主策略不允许选择训练中没有动作支持的选项。

`no_action` 直接返回 None；其他策略为 random、cheapest、uncertainty_per_minute、disagreement_rule、learned_gain_per_minute、learned_gain_without_cost 或 `fixed_<action>`。单动作／病例约束在预算执行器实施；失败也用完本病例本轮机会。继续执行所有满足策略与预算条件的有支持动作，不包含可学习早停机制。

locked_timeline 内 checkpoint／paid_state 路径相对该 JSON 的父目录。调用评价入口时优先传文件路径，解压到新位置仍可解析；`artifact_base` 仅供同进程字典调用时使用。`paid_actions` 包含所有付费尝试，`completed_actions` 只数 success，`failed_actions` 单列。

## 独立评估与统计

```python
load_truth(manifest_path, split=None, truth_path=None) -> dict[str, numpy.ndarray]
evaluate_detector(model, cases, truth, config=None) -> dict
trajectory_summary(events, horizon_minutes, n_bootstrap=2000, seed=17) -> dict
compare_trajectories(events_a, events_b, horizon_minutes, n_bootstrap=2000,
                     n_permutations=2000, seed=17) -> dict
evaluate_ope(records, target_probabilities=None, *, target_policy_provenance=None,
             q_predictions=None, q_provenance=None, max_weight=None,
             n_bootstrap=2000, seed=17) -> dict
```

风险为每图真实出现类别的加权错误率，先图→试样，再试样等权平均。结果包含完整 `per_specimen`、参考哈希和所有分母。未知 255 不重编码为背景。事件为 `minutes`、`risk`、`per_specimen`；必须从 t=0 开始，预算点风险不得从未来插值。比较方向始终为 **B−A**，主比较令 A=不确定性基线，B=学习策略。

OPE target/q 是 `{event_id: {action: value}}`，必须有独立训练／调参来源。先在 policy_val 调参的模型不能再把同一 policy_val 当作独立 OPE 集；提供的 CLI 改用固定 ridge=1 的独立模型。

## 真实实验文件协议

见 [代码与真实实验边界](docs/代码与真实实验边界.md)。`DetectorCommandAdapter` 使用参数列表运行本地 wrapper，无 shell 拼接。训练导出 `export_current_training_manifest()` 包含实际当前标签及 paid revisions；预测物化仅有图像的白名单 manifest。`person_minutes()` 汇总所有人的 active intervals；`import_human_completion()` 追加失败和超时，仍不生成 reward；`attach_measured_reward()` 仅合并配对的开发集实际评价。
