# 焊缝影像标注修订策略研究

研究在固定专家人时预算下，选择病例和标注修订动作，以降低独立测试集上的焊缝指示检测风险。

仓库现包含可运行的 CPU 合成实验代码、测试、结果和图表。真实焊缝影像、专家动作、人时及工业检测器实验尚未完成。合成结果不能用于真实研究结论。

从 [AGENTS.md](AGENTS.md) 了解路径与入口，从 [四周初稿计划](docs/四周初稿计划.md) 了解交付顺序。原实验包的完整使用说明保存在 [参考实现说明](REFERENCE_PACKAGE_README.md)。

运行参考流程可执行 `python scripts/smoke_test.py --output-dir <新输出目录>`。已有合成结果在 `runs/smoke/`，17项矩阵结果在 `runs/matrix/`。论文初稿骨架在 `paper/main.tex`。
