#!/usr/bin/env python3
"""Train the actual NumPy downstream fixture from visible seed annotations."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import yaml
from annotation_policy.dataset import load_public_cases
from annotation_policy.detector import train_detector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/synthetic.yaml"))
    p.add_argument("--manifest")
    p.add_argument("--output-dir")
    p.add_argument("--splits", nargs="+", default=["seed_train"])
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifest = args.manifest or cfg["dataset"]["manifest"]
    output = Path(args.output_dir or cfg.get("output_dir", "runs/reference_detector")); output.mkdir(parents=True, exist_ok=True)
    cases = load_public_cases(manifest, args.splits)
    model = train_detector(cases, cfg, seed=int(cfg.get("seed", 17)))
    checkpoint = model.save(output / "detector.npz")
    result = {"status": "ACTUALLY_TRAINED_NUMPY_SYNTHETIC_REFERENCE_ONLY", "checkpoint": str(checkpoint),
              "diagnostics": model.diagnostics, "metadata": model.metadata, "real_detector_training": False}
    (output / "training_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint), "initial_loss": model.diagnostics["initial_loss"],
                      "final_loss": model.diagnostics["final_loss"], "real_detector_training": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
