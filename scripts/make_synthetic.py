#!/usr/bin/env python3
"""Generate noisy synthetic multi-rater images and isolated simulation truth."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import yaml
from annotation_policy.dataset import generate_synthetic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/synthetic.yaml"))
    p.add_argument("--output-dir")
    p.add_argument("--n-specimens", type=int)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset = cfg.get("dataset", {})
    print(generate_synthetic(args.output_dir or dataset.get("output_dir", "runs/synthetic_data"),
                             seed=int(cfg.get("seed", 17)),
                             n_specimens=args.n_specimens or int(dataset.get("n_specimens", 180)),
                             image_size=int(dataset.get("image_size", 48))))


if __name__ == "__main__":
    main()
