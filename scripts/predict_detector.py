#!/usr/bin/env python3
"""Image-only prediction from a saved synthetic reference detector."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from annotation_policy.dataset import load_public_cases
from annotation_policy.detector import Detector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test"); p.add_argument("--output", required=True)
    args = p.parse_args(); model = Detector.load(args.checkpoint)
    cases = load_public_cases(args.manifest, args.split)
    if not cases:
        p.error("Requested prediction role is empty")
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for case in cases.values():
            handle.write(json.dumps({"image_id": case.image_id, "specimen_id": case.specimen_id,
                                     "split": case.split, "is_synthetic": case.is_synthetic,
                                     "probabilities": model.predict(case).tolist(), "grid_size": model.grid_size,
                                     "backend": "numpy_synthetic_patch_classifier"}, allow_nan=False) + "\n")
    print(str(target))


if __name__ == "__main__":
    main()
