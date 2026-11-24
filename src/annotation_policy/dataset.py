"""Synthetic images and noisy simulated raters, with private truth separation.

The generator is a controlled *software fixture*. Its error distributions are
assumptions, not estimates of natural expert disagreement or industrial costs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from .annotation_state import AnnotationRead, Case, ACTIONS, DEVELOPMENT_ACTION_SPLITS

SPLITS = ("seed_train", "policy_train", "policy_val", "reward", "acquisition_pool", "detector_val", "test", "device_test")
DEFAULT_COUNTS = np.array([32, 32, 16, 20, 32, 16, 16, 16])


def stable_seed(seed: int, *parts) -> int:
    """Order-independent potential outcomes; event IDs intentionally not included."""
    payload = "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _split_counts(n: int) -> np.ndarray:
    if n < 40:
        raise ValueError("At least 40 synthetic specimens are required for eight disjoint roles")
    desired = DEFAULT_COUNTS / DEFAULT_COUNTS.sum() * n
    counts = np.maximum(np.floor(desired).astype(int), 3)
    while counts.sum() < n:
        counts[np.argmax(desired - counts)] += 1
    while counts.sum() > n:
        eligible = np.where(counts > 3, counts - desired, -np.inf)
        counts[np.argmax(eligible)] -= 1
    return counts


def _objects(labelmap):
    found = []
    for class_id in (1, 2):
        components, count = ndimage.label(np.asarray(labelmap) == class_id)
        for k in range(1, count + 1):
            mask = components == k
            if mask.any():
                found.append((class_id, mask))
    return found


def _deform(mask, rng, severity=2):
    shift = rng.integers(-severity, severity + 1, size=2)
    moved = ndimage.shift(mask.astype(float), shift, order=0, mode="constant", cval=0) > .5
    choice = int(rng.integers(3))
    if choice == 0:
        moved = ndimage.binary_dilation(moved, iterations=1)
    elif choice == 1:
        smaller = ndimage.binary_erosion(moved, iterations=1)
        if smaller.any():
            moved = smaller
    return moved


def make_simulated_read(truth, image, rng, *, mode="mixed", quality=.55, difficulty=.5):
    """Oracle-only simulation of a rater, not a real annotation model.

    Caller must never pass truth into the selector or detector. Shared mode
    creates correlated mistakes across raters, so plurality is not an oracle.
    """
    result = np.zeros_like(truth, dtype=np.uint8)
    miss = np.clip(.02 + .20 * (1 - quality) + .12 * difficulty + (.32 if mode == "missing" else 0), 0, .75)
    wrong_class = np.clip(.02 + .12 * (1 - quality) + (.45 if mode == "class" else 0), 0, .65)
    boundary = np.clip(.12 + .35 * (1 - quality) + (.40 if mode == "boundary" else 0), 0, .90)
    if mode == "uncertain":
        miss += .12; wrong_class += .12
    if mode == "clean":
        miss *= .2; wrong_class *= .2; boundary *= .25
    for class_id, mask in _objects(truth):
        if rng.random() < miss:
            continue
        visible = _deform(mask, rng, 2) if rng.random() < boundary else mask.copy()
        assigned = 3 - class_id if rng.random() < wrong_class else class_id
        result[visible] = assigned
    # Occasional spurious annotation, not a latent defect. Reviews may retain it.
    if rng.random() < .025 + .055 * (1 - quality):
        height, width = result.shape
        x = int(rng.integers(3, width - 3)); y = int(rng.integers(3, height - 3))
        yy, xx = np.ogrid[:height, :width]
        result[(xx - x) ** 2 + (yy - y) ** 2 <= 4] = int(rng.integers(1, 3))
    # Self-reported uncertainty is simulated from visible quality, not accuracy.
    highpass = np.asarray(image) - ndimage.gaussian_filter(image, .8)
    noise = float(np.median(np.abs(highpass - np.median(highpass))))
    contrast = float(np.percentile(image, 95) - np.percentile(image, 5))
    reported = float(np.clip(noise / (contrast + .01) * 2.5 + rng.normal(.15, .07), 0, 1))
    return result, reported


def _render_truth(rng, size, device):
    truth_image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(truth_image)
    category = int(rng.choice(4, p=[.18, .36, .34, .12]))
    classes = [] if category == 0 else [1] if category == 1 else [2] if category == 2 else [1, 2]
    centers = []
    for class_id in classes:
        for _ in range(30):
            x, y = rng.integers(size // 4, 3 * size // 4 + 1, size=2)
            if all(np.linalg.norm(np.asarray([x, y]) - c) > size * .28 for c in centers):
                break
        centers.append(np.array([x, y]))
        if class_id == 1:
            r = int(rng.integers(max(3, size // 12), max(4, size // 7) + 1))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=class_id)
        else:
            length = rng.uniform(size * .28, size * .43)
            angle = rng.uniform(-np.pi, np.pi)
            dx, dy = length * np.cos(angle) / 2, length * np.sin(angle) / 2
            draw.line((x - dx, y - dy, x + dx, y + dy), fill=class_id,
                      width=int(rng.integers(max(3, size // 12), max(4, size // 8) + 1)))
    truth = np.asarray(truth_image, dtype=np.uint8)
    yy, xx = np.mgrid[:size, :size] / size
    background = .62 + .05 * (xx - .5) + .04 * np.exp(-((xx - .5) / .24) ** 2)
    background += .012 * np.sin(yy * 10 + rng.uniform(0, np.pi))
    difficulty = float(rng.beta(1.8, 1.8))
    contrast = .09 + .22 * (1 - difficulty)
    clean = background - contrast * ndimage.gaussian_filter((truth > 0).astype(float), .35)
    clean = np.clip(clean, 0, 1)
    noise_sigma = .008 + .025 * difficulty + (.012 if device == "SYNTH_DEVICE_3" else 0)
    blur_sigma = .35 + .7 * difficulty + (.15 if device == "SYNTH_DEVICE_3" else 0)
    observed = ndimage.gaussian_filter(clean, blur_sigma) + rng.normal(0, noise_sigma, size=(size, size))
    if device == "SYNTH_DEVICE_3":
        observed = observed * .88 + .05
    return np.clip(observed, 0, 1), clean, truth, difficulty


def _save_gray(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)).save(path)


def generate_synthetic(output_dir, seed=17, n_specimens=180, image_size=48) -> Path:
    """Write public noisy data and explicitly separate private synthetic truth."""
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    if image_size < 32 or image_size % 8:
        raise ValueError("image_size must be >=32 and divisible by 8")
    counts = _split_counts(int(n_specimens))
    assignments = np.concatenate([np.repeat(split, count) for split, count in zip(SPLITS, counts)])
    rng = np.random.default_rng(seed); rng.shuffle(assignments)
    records, oracle_records = [], []
    role_indices = {split: 0 for split in SPLITS}
    for i, split in enumerate(assignments):
        iid = f"synthetic_image_{i:04d}"; specimen = f"synthetic_specimen_{i:04d}"
        local = np.random.default_rng(stable_seed(seed, iid, "image"))
        role_index = role_indices[split]; role_indices[split] += 1
        device = "SYNTH_DEVICE_3" if split == "device_test" else f"SYNTH_DEVICE_{role_index % 3}"
        observed, clean, truth, difficulty = _render_truth(local, image_size, device)
        image_path = f"images/{iid}.png"; truth_path = f"oracle/masks/{iid}.png"; clean_path = f"oracle/clean/{iid}.png"
        _save_gray(output / image_path, observed); _save_gray(output / clean_path, clean)
        (output / truth_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(truth).save(output / truth_path)
        # Held-out labels are not present in the public training view.
        eligible_role = split in DEVELOPMENT_ACTION_SPLITS
        initially_active = split == "seed_train" or (eligible_role and role_index % 4 != 0)
        mode = str(local.choice(["boundary", "class", "missing", "mixed", "uncertain", "clean"],
                                p=[.22, .20, .20, .14, .14, .10]))
        annotations = []
        rater_ids = [f"SYNTH_RATER_{k}" for k in local.choice(6, size=3, replace=False)]
        if initially_active:
            for j, rater in enumerate(rater_ids):
                read_rng = np.random.default_rng(stable_seed(seed, iid, rater, "initial"))
                labelmap, uncertainty = make_simulated_read(truth, observed, read_rng, mode=mode,
                                                           quality=.45 + .08 * (int(rater[-1]) % 3), difficulty=difficulty)
                path = f"annotations/{iid}_{rater}_read0.png"
                (output / path).parent.mkdir(parents=True, exist_ok=True); Image.fromarray(labelmap).save(output / path)
                annotations.append({"annotator_id": rater, "read_id": "read0", "labelmap_path": path,
                                    "is_repeat": False, "uncertainty": uncertainty,
                                    "provenance": "synthetic_rater_not_human"})
            if role_index % 5 == 0:
                read_rng = np.random.default_rng(stable_seed(seed, iid, rater_ids[0], "repeat"))
                labelmap, uncertainty = make_simulated_read(truth, observed, read_rng, mode=mode, quality=.55, difficulty=difficulty)
                path = f"annotations/{iid}_{rater_ids[0]}_repeat.png"
                Image.fromarray(labelmap).save(output / path)
                annotations.append({"annotator_id": rater_ids[0], "read_id": "repeat_after_interval",
                                    "labelmap_path": path, "is_repeat": True, "uncertainty": uncertainty,
                                    "provenance": "synthetic_repeat_not_actual_human_reread"})
        reimage_available = bool(role_index % 3 != 1)
        if eligible_role:
            available = ["boundary", "missing", "class", "review"] if initially_active else ["new_image"]
            if initially_active and reimage_available:
                available.append("reimage")
        else:
            available = []
        # Quote depends only on visible geometry and public acquisition logistics.
        size_factor = image_size * image_size / (48 * 48)
        quotes = {"new_image": 4.5 * size_factor, "boundary": 2.0 * size_factor,
                  "missing": 3.2 * size_factor, "class": 1.4 * size_factor,
                  "review": 2.5 * size_factor, "reimage": 5.5 * size_factor}
        row = {"schema_version": "1.0", "image_id": iid, "image_path": image_path,
               "image_sha256": hashlib.sha256((output / image_path).read_bytes()).hexdigest(),
               "specimen_id": specimen, "weld_id": f"synthetic_weld_{i:04d}", "source_image_id": iid,
               "device_id": device, "site_id": "SYNTH_SITE_EXTERNAL" if split == "device_test" else "SYNTH_SITE_MAIN",
               "acquisition_session": f"synthetic_session_{i:04d}", "split": str(split),
               "width": image_size, "height": image_size, "is_synthetic": True,
               "initial_labeled": bool(initially_active), "annotations": annotations,
               "available_actions": available, "action_quotes_minutes": {a: quotes[a] for a in available},
               "quote_semantics": "synthetic_predeclared_maximum_not_observed_future_cost",
               "annotation_provenance": "all_rater_reads_are_simulated_no_human_annotation_performed"}
        records.append(row)
        oracle_records.append({"image_id": iid, "specimen_id": specimen, "split": str(split), "is_synthetic": True,
                               "labelmap_path": truth_path, "clean_image_path": clean_path,
                               "latent_difficulty": difficulty, "simulation_error_mode": mode,
                               "oracle_scope": "synthetic_environment_and_independent_evaluator_only"})
    for name, rows in (("manifest.jsonl", records), ("oracle/truth.jsonl", oracle_records)):
        with (output / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    info = {"status": "SYNTHETIC_SOFTWARE_FIXTURE_NOT_REAL_EXPERT_DATA", "seed": seed,
            "n_specimens": n_specimens, "image_size": image_size,
            "role_counts": {k: int(v) for k, v in zip(SPLITS, counts)},
            "class_ontology": {"0": "background", "1": "round_synthetic_indication", "2": "linear_synthetic_indication"},
            "ground_truth_access": "oracle/truth.jsonl is forbidden as a policy/detector feature source",
            "publication_claim": "No human disagreement distribution, industrial risk reduction or true labor cost is established."}
    (output / "generation_provenance.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output / "manifest.jsonl"


def load_public_cases(manifest_path, splits: Iterable[str] | str | None = None) -> dict[str, Case]:
    """Load pixels and currently visible rater annotations; NEVER opens oracle/."""
    manifest = Path(manifest_path).resolve(); root = manifest.parent
    if isinstance(splits, str):
        splits = {splits}
    elif splits is not None:
        splits = set(splits)
    result = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if splits is not None and record["split"] not in splits:
            continue
        if record["image_id"] in result:
            raise ValueError("Duplicate image_id in public manifest")
        def public_path(relative):
            p = Path(relative)
            if p.is_absolute() or ".." in p.parts or "oracle" in p.parts:
                raise ValueError("Public loader rejects absolute, traversal and oracle paths")
            resolved = (root / p).resolve()
            if not resolved.is_relative_to(root) or "oracle" in resolved.relative_to(root).parts:
                raise ValueError("A public path cannot resolve outside the dataset or into oracle/")
            return resolved
        with Image.open(public_path(record["image_path"])) as im:
            if im.mode not in ("L", "RGB", "RGBA"):
                raise ValueError("Reference loader requires an audited 8-bit image")
            image = np.asarray(im.convert("L"), dtype=np.float64) / 255.0
        if image.shape != (record["height"], record["width"]):
            raise ValueError("Image geometry disagrees with manifest")
        reads = []
        for item in record["annotations"]:
            with Image.open(public_path(item["labelmap_path"])) as im:
                labelmap = np.asarray(im.convert("L"), dtype=np.uint8).copy()
            if labelmap.shape != image.shape or not np.isin(labelmap, (0, 1, 2, 255)).all():
                raise ValueError("Invalid public annotation geometry or class ID")
            reads.append(AnnotationRead(item["annotator_id"], item["read_id"], labelmap,
                                        bool(item.get("is_repeat", False)), float(item.get("uncertainty", 0)),
                                        item.get("provenance", "unspecified")))
        metadata = {k: v for k, v in record.items() if k not in ("annotations",)}
        result[record["image_id"]] = Case(record["image_id"], record["specimen_id"], record["split"],
                                         record["device_id"], image, reads, bool(record["initial_labeled"]),
                                         tuple(record["available_actions"]),
                                         {k: float(v) for k, v in record["action_quotes_minutes"].items()}, metadata)
    return result
