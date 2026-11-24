"""Connected provenance-group isolation for annotation-action experiments.

Random crops, repeat reads, and additional views are never independent splits.
This module checks identifiers; a source audit must establish that those
identifiers are truthful and that unrecorded near-duplicates do not exist.
"""
from __future__ import annotations
from collections import defaultdict

SPLITS = ("seed_train", "policy_train", "policy_val", "reward", "acquisition_pool",
          "detector_val", "test", "device_test")
LEARNING_SPLITS = {"seed_train", "policy_train", "policy_val", "reward", "acquisition_pool", "detector_val"}


class UnionFind:
    def __init__(self, keys):
        self.parent = {key: key for key in keys}

    def find(self, key):
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def check_group_leakage(records, protocol="within_domain"):
    """Require every connected specimen/source/read family to share one role.

    ``device_test`` is a device-held-out role even under the default protocol.
    ``device_holdout`` additionally requires the main test devices to be absent
    from every development/acquisition role. A site holdout is analogous.
    """
    if protocol not in {"within_domain", "device_holdout", "site_holdout"}:
        raise ValueError("Unknown split protocol")
    errors, warnings = [], []
    ids = [r.get("image_id") for r in records]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        return {"ok": False, "errors": ["image_id must be present and unique before checking groups"], "warnings": []}
    index = {r["image_id"]: r for r in records}
    uf = UnionFind(ids)
    owners = {}
    group_fields = ("specimen_id", "weld_id", "source_image_id", "acquisition_session",
                    "repeat_acquisition_group", "duplicate_group", "near_duplicate_group")
    for row in records:
        image_id = row["image_id"]
        for field in group_fields:
            value = row.get(field)
            if value is not None and value != "":
                token = (field, str(value))
                if token in owners:
                    uf.union(image_id, owners[token])
                else:
                    owners[token] = image_id
        for field in ("parent_image_id", "original_image_id"):
            parent = row.get(field)
            if parent is not None:
                if parent in index:
                    uf.union(image_id, parent)
                else:
                    errors.append(f"{image_id}: {field}={parent!r} is absent from the manifest")
        if row.get("split") not in SPLITS:
            errors.append(f"{image_id}: invalid split {row.get('split')!r}")
    components = defaultdict(list)
    for image_id in ids:
        components[uf.find(image_id)].append(image_id)
    leaked = []
    for group in components.values():
        roles = sorted({index[i].get("split", "MISSING") for i in group})
        if len(roles) > 1:
            leaked.append({"image_ids": sorted(group), "splits": roles})
    if leaked:
        errors.append(f"{len(leaked)} provenance-connected groups cross data roles")
    summary = {}
    for split in SPLITS:
        subset = [r for r in records if r.get("split") == split]
        summary[split] = {"images": len(subset),
                          "specimens": len({r.get("specimen_id") for r in subset}),
                          "devices": sorted({str(r.get("device_id")) for r in subset}),
                          "sites": sorted({str(r.get("site_id")) for r in subset})}
        if not subset:
            warnings.append(f"empty {split} role")
    held = {r.get("device_id") for r in records if r.get("split") == "device_test"}
    other = {r.get("device_id") for r in records if r.get("split") != "device_test"}
    overlap = sorted(str(x) for x in held & other)
    if overlap:
        errors.append("device_test devices also occur in other roles: " + ", ".join(overlap))
    if protocol in {"device_holdout", "site_holdout"}:
        field = "device_id" if protocol == "device_holdout" else "site_id"
        test_ids = {r.get(field) for r in records if r.get("split") in {"test", "device_test"}}
        development_ids = {r.get(field) for r in records if r.get("split") in LEARNING_SPLITS}
        repeated = sorted(str(x) for x in test_ids & development_ids)
        if repeated:
            errors.append(f"{protocol}: held-out test {field}s appear in development: {repeated}")
    return {"ok": not errors, "protocol": protocol, "errors": errors, "warnings": warnings,
            "connected_groups": len(components), "leaked_groups": leaked, "split_summary": summary,
            "limitations": ["Identifier checks cannot certify provenance correctness.",
                            "Perceptual near-duplicates need an independent source audit.",
                            "Repeated expert reads remain correlated with the same specimen."]}
