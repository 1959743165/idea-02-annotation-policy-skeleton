#!/usr/bin/env python3
"""Check package files without importing package code or running any experiment.

Examples (run from package root):
  python scripts/check_package.py --write-manifest
  python scripts/check_package.py --verify-manifest
  python scripts/check_package.py --root /tmp/unpacked_package --verify-manifest \
      --report /tmp/unpacked_package_integrity.json

The manifest and human-readable file list never hash themselves. Volatile
checker reports are also explicitly excluded; see the emitted exclusions.
This verifies file integrity, not research validity, schema-instance validity,
trusted provenance, model performance, or a cryptographic signature.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import tokenize
from urllib.parse import unquote, urlsplit


CHECKER_VERSION = "1.0.0"
MANIFEST_VERSION = "1.0"
MANIFEST_RELATIVE = "checks/file_manifest.json"
FILE_LIST_RELATIVE = "checks/FILE_LIST.txt"
DEFAULT_REPORT_RELATIVE = "checks/package_integrity.json"
EXCLUDED_EXACT = {MANIFEST_RELATIVE, FILE_LIST_RELATIVE}
EXCLUDED_PATTERN_DESCRIPTIONS = [
    "any path component named __pycache__",
    "any file with suffix .pyc",
    "checks/package_integrity*.json",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def reject_nonfinite(value):
    raise ValueError(f"Nonstandard JSON numeric constant: {value}")


def strict_json(value):
    return json.loads(value, parse_constant=reject_nonfinite)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def exclusions_for(root, report_path):
    exact = set(EXCLUDED_EXACT)
    if report_path.is_relative_to(root):
        exact.add(report_path.relative_to(root).as_posix())
    return exact


def is_excluded(relative, excluded_exact):
    path = PurePosixPath(relative)
    return (relative in excluded_exact or "__pycache__" in path.parts or path.suffix == ".pyc"
            or (path.parent.as_posix() == "checks" and path.name.startswith("package_integrity")
                and path.suffix == ".json"))


def without_markdown_code(text):
    """Ignore fenced/inline code so example link syntax is not treated as a file."""
    kept = []
    fence = None
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            continue
        if fence is None:
            kept.append(re.sub(r"(`+).*?\1", "", line))
    return "\n".join(kept)


def markdown_destinations(text):
    """Recognize ordinary Markdown inline/image links and reference definitions.

    Not a complete CommonMark parser: unusual nested destinations can be skipped.
    It intentionally does not interpret bare filenames or explanatory prose.
    """
    cleaned = without_markdown_code(text)
    inline = re.compile(r"!?\[[^\]\n]*\]\(\s*(?:<([^>\n]+)>|([^\s()]+))(?:\s+[\"'][^\n]*?[\"'])?\s*\)")
    reference = re.compile(r"^\s{0,3}\[[^\]\n]+\]:\s*(?:<([^>\n]+)>|(\S+))", re.MULTILINE)
    for matcher in (inline, reference):
        for match in matcher.finditer(cleaned):
            yield match.group(1) or match.group(2)


def check_markdown_links(path, root, text, errors):
    checked = 0
    for destination in markdown_destinations(text):
        if not destination or destination.startswith(("#", "//")):
            continue
        try:
            parsed = urlsplit(destination)
        except ValueError:
            continue  # A non-URI explanatory construct is outside this parser.
        if parsed.scheme or not parsed.path:
            continue
        decoded = unquote(parsed.path)
        if Path(decoded).is_absolute():
            continue  # This check is intentionally limited to relative links.
        checked += 1
        target = (path.parent / decoded).resolve()
        relative = path.relative_to(root).as_posix()
        if not target.is_relative_to(root):
            errors.append({"path": relative, "check": "markdown_local_link",
                           "message": f"Relative target escapes package: {destination}"})
        elif not target.exists():
            errors.append({"path": relative, "check": "markdown_local_link",
                           "message": f"Missing relative target: {destination}"})
    return checked


def scan_package(root, report_path):
    errors, entries, excluded = [], [], []
    counts = {"files": 0, "bytes": 0, "python_ast": 0, "json": 0, "jsonl": 0,
              "jsonl_records": 0, "yaml": 0, "csv_with_header": 0,
              "schema_documents_parsed": 0, "markdown": 0, "local_markdown_links_checked": 0}
    exact = exclusions_for(root, report_path)
    try:
        import yaml
        yaml_version = getattr(yaml, "__version__", "unknown")
    except ImportError:
        yaml = None
        yaml_version = "unavailable"
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if is_excluded(relative, exact):
            if path.is_file() or path.is_symlink():
                excluded.append(relative)
            continue
        if path.is_symlink():
            errors.append({"path": relative, "check": "filesystem",
                           "message": "Symlinks are not accepted in a self-contained package"})
            continue
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            errors.append({"path": relative, "check": "read", "message": str(exc)})
            continue
        counts["files"] += 1
        counts["bytes"] += len(data)
        entries.append({"path": relative, "bytes": len(data), "sha256": sha256_bytes(data)})
        if not data:
            errors.append({"path": relative, "check": "nonempty", "message": "Zero-byte file"})
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".py":
                # Honor the source's Python encoding cookie; do not execute it.
                with tokenize.open(path) as stream:
                    ast.parse(stream.read(), filename=relative)
                counts["python_ast"] += 1
            elif suffix == ".json":
                parsed = strict_json(data.decode("utf-8-sig"))
                counts["json"] += 1
                if ".schema." in path.name or "schemas" in path.parts or (isinstance(parsed, dict) and "$schema" in parsed):
                    if not isinstance(parsed, (dict, bool)):
                        raise ValueError("A JSON Schema document must be an object or boolean")
                    counts["schema_documents_parsed"] += 1
            elif suffix in (".jsonl", ".ndjson"):
                lines = [line for line in data.decode("utf-8-sig").splitlines() if line.strip()]
                if not lines:
                    raise ValueError("JSONL has no records")
                for line_number, line in enumerate(lines, start=1):
                    try:
                        strict_json(line)
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"JSONL record {line_number}: {exc}") from exc
                counts["jsonl"] += 1
                counts["jsonl_records"] += len(lines)
            elif suffix in (".yaml", ".yml"):
                if yaml is None:
                    raise ValueError("PyYAML is required; install requirements.txt before checking YAML")
                list(yaml.safe_load_all(data.decode("utf-8-sig")))
                counts["yaml"] += 1
            elif suffix == ".csv":
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    rows = csv.reader(stream, strict=True)
                    header = next(rows, None)
                    if not header or any(not cell.strip() for cell in header):
                        raise ValueError("CSV requires a nonempty named header for every column")
                    if len(set(header)) != len(header):
                        raise ValueError("CSV header contains duplicate names")
                    for _ in rows:
                        pass  # Parse remaining quoting without inventing a row schema.
                counts["csv_with_header"] += 1
            elif suffix in (".md", ".markdown"):
                counts["markdown"] += 1
                counts["local_markdown_links_checked"] += check_markdown_links(
                    path, root, data.decode("utf-8-sig"), errors)
        except (UnicodeError, SyntaxError, ValueError, TypeError, csv.Error, OSError) as exc:
            errors.append({"path": relative, "check": "syntax_or_format", "message": str(exc)})
        except Exception as exc:
            # PyYAML errors are not necessarily ValueError; make the check fail
            # visibly rather than preventing the rest of the package audit.
            errors.append({"path": relative, "check": "syntax_or_format",
                           "message": f"{type(exc).__name__}: {exc}"})
    if not entries:
        errors.append({"path": ".", "check": "nonempty", "message": "No non-excluded package files found"})
    return entries, counts, errors, excluded, exact, yaml_version


def verify_manifest(root, entries, excluded_exact, errors):
    manifest_path = root / MANIFEST_RELATIVE
    details = {"requested": True, "manifest_path": MANIFEST_RELATIVE, "missing": [], "added": [], "changed": []}
    if not manifest_path.is_file():
        errors.append({"path": MANIFEST_RELATIVE, "check": "manifest", "message": "Manifest does not exist"})
        return details
    try:
        raw = manifest_path.read_bytes()
        manifest = strict_json(raw.decode("utf-8-sig"))
        details["manifest_sha256"] = sha256_bytes(raw)
        if not isinstance(manifest, dict) or manifest.get("manifest_version") != MANIFEST_VERSION or not isinstance(manifest.get("files"), list):
            raise ValueError("Unsupported manifest format/version")
        expected = {}
        for item in manifest["files"]:
            path = item["path"]
            relative = PurePosixPath(path)
            if relative.is_absolute() or ".." in relative.parts or path in expected or path != relative.as_posix():
                raise ValueError(f"Invalid or duplicate manifest path: {path}")
            if is_excluded(path, excluded_exact):
                raise ValueError(f"Manifest must not hash an excluded/self-referential file: {path}")
            if not isinstance(item["bytes"], int) or item["bytes"] <= 0:
                raise ValueError(f"Invalid manifest byte count: {path}")
            if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
                raise ValueError(f"Invalid SHA256: {path}")
            expected[path] = item
        actual = {entry["path"]: entry for entry in entries}
        details["missing"] = sorted(set(expected) - set(actual))
        details["added"] = sorted(set(actual) - set(expected))
        details["changed"] = sorted(path for path in set(expected) & set(actual)
                                    if actual[path]["bytes"] != expected[path]["bytes"]
                                    or actual[path]["sha256"] != expected[path]["sha256"])
        for kind in ("missing", "added", "changed"):
            for path in details[kind]:
                errors.append({"path": path, "check": "manifest", "message": f"Manifest comparison: {kind}"})
    except (ValueError, TypeError, KeyError, OSError, UnicodeError) as exc:
        errors.append({"path": MANIFEST_RELATIVE, "check": "manifest", "message": str(exc)})
    return details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Package directory to scan")
    parser.add_argument("--report", type=Path, help="Report path; relative paths are relative to --root. Use an absolute path outside the package to keep it untouched.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write-manifest", action="store_true", help="After a successful check, freeze checks/file_manifest.json and checks/FILE_LIST.txt")
    group.add_argument("--verify-manifest", action="store_true", help="Compare every non-excluded file with the existing frozen manifest")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"--root is not a directory: {root}")
    requested_report = args.report if args.report is not None else Path(DEFAULT_REPORT_RELATIVE)
    report_path = (requested_report if requested_report.is_absolute() else root / requested_report).resolve()
    if report_path in (root / MANIFEST_RELATIVE, root / FILE_LIST_RELATIVE):
        parser.error("--report cannot overwrite the manifest or file list")
    started = utc_now()
    entries, counts, errors, excluded, exact, yaml_version = scan_package(root, report_path)
    verification = verify_manifest(root, entries, exact, errors) if args.verify_manifest else {"requested": False}
    exclusion_record = {"exact_relative_paths": sorted(exact), "patterns": EXCLUDED_PATTERN_DESCRIPTIONS,
                        "existing_files_excluded": excluded,
                        "reason": "Generated bytecode and self-referential/volatile inventory reports are not package payload hashes."}
    wrote_manifest = False
    if args.write_manifest and not errors:
        write_json(root / MANIFEST_RELATIVE, {"manifest_version": MANIFEST_VERSION,
            "checker_version": CHECKER_VERSION, "created_utc": utc_now(), "root_name": root.name,
            "exclusions": exclusion_record, "files": entries,
            "scope": "Byte-for-byte payload inventory; manifest/list/checker reports explicitly excluded"})
        file_list = root / FILE_LIST_RELATIVE
        lines = ["PACKAGE PAYLOAD FILE LIST", f"Generated UTC: {utc_now()}",
                 f"Files: {counts['files']}; bytes: {counts['bytes']}",
                 "Exclusions (not self-hashed): " + json.dumps(exclusion_record, ensure_ascii=False),
                 "", "relative_path\tbytes\tsha256"]
        lines.extend(f"{item['path']}\t{item['bytes']}\t{item['sha256']}" for item in entries)
        file_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
        wrote_manifest = True
    report = {"checker_version": CHECKER_VERSION, "started_utc": started, "finished_utc": utc_now(),
              "package_root": str(root), "report_path": str(report_path),
              "python_version": platform.python_version(), "python_implementation": platform.python_implementation(),
              "pyyaml_version": yaml_version, "mode": "write-manifest" if args.write_manifest else "verify-manifest" if args.verify_manifest else "check",
              "statistics": counts, "exclusions": exclusion_record, "manifest_verification": verification,
              "manifest_written": wrote_manifest, "errors": errors, "passed": not errors,
              "scope": {"python": "ast.parse only; never import/execute project code", "json_schema": "Syntax/root-type parsing only; no schema-instance validation",
                        "markdown_links": "Relative inline/image/reference destinations outside code; anchors and URLs skipped; not full CommonMark",
                        "smoke_tests_run": False, "unit_tests_run": False, "real_training_performed": False}}
    write_json(report_path, report)
    print(json.dumps({"passed": report["passed"], "files": counts["files"], "python_ast": counts["python_ast"],
                      "errors": len(errors), "manifest_written": wrote_manifest, "report": str(report_path)}, ensure_ascii=False))
    for error in errors[:8]:
        print(f"ERROR {error['path']} [{error['check']}]: {error['message']}", file=sys.stderr)
    if len(errors) > 8:
        print(f"Additional errors are in {report_path}", file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
