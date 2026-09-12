#!/usr/bin/env python3
"""Build and audit an allow-listed DenMark source release tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


ROOT_FILES = (
    ".gitignore",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    "pyproject.toml",
)
TREE_PATTERNS = (
    (".github/workflows", "*.yml"),
    ("denmark", "*.py"),
    ("configs", "*.json"),
    ("docs", "*.md"),
    ("tests", "*.py"),
)
MAX_FILE_BYTES = 5 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".bin", ".ckpt", ".gz", ".npy", ".npz", ".pdf", ".pt", ".pth",
    ".safetensors", ".tar", ".tgz", ".zip",
}
TEXT_PATTERNS = {
    "OpenRouter key": re.compile(r"sk-or-v1-[A-Za-z0-9]{20,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    # Split the literal so the release scanner does not flag its own source.
    "macOS absolute path": re.compile("/" + r"Users/[^/\s]+/"),
    "cluster absolute path": re.compile(r"/(?:work|home)/[A-Za-z0-9_.-]+/"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_files(root: Path) -> list[Path]:
    files = [root / name for name in ROOT_FILES]
    for directory, pattern in TREE_PATTERNS:
        files.extend((root / directory).rglob(pattern))
    unique = sorted({path.resolve() for path in files})
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"release input is missing: {missing[0]}")
    return unique


def audit_file(path: Path, relative: Path) -> list[str]:
    issues: list[str] = []
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        issues.append(f"{relative}: file is larger than {MAX_FILE_BYTES} bytes")
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        issues.append(f"{relative}: forbidden generated/binary suffix")
    if path.is_symlink():
        issues.append(f"{relative}: symbolic links are not allowed")
        return issues
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        issues.append(f"{relative}: file is not UTF-8 text")
        return issues
    for label, pattern in TEXT_PATTERNS.items():
        if pattern.search(text):
            issues.append(f"{relative}: contains {label}")
    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.source_root.resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to replace existing output: {output}; choose a new directory"
        )
    if output == root or root in output.parents:
        raise ValueError("release output must be outside the source repository")

    sources = selected_files(root)
    issues: list[str] = []
    manifest: list[dict[str, object]] = []
    for source in sources:
        relative = source.relative_to(root)
        issues.extend(audit_file(source, relative))
        manifest.append(
            {
                "path": relative.as_posix(),
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
            }
        )
    if issues:
        raise RuntimeError("release audit failed:\n" + "\n".join(issues))

    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"stale staging directory exists: {staging}")
    for source in sources:
        relative = source.relative_to(root)
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (staging / "RELEASE_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_root": "omitted",
                "file_count": len(manifest),
                "files": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    staging.rename(output)
    print(json.dumps({"output": str(output), "files": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
