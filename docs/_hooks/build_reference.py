"""mkdocs pre-build hook: regenerate docgen-owned pages.

Runs `cargo run -p trxviz-docgen` before every `mkdocs build` /
`mkdocs serve` rebuild so the per-op pages, SVG examples, and methods
boilerplate stay in sync with the Rust source.

The generated paths (`docs/reference/ops/`, `docs/examples/{*.svg,*.md}`)
are gitignored; this hook is how they materialize for anyone running
mkdocs without invoking cargo manually.

Opt out by setting `TRXVIZ_DOCGEN_SKIP=1` (useful in CI jobs that
generate docs separately, or when iterating on CSS and you don't want
each save to retrigger a cargo build).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def on_pre_build(config, **_kwargs):  # noqa: ARG001 - mkdocs passes config; unused
    if os.environ.get("TRXVIZ_DOCGEN_SKIP"):
        print("[trxviz-docgen] skipped (TRXVIZ_DOCGEN_SKIP is set)", file=sys.stderr)
        return

    docs_dir = Path(config["docs_dir"]).resolve()
    repo_root = docs_dir.parent
    if not (repo_root / "Cargo.toml").exists():
        # Docgen needs the workspace; bail loud rather than silently skip.
        raise RuntimeError(
            f"trxviz-docgen hook: no Cargo.toml at {repo_root}; can't regenerate docs"
        )

    cmd = [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "trxviz-docgen",
        "--",
        str(docs_dir),
    ]
    print(f"[trxviz-docgen] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=repo_root, check=True)
