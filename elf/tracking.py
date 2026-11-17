"""Experiment tracking: a thin wrapper over runboard.Run (github.com/kurmetov/runboard).

Runs go to <repo>/runs/ (one folder per run: config, git state, metrics, summary).
`git_dirty` is judged on code only: run folders and results not yet committed do not
make the next run dirty.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from runboard import Run as _Run

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
_NOT_CODE = (":(exclude)runs", ":(exclude)results", ":(exclude)data")


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True).stdout


def _code_diff() -> str:
    return _git("diff", "HEAD", "--", ".", *_NOT_CODE)


def _code_dirty() -> bool:
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", ".", *_NOT_CODE)
    return bool(_code_diff().strip() or untracked.strip())


class Run(_Run):
    def __init__(self, experiment: str, config: dict[str, Any], tags: dict[str, Any] | None = None, **kwargs: Any):
        dirty = _code_dirty()
        if not dirty:
            kwargs.setdefault("snapshot", False)
        super().__init__(experiment, config, tags=tags, root=RUNS_DIR, repo=REPO_ROOT, **kwargs)
        self.meta["git_dirty"] = dirty
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2, ensure_ascii=False))
        (self.dir / "git.diff").unlink(missing_ok=True)
        if diff := _code_diff().strip():
            (self.dir / "git.diff").write_text(diff + "\n")
