"""Phase 6: config & experiment infrastructure.

The roadmap's exit criteria was "every eval run reproducible from a single
config file/commit." A `config_path` field in the results JSON (added in
Phase 1) gets you halfway there, but it's not actually sufficient on its
own -- a config file can be edited after a run without creating a new
commit, and nothing before this phase recorded whether the working tree was
even clean when a result was produced. Both gaps are closed here, cheaply,
with functions that never raise: eval scripts run the same whether or not
this happens to be a git checkout, so `git log`/`git status` failing just
means these fields come back `None` rather than crashing an eval run over
a metadata nicety.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_commit_sha(repo_root: Path) -> str | None:
    return _run_git(["rev-parse", "HEAD"], repo_root)


def git_is_dirty(repo_root: Path) -> bool | None:
    """True if there are uncommitted changes (staged or not) -- a result
    produced against a dirty tree is only reproducible from the config
    file's exact bytes, not from `git show <commit>:config/...`, which is
    worth flagging rather than silently implying full reproducibility."""
    status = _run_git(["status", "--porcelain"], repo_root)
    if status is None:
        return None
    return len(status) > 0


def config_content_hash(config_path: Path) -> str:
    """First 12 hex chars of the config file's own sha256 -- catches config
    drift (an edit that changed content but not the filename) even when the
    commit-sha/dirty-flag combination above is unavailable or ambiguous."""
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return digest[:12]


def repro_metadata(repo_root: Path, config_path: Path) -> dict:
    return {
        "git_commit": git_commit_sha(repo_root),
        "git_dirty": git_is_dirty(repo_root),
        "config_content_hash": config_content_hash(config_path),
    }
