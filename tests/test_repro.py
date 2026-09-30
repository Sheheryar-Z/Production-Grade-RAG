"""Phase 6: config/experiment reproducibility metadata. Tests the pure
hashing logic directly (deterministic, no git needed) and checks that the
git-dependent functions degrade to None rather than raising when run
outside a git checkout -- eval scripts must never crash over this."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from k8s_rag.repro import config_content_hash, git_commit_sha, git_is_dirty  # noqa: E402


def test_config_content_hash_deterministic(tmp_path):
    f = tmp_path / "a.yaml"
    f.write_text("run_name: x\n")
    assert config_content_hash(f) == config_content_hash(f)


def test_config_content_hash_changes_with_content(tmp_path):
    f = tmp_path / "a.yaml"
    f.write_text("run_name: x\n")
    h1 = config_content_hash(f)
    f.write_text("run_name: y\n")
    h2 = config_content_hash(f)
    assert h1 != h2


def test_config_content_hash_is_short_hex():
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tf:
        tf.write(b"hello")
        path = Path(tf.name)
    try:
        h = config_content_hash(path)
        assert len(h) == 12
        int(h, 16)  # raises if not valid hex
    finally:
        path.unlink()


def test_git_commit_sha_never_raises_outside_a_repo(tmp_path):
    # tmp_path is not a git checkout -- this must degrade to None, not raise
    result = git_commit_sha(tmp_path)
    assert result is None


def test_git_is_dirty_never_raises_outside_a_repo(tmp_path):
    result = git_is_dirty(tmp_path)
    assert result is None


def test_git_commit_sha_on_this_repo_looks_like_a_sha():
    repo_root = Path(__file__).resolve().parents[1]
    sha = git_commit_sha(repo_root)
    # this repo IS a git checkout (verified earlier in the session) -- if
    # that ever stops being true this assertion should fail loudly, not
    # silently pass on a None
    assert sha is not None
    assert len(sha) == 40
    int(sha, 16)
