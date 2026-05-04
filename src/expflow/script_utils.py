"""
Script generation utilities.

Helpers used by ``BaseExperimentManager`` and user-written script generators
to safely embed config values into shell scripts (``quote_bash``,
``assert_safe_identifier``) and to wrap a job in an isolated git worktree
so the login-node checkout is never mutated by a running job.

These were extracted from navsim_manager.py so any manager subclass can use
them without copy-pasting the implementation.
"""

from __future__ import annotations

import re
import shlex
from typing import Any, Dict, Optional

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


def quote_bash(value: Any) -> str:
    """
    Shell-quote a value for safe embedding in a bash script.

    Wraps the value in single quotes and escapes any embedded single quotes
    using the standard ``'\"'\"'`` trick. Numbers and booleans are coerced
    to strings first.

    Example:
        quote_bash("a'b; rm -rf /") -> "'a'\"'\"'b; rm -rf /'"
    """
    return shlex.quote(str(value))


def assert_safe_identifier(name: str) -> str:
    """
    Validate that ``name`` is a safe SQL/shell identifier.

    Returns the name on success. Raises ``ValueError`` otherwise. Used to
    guard table/column names and any user-supplied tokens that get
    interpolated into SQL or shell without further quoting.
    """
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Unsafe identifier {name!r}: must match {_IDENTIFIER_RE.pattern}"
        )
    return name


def git_worktree_block(
    config: Dict[str, Any],
    repo_root_var: str = "REPO_ROOT",
    worktree_base_var: str = "WORKTREE_BASE",
    code_root_var: str = "NAVSIM_DEVKIT_ROOT",
    devkit_subdir: Optional[str] = None,
) -> str:
    """
    Generate a self-cleaning ``git worktree`` shell block for a SLURM script.

    The block (a) creates a worktree at ``${worktree_base_var}/{exp_id}_${SLURM_JOB_ID}``
    pinned to ``config['git_branch']``, (b) overrides ``${code_root_var}`` to
    point inside it, (c) registers a ``trap`` cleanup so the worktree is
    removed on job exit. If ``git_branch`` is missing from ``config`` the
    block is a no-op comment. Falls back to the original ``${code_root_var}``
    if ``git worktree add`` fails.

    Args:
        config: Experiment config dict. Reads ``exp_id``, ``git_branch``,
                ``git_commit``.
        repo_root_var: Shell variable that points at the repo containing
                       the manager (default: ``REPO_ROOT``). The block
                       computes it from ``${code_root_var}`` if not set.
        worktree_base_var: Shell variable pointing at the worktree-parent
                           directory (default: ``WORKTREE_BASE``).
        code_root_var: Shell variable that the build/eval command uses to
                       locate the code (e.g. ``NAVSIM_DEVKIT_ROOT``,
                       ``CODE_ROOT``). Overridden by this block to point
                       into the worktree.
        devkit_subdir: Optional subdirectory inside the worktree to use as
                       the new ``${code_root_var}`` (e.g. ``"navsim"`` for
                       a monorepo). If ``None``, the worktree root is used.

    Returns:
        Multi-line bash block ready to splice into a generated SLURM script.
    """
    git_branch = str(config.get("git_branch") or "").strip()
    if not git_branch:
        return "# Git worktree: skipped (no git_branch in config)"

    exp_id = str(config.get("exp_id", "exp")).strip() or "exp"
    git_commit = str(config.get("git_commit") or "").strip()

    if devkit_subdir:
        # devkit_subdir is a controlled token (e.g. "navsim", "src");
        # validate as identifier-ish path component before splicing.
        if not re.match(r"^[A-Za-z0-9_./-]+$", devkit_subdir):
            raise ValueError(f"Unsafe devkit_subdir: {devkit_subdir!r}")
        pin_to_subdir = (
            f'    export {code_root_var}="${{WORKTREE_DIR}}/{devkit_subdir}"'
        )
    else:
        pin_to_subdir = f'    export {code_root_var}="${{WORKTREE_DIR}}"'

    return f"""# =============================================================================
# GIT WORKTREE ISOLATION
# Creates an isolated working tree so the login-node checkout is untouched.
# =============================================================================
ORIG_{code_root_var}="${{{code_root_var}}}"
{repo_root_var}="${{{repo_root_var}:-$(dirname "${{{code_root_var}}}")}}"
GIT_BRANCH={quote_bash(git_branch)}
GIT_COMMIT={quote_bash(git_commit)}
{worktree_base_var}="${{{worktree_base_var}:-${{{repo_root_var}}}/.expflow_worktrees}}"
WORKTREE_DIR="${{{worktree_base_var}}}/{exp_id}_${{SLURM_JOB_ID:-$$}}"

mkdir -p "${{{worktree_base_var}}}"
cd "${{{repo_root_var}}}"

git fetch origin "${{GIT_BRANCH}}" 2>/dev/null || true
git merge --ff-only "origin/${{GIT_BRANCH}}" 2>/dev/null || true

if git worktree add "${{WORKTREE_DIR}}" "${{GIT_BRANCH}}" 2>/dev/null; then
    echo "[git-worktree] created from local branch ${{GIT_BRANCH}}"
elif git worktree add "${{WORKTREE_DIR}}" -b "__wt_{exp_id}_${{SLURM_JOB_ID:-$$}}" "origin/${{GIT_BRANCH}}" 2>/dev/null; then
    echo "[git-worktree] created from origin/${{GIT_BRANCH}}"
else
    echo "[git-worktree] WARN: could not create worktree, using login-node checkout"
    cd "${{ORIG_{code_root_var}}}"
    WORKTREE_DIR=""
fi

if [ -n "${{WORKTREE_DIR}}" ]; then
{pin_to_subdir}
    echo "[git-worktree] {code_root_var}=${{{code_root_var}}}"
    cleanup_worktree() {{
        cd "${{{repo_root_var}}}" 2>/dev/null || true
        git worktree remove "${{WORKTREE_DIR}}" --force 2>/dev/null || rm -rf "${{WORKTREE_DIR}}" 2>/dev/null || true
        git branch -D "__wt_{exp_id}_${{SLURM_JOB_ID:-$$}}" 2>/dev/null || true
    }}
    trap cleanup_worktree EXIT
fi
"""
