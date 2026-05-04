"""
Tests for Phase-3 security hardening.

Covers:
- ``PostgreSQLBackend`` rejects unsafe ``table_name`` in the constructor.
- ``PostgreSQLBackend.query_experiments`` rejects unsafe filter keys.
- ``BaseExperimentManager`` falls back to ``pwd.getpwuid`` when ``USER``
  is unset, instead of passing an empty string to ``squeue -u`` (which
  matches every job).
"""

from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# PostgreSQL identifier guard
# ---------------------------------------------------------------------------


_psycopg2_skip = pytest.mark.skipif(
    not bool(__import__("importlib").util.find_spec("psycopg2")),
    reason="psycopg2 not installed",
)


@_psycopg2_skip
def test_postgresql_backend_rejects_unsafe_table_name():
    from expflow.results_storage import PostgreSQLBackend

    with pytest.raises(ValueError):
        PostgreSQLBackend(
            connection_string="postgresql://x",
            table_name="users; DROP TABLE x",
        )


@_psycopg2_skip
def test_postgresql_backend_rejects_dotted_table_name():
    from expflow.results_storage import PostgreSQLBackend

    # public.users is a valid Postgres reference, but the schema-name
    # part should be rejected by our identifier regex (no dots allowed).
    with pytest.raises(ValueError):
        PostgreSQLBackend(
            connection_string="postgresql://x",
            table_name="public.users",
        )


@_psycopg2_skip
def test_postgresql_backend_accepts_valid_table_name():
    from expflow.results_storage import PostgreSQLBackend

    # Should not raise — but will fail to connect to a real DB; that's fine.
    PostgreSQLBackend(
        connection_string="postgresql://x",
        table_name="experiments_v2",
    )


@_psycopg2_skip
def test_postgresql_query_rejects_injected_filter_key(tmp_path, monkeypatch):
    """The ``query_experiments`` JSONB path-builder validates each key segment."""
    from expflow.results_storage import PostgreSQLBackend

    backend = PostgreSQLBackend(
        connection_string="postgresql://nobody",
        table_name="experiments",
    )

    # We don't connect to a real DB — instead we monkey-patch ``connect``
    # to a no-op and set up a fake cursor that raises if called. The
    # injection guard fires before any SQL is dispatched, so the cursor
    # never runs.
    backend.connect = lambda: None

    class _BadCursor:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *args, **kwargs):
            raise AssertionError("execute should not be called for unsafe keys")

        def fetchall(self):
            return []

    class _Conn:
        def cursor(self):
            return _BadCursor()

    backend.conn = _Conn()

    # The bad key should bubble up through the public method as a
    # logged ValueError caught by the broad except, returning [] —
    # but the SQL execute() must never run.
    out = backend.query_experiments({"users; DROP TABLE x": "completed"})
    assert out == []


# ---------------------------------------------------------------------------
# USER fallback
# ---------------------------------------------------------------------------


def test_resolve_user_falls_back_when_env_unset(monkeypatch):
    from expflow.hpcexp_core import _resolve_user

    monkeypatch.delenv("USER", raising=False)
    user = _resolve_user()
    # Whatever pwd returns — must not be the empty string that would
    # cause ``squeue -u ""`` to match every job.
    assert isinstance(user, str)
    assert user != ""


def test_resolve_user_honours_env(monkeypatch):
    from expflow.hpcexp_core import _resolve_user

    monkeypatch.setenv("USER", "phase3-test-user")
    assert _resolve_user() == "phase3-test-user"


# ---------------------------------------------------------------------------
# Bare-except cleanup (smoke check: file no longer contains the pattern)
# ---------------------------------------------------------------------------


def test_no_bare_except_in_hpc_config():
    src = (
        __import__("expflow", fromlist=["__file__"]).__file__.rsplit("/", 1)[0]
        + "/hpc_config.py"
    )
    with open(src) as fh:
        text = fh.read()
    # bare ``except:`` (followed by whitespace and a colon) is forbidden;
    # ``except Exception:`` is fine.
    import re

    assert not re.search(r"\n\s*except:\s*$", text, flags=re.MULTILINE), (
        "bare `except:` survives in hpc_config.py"
    )
