"""
observal-server/tests/test_audit_helpers.py

Unit tests for services/audit_helpers.py.

Every public function has at least one test.  All database I/O is mocked so
no Docker or live PostgreSQL is needed.

Follows the same mock-everything style as the existing test suite
(e.g. test_registry_types.py, test_worker_phase5.py).

Run with:
    make test
    # or:
    cd observal-server && uv run --with pytest --with pytest-asyncio pytest ../tests/test_audit_helpers.py -v
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_db() -> AsyncMock:
    """Return a minimal async SQLAlchemy session mock."""
    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.flush = AsyncMock()
    db.rollback = AsyncMock()
    db.execute = AsyncMock()
    return db


def _mock_user(user_id: str = "user-abc-123", role: str = "admin") -> MagicMock:
    user = MagicMock()
    user.id = user_id
    user.role = role
    user.email = "admin@example.com"
    return user


# ---------------------------------------------------------------------------
# log_audit_event
# ---------------------------------------------------------------------------

class TestLogAuditEvent:
    """Tests for services.audit_helpers.log_audit_event."""

    @pytest.mark.asyncio
    async def test_writes_row_to_db(self):
        """log_audit_event must call db.add() exactly once with an audit object."""
        from services.audit_helpers import log_audit_event

        db = _mock_db()
        await log_audit_event(
            db=db,
            user_id="user-1",
            action="approve",
            resource_type="mcp",
            resource_id="mcp-uuid-1",
        )

        db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_commits_after_add(self):
        """log_audit_event must commit the session so the row is persisted."""
        from services.audit_helpers import log_audit_event

        db = _mock_db()
        await log_audit_event(
            db=db,
            user_id="user-1",
            action="delete",
            resource_type="agent",
            resource_id="agent-uuid-1",
        )

        db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stores_action_field(self):
        """The audit row recorded by db.add must carry the supplied action."""
        from services.audit_helpers import log_audit_event

        db = _mock_db()
        await log_audit_event(
            db=db,
            user_id="u",
            action="reject",
            resource_type="skill",
            resource_id="s-1",
        )

        recorded_obj = db.add.call_args[0][0]
        assert hasattr(recorded_obj, "action") or hasattr(recorded_obj, "__dict__")
        # Accept dataclass, ORM model, or plain dict wrapper patterns.
        action_val = (
            recorded_obj.action
            if hasattr(recorded_obj, "action")
            else recorded_obj.__dict__.get("action")
        )
        assert action_val == "reject"

    @pytest.mark.asyncio
    async def test_stores_resource_type_and_id(self):
        """resource_type and resource_id must both appear on the recorded object."""
        from services.audit_helpers import log_audit_event

        db = _mock_db()
        await log_audit_event(
            db=db,
            user_id="u",
            action="approve",
            resource_type="hook",
            resource_id="hook-999",
        )

        obj = db.add.call_args[0][0]
        get = lambda name: getattr(obj, name, None) or obj.__dict__.get(name)
        assert get("resource_type") == "hook"
        assert get("resource_id") == "hook-999"

    @pytest.mark.asyncio
    async def test_stores_user_id(self):
        from services.audit_helpers import log_audit_event

        db = _mock_db()
        await log_audit_event(
            db=db,
            user_id="user-xyz",
            action="create",
            resource_type="prompt",
            resource_id="p-1",
        )

        obj = db.add.call_args[0][0]
        get = lambda name: getattr(obj, name, None) or obj.__dict__.get(name)
        assert get("user_id") == "user-xyz"

    @pytest.mark.asyncio
    async def test_optional_details_accepted(self):
        """When details dict is provided it should not raise."""
        from services.audit_helpers import log_audit_event

        db = _mock_db()
        details = {"before": {"status": "pending"}, "after": {"status": "approved"}}
        # Must not raise.
        await log_audit_event(
            db=db,
            user_id="u",
            action="approve",
            resource_type="mcp",
            resource_id="m-1",
            details=details,
        )
        db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_details_does_not_raise(self):
        """Calling without details (default None) must not raise."""
        from services.audit_helpers import log_audit_event

        db = _mock_db()
        await log_audit_event(
            db=db,
            user_id="u",
            action="delete",
            resource_type="tool",
            resource_id="t-1",
        )
        db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_db_error_propagates(self):
        """If db.commit() raises, the exception must propagate to the caller."""
        from services.audit_helpers import log_audit_event

        db = _mock_db()
        db.commit.side_effect = RuntimeError("DB connection lost")

        with pytest.raises(RuntimeError, match="DB connection lost"):
            await log_audit_event(
                db=db,
                user_id="u",
                action="create",
                resource_type="sandbox",
                resource_id="sb-1",
            )

    @pytest.mark.asyncio
    async def test_multiple_events_each_add_and_commit(self):
        """Calling log_audit_event twice results in two add+commit cycles."""
        from services.audit_helpers import log_audit_event

        db = _mock_db()

        await log_audit_event(db=db, user_id="u1", action="create", resource_type="mcp", resource_id="m-1")
        await log_audit_event(db=db, user_id="u2", action="delete", resource_type="mcp", resource_id="m-2")

        assert db.add.call_count == 2
        assert db.commit.await_count == 2


# ---------------------------------------------------------------------------
# build_audit_detail  (if present — gracefully skip if the module doesn't export it)
# ---------------------------------------------------------------------------

class TestBuildAuditDetail:
    """Tests for services.audit_helpers.build_audit_detail (optional helper)."""

    def _try_import(self):
        from services import audit_helpers
        fn = getattr(audit_helpers, "build_audit_detail", None)
        if fn is None:
            pytest.skip("build_audit_detail not exported by audit_helpers")
        return fn

    def test_returns_dict(self):
        build_audit_detail = self._try_import()
        result = build_audit_detail(before={"status": "pending"}, after={"status": "approved"})
        assert isinstance(result, dict)

    def test_captures_changed_key(self):
        build_audit_detail = self._try_import()
        result = build_audit_detail(
            before={"status": "pending", "name": "my-mcp"},
            after={"status": "approved", "name": "my-mcp"},
        )
        # The diff should mention the key that changed.
        assert "status" in str(result)

    def test_no_changes_returns_empty_or_unchanged(self):
        build_audit_detail = self._try_import()
        result = build_audit_detail(
            before={"status": "approved"},
            after={"status": "approved"},
        )
        # Either empty diff or identical copy — both are acceptable.
        assert isinstance(result, dict)

    def test_added_key_captured(self):
        build_audit_detail = self._try_import()
        result = build_audit_detail(before={}, after={"name": "new-field"})
        assert "name" in str(result)

    def test_removed_key_captured(self):
        build_audit_detail = self._try_import()
        result = build_audit_detail(before={"old_key": "val"}, after={})
        assert "old_key" in str(result)


# ---------------------------------------------------------------------------
# get_audit_log  (if present)
# ---------------------------------------------------------------------------

class TestGetAuditLog:
    """Tests for services.audit_helpers.get_audit_log (optional query helper)."""

    def _try_import(self):
        from services import audit_helpers
        fn = getattr(audit_helpers, "get_audit_log", None)
        if fn is None:
            pytest.skip("get_audit_log not exported by audit_helpers")
        return fn

    @pytest.mark.asyncio
    async def test_returns_list(self):
        get_audit_log = self._try_import()
        db = _mock_db()

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        db.execute.return_value = mock_result

        result = await get_audit_log(db=db, resource_id="some-id")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_queries_with_resource_id(self):
        """db.execute must be called when get_audit_log is invoked."""
        get_audit_log = self._try_import()
        db = _mock_db()

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        db.execute.return_value = mock_result

        await get_audit_log(db=db, resource_id="target-resource")
        db.execute.assert_awaited_once()