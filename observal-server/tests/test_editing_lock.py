# SPDX-FileCopyrightText: 2026 Satej More <satejmore28@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-only
"""
observal-server/tests/test_editing_lock.py

Unit tests for services/editing_lock.py.

Covers every public function with a focus on acquire, release, and expiry
behaviour as requested in issue #852.  All Redis I/O is mocked — no live
Redis instance is needed.

Run with:
    make test
    # or:
    cd observal-server && uv run --with pytest --with pytest-asyncio pytest ../tests/test_editing_lock.py -v
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call


# ---------------------------------------------------------------------------
# Redis mock factory
# ---------------------------------------------------------------------------

def _mock_redis() -> AsyncMock:
    """
    Return an async Redis client mock with the methods editing_lock.py uses:
        set, get, delete, expire, ttl, exists
    """
    r = AsyncMock()
    r.set = AsyncMock(return_value=True)   # SET NX returns True on success
    r.get = AsyncMock(return_value=None)
    r.delete = AsyncMock(return_value=1)
    r.expire = AsyncMock(return_value=True)
    r.ttl = AsyncMock(return_value=-2)     # -2 = key does not exist
    r.exists = AsyncMock(return_value=0)
    return r


# ---------------------------------------------------------------------------
# acquire_lock
# ---------------------------------------------------------------------------

class TestAcquireLock:
    """Tests for services.editing_lock.acquire_lock."""

    @pytest.mark.asyncio
    async def test_returns_true_on_first_acquire(self):
        """acquire_lock must return True when the resource is not yet locked."""
        from services.editing_lock import acquire_lock

        redis = _mock_redis()
        # SET NX returns the token / True when the key is new.
        redis.set.return_value = True

        result = await acquire_lock(redis, resource_id="doc-1", user_id="alice")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_already_locked(self):
        """acquire_lock must return False when another user holds the lock."""
        from services.editing_lock import acquire_lock

        redis = _mock_redis()
        # Redis SET NX returns None / False when the key already exists.
        redis.set.return_value = None

        result = await acquire_lock(redis, resource_id="doc-1", user_id="bob")

        assert result is False

    @pytest.mark.asyncio
    async def test_calls_set_with_nx_flag(self):
        """acquire_lock must use the NX option to prevent overwriting existing locks."""
        from services.editing_lock import acquire_lock

        redis = _mock_redis()
        await acquire_lock(redis, resource_id="doc-1", user_id="alice")

        redis.set.assert_awaited_once()
        kwargs = redis.set.call_args.kwargs
        # The NX flag must be set to True to ensure atomic compare-and-set.
        assert kwargs.get("nx") is True or kwargs.get("nx") == 1, (
            "SET must use nx=True so two users cannot acquire the same lock simultaneously"
        )

    @pytest.mark.asyncio
    async def test_calls_set_with_ttl(self):
        """acquire_lock must set a TTL so locks expire automatically."""
        from services.editing_lock import acquire_lock

        redis = _mock_redis()
        await acquire_lock(redis, resource_id="doc-1", user_id="alice", ttl=120)

        kwargs = redis.set.call_args.kwargs
        # Accept either 'ex' (seconds) or 'px' (milliseconds) — both are valid.
        has_expiry = "ex" in kwargs or "px" in kwargs
        assert has_expiry, "SET must include an expiry (ex= or px=) to prevent locks from lasting forever"

    @pytest.mark.asyncio
    async def test_default_ttl_is_positive(self):
        """When called without an explicit ttl the default must be a positive integer."""
        from services.editing_lock import acquire_lock

        redis = _mock_redis()
        await acquire_lock(redis, resource_id="doc-1", user_id="alice")

        kwargs = redis.set.call_args.kwargs
        ttl_val = kwargs.get("ex") or kwargs.get("px", 0)
        assert ttl_val > 0, "Default TTL must be positive"

    @pytest.mark.asyncio
    async def test_lock_key_contains_resource_id(self):
        """The Redis key used for the lock must include the resource_id."""
        from services.editing_lock import acquire_lock

        redis = _mock_redis()
        await acquire_lock(redis, resource_id="my-document", user_id="alice")

        key_used = redis.set.call_args.args[0]
        assert "my-document" in key_used, (
            f"Redis key '{key_used}' should include the resource_id 'my-document'"
        )

    @pytest.mark.asyncio
    async def test_value_encodes_user_id(self):
        """The value stored in Redis must encode the user_id so ownership can be verified."""
        from services.editing_lock import acquire_lock

        redis = _mock_redis()
        await acquire_lock(redis, resource_id="doc-1", user_id="user-999")

        raw_value = redis.set.call_args.args[1]
        # Accept JSON string, plain string, or bytes.
        if isinstance(raw_value, bytes):
            raw_value = raw_value.decode()
        assert "user-999" in raw_value, (
            "The Redis value must contain the user_id so release_lock can verify ownership"
        )

    @pytest.mark.asyncio
    async def test_same_user_second_acquire_returns_false(self):
        """
        A second acquire by the same user must also return False because the
        key already exists in Redis (nx=True prevents re-entry).
        """
        from services.editing_lock import acquire_lock

        redis = _mock_redis()
        redis.set.return_value = None  # key already present

        result = await acquire_lock(redis, resource_id="doc-1", user_id="alice")
        assert result is False

    @pytest.mark.asyncio
    async def test_redis_error_propagates(self):
        """If Redis raises an exception acquire_lock must let it propagate."""
        from services.editing_lock import acquire_lock

        redis = _mock_redis()
        redis.set.side_effect = ConnectionError("Redis is down")

        with pytest.raises(ConnectionError):
            await acquire_lock(redis, resource_id="doc-1", user_id="alice")


# ---------------------------------------------------------------------------
# release_lock
# ---------------------------------------------------------------------------

class TestReleaseLock:
    """Tests for services.editing_lock.release_lock."""

    def _lock_value(self, user_id: str) -> bytes:
        """Build the JSON bytes that acquire_lock would have stored."""
        return json.dumps({"user_id": user_id}).encode()

    @pytest.mark.asyncio
    async def test_owner_can_release(self):
        """The lock owner must be able to release their own lock."""
        from services.editing_lock import release_lock

        redis = _mock_redis()
        redis.get.return_value = self._lock_value("alice")

        result = await release_lock(redis, resource_id="doc-1", user_id="alice")

        assert result is True

    @pytest.mark.asyncio
    async def test_owner_release_deletes_key(self):
        """release_lock must call redis.delete when the caller owns the lock."""
        from services.editing_lock import release_lock

        redis = _mock_redis()
        redis.get.return_value = self._lock_value("alice")

        await release_lock(redis, resource_id="doc-1", user_id="alice")

        redis.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_owner_cannot_release(self):
        """A user who does not own the lock must not be able to release it."""
        from services.editing_lock import release_lock

        redis = _mock_redis()
        redis.get.return_value = self._lock_value("alice")  # alice holds the lock

        result = await release_lock(redis, resource_id="doc-1", user_id="bob")

        assert result is False

    @pytest.mark.asyncio
    async def test_non_owner_does_not_delete_key(self):
        """redis.delete must NOT be called when the caller is not the lock owner."""
        from services.editing_lock import release_lock

        redis = _mock_redis()
        redis.get.return_value = self._lock_value("alice")

        await release_lock(redis, resource_id="doc-1", user_id="bob")

        redis.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_release_nonexistent_lock_returns_false(self):
        """Releasing a lock that was never acquired must return False gracefully."""
        from services.editing_lock import release_lock

        redis = _mock_redis()
        redis.get.return_value = None  # key does not exist

        result = await release_lock(redis, resource_id="doc-1", user_id="alice")

        assert result is False

    @pytest.mark.asyncio
    async def test_release_nonexistent_lock_does_not_delete(self):
        from services.editing_lock import release_lock

        redis = _mock_redis()
        redis.get.return_value = None

        await release_lock(redis, resource_id="doc-1", user_id="alice")

        redis.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delete_uses_same_key_as_acquire(self):
        """The key passed to delete must be the same key that was used on acquire."""
        from services.editing_lock import acquire_lock, release_lock

        redis_acquire = _mock_redis()
        redis_acquire.set.return_value = True
        await acquire_lock(redis_acquire, resource_id="doc-99", user_id="alice")
        acquire_key = redis_acquire.set.call_args.args[0]

        redis_release = _mock_redis()
        redis_release.get.return_value = self._lock_value("alice")
        await release_lock(redis_release, resource_id="doc-99", user_id="alice")
        delete_key = redis_release.delete.call_args.args[0]

        assert acquire_key == delete_key, (
            f"acquire used key '{acquire_key}' but release called delete on '{delete_key}'"
        )


# ---------------------------------------------------------------------------
# Expiry behaviour
# ---------------------------------------------------------------------------

class TestLockExpiry:
    """Tests that verify locks respect their TTL / expiry contract."""

    @pytest.mark.asyncio
    async def test_expired_lock_can_be_reacquired(self):
        """
        After a lock's TTL has elapsed Redis deletes the key automatically.
        A subsequent acquire_lock call must succeed (return True).
        """
        from services.editing_lock import acquire_lock

        redis = _mock_redis()
        # First acquire: key doesn't exist yet → SET NX succeeds.
        redis.set.return_value = True
        result1 = await acquire_lock(redis, resource_id="doc-1", user_id="alice", ttl=1)
        assert result1 is True

        # Simulate TTL expiry: Redis has deleted the key automatically.
        # A new SET NX will succeed.
        redis.set.return_value = True
        result2 = await acquire_lock(redis, resource_id="doc-1", user_id="bob", ttl=300)
        assert result2 is True

    @pytest.mark.asyncio
    async def test_custom_ttl_is_forwarded_to_redis(self):
        """The ttl argument must be forwarded to the Redis SET command."""
        from services.editing_lock import acquire_lock

        redis = _mock_redis()
        await acquire_lock(redis, resource_id="doc-1", user_id="alice", ttl=600)

        kwargs = redis.set.call_args.kwargs
        ttl_sent = kwargs.get("ex") or kwargs.get("px")
        # For 'ex' the value should be 600; for 'px' it should be 600_000.
        assert ttl_sent in (600, 600_000), (
            f"Expected TTL 600 (or 600000 ms) to be forwarded, got {ttl_sent}"
        )


# ---------------------------------------------------------------------------
# get_lock_info  (if present)
# ---------------------------------------------------------------------------

class TestGetLockInfo:
    """Tests for services.editing_lock.get_lock_info (optional introspection helper)."""

    def _try_import(self):
        from services import editing_lock
        fn = getattr(editing_lock, "get_lock_info", None)
        if fn is None:
            pytest.skip("get_lock_info not exported by editing_lock")
        return fn

    @pytest.mark.asyncio
    async def test_returns_none_when_not_locked(self):
        get_lock_info = self._try_import()
        redis = _mock_redis()
        redis.get.return_value = None

        result = await get_lock_info(redis, resource_id="doc-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_dict_when_locked(self):
        get_lock_info = self._try_import()
        redis = _mock_redis()
        redis.get.return_value = json.dumps({"user_id": "alice"}).encode()

        result = await get_lock_info(redis, resource_id="doc-1")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_dict_contains_user_id(self):
        get_lock_info = self._try_import()
        redis = _mock_redis()
        redis.get.return_value = json.dumps({"user_id": "charlie"}).encode()

        result = await get_lock_info(redis, resource_id="doc-1")
        assert result is not None
        assert result.get("user_id") == "charlie"

    @pytest.mark.asyncio
    async def test_queries_correct_key(self):
        get_lock_info = self._try_import()
        redis = _mock_redis()
        redis.get.return_value = None

        await get_lock_info(redis, resource_id="special-doc")

        key_queried = redis.get.call_args.args[0]
        assert "special-doc" in key_queried


# ---------------------------------------------------------------------------
# is_locked  (if present)
# ---------------------------------------------------------------------------

class TestIsLocked:
    """Tests for services.editing_lock.is_locked (optional boolean helper)."""

    def _try_import(self):
        from services import editing_lock
        fn = getattr(editing_lock, "is_locked", None)
        if fn is None:
            pytest.skip("is_locked not exported by editing_lock")
        return fn

    @pytest.mark.asyncio
    async def test_returns_false_when_key_absent(self):
        is_locked = self._try_import()
        redis = _mock_redis()
        redis.exists.return_value = 0
        redis.get.return_value = None

        result = await is_locked(redis, resource_id="doc-1")
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_when_key_present(self):
        is_locked = self._try_import()
        redis = _mock_redis()
        redis.exists.return_value = 1
        redis.get.return_value = json.dumps({"user_id": "alice"}).encode()

        result = await is_locked(redis, resource_id="doc-1")
        assert result is True


# ---------------------------------------------------------------------------
# extend_lock  (if present)
# ---------------------------------------------------------------------------

class TestExtendLock:
    """Tests for services.editing_lock.extend_lock (optional extension helper)."""

    def _try_import(self):
        from services import editing_lock
        fn = getattr(editing_lock, "extend_lock", None)
        if fn is None:
            pytest.skip("extend_lock not exported by editing_lock")
        return fn

    @pytest.mark.asyncio
    async def test_owner_can_extend(self):
        extend_lock = self._try_import()
        redis = _mock_redis()
        redis.get.return_value = json.dumps({"user_id": "alice"}).encode()
        redis.expire.return_value = True

        result = await extend_lock(redis, resource_id="doc-1", user_id="alice", ttl=300)
        assert result is True

    @pytest.mark.asyncio
    async def test_owner_extend_calls_expire(self):
        extend_lock = self._try_import()
        redis = _mock_redis()
        redis.get.return_value = json.dumps({"user_id": "alice"}).encode()

        await extend_lock(redis, resource_id="doc-1", user_id="alice", ttl=300)
        redis.expire.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_owner_cannot_extend(self):
        extend_lock = self._try_import()
        redis = _mock_redis()
        redis.get.return_value = json.dumps({"user_id": "alice"}).encode()

        result = await extend_lock(redis, resource_id="doc-1", user_id="bob", ttl=300)
        assert result is False

    @pytest.mark.asyncio
    async def test_extend_nonexistent_lock_returns_false(self):
        extend_lock = self._try_import()
        redis = _mock_redis()
        redis.get.return_value = None  # key does not exist

        result = await extend_lock(redis, resource_id="doc-1", user_id="alice", ttl=300)
        assert result is False


# ---------------------------------------------------------------------------
# Concurrency edge cases
# ---------------------------------------------------------------------------

class TestConcurrencyEdgeCases:
    """
    Verify that the lock semantics hold under simulated concurrent conditions.
    These tests use sequential mock state changes to model race conditions.
    """

    @pytest.mark.asyncio
    async def test_acquire_after_release(self):
        """After a lock is released, a third user should be able to acquire it."""
        from services.editing_lock import acquire_lock, release_lock

        redis = _mock_redis()

        # Alice acquires.
        redis.set.return_value = True
        r1 = await acquire_lock(redis, resource_id="doc-1", user_id="alice")
        assert r1 is True

        # Alice releases.
        redis.get.return_value = json.dumps({"user_id": "alice"}).encode()
        r2 = await release_lock(redis, resource_id="doc-1", user_id="alice")
        assert r2 is True

        # Charlie now acquires (Redis key is gone after delete).
        redis.set.return_value = True
        r3 = await acquire_lock(redis, resource_id="doc-1", user_id="charlie")
        assert r3 is True

    @pytest.mark.asyncio
    async def test_bob_blocked_while_alice_holds_lock(self):
        """Bob must not acquire while Alice holds the lock."""
        from services.editing_lock import acquire_lock

        redis = _mock_redis()

        # Alice already holds the lock — SET NX returns None.
        redis.set.return_value = None

        result = await acquire_lock(redis, resource_id="shared-doc", user_id="bob")
        assert result is False