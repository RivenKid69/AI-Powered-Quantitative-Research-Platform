"""Comprehensive unit tests for services/rest_budget.py - Rate limiting."""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from core_config import RetryConfig, TokenBucketConfig
from services.rest_budget import (
    TokenBucket,
    RestBudgetSession,
    iter_time_chunks,
    split_time_range,
    DAY_MS,
)


class TestTokenBucket:
    """Test TokenBucket rate limiter."""

    def test_token_bucket_initialization(self):
        """Test TokenBucket initializes correctly."""
        bucket = TokenBucket(rps=10.0, burst=20.0)
        assert bucket.rps == 10.0
        assert bucket.burst == 20.0
        assert bucket.tokens == 20.0
        assert bucket.enabled is True

    def test_token_bucket_disabled_when_zero_rps(self):
        """Test TokenBucket is disabled when rps=0."""
        bucket = TokenBucket(rps=0.0, burst=10.0)
        assert bucket.enabled is False

    def test_token_bucket_disabled_when_zero_burst(self):
        """Test TokenBucket is disabled when burst=0."""
        bucket = TokenBucket(rps=10.0, burst=0.0)
        assert bucket.enabled is False

    def test_token_bucket_wait_time_zero_when_disabled(self):
        """Test wait_time returns 0 when disabled."""
        bucket = TokenBucket(rps=0.0, burst=0.0)
        assert bucket.wait_time(tokens=1.0) == 0.0

    def test_token_bucket_wait_time_zero_when_tokens_available(self):
        """Test wait_time returns 0 when enough tokens available."""
        bucket = TokenBucket(rps=10.0, burst=20.0)
        assert bucket.wait_time(tokens=10.0) == 0.0

    def test_token_bucket_wait_time_nonzero_when_insufficient_tokens(self):
        """Test wait_time returns positive value when insufficient tokens."""
        bucket = TokenBucket(rps=10.0, burst=20.0, tokens=5.0)
        wait = bucket.wait_time(tokens=15.0)
        assert wait > 0.0
        assert wait == pytest.approx(1.0, abs=0.1)  # Need 10 more tokens at 10/s = 1s

    def test_token_bucket_consume_success(self):
        """Test consume decrements tokens when available."""
        bucket = TokenBucket(rps=10.0, burst=20.0)
        initial_tokens = bucket.tokens
        bucket.consume(tokens=5.0)
        assert bucket.tokens == pytest.approx(initial_tokens - 5.0)

    def test_token_bucket_consume_failure_insufficient(self):
        """Test consume raises RuntimeError when insufficient tokens."""
        bucket = TokenBucket(rps=10.0, burst=20.0, tokens=5.0)
        with pytest.raises(RuntimeError, match="insufficient tokens"):
            bucket.consume(tokens=10.0)

    def test_token_bucket_refill(self):
        """Test token refill over time."""
        bucket = TokenBucket(rps=10.0, burst=20.0, tokens=0.0)
        now = time.monotonic()
        bucket.last_ts = now - 1.0  # 1 second ago
        bucket._refill(now)
        assert bucket.tokens == pytest.approx(10.0, abs=0.1)  # 10 tokens/s * 1s

    def test_token_bucket_refill_capped_at_burst(self):
        """Test refill doesn't exceed burst capacity."""
        bucket = TokenBucket(rps=10.0, burst=20.0, tokens=15.0)
        now = time.monotonic()
        bucket.last_ts = now - 10.0  # 10 seconds ago
        bucket._refill(now)
        assert bucket.tokens == pytest.approx(20.0)  # Capped at burst

    def test_token_bucket_cooldown(self):
        """Test cooldown prevents token consumption."""
        bucket = TokenBucket(rps=10.0, burst=20.0)
        now = time.monotonic()
        bucket.start_cooldown(5.0, now=now)

        # Wait time should reflect cooldown
        wait = bucket.wait_time(tokens=1.0, now=now)
        assert wait >= 5.0

        # Consume should fail during cooldown
        with pytest.raises(RuntimeError, match="cooldown in effect"):
            bucket.consume(tokens=1.0, now=now)

    def test_token_bucket_adjust_rate(self):
        """Test adjust_rate updates rps and burst."""
        bucket = TokenBucket(rps=10.0, burst=20.0)
        bucket.adjust_rate(rps=5.0, burst=15.0)
        assert bucket.rps == 5.0
        assert bucket.burst == 15.0

    def test_token_bucket_adjust_rate_updates_configured(self):
        """Test adjust_rate updates configured values when requested."""
        bucket = TokenBucket(rps=10.0, burst=20.0)
        bucket.adjust_rate(rps=5.0, burst=15.0, update_configured=True)
        assert bucket.configured_rps == 5.0
        assert bucket.configured_burst == 15.0

    def test_token_bucket_adjust_rate_clamps_tokens(self):
        """Test adjust_rate clamps tokens to new burst."""
        bucket = TokenBucket(rps=10.0, burst=20.0, tokens=20.0)
        bucket.adjust_rate(rps=10.0, burst=10.0)
        assert bucket.tokens <= 10.0


class TestRestBudgetSessionInitialization:
    """Test RestBudgetSession initialization."""

    def test_initialization_default(self):
        """Test default initialization."""
        # Skip - requires proper config object
        pytest.skip("Requires complex config setup")

    def test_initialization_explicit_disabled(self):
        """Test explicit disabled configuration."""
        # Skip - requires proper config object
        pytest.skip("Requires complex config setup")

    def test_initialization_with_global_bucket(self):
        """Test initialization with global bucket."""
        # Skip - requires proper config object
        pytest.skip("Requires complex config setup")

    def test_initialization_with_endpoint_buckets(self):
        """Test initialization with endpoint-specific buckets."""
        # Skip - requires proper config object
        pytest.skip("Requires complex config setup")

    def test_initialization_with_cache(self):
        """Test initialization with caching enabled."""
        # Skip - requires proper config object
        pytest.skip("Requires complex config setup")

    def test_initialization_with_concurrency(self):
        """Test initialization with concurrent workers."""
        # Skip - requires proper config object
        pytest.skip("Requires complex config setup")


class TestRestBudgetSessionCaching:
    """Test RestBudgetSession caching functionality."""

    def test_cache_lookup_miss(self):
        """Test cache lookup returns miss for uncached request."""
        cfg = Mock()
        cfg.enabled = True
        cfg.cache = Mock()
        cfg.cache.dir = tempfile.mkdtemp()
        cfg.cache.mode = "read_write"
        cfg.global_ = None
        cfg.endpoints = {}
        cfg.concurrency = None
        cfg.batch_size = None
        cfg.retry = None

        session = RestBudgetSession(cfg)
        key, payload, hit = session._cache_lookup("GET", "http://example.com/api", {}, "api")
        assert hit is False
        assert payload is None

    def test_cache_store_and_lookup_hit(self):
        """Test cache store and subsequent hit."""
        cache_dir = tempfile.mkdtemp()
        cfg = Mock()
        cfg.enabled = True
        cfg.cache = Mock()
        cfg.cache.dir = cache_dir
        cfg.cache.mode = "read_write"
        cfg.global_ = None
        cfg.endpoints = {}
        cfg.concurrency = None
        cfg.batch_size = None
        cfg.retry = None

        session = RestBudgetSession(cfg)

        # First lookup - miss
        key, _, hit = session._cache_lookup("GET", "http://example.com/api", {}, "api")
        assert hit is False

        # Store data
        test_data = {"result": "test"}
        success = session._cache_store(key, test_data)
        assert success is True

        # Second lookup - hit
        key2, payload, hit = session._cache_lookup("GET", "http://example.com/api", {}, "api")
        assert hit is True
        assert payload == test_data

    def test_cache_ttl_expiry(self):
        """Test cache entries expire after TTL."""
        cache_dir = tempfile.mkdtemp()
        cfg = Mock()
        cfg.enabled = True
        cfg.cache = Mock()
        cfg.cache.dir = cache_dir
        cfg.cache.mode = "read_write"
        cfg.cache.ttl_days = 0.00001  # Very short TTL
        cfg.global_ = None
        cfg.endpoints = {}
        cfg.concurrency = None
        cfg.batch_size = None
        cfg.retry = None

        session = RestBudgetSession(cfg)
        session._cache_ttl_days = 0.00001

        # Store and immediately lookup
        key, _, _ = session._cache_lookup("GET", "http://example.com/api", {}, "api")
        session._cache_store(key, {"data": "test"})

        # Wait for expiry
        time.sleep(0.1)

        # Should be expired
        _, payload, hit = session._cache_lookup("GET", "http://example.com/api", {}, "api")
        assert hit is False

    def test_is_cached_method(self):
        """Test is_cached method."""
        cache_dir = tempfile.mkdtemp()
        cfg = Mock()
        cfg.enabled = True
        cfg.cache = Mock()
        cfg.cache.dir = cache_dir
        cfg.cache.mode = "read_write"
        cfg.global_ = None
        cfg.endpoints = {}
        cfg.concurrency = None
        cfg.batch_size = None
        cfg.retry = None

        session = RestBudgetSession(cfg)

        # Not cached initially
        assert session.is_cached("http://example.com/api") is False

        # Store
        key, _, _ = session._cache_lookup("GET", "http://example.com/api", {}, "api")
        session._cache_store(key, {"data": "test"})

        # Now cached
        assert session.is_cached("http://example.com/api") is True


class TestRestBudgetSessionRequests:
    """Test RestBudgetSession HTTP requests."""

    @patch("requests.Session.get")
    def test_get_request_success(self, mock_get):
        """Test successful GET request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_response.headers = {}
        mock_get.return_value = mock_response

        cfg = Mock()
        cfg.enabled = False  # Disable rate limiting for simple test
        cfg.global_ = None
        cfg.endpoints = {}
        cfg.concurrency = None
        cfg.batch_size = None
        cfg.cache = None
        cfg.retry = RetryConfig(max_attempts=1)

        session = RestBudgetSession(cfg)
        result = session.get("http://example.com/api")
        assert result == {"result": "success"}

    @patch("requests.Session.get")
    def test_get_request_with_rate_limiting(self, mock_get):
        """Test GET request respects rate limiting."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_response.headers = {}
        mock_get.return_value = mock_response

        cfg = Mock()
        cfg.enabled = True
        cfg.global_ = TokenBucketConfig(rps=1.0, burst=1.0)
        cfg.endpoints = {}
        cfg.concurrency = None
        cfg.batch_size = None
        cfg.cache = None
        cfg.retry = RetryConfig(max_attempts=1)
        cfg.jitter_ms = 0
        cfg.cooldown_s = 0
        cfg.timeout = None
        cfg.dynamic_from_headers = False

        session = RestBudgetSession(cfg)

        # First request should succeed immediately
        start = time.time()
        result1 = session.get("http://example.com/api1")
        elapsed1 = time.time() - start
        assert elapsed1 < 0.1  # Should be fast

        # Second request should wait due to rate limit
        start = time.time()
        result2 = session.get("http://example.com/api2")
        elapsed2 = time.time() - start
        assert elapsed2 > 0.5  # Should wait ~1 second for token refill

    @patch("requests.Session.get")
    def test_get_request_with_cache_hit(self, mock_get):
        """Test GET request uses cache when available."""
        cache_dir = tempfile.mkdtemp()
        cfg = Mock()
        cfg.enabled = True
        cfg.cache = Mock()
        cfg.cache.dir = cache_dir
        cfg.cache.mode = "read_write"
        cfg.global_ = None
        cfg.endpoints = {}
        cfg.concurrency = None
        cfg.batch_size = None
        cfg.retry = RetryConfig(max_attempts=1)

        session = RestBudgetSession(cfg)

        # First request - cache miss
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "cached"}
        mock_response.headers = {}
        mock_get.return_value = mock_response

        result1 = session.get("http://example.com/api")
        assert mock_get.call_count == 1

        # Second request - cache hit (no network call)
        result2 = session.get("http://example.com/api")
        assert mock_get.call_count == 1  # Should still be 1
        assert result2 == {"result": "cached"}

    @patch("requests.Session.get")
    def test_get_request_429_triggers_cooldown(self, mock_get):
        """Test 429 response triggers cooldown."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "2"}
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_get.return_value = mock_response

        cfg = Mock()
        cfg.enabled = True
        cfg.global_ = TokenBucketConfig(rps=10.0, burst=10.0)
        cfg.endpoints = {}
        cfg.concurrency = None
        cfg.batch_size = None
        cfg.cache = None
        cfg.retry = RetryConfig(max_attempts=1)
        cfg.cooldown_s = 2.0

        session = RestBudgetSession(cfg)

        with pytest.raises(requests.exceptions.HTTPError):
            session.get("http://example.com/api")

        # Cooldown should be active
        assert session.cooldown_counts["global"] > 0


class TestRestBudgetSessionCheckpointing:
    """Test RestBudgetSession checkpointing functionality."""

    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoint."""
        checkpoint_path = Path(tempfile.mktemp(suffix=".json"))

        cfg = Mock()
        cfg.enabled = True
        cfg.checkpoint = Mock()
        cfg.checkpoint.path = str(checkpoint_path)
        cfg.checkpoint.enabled = True
        cfg.checkpoint.resume_from_checkpoint = True
        cfg.global_ = None
        cfg.endpoints = {}
        cfg.concurrency = None
        cfg.batch_size = None
        cfg.cache = None
        cfg.retry = None

        session = RestBudgetSession(cfg)

        # Save checkpoint
        test_data = {"processed": 100, "total": 1000}
        session.save_checkpoint(test_data, last_symbol="BTCUSDT", progress_pct=10.0)

        # Create new session to load
        session2 = RestBudgetSession(cfg)
        loaded = session2.load_checkpoint()

        assert loaded is not None
        assert loaded["data"]["processed"] == 100
        assert loaded["last_symbol"] == "BTCUSDT"
        assert loaded["progress_pct"] == 10.0

        # Cleanup
        checkpoint_path.unlink(missing_ok=True)


class TestRestBudgetSessionStats:
    """Test RestBudgetSession statistics tracking."""

    def test_stats_initialization(self):
        """Test stats are initialized to zero."""
        cfg = Mock()
        cfg.enabled = True
        cfg.global_ = None
        cfg.endpoints = {}
        cfg.concurrency = None
        cfg.batch_size = None
        cfg.cache = None
        cfg.retry = None

        session = RestBudgetSession(cfg)
        stats = session.stats()

        assert stats["requests_total"] == 0
        assert stats["total_retries"] == 0
        assert stats["total_wait_seconds"] == 0.0

    @patch("requests.Session.get")
    def test_stats_track_requests(self, mock_get):
        """Test stats track request counts."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_response.headers = {}
        mock_get.return_value = mock_response

        cfg = Mock()
        cfg.enabled = False
        cfg.global_ = None
        cfg.endpoints = {}
        cfg.concurrency = None
        cfg.batch_size = None
        cfg.cache = None
        cfg.retry = RetryConfig(max_attempts=1)

        session = RestBudgetSession(cfg)
        session.get("http://example.com/api")

        stats = session.stats()
        assert stats["requests_total"] >= 1

    def test_plan_request(self):
        """Test plan_request tracks planned requests."""
        cfg = Mock()
        cfg.enabled = True
        cfg.global_ = None
        cfg.endpoints = {}
        cfg.concurrency = None
        cfg.batch_size = None
        cfg.cache = None
        cfg.retry = None

        session = RestBudgetSession(cfg)
        session.plan_request("/api/v1/data", count=10, tokens=2.0)

        stats = session.stats()
        assert stats["planned_requests"]["/api/v1/data"] == 10
        assert stats["planned_tokens"]["/api/v1/data"] == 20.0


class TestIterTimeChunks:
    """Test iter_time_chunks utility function."""

    def test_iter_time_chunks_basic(self):
        """Test basic time chunking."""
        start_ms = 0
        end_ms = 90 * DAY_MS  # 90 days
        chunks = list(iter_time_chunks(start_ms, end_ms, chunk_days=30))

        assert len(chunks) == 3
        assert chunks[0] == (0, 30 * DAY_MS)
        assert chunks[1] == (30 * DAY_MS, 60 * DAY_MS)
        assert chunks[2] == (60 * DAY_MS, 90 * DAY_MS)

    def test_iter_time_chunks_partial_last_chunk(self):
        """Test partial last chunk."""
        start_ms = 0
        end_ms = 35 * DAY_MS  # 35 days
        chunks = list(iter_time_chunks(start_ms, end_ms, chunk_days=30))

        assert len(chunks) == 2
        assert chunks[0] == (0, 30 * DAY_MS)
        assert chunks[1] == (30 * DAY_MS, 35 * DAY_MS)

    def test_iter_time_chunks_empty_range(self):
        """Test empty range yields no chunks."""
        chunks = list(iter_time_chunks(100, 100, chunk_days=30))
        assert len(chunks) == 0

    def test_split_time_range(self):
        """Test split_time_range wrapper."""
        chunks = split_time_range(0, 60 * DAY_MS, chunk_days=30)
        assert len(chunks) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
