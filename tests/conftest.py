from __future__ import annotations

import sys
import types
from pathlib import Path

# Project root contains this file; tests may live alongside sources or under tests/
PROJECT_ROOT = Path(__file__).resolve().parent
TESTS = PROJECT_ROOT / "tests"

# Load stdlib logging before project paths are added
sys.path = [p for p in sys.path if p not in {str(PROJECT_ROOT), str(TESTS)}]
import logging  # noqa: F401
sys.path.extend([str(PROJECT_ROOT), str(TESTS)])

_requests_stub = types.ModuleType("requests")


def _unavailable(*args, **kwargs):  # pragma: no cover - network calls disabled in tests
    raise RuntimeError("requests module is not available in the test environment")


_requests_stub.get = _unavailable
_requests_stub.post = _unavailable
_requests_stub.put = _unavailable
_requests_stub.delete = _unavailable
_requests_stub.request = _unavailable

# Create stub exceptions module for testing
_requests_exceptions_stub = types.ModuleType("requests.exceptions")


class RequestException(Exception):
    """Base exception for requests."""
    pass


class HTTPError(RequestException):
    """HTTP error occurred."""
    pass


class ConnectionError(RequestException):
    """Connection error occurred."""
    pass


class Timeout(RequestException):
    """Request timed out."""
    pass


class TooManyRedirects(RequestException):
    """Too many redirects."""
    pass


# Add exception classes to both modules
_requests_exceptions_stub.RequestException = RequestException
_requests_exceptions_stub.HTTPError = HTTPError
_requests_exceptions_stub.ConnectionError = ConnectionError
_requests_exceptions_stub.Timeout = Timeout
_requests_exceptions_stub.TooManyRedirects = TooManyRedirects

_requests_stub.exceptions = _requests_exceptions_stub
_requests_stub.RequestException = RequestException
_requests_stub.HTTPError = HTTPError
_requests_stub.ConnectionError = ConnectionError
_requests_stub.Timeout = Timeout

sys.modules.setdefault("requests", _requests_stub)
sys.modules.setdefault("requests.exceptions", _requests_exceptions_stub)
