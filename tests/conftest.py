"""
tests/conftest.py — pytest configuration and shared fixtures.
"""

import socket
import pytest


def _has_network() -> bool:
    """Return True if the HuggingFace Hub is reachable."""
    try:
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("huggingface.co", 443))
        return True
    except OSError:
        return False


# Skip marker for tests that require downloading models from HuggingFace Hub
requires_network = pytest.mark.skipif(
    not _has_network(),
    reason="HuggingFace Hub not reachable in this environment",
)


@pytest.fixture(scope="session")
def bert_tokenizer():
    """Session-scoped BERT tokenizer (requires network)."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.model import get_tokenizer
    return get_tokenizer()
