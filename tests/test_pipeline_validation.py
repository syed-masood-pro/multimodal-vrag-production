import pytest

from src.core.exceptions import ValidationError
from src.rag.pipeline import _validate


def test_validate_empty():
    with pytest.raises(ValidationError):
        _validate("   ")


def test_validate_too_long():
    with pytest.raises(ValidationError):
        _validate("x" * 100000)


def test_validate_ok():
    assert _validate("  hello  ") == "hello"
