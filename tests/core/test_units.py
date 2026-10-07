import numpy as np
import pytest

from modules.core.utils.units import Bytes


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (0, "0 B"),
        (1023, "1023 B"),
        (1024, "1.00 KiB"),
        (1536, "1.50 KiB"),
        (5 * 2**20, "5.00 MiB"),
        (int(681.3 * 2**30), "681.30 GiB"),
        (3 * 2**40, "3.00 TiB"),
        (2**70, "1024.00 EiB"),
        (np.int64(1536), "1.50 KiB"),
        (1e7, "9.54 MiB"),
        (512.0, "512 B"),
        (np.float32(2048), "2.00 KiB"),
    ],
)
def test_whole_numbers_print_in_binary_units(value, text):
    size = Bytes(value)
    expected = Bytes(int(value))
    assert f"{size}" == text
    assert int(size) == int(value) and type(int(size)) is int
    assert size == expected and hash(size) == hash(expected)


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (1.5, TypeError),
        (float("inf"), TypeError),
        (float("nan"), TypeError),
        ("1024", TypeError),
        (None, TypeError),
        (True, TypeError),
    ],
)
def test_rejects_invalid_sizes(value, error):
    with pytest.raises(error):
        Bytes(value)
