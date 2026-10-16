import numbers
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Bytes:
    """A human-readable byte object."""

    UNITS: ClassVar[tuple[str, ...]] = ("KiB", "MiB", "GiB", "TiB", "PiB", "EiB")

    bytes: int | float

    def __post_init__(self) -> None:
        if (
            isinstance(self.bytes, bool)
            or not isinstance(self.bytes, numbers.Real)
            or self.bytes % 1
        ):
            raise TypeError(f"bytes must be a whole number, got {self.bytes!r}")

    def __int__(self) -> int:
        # `int()` requires a plain int back; `bytes` may be a float or numpy number.
        return int(self.bytes)

    def __str__(self) -> str:
        """
        Human readable byte count.

        Example:
            >>> str(Bytes(1024))
            '1.00 KiB'
        """
        if self.bytes < 1024:
            return f"{int(self)} B"
        for power, unit in enumerate(self.UNITS[:-1], start=1):
            if self.bytes < 1024 ** (power + 1):
                return f"{self.bytes / 1024**power:.2f} {unit}"
        return f"{self.bytes / 1024 ** len(self.UNITS):.2f} {self.UNITS[-1]}"
