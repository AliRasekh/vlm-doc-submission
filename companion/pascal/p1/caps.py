"""Tile-cap candidate derivation for P1 (no raise above original M)."""

from __future__ import annotations

from typing import Iterable


def candidate_max_nums(original_max_num: int, *, min_num: int = 1) -> list[int]:
    """Return unique sorted caps in {1, min(4, M), M} that are legal vs min_num."""
    m = int(original_max_num)
    if m < int(min_num):
        raise ValueError(f"original_max_num={m} < min_num={min_num}")
    raw = {1, min(4, m), m}
    valid = sorted(c for c in raw if c >= int(min_num))
    omitted = sorted(c for c in raw if c < int(min_num))
    return valid


def omitted_caps(original_max_num: int, *, min_num: int = 1) -> list[int]:
    m = int(original_max_num)
    raw = {1, min(4, m), m}
    return sorted(c for c in raw if c < int(min_num))


def execution_order(
    caps: Iterable[int], *, original_max_num: int
) -> list[int]:
    """Control (original M) first, then remaining caps descending."""
    caps_set = set(int(c) for c in caps)
    ordered = [int(original_max_num)]
    for c in sorted(caps_set, reverse=True):
        if c != int(original_max_num):
            ordered.append(c)
    return ordered
