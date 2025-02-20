# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.dummy import add, ag2_ceil, div, mul, sub


def test_add() -> None:
    actual = add(3, 5)
    expected = 8
    assert actual == expected


def test_sub() -> None:
    actual = sub(5, 3)
    expected = 2
    assert actual == expected


def test_mul() -> None:
    actual = mul(3, 5)
    expected = 15
    assert actual == expected


def test_div() -> None:
    actual = div(10, 5)
    expected = 2
    assert actual == expected


def test_ag2_ceil() -> None:
    actual = ag2_ceil(0.9)
    expected = 1
    assert actual == expected
