from typing import Optional

import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import integers, tuples
from numpy.testing import assert_allclose

from .codecs import PackGeneticBits

array_t = np.ndarray


@pytest.fixture(scope="session")
def codec() -> PackGeneticBits:
    return PackGeneticBits()


def validate_roundtrip(
    codec: PackGeneticBits, x: array_t, o: Optional[array_t]
) -> None:
    o = codec.decode(codec.encode(x), o)
    assert o.shape == x.shape
    assert_allclose(o, x)


# NOTE: strategy for array shapes, positive tuple, max 42 is arbitrary
twoD_shaped_arrays = tuples(
    integers(min_value=1, max_value=42), integers(min_value=1, max_value=42)
)


def test_cant_cast_to_int_via_output(codec: PackGeneticBits) -> None:
    x = np.array([[0, 1], [1, 0]], dtype=np.int)
    # this works (note that the output is a vector):
    o = codec.decode(codec.encode(x))
    assert_allclose(o, np.array([0, 1, 1, 0]))
    # but this doesn't (when we try to preserve the shape via output array):
    with pytest.raises(ValueError, match="When changing to a larger dtype"):
        x = np.array([[0, 1], [1, 0]], dtype=np.int)
        validate_roundtrip(codec, x, np.zeros_like(x))


def test_simple_encoding(codec: PackGeneticBits) -> None:
    assert_allclose(codec.encode(np.array([[1]])), np.array([6, 128]))
    assert_allclose(codec.encode(np.array([[0, 1]])), np.array([4, 32]))
    assert_allclose(codec.encode(np.array([[1, 0]])), np.array([4, 128]))
    assert_allclose(codec.encode(np.array([[1, 1]])), np.array([4, 160]))
    assert_allclose(codec.encode(np.array([[0, 2, 0]])), np.array([2, 16]))


def test_simple_simple_roundtrip(codec: PackGeneticBits) -> None:
    assert_allclose(codec.decode(codec.encode(np.array([[1]]))), np.array([1]))
    assert_allclose(codec.decode(codec.encode(np.array([[0, 1]]))), np.array([0, 1]))
    assert_allclose(
        codec.decode(codec.encode(np.array([[0, 2, 0]]))), np.array([0, 2, 0])
    )


@given(e=arrays(np.int8, shape=twoD_shaped_arrays, elements=integers(0, 2)))
def test_positive_roundtrips(codec: PackGeneticBits, e: array_t) -> None:
    validate_roundtrip(codec, e, np.zeros_like(e))


@given(e=arrays(np.int8, shape=twoD_shaped_arrays, elements=integers(-1, 2)))
def test_all_roundtrips(codec: PackGeneticBits, e: array_t) -> None:
    validate_roundtrip(codec, e, np.zeros_like(e))
