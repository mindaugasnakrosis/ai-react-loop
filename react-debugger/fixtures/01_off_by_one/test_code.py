import pytest
from buggy_code import find_max, find_min


def test_find_max_basic():
    assert find_max([3, 1, 2]) == 3


def test_find_max_last_element_is_largest():
    # This test fails: the loop skips the last element
    assert find_max([1, 2, 3, 4, 5]) == 5


def test_find_max_single():
    assert find_max([42]) == 42


def test_find_max_negatives():
    assert find_max([-5, -1, -3]) == -1


def test_find_max_empty():
    with pytest.raises(ValueError):
        find_max([])


def test_find_min_last_element_is_smallest():
    # This test fails: same off-by-one
    assert find_min([5, 4, 3, 2, 1]) == 1


def test_find_min_basic():
    assert find_min([3, 1, 2]) == 1
