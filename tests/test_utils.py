import pytest
from slurmomatic import batch  # Replace 'your_module' with your actual module name


def test_single_list_batching():
    data = [1, 2, 3, 4, 5]
    batches = list(batch(2, data))
    assert batches == [([1, 2],), ([3, 4],), ([5],)]


def test_multiple_lists_batching():
    a = [1, 2, 3, 4]
    b = ['a', 'b', 'c', 'd']
    expected = [
        ([1, 2], ['a', 'b']),
        ([3, 4], ['c', 'd'])
    ]
    assert list(batch(2, a, b)) == expected


def test_batch_size_larger_than_list():
    a = [1, 2]
    b = ['x', 'y']
    expected = [([1, 2], ['x', 'y'])]
    assert list(batch(10, a, b)) == expected


def test_empty_lists():
    assert list(batch(2, [], [])) == []


def test_mismatched_list_lengths():
    with pytest.raises(ValueError, match="same length"):
        list(batch(2, [1, 2], [3]))


def test_no_input_lists():
    assert list(batch(2)) == []


def test_batch_exact_division():
    a = [1, 2, 3, 4]
    b = ['a', 'b', 'c', 'd']
    result = list(batch(2, a, b))
    assert result == [([1, 2], ['a', 'b']), ([3, 4], ['c', 'd'])]
