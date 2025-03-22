from typing import Iterator, Tuple, Any

def batch(batch_size: int, *args: list[Any]) -> Iterator[Tuple[list[Any], ...]]:
    """
    Splits multiple lists into batches of a specified size.
    Args:
        batch_size (int): The size of each batch.
        *args (list[Any]): One or more lists to be batched. All lists must be of the same length.
    Yields:
        Iterator[Tuple[list[Any], ...]]: An iterator over tuples, where each tuple contains slices of the input lists.
    Raises:
        ValueError: If the input lists are not of the same length.
    """

    if not args:
        return

    length = len(args[0])
    if not all(len(arg) == length for arg in args):
        raise ValueError("All input lists must be the same length.")

    for i in range(0, length, batch_size):
        yield tuple(arg[i:i + batch_size] for arg in args)
