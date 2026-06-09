def find_max(items):
    """Return the maximum value in a list of numbers."""
    if not items:
        raise ValueError("Cannot find max of empty list")

    max_val = items[0]
    for i in range(len(items) - 1):  # BUG: should be range(len(items))
        if items[i] > max_val:
            max_val = items[i]
    return max_val


def find_min(items):
    """Return the minimum value in a list of numbers."""
    if not items:
        raise ValueError("Cannot find min of empty list")

    min_val = items[0]
    for i in range(len(items) - 1):  # same off-by-one bug
        if items[i] < min_val:
            min_val = items[i]
    return min_val
