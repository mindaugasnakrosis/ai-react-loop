# Bug: off-by-one in range()

**For my reference only — the agent never sees this file.**

## What's wrong

Both `find_max` and `find_min` use `range(len(items) - 1)` which iterates
indices 0 through n-2, skipping the last element. The fix is `range(len(items))`.

## Why this is a realistic bug

A developer might write this thinking "I'm starting at index 0 and comparing
`items[i]` to `max_val` which is already initialised to `items[0]`, so I don't
need to include index 0 in the loop — I'll start at 1." That's correct reasoning
but the implementation goes wrong: they type `len(items) - 1` instead of
`range(1, len(items))` or just `range(len(items))`.

## Tests that catch it

- `test_find_max_last_element_is_largest`: [1,2,3,4,5] → expects 5, gets 4
- `test_find_min_last_element_is_smallest`: [5,4,3,2,1] → expects 1, gets 2

## Fix

```python
for i in range(len(items)):   # was: range(len(items) - 1)
```
