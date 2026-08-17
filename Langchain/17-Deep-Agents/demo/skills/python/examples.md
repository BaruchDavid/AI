# Python — Examples

## Example 1: "Write a function that reads a CSV and returns the average of a column"

Good response — typed, uses `pathlib`, minimal error handling at the
boundary (file may not exist), then actually run it against a sample file:

```python
import csv
from pathlib import Path


def average_column(csv_path: Path, column: str) -> float:
    """Return the mean of `column` across all rows in csv_path."""
    values: list[float] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row[column]))
    if not values:
        raise ValueError(f"No rows found in {csv_path}")
    return sum(values) / len(values)
```

Then: `uv run python -c "from mod import average_column; print(average_column(Path('data.csv'), 'price'))"`
to confirm it works before answering.

## Example 2: Debugging a traceback

User pastes:
```
Traceback (most recent call last):
  File "app.py", line 42, in <module>
    total = sum(order.amount for order in orders)
AttributeError: 'dict' object has no attribute 'amount'
```

Good response: "`orders` contains plain dicts, not objects with an
`.amount` attribute — line 42 assumes attribute access. Fix: use
`order['amount']`, or convert the dicts to a dataclass/`SimpleNamespace`
first if you want attribute access elsewhere too." Then show the one-line
fix, don't rewrite unrelated code.

## Example 3: Adding a dependency

User: "add pandas to the project"

Good response: run `uv add pandas` (not `pip install pandas`), then confirm
by checking `pyproject.toml` was updated. Do not manually edit the
`dependencies` array.

## Example 4: Refactor request scope

User: "this function is doing too much, can you clean it up?"

Good response: refactor only the function in question (extract 2-3 small
helpers with clear names), run existing tests to confirm behavior didn't
change. Don't also reformat unrelated files or introduce a new framework.
