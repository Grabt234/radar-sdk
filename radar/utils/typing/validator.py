from functools import wraps
import polars as pl
from typing import TypeAlias


from .units import Angle, Phase


Position: TypeAlias = tuple[float, float]
AngleBound: TypeAlias = tuple[Phase | Angle, Phase | Angle]


def require_columns(*expected_cols, allow_none):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 1. Locate the DataFrame in args or kwargs
            df = None
            for arg in args:
                if isinstance(arg, pl.DataFrame):
                    df = arg
                    break
            if df is None:
                for val in kwargs.values():
                    if isinstance(val, pl.DataFrame):
                        df = val
                        break

            if df is None:
                raise TypeError(
                    f"Function '{func.__name__}' expects a DataFrame, "
                    "but none was found in the arguments."
                )

            # 2. Check if the DataFrame is empty
            if len(df) == 0:
                raise ValueError(
                    f"Validation failed for '{func.__name__}': "
                    "The provided DataFrame is empty (contains 0 rows)."
                )

            # 3. Check columns (Handles both Pandas Index and Polars list of strings)
            expected_set = set(expected_cols)
            actual_set = set(df.columns)

            missing_cols = expected_set - actual_set
            extra_cols = actual_set - expected_set

            error_messages = []
            if missing_cols:
                error_messages.append(f"missing columns: {list(missing_cols)}")
            if extra_cols:
                error_messages.append(
                    f"unexpected additional columns: {list(extra_cols)}"
                )

            if error_messages:
                raise ValueError(
                    f"DataFrame schema mismatch in '{func.__name__}'. "
                    f"Expected exactly {list(expected_cols)}. Found: "
                    + " AND ".join(error_messages)
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator
