import polars as pl
import pytest

from radar.utils.typing.validator import require_columns


def test_require_columns_success():
    @require_columns("a", "b", allow_none=False)
    def my_func(df):
        return "ok"

    # Positional argument DataFrame
    df_valid = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert my_func(df_valid) == "ok"

    # Keyword argument DataFrame
    assert my_func(df=df_valid) == "ok"


def test_require_columns_missing_and_extra():
    @require_columns("a", "b", allow_none=False)
    def my_func(df):
        return "ok"

    # Missing column 'b'
    df_missing = pl.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="missing columns: \\['b'\\]"):
        my_func(df_missing)

    # Extra column 'c'
    df_extra = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    with pytest.raises(ValueError, match="unexpected additional columns: \\['c'\\]"):
        my_func(df_extra)

    # Both missing and extra
    df_both = pl.DataFrame({"a": [1, 2], "c": [5, 6]})
    with pytest.raises(ValueError) as excinfo:
        my_func(df_both)
    assert "missing columns: ['b']" in str(excinfo.value)
    assert "unexpected additional columns: ['c']" in str(excinfo.value)


def test_require_columns_empty_df():
    @require_columns("a", "b", allow_none=False)
    def my_func(df):
        return "ok"

    df_empty = pl.DataFrame({"a": [], "b": []})
    with pytest.raises(ValueError, match="The provided DataFrame is empty"):
        my_func(df_empty)


def test_require_columns_no_df():
    @require_columns("a", "b", allow_none=False)
    def my_func(x):
        return "ok"

    with pytest.raises(TypeError, match="expects a DataFrame, but none was found"):
        my_func(123)
