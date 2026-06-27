from enum import Enum
from typing import Literal, cast
import polars as pl
from radar.utils.typing.enums import BoundType, FilterTuple


class Statistic:
    @classmethod
    def std(
        cls,
        df: pl.DataFrame,
        header: str,
        filters: list[FilterTuple] | None = None,
        combine_method: Literal["AND", "OR"] = "AND",
    ) -> float:
        if filters:
            df = cls._filter_by_bounds(df, filters, combine_method)
        val = df[header].std()
        val = cast(float | None, val)
        return val if val is not None else 0.0

    @classmethod
    def mean(
        cls,
        df: pl.DataFrame,
        header: str,
        filters: list[FilterTuple] | None = None,
        combine_method: Literal["AND", "OR"] = "AND",
    ) -> float:
        if filters:
            df = cls._filter_by_bounds(df, filters, combine_method)
        val = df[header].mean()
        val = cast(float | None, val)
        return val if val is not None else 0.0

    @classmethod
    def max(
        cls,
        df: pl.DataFrame,
        header: str,
        filters: list[FilterTuple] | None = None,
        combine_method: Literal["AND", "OR"] = "AND",
    ) -> float:
        if filters:
            df = cls._filter_by_bounds(df, filters, combine_method)
        val = df[header].max()
        val = cast(float | None, val)
        return val if val is not None else 0.0

    @classmethod
    def min(
        cls,
        df: pl.DataFrame,
        header: str,
        filters: list[FilterTuple] | None = None,
        combine_method: Literal["AND", "OR"] = "AND",
    ) -> float:
        if filters:
            df = cls._filter_by_bounds(df, filters, combine_method)
        val = df[header].max()
        val = cast(float | None, val)
        return val if val is not None else 0.0
    
    @classmethod
    def _filter_by_bounds(
        cls,
        df: pl.DataFrame,
        filters: list[FilterTuple],
        combine_method: Literal["AND", "OR"],
    ) -> pl.DataFrame:
        exprs = []
        for header, lower, upper, bound_type in filters:
            if bound_type == BoundType.INCLUSIVE:
                expr = pl.col(header).is_between(lower, upper, closed="both")
            else:
                expr = (pl.col(header) < lower) | (pl.col(header) > upper)
            exprs.append(expr)

        if not exprs:
            return df

        # If "AND", all conditions must be true (Four Corners)
        if combine_method == "AND":
            return df.filter(exprs)

        # If "OR", at least one condition must be true (Donut/Cross-exclusion)
        else:
            combined_expr = exprs[0]
            for expr in exprs[1:]:
                combined_expr = combined_expr | expr
            return df.filter(combined_expr)
