import random
import uuid
import numpy as np
import polars as pl
from typing import Callable, List, Tuple

class Chromosome:
    """Represents a single-column genetic sequence that can undergo mutations.

    This class wraps a single-column Polars DataFrame and provides optimized
    operations to simulate genetic drift (minor mutations) and randomized resets
    (major mutations) within strict numeric boundaries.
    """

    def __init__(self, header: str, min_val: float, max_val: float, num_values: int):
        """Initializes the Chromosome by generating a single-column DataFrame."""
        if min_val >= max_val:
            raise ValueError(f"min_val ({min_val}) must be strictly less than max_val ({max_val}).")
        if num_values < 0:
            raise ValueError("num_values must be a non-negative integer.")

        self._id = uuid.uuid4()
        self.min_val: float = min_val
        self.max_val: float = max_val
        self._col_name: str = header

        rng = np.random.default_rng()
        random_values = rng.uniform(self.min_val, self.max_val, num_values)
        self._df: pl.DataFrame = pl.DataFrame({self._col_name: random_values})

    @property
    def df(self) -> pl.DataFrame:
        """pl.DataFrame: Provides read-only access to the internal DataFrame."""
        return self._df

    def clone(self) -> "Chromosome":
        """Returns a deep clone of the chromosome to prevent side-effects during crossover."""
        new_chrom = Chromosome(self._col_name, self.min_val, self.max_val, 0)
        new_chrom._df = self._df.clone()
        return new_chrom

    def mutate_major(self, num_rows: int) -> None:
        """Replaces values in randomly selected rows with entirely new random numbers."""
        total_rows = self._df.height
        if num_rows <= 0 or total_rows == 0:
            return

        sample_size = min(num_rows, total_rows)
        indices = np.random.choice(total_rows, size=sample_size, replace=False)
        new_values = np.random.uniform(self.min_val, self.max_val, size=sample_size)
        
        # .copy() decouples array from Polars' internal read-only memory buffer
        series = self._df[self._col_name].to_numpy().copy()
        series[indices] = new_values
        self._df = pl.DataFrame({self._col_name: series})

    def mutate_minor(self, num_rows: int, pct: float) -> None:
        """Shifts values in randomly selected rows by a random percentage."""
        if pct < 0:
            raise ValueError("Mutation percentage (pct) cannot be negative.")
        total_rows = self._df.height
        if num_rows <= 0 or total_rows == 0:
            return

        sample_size = min(num_rows, total_rows)
        indices = np.random.choice(total_rows, size=sample_size, replace=False)
        
        series = self._df[self._col_name].to_numpy().copy()
        modifiers = np.random.uniform(1 - pct, 1 + pct, size=sample_size)
        
        # Apply mutation and clamp inside safe boundaries
        series[indices] = np.clip(series[indices] * modifiers, self.min_val, self.max_val)
        self._df = pl.DataFrame({self._col_name: series})