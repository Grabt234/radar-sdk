from typing import List, Tuple
import random
import uuid
import polars as pl
from .chromosome import Chromosome


class Organism:
    """Represents an individual organism containing a set of chromosomes.

    Handles genetic operations at the organism level, including independent
    crossover (mating) and rolling mutations.
    """

    def __init__(
        self,
        chromosomes: List[Chromosome],
        mutation_chance: float = 0.1,
        major_mutation_chance: float = 0.1,
    ):
        """Initializes the Organism with a collection of chromosomes and mutation rates.

        Args:
            chromosomes (List[Chromosome]): The set of traits representing the organism's configuration.
            mutation_chance (float, optional): Probability of a mutation occurring on each chromosome. Defaults to 0.45.
            major_mutation_chance (float, optional): Probability that an occurred mutation is a major reset rather than minor drift. Defaults to 0.30.
        """
        self._chromosomes = chromosomes
        self.mutation_chance = mutation_chance
        self.major_mutation_chance = major_mutation_chance
        self._id = uuid.uuid4()

    @property
    def chromosomes(self) -> List[Chromosome]:
        """List[Chromosome]: Read-only access to the organism's chromosomes."""
        return self._chromosomes

    def clone(self) -> "Organism":
        """Returns a deep clone of the organism and its chromosomes."""
        cloned_chroms = [c.clone() for c in self._chromosomes]
        return Organism(cloned_chroms, self.mutation_chance, self.major_mutation_chance)

    def mate(self, organism: "Organism") -> Tuple["Organism", "Organism"]:
        """Mates this organism with another, producing two offspring."""
        # 1. Perform genetic crossover to get clean, isolated child configurations
        child_1_chroms, child_2_chroms = self._cross_over(organism)

        child_1 = Organism(
            child_1_chroms, self.mutation_chance, self.major_mutation_chance
        )
        child_2 = Organism(
            child_2_chroms, self.mutation_chance, self.major_mutation_chance
        )

        # 2. Mutate both children independently
        child_1._mutate()
        child_2._mutate()

        return child_1, child_2

    def _cross_over(
        self, organism: "Organism"
    ) -> Tuple[List[Chromosome], List[Chromosome]]:
        """Executes a single-point crossover between chromosome lists."""
        # Clone chromosomes to completely isolate child genetic structures from parent arrays
        chroms_a = [c.clone() for c in self._chromosomes]
        chroms_b = [c.clone() for c in organism.chromosomes]

        min_length = min(len(chroms_a), len(chroms_b))

        if min_length < 2:
            if random.random() > 0.5:
                return chroms_a, chroms_b
            return chroms_b, chroms_a

        crossover_point = random.randint(1, min_length - 1)

        child_1_chromosomes = chroms_a[:crossover_point] + chroms_b[crossover_point:]
        child_2_chromosomes = chroms_b[:crossover_point] + chroms_a[crossover_point:]

        return child_1_chromosomes, child_2_chromosomes

    def _mutate(self) -> None:
        """Randomly triggers major or minor mutations independently across chromosomes."""
        # Evaluated per-chromosome so mutations are decoupled across traits
        for chromosome in self._chromosomes:
            if random.random() < self.mutation_chance:
                if random.random() < self.major_mutation_chance:
                    chromosome.mutate_major(num_rows=5)
                else:
                    chromosome.mutate_minor(num_rows=5, pct=0.05)

    def to_df(self) -> pl.DataFrame:
        """Converts the organism's chromosomes into a single horizontal Polars DataFrame.

        Returns:
            pl.DataFrame: A combined DataFrame where each column represents a chromosome.
        """
        dfs = [c.df for c in self._chromosomes]
        return pl.concat(dfs, how="horizontal")

    @classmethod
    def from_df(
        cls,
        df: pl.DataFrame,
        min_vals: List[float],
        max_vals: List[float],
        mutation_chance: float = 0.45,
        major_mutation_chance: float = 0.30,
    ) -> "Organism":
        """Reconstructs an Organism from a Polars DataFrame.

        Args:
            df (pl.DataFrame): The DataFrame containing the chromosome columns.
            min_vals (List[float]): The minimum value boundaries for each column/chromosome.
            max_vals (List[float]): The maximum value boundaries for each column/chromosome.
            mutation_chance (float, optional): Mutation probability. Defaults to 0.45.
            major_mutation_chance (float, optional): Major mutation probability. Defaults to 0.30.

        Returns:
            Organism: The reconstructed Organism.
        """
        chromosomes = []
        for col_name, min_val, max_val in zip(
            df.columns, min_vals, max_vals, strict=True
        ):
            chrom = Chromosome(col_name, min_val, max_val, 0)
            chrom._df = df.select(col_name)
            chromosomes.append(chrom)
        return cls(chromosomes, mutation_chance, major_mutation_chance)

    def save(self, file_path: str) -> None:
        """Saves the organism's data as a CSV or Parquet file based on the file extension.

        Args:
            file_path (str): Path to the destination file (supports .csv, .parquet, .ipc/.arrow).
        """
        df = self.to_df()
        if file_path.endswith(".parquet"):
            df.write_parquet(file_path)
        elif file_path.endswith(".csv"):
            df.write_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Use .csv, .parquet, or .ipc")

    @classmethod
    def load(
        cls,
        file_path: str,
        min_vals: List[float],
        max_vals: List[float],
        mutation_chance: float = 0.45,
        major_mutation_chance: float = 0.30,
    ) -> "Organism":
        """Loads an organism from a saved CSV, Parquet, or IPC file.

        Args:
            file_path (str): Path to the file containing the organism DataFrame.
            min_vals (List[float]): The minimum value boundaries for each chromosome.
            max_vals (List[float]): The maximum value boundaries for each chromosome.
            mutation_chance (float, optional): Mutation probability. Defaults to 0.45.
            major_mutation_chance (float, optional): Major mutation probability. Defaults to 0.30.

        Returns:
            Organism: The loaded Organism.
        """
        if file_path.endswith(".parquet"):
            df = pl.read_parquet(file_path)
        elif file_path.endswith(".csv"):
            df = pl.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Use .csv, .parquet, or .ipc")

        return cls.from_df(
            df, min_vals, max_vals, mutation_chance, major_mutation_chance
        )
