import random
from typing import Callable, List
from .organism import Organism
from .chromosome import Chromosome
import uuid

# Define a helper function at the module level (required for process pooling)
def create_community(size, headers, num_vals, mins, maxs, kill_p):
    return Community(
        community_size=size, 
        headers=headers, 
        num_values_per_chrom=num_vals, 
        min_vals=mins, 
        max_vals=maxs,
        kill_prop=kill_p
    )

class Community:
    """Manages a collection of Organisms, executing selection, sorting,
    mating, and generational propagation based on fitness scores.
    """

    def __init__(
        self,
        community_size: int,
        headers: List[str],
        num_values_per_chrom: int,
        min_vals: List[float],
        max_vals: List[float],
        kill_prop: float = 0.3,
    ):
        """Initializes the population by generating a specified number of organisms."""
        if not (0.0 <= kill_prop < 1.0):
            raise ValueError("kill_prop must be between 0.0 (inclusive) and 1.0 (exclusive).")
            
        self.community_size = community_size
        self.kill_prop = kill_prop
        
        self._headers = headers
        self._num_values = num_values_per_chrom
        self._min_vals = min_vals
        self._max_vals = max_vals

        self._organisms: List[Organism] = []

        for _ in range(self.community_size):
            zipped = zip(headers, min_vals, max_vals, strict=True)
            chromosomes = [
                Chromosome(items[0], items[1], items[2], num_values_per_chrom)
                for items in zipped
            ]
            self._organisms.append(Organism(chromosomes))

        self._fitness_map: dict[Organism, float] = {}
        self._fittest_organism = (None, None)
        self._id = uuid.uuid4()

    @property
    def organisms(self) -> List[Organism]:
        """Returns the current list of organisms in the population."""
        return self._organisms

    def sort(self, fitness_fn: Callable[[Organism], dict]) -> None:
        """Evaluates and reorders the population. d]) -
        
        Default behavior sorts by absolute value (closest to 0 comes first).
        """
        # 1. Map organisms to their raw scores
        self._fitness_map = {org: fitness_fn(org)["fitness"] for org in self._organisms}

        # 2. Sort the actual list in place based on ABSOLUTE value (distance from 0)
        # reverse=False means smallest distance (closest to 0) comes first
        self._organisms.sort(key=lambda org: abs(self._fitness_map[org]), reverse=False)

        self._fittest_organism = (self._fitness_map[self._organisms[0]], self._organisms[0])

    def select(self) -> None:
        """Culls a proportion of the active population pool based on `kill_prop`."""
        current_size = len(self._organisms)
        num_to_keep = int(current_size * (1.0 - self.kill_prop))

        # Ensure at least 2 organisms always survive to keep breeding viable
        num_to_keep = max(2, num_to_keep)
        self._organisms = self._organisms[:num_to_keep]
        

    def mate(self) -> None:
        """Pairs up the remaining organisms and mates them to produce offspring."""
        offspring: List[Organism] = []
        survivors = self._organisms.copy()
        
        # Shuffle a copy so breeding pairs are randomized, 
        # without destroying the elite ordering in self._organisms yet
        random.shuffle(survivors)
        
        for i in range(0, len(survivors) - 1, 2):
            parent_1 = survivors[i]
            parent_2 = survivors[i+1]
            
            child_1, child_2 = parent_1.mate(parent_2)
            offspring.extend([child_1, child_2])
        
        # Handle odd-numbered survivor pools safely
        if len(survivors) % 2 != 0 and len(survivors) > 1:
            parent_1 = survivors[-1]
            parent_2 = random.choice(survivors[:-1])
            child_1, child_2 = parent_1.mate(parent_2)
            offspring.extend([child_1, child_2])

        self._organisms.extend(offspring)

    def fill(self) -> None:
        """Replenishes leftover layout vacancies with fresh genetic structures if necessary."""
        while len(self._organisms) < self.community_size:
            zipped = zip(self._headers, self._min_vals, self._max_vals, strict=True)
            fresh_chromosomes = [
                Chromosome(item[0], item[1], item[2], self._num_values)
                for item in zipped
            ]
            self._organisms.append(Organism(fresh_chromosomes))

        self._organisms = self._organisms[:self.community_size]     

    def propagate(self, fitness_fn: Callable[[Organism], dict]) -> None:
        """Executes a full generational cycle while preserving elite performance thresholds."""
        # Ensure the population is sorted before first selection or if new organisms have been seeded
        if self._fittest_organism[0] is None or len(self._organisms) > self.community_size:
            self.sort(fitness_fn)
            
        # Step 3: Cull the weak performers
        self.select() 
        
        # Step 4: Pair up and cross-breed survivors
        self.mate()

        # Step 5: Merge in brand new genetic material if numbers are low
        self.fill()

        # Step 6: Grade and sort the newly combined population (Survivors + Kids + Fresh)
        self.sort(fitness_fn)

        # Clean up mapping references for the garbage collector
        self._fitness_map.clear()

    def seed(self, org : Organism) -> None:
        """Seeds the community with a specific pre-configured organism.

        Args:
            org (Organism): The organism to insert into the community.
        """
        self._organisms.append(org)
        

