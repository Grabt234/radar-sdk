from typing import List, Tuple
import random
import uuid
from .chromosome import Chromosome
from typing import List, Tuple

class Organism:
    """Represents an individual organism containing a set of chromosomes.
    
    Handles genetic operations at the organism level, including independent
    crossover (mating) and rolling mutations.
    """
    
    def __init__(
        self, 
        chromosomes: List[Chromosome], 
        mutation_chance: float = 0.45, 
        major_mutation_chance: float = 0.30
    ):
        self._chromosomes = chromosomes
        self.mutation_chance = mutation_chance
        self.major_mutation_chance = major_mutation_chance
        self._id = uuid.uuid4()

    @property
    def chromosomes(self) -> List[Chromosome]:
        """List[Chromosome]: Read-only access to the organism's chromosomes."""
        return self._chromosomes

    def mate(self, organism: "Organism") -> Tuple["Organism", "Organism"]:
        """Mates this organism with another, producing two offspring."""
        # 1. Perform genetic crossover to get clean, isolated child configurations
        child_1_chroms, child_2_chroms = self._cross_over(organism)        
        
        child_1 = Organism(child_1_chroms, self.mutation_chance, self.major_mutation_chance)
        child_2 = Organism(child_2_chroms, self.mutation_chance, self.major_mutation_chance)
        
        # 2. Mutate both children independently
        child_1._mutate()
        child_2._mutate()

        return child_1, child_2
    
    def _cross_over(self, organism: "Organism") -> Tuple[List[Chromosome], List[Chromosome]]:
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
                    chromosome.mutate_minor(num_rows=5, pct=0.1)