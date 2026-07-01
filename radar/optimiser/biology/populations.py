from .community import Community
from .organism import Organism
from typing import Callable, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from radar.utils.plotter import Generation
import time


def _worker_evolve_isolated_community(
    community_sizes: int,
    headers: List[str],
    num_values_per_chrom: int,
    min_vals: List[float],
    max_vals: List[float],
    kill_prop: float,
    fitness_fn: Callable[[Organism], dict],
    generations: int,
) -> Organism:
    """Worker function that instantiates and evolves a fresh island community entirely inside the process.

    Returns only its single top performing Organism.
    """
    community = Community(
        community_size=community_sizes,
        headers=headers,
        num_values_per_chrom=num_values_per_chrom,
        min_vals=min_vals,
        max_vals=max_vals,
        kill_prop=kill_prop,
    )

    for _ in range(generations):
        community.propagate(fitness_fn)

    return community.organisms[0]


def _worker_evolve_elite_community(
    elite_community: Community, fitness_fn: Callable[[Organism], dict], generations: int
) -> Community:
    """Worker function that takes the existing elite community and evolves it in its own process.

    Returns the fully updated and evolved Elite Community object back.
    """
    for _ in range(generations):
        elite_community.propagate(fitness_fn)

    return elite_community


class Population:
    """Manages multi-community (island model) genetic optimization using multi-processing.

    Delegates both the island lifecycles and the elite community tracking to background
    worker processes for full parallel computation.
    """

    def __init__(
        self,
        num_communities: int,
        community_sizes: int,
        headers: List[str],
        num_values_per_chrom: int,
        min_vals: List[float],
        max_vals: List[float],
        kill_prop: float,
    ) -> None:
        """Initializes the population meta-framework configuration."""
        self.generation = Generation()
        self.generation.show()

        self._num_communities = num_communities
        self._community_sizes = community_sizes
        self._headers = headers
        self._num_values_per_chrom = num_values_per_chrom
        self._min_vals = min_vals
        self._max_vals = max_vals
        self._kill_prop = kill_prop

        # Persistent elite tracking community
        self._elite_community = Community(
            community_size=community_sizes,
            headers=headers,
            num_values_per_chrom=num_values_per_chrom,
            min_vals=min_vals,
            max_vals=max_vals,
            kill_prop=0.5,
        )

    def propagate_epochs(
        self,
        fitness_fn: Callable[[Organism], dict],
        generations_per_epoch: int,
        current_epoch: int = 0,
    ) -> Organism:
        """Spawns background processes where both the islands and the elite community
        are simultaneously evolved in parallel.
        """
        start_time = time.perf_counter()
        new_immigrants = []

        # Total workers = N islands + 1 elite process
        with ProcessPoolExecutor(max_workers=self._num_communities + 1) as executor:
            futures = {}

            # 1. Submit the N fresh island communities
            for _ in range(self._num_communities):
                f = executor.submit(
                    _worker_evolve_isolated_community,
                    self._community_sizes,
                    self._headers,
                    self._num_values_per_chrom,
                    self._min_vals,
                    self._max_vals,
                    self._kill_prop,
                    fitness_fn,
                    generations_per_epoch,
                )
                futures[f] = "island"

            # 2. Submit the elite community to evolve concurrently in its own process
            fe = executor.submit(
                _worker_evolve_elite_community,
                self._elite_community,
                fitness_fn,
                generations_per_epoch,
            )
            futures[fe] = "elite"

            # Gather results as they complete
            for future in as_completed(futures):
                task_type = futures[future]
                try:
                    if task_type == "island":
                        best_island_organism = future.result()
                        new_immigrants.append(best_island_organism)
                    elif task_type == "elite":
                        # Retain the updated elite community instance sent back
                        self._elite_community = future.result()
                except Exception as exc:
                    print(
                        f"Background task ({task_type}) generated an exception: {exc}"
                    )
                    raise exc

        # 3. Migrate the top island performers into the newly evolved elite community
        self._elite_community._organisms.extend(new_immigrants)

        # 4. Filter, breed, and sort survivors inside the elite group on the main process
        self._elite_community.sort(fitness_fn)
        self._elite_community._organisms = self._elite_community._organisms[
            : self._community_sizes
        ]

        self._elite_community.mate()
        self._elite_community.sort(fitness_fn)
        self._elite_community.select()

        # 5. Log telemetry metrics
        best_overall_organism = self._elite_community._organisms[0]
        results = fitness_fn(best_overall_organism)

        for key, value in results.items():
            self.generation.append(label=key, x=current_epoch, y=value)

        end_time = time.perf_counter()
        self.generation.append(label="timer", x=current_epoch, y=end_time - start_time)

        return best_overall_organism

    def seed(self, org: Organism) -> None:
        """Seeds the elite community with a pre-selected organism configuration."""
        self._elite_community.seed(org.clone())
