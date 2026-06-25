from .community import Community
from .organism import Organism
from typing import Callable, List
from concurrent.futures import ThreadPoolExecutor, wait
from radar.utils.plotter import Generation
import time


class Population:
    """Manages multi-community (island model) genetic optimization for radar sdk designs.

    This class coordinates several isolated communities (islands) that undergo standard
    generational cycles, and periodically merges the best candidates into a central
    elite community to preserve and cross-breed the highest performing configurations.
    """

    def __init__(
        self,
        num_communities: int,
        community_sizes: int,
        headers: List[str],
        num_values_per_chrom,
        min_vals: List[float],
        max_vals: List[float],
        kill_prop,
    ) -> None:
        """Initializes the multi-community population framework.

        Args:
            num_communities (int): Number of independent communities (islands) to evolve.
            community_sizes (int): Number of individual organisms inside each community.
            headers (List[str]): List of column names representing the traits/parameters.
            num_values_per_chrom (int): Length of the genomic sequence (number of rows) for each chromosome.
            min_vals (List[float]): Minimum numeric boundary values for each chromosome parameter.
            max_vals (List[float]): Maximum numeric boundary values for each chromosome parameter.
            kill_prop (float): Culling proportion (between 0.0 and 1.0) applied to weak performers each generation.
        """
        self.generation = Generation()
        self.generation.show()

        self._num_communities = num_communities
        self._community_sizes = community_sizes
        self._headers = headers
        self._num_values_per_chrom = num_values_per_chrom
        self._min_vals = min_vals
        self._max_vals = max_vals
        self._kill_prop = kill_prop

        self._init_communitites(
            num_communities,
            community_sizes,
            headers,
            num_values_per_chrom,
            min_vals,
            max_vals,
            kill_prop,
        )
        self._elite_community = Community(
            community_size=community_sizes,
            headers=headers,
            num_values_per_chrom=num_values_per_chrom,
            min_vals=min_vals,
            max_vals=max_vals,
            kill_prop=0.5,
        )

    def _init_communitites(
        self,
        num_communities: int,
        community_sizes: int,
        headers: List[str],
        num_values_per_chrom,
        min_vals: List[float],
        max_vals: List[float],
        kill_prop,
    ):
        """Helper method to initialize all individual communities with fresh random candidates."""
        self._communities = []
        self._communities = [
            Community(
                community_size=community_sizes,
                headers=headers,
                num_values_per_chrom=num_values_per_chrom,
                min_vals=min_vals,
                max_vals=max_vals,
                kill_prop=kill_prop,
            )
            for _ in range(num_communities)
        ]

    def propogate(
        self,
        fitness_fn: Callable[[Organism], dict],
        elite=False,
        plot=True,
        generation=0,
    ):
        """Executes a single generation step across all communities.

        When elite is False, it propagates all communities (including the elite community)
        independently in parallel.
        When elite is True, it migrates the best organism of each community into the
        elite community, and then completely re-initializes the other communities.

        Args:
            fitness_fn (Callable[[Organism], float]): The grading function used to calculate organism scores.
            elite (bool, optional): If True, triggers the elite migration and community reset step. Defaults to False.
            plot (bool, optional): Unused parameter, kept for compatibility. Defaults to True.
            generation (int, optional): The current generational index for tracking. Defaults to 0.
        """
        start_time = time.perf_counter()

        if not elite:
            with ThreadPoolExecutor(max_workers=50) as executor:
                # Submits all jobs to the pool at once
                futures = [
                    executor.submit(community.propagate, fitness_fn)
                    for community in self._communities
                ]

                self._elite_community.propagate(fitness_fn)
                wait(futures)

        else:
            # Gather the best organisms from all communities
            new_immigrants = [community.organisms[0] for community in self._communities]
            self._elite_community._organisms.extend(new_immigrants)

            # Re-initialize other communities to inject fresh diversity
            self._init_communitites(
                self._num_communities,
                self._community_sizes,
                self._headers,
                self._num_values_per_chrom,
                self._min_vals,
                self._max_vals,
                self._kill_prop,
            )

            # Sort combined list and keep the best community_size organisms
            self._elite_community.sort(fitness_fn)
            self._elite_community._organisms = self._elite_community._organisms[
                : self._community_sizes
            ]

            # Breed elite organisms and select survivors based on elite community's kill_prop
            self._elite_community.mate()
            self._elite_community.sort(fitness_fn)
            self._elite_community.select()

        results = fitness_fn(self._elite_community._organisms[0])
        for key, value in results.items():
            self.generation.append(label=key, x=generation, y=value)

        end_time = time.perf_counter()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        self.generation.append(label="timer", x=generation, y=elapsed_time)

    # Alias to support correct spelling
    propagate = propogate

    def seed(self, org: Organism) -> None:
        """Seeds the elite community with a pre-selected organism configuration.

        Args:
            org (Organism): The pre-configured organism to seed into the elite community.
        """
        self._elite_community.seed(org.clone())
