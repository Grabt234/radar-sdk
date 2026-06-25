import datetime

import logging
from time import sleep

from radar.components import geometry
from radar.utils.calculate import convert
from radar.utils.typing.enums import FrequencyUnit
from radar.utils.typing.units import Frequency

from radar.components import Element
from radar.utils.calculate import pattern
from radar.utils.typing import (
    PhaseUnit,
    DirectionDomain,
    FigureType,
    AmplitudeDomain,
    Angle,
    AmplitudeUnit,
)

from radar.components.array import Array
import polars as pl

from radar.utils.typing.constants import DataHeader


import array
from typing import Literal, cast
import polars as pl

from radar.optimiser.biology import Organism
from IPython.display import clear_output
from radar.utils.typing.enums import BoundType

def calculate_genetic_health3(org: Organism, generation : int | None = None, plot = False) -> dict:

    x = org._chromosomes[0]
    y = org._chromosomes[1]
    g = org._chromosomes[2]

    # print(g.df)
    
    az_bound = 90
    el_bound = 90
    az_bound_tuple = (Angle(-az_bound, PhaseUnit.DEGREE), Angle(az_bound, PhaseUnit.DEGREE))
    el_bound_tuple = (Angle(-el_bound, PhaseUnit.DEGREE), Angle(el_bound, PhaseUnit.DEGREE))


    element_pattern = pattern.Isotropic()
    freq = Frequency(1, FrequencyUnit.GIGAHERTZ)
    antenna_element = Element(element_pattern, az_bound_tuple, el_bound_tuple, freq, 1)

    cf = Frequency(1, FrequencyUnit.GIGAHERTZ)
    array_geometry = geometry.CustomGeometry(x.df.to_numpy().ravel(), y.df.to_numpy().ravel())
    
    # Geometry
    df = array_geometry.df
    updated_df = (
            df.with_columns(
                g.df.to_series().alias(DataHeader.GEOM_AMP_GAIN_DB)
            )
            .drop(DataHeader.GEOM_AMP_GAIN_LIN)
        )
    array_geometry.gains = updated_df

    arr = Array(antenna_element, array_geometry)
    
    beam = arr.beam_pattern(cf, None)
    beam2 = beam.clone()

    beam2 = beam2.with_columns(
        pl.when(
            (pl.col(DataHeader.AZIMUTH_DEG).abs() > 10) & 
            (pl.col(DataHeader.ELEVATION_DEG).abs() > 10)
        )
        .then(1)     # If condition is True -> 0
        .otherwise(0.01)  # If condition is False -> -20
        .alias(DataHeader.BEAM_GAIN_LINEAR)
    )

    average_diff = (
        beam2.select(
            (pl.col(DataHeader.BEAM_GAIN_LINEAR) - pl.lit(beam[DataHeader.BEAM_GAIN_LINEAR]))
            .abs()
            .mean()
        )
        .item()  # Extracts the single value from the resulting 1x1 DataFrame
    )

    filters = [
        (DataHeader.AZIMUTH_DEG, -20.0, 20, BoundType.EXCLUSIVE),
        (DataHeader.ELEVATION_DEG, -20, 20, BoundType.EXCLUSIVE),
        
    ]
    ave_db = arr.statistic.mean(DataHeader.ANTENNA_FACTOR_DB, freq, filters, None, 'OR')
    std_db = arr.statistic.std(DataHeader.ANTENNA_FACTOR_DB, freq, filters, None, 'OR' )
    max_db = arr.statistic.max(DataHeader.ANTENNA_FACTOR_DB, freq, filters, None, 'OR')


    ave_prop = 200*(10 ** (ave_db / 10.0))
    std_dev_prop = 40*(10 ** (std_db / 10.0))
    diff_prop =  8*average_diff
    max_prop = 4*(10**(max_db/10))
    fitness = (ave_prop  + std_dev_prop + diff_prop)*max_prop
   
    return {"fitness": fitness, "average_db" : ave_db, "standard_deviation_db" : std_db, "ave_diff_lin" : average_diff, "ave_prop" : ave_prop, "std_dev_prop" : std_dev_prop, "diff_prop" : diff_prop, "max_db" : max_db, "max_prop" : max_prop}  # arr.statistic.ave(cf)


#1. Initialize your population
from radar.optimiser.biology import Population

pop = Population(50, 500,[DataHeader.X_POS_M, DataHeader.Y_POS_M, DataHeader.GEOM_AMP_GAIN_DB], 25, [-0.5, -0.5, -3], [0.5, 0.5, 0], 0.60)
if "generation" in locals():
    del generation

#. Define how many generations you want to evolve
num_generations = 500000

print("Starting optimization loototal_diffp...\n")

start = 1
if "generation" in locals():
    start = generation
    
for generation in range(start, num_generations + 1):
    # 3. Propagate the population to the next generation
    # Pass the function name without parentheses!

    if generation % 20:
        pop.propogate(fitness_fn=calculate_genetic_health3, elite=False,generation=generation)
        print(f"Generation {generation}/{num_generations}")
    else:
        pop.propogate(fitness_fn=calculate_genetic_health3, elite=True, generation=generation)
        best = pop._elite_community._organisms[0]
        best.save("tmp5.csv")
        a = calculate_genetic_health3(best, generation, True)
        print(f"Generation {generation}/{num_generations}: - fiotness f{a["fitness"]}")


while True:
    print("sleeping")
    sleep(10)
