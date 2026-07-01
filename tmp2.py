


from radar.utils.typing.constants import DataHeader


from radar.optimiser.biology import Population

if __name__ == "__main__":
    pop = Population(
        40,
        400,
        [DataHeader.X_POS_M, DataHeader.Y_POS_M, DataHeader.GEOM_AMP_GAIN_DB],
        25,
        [-0.5, -0.5, -3],
        [0.5, 0.5, 0],
        0.60,
    )
    if "generation" in locals():
        del generation

    from radar.optimiser.biology.fitness import fitness_function


    import logging

    # Mute the kaleido logger specifically
    logging.getLogger("kaleido").setLevel(logging.ERROR)
    # # 2. Define how many generations you want to evolve
    num_epoch = 500000

    print("Starting optimization loototal_diffp...\n")

    start = 1

    for epoch in range(start, num_epoch + 1):
        # 3. Propagate the population to the next generation
        # Pass the function name without parentheses!

        pop.propagate_epochs(
            fitness_fn=fitness_function, generations_per_epoch=5, current_epoch=epoch
        )

        best = pop._elite_community._organisms[0]
        best.save("tmp7.csv")
        # tmp = pop._elite_community._organisms[0]
        # calculate_genetic_health3(tmp, generation, True)
        fitness = fitness_function(best)
        print(fitness)
        print(f"Generation {epoch}/{num_epoch}:")
