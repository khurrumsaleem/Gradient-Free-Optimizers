# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np

from .base_population_optimizer import BasePopulationOptimizer
from ._individual import Individual


class GeneticAlgorithmOptimizer(BasePopulationOptimizer):
    name = "Genetic Algorithm"
    _name_ = "genetic_algorithm"
    __name__ = "GeneticAlgorithmOptimizer"

    optimizer_type = "population"
    computationally_expensive = False

    def __init__(
        self,
        *args,
        population=10,
        offspring=20,
        crossover="discrete-recombination",
        n_parents=2,
        replace_parents=False,
        mutation_rate=0.7,
        crossover_rate=0.3,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.population = population
        self.offspring = offspring
        self.crossover = crossover
        self.n_parents = n_parents
        self.replace_parents = replace_parents
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.individuals = self._create_population(Individual)
        self.optimizers = self.individuals

        self.offspring_l = []

    def discrete_recombination(self, parent_l):
        n_arrays = len(parent_l)
        size = parent_l[0].pos_new.size

        if random.choice([True, False]):
            choice = [True, False]
        else:
            choice = [False, True]
        if size > 2:
            add_choice = np.random.randint(n_arrays, size=size - 2).astype(bool)
            choice += list(add_choice)
        return np.choose(choice, parent_l)

    def fittest_parents(self):
        fittest_parents_f = 0.5

        self.sort_pop_best_score()

        n_fittest = int(len(self.pop_sorted) * fittest_parents_f)
        return self.pop_sorted[:n_fittest]

    def _crossover(self):
        fittest_parents = self.fittest_parents()
        selected_parents = random.sample(fittest_parents, self.n_parents)

        for _ in range(self.offspring):
            offspring = self.discrete_recombination(selected_parents)
            offspring = self._constraint_loop(offspring)
            self.offspring_l.append(offspring)

            print("\n offspring \n", offspring, "\n")

    def _constraint_loop(self, position):
        print("\n position \n", position, "\n")
        while True:
            if self.conv.not_in_constraint(position):
                return position
            position = self.p_current.move_climb(position, epsilon_mod=0.3)

    @BasePopulationOptimizer.track_new_pos
    def init_pos(self):
        nth_pop = self.nth_trial % len(self.individuals)

        self.p_current = self.individuals[nth_pop]
        return self.p_current.init_pos()

    def _iterate_existing_offspring(self):
        pass

    def _iterate_no_offspring(self):
        pass

    def _iterate_pure_mutation(self):
        pass

    @BasePopulationOptimizer.track_new_pos
    def iterate(self):
        n_ind = len(self.individuals)

        if n_ind == 1:
            self.p_current = self.individuals[0]
            return self.p_current.iterate()

        self.sort_pop_best_score()
        rnd_int = random.randint(0, len(self.pop_sorted) - 1)
        self.p_current = self.pop_sorted[rnd_int]

        total_rate = self.mutation_rate + self.crossover_rate
        rand = np.random.uniform(low=0, high=total_rate)

        if rand <= self.mutation_rate:
            return self.p_current.iterate()
        else:
            return self._crossover()

    @BasePopulationOptimizer.track_new_score
    def evaluate(self, score_new):
        self.p_current.evaluate(score_new)