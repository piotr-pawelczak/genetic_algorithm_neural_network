"""
TODO module docstring
"""

import numpy as np
from neural_network import NeuralNetwork

#pylint: disable=too-many-arguments, line-too-long

class GeneticAlgorithm():
    """TODO class docstring
    """

    def __init__(self, network: NeuralNetwork, x_train: np.ndarray, y_train: np.ndarray, population_size: int, num_parents: int, metric_type: str):
        self.network = network
        self.population_size = population_size
        self.x_train = x_train
        self.y_train = y_train
        self.num_parents = num_parents
        self.metric_type = metric_type

        self.population = None

    def generate_population(self, min_value: np.float32 = -1, max_value: np.float32 = 1):
        """
        Create random population with given boundary values based on attribute population_size

        Args:
            min_value (np.float32, optional): minimum value of gene. Defaults to -1.
            max_value (np.float32, optional): maximum value of gene. Defaults to 1.
        """
        random_population = np.random.uniform(low=min_value, high=max_value, size=(self.population_size, 151))
        self.population = random_population

    def get_fitness(self, chromosome: np.ndarray, batch_size=10) -> float:
        """
        Return fitness value of given chromosome

        Args:
            network (NeuralNetwork): NeuralNetwork object
            chromosome (np.ndarray): chromosome contains weights which are passed to model
            batch_size (int, optional): batch_size. Defaults to 10.

        Raises:
            ValueError: Raises ValueError if metric_type is other than loss or accuracy

        Returns:
            float: value of loss or accuracy transformed to fitness
        """

        self.network.set_weights(chromosome)

        if self.metric_type not in ['loss', 'accuracy']:
            raise ValueError('metric_type should be loss or accuracy')

        if self.metric_type == 'accuracy':
            fitness = self.network.get_accuracy(self.x_train, self.y_train, batch_size)
        elif self.metric_type == 'loss':
            loss = self.network.get_loss(self.x_train, self.y_train, batch_size)
            fitness = 1 / (loss + 0.000001)
        return fitness

    def select_roulette(self) -> np.ndarray:
        """
        Args:
            num_parents (int): number of parents

        Returns:
            np.ndarray: array of parents selected by roulette wheel selection method
        """
        population_fitness = list(map(lambda chromosome: self.get_fitness(chromosome, self.metric_type), self.population))
        total_fitness = sum(population_fitness)

        chromosome_probabilities = [chromosome_fitness/total_fitness for chromosome_fitness in population_fitness]
        return np.random.choice(self.population, size=self.num_parents, p=chromosome_probabilities, replace=False)
