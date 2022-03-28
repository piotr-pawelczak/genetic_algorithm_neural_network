"""
TODO module docstring
"""

from operator import index
import numpy as np
from neural_network import NeuralNetwork
from typing import Tuple
import random

#pylint: disable=too-many-arguments, line-too-long

class GeneticAlgorithm():
    """TODO class docstring

    Supported crossover operations:

    Single point: Implemented using the single_point_crossover() method.
    Two points: Implemented using the two_points_crossover() method.
    Uniform: Implemented using the uniform_crossover() method.
    """

    def __init__(self, network: NeuralNetwork, x_train: np.ndarray, y_train: np.ndarray, population_size: int,
                 num_parents: int, metric_type: str, crossover_type: str = "single_point", mutation_type: str = "uniform"):
        self.network = network
        self.population_size = population_size
        self.x_train = x_train
        self.y_train = y_train
        self.num_parents = num_parents
        self.metric_type = metric_type
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type

        self.population = None

    def generate_population(self, min_value: np.float32 = -1, max_value: np.float32 = 1):
        """
        Create random population with given boundary values based on attribute population_size

        Args:
            min_value (np.float32, optional): minimum value of gene. Defaults to -1.
            max_value (np.float32, optional): maximum value of gene. Defaults to 1.
        """
        # TODO change hardcoded value (151) to class parameter or get it directly from model object
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

    """ PARENT SELECT """

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

    def select_elite(self) -> np.ndarray:
        """
        Args:
            num_parents (int): number of parents

        Returns:
            np.ndarray: array of parents selected by roulette wheel selection method
        """
        elite_population = []
        population_fitness = list(map(lambda chromosome: self.get_fitness(chromosome, self.metric_type), self.population))
        for i in range(self.num_parents):
            index = population_fitness.index(max(population_fitness))
            elite_population.append(self.population[index])
            population_fitness[index]=-99999999
        return elite_population

    """ CROSSOVER """
    # TODO consider adding crossover_probability param to class

    def make_crossover(self, selected_parents: np.ndarray) -> None:
        """
        Make crossover on selected parents array
        :param selected_parents: array with best possible parents chromosomes
        :return:
        """
        child_generation = None
        first_parent_chromosome, second_parent_chromosome = self.get_parents_for_crossover(selected_parents)

        valid_crossover_types = ["single_point", "two_points", "uniform"]

        if self.crossover_type not in valid_crossover_types:
            raise ValueError(f"Given crossover_type: {self.crossover_type} is invalid.\n Valid crossover types: "
                             f"{valid_crossover_types}")

        elif self.crossover_type == "single_point":
            child_generation = self.single_point_crossover(first_parent_chromosome, second_parent_chromosome)
        elif self.crossover_type == "two_points":
            child_generation = self.two_points_crossover(first_parent_chromosome, second_parent_chromosome)
        elif self.crossover_type == "uniform":
            child_generation = self.uniform_crossover(first_parent_chromosome, second_parent_chromosome)

        # TODO consider changing child_generation/new_generation variables in crossover methods to self.population
        return child_generation

    @staticmethod
    def get_parents_for_crossover(selected_parents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly choose parents for crossover
        :param selected_parents: array with best possible parents chromosomes
        :return: Tuple of selected parents
        """
        parents = selected_parents[np.random.randint(selected_parents.shape[0], size=2), :]
        return parents[0], parents[1]

    def create_new_population(self, first_child_chromosome: np.ndarray,
                              second_child_chromosome: np.ndarray) -> np.ndarray:
        """
        Create new population based on children's chromosomes
        :param first_child_chromosome: array with first child genes
        :param second_child_chromosome: array with second child genes
        :return:
        """
        # TODO check odd population values - possible errors
        new_population = np.zeros((self.population_size, first_child_chromosome.size))
        for population_cnt in range(0, self.population_size, 2):
            new_population[population_cnt] = first_child_chromosome
            new_population[population_cnt+1] = second_child_chromosome
        return new_population

    def single_point_crossover(self, first_parent_chromosome: np.ndarray,
                               second_parent_chromosome: np.ndarray) -> np.ndarray:
        """
        Applies the single-point crossover.
        It selects a point randomly at which crossover takes place between the pairs of parents.

        :param first_parent_chromosome: array with first parent genes
        :param second_parent_chromosome: array with second parent genes
        :return: new population of chromosomes
        """
        valid_chromosome_size = 2
        chromosome_size = first_parent_chromosome.size
        if chromosome_size < valid_chromosome_size:
            raise ValueError(f"Chromosome should have bigger size than {valid_chromosome_size} "
                             f"to perform valid crossover operation")
        crossover_point = np.random.randint(1, chromosome_size)
        first_child_chromosome = np.concatenate((first_parent_chromosome[0:crossover_point],
                                                second_parent_chromosome[crossover_point:]))
        second_child_chromosome = np.concatenate((second_parent_chromosome[0:crossover_point],
                                                 first_parent_chromosome[crossover_point:]))

        new_population = self.create_new_population(first_child_chromosome, second_child_chromosome)

        return new_population

    def two_points_crossover(self, first_parent_chromosome: np.ndarray,
                             second_parent_chromosome: np.ndarray) -> np.ndarray:
        """
        Applies the 2 points crossover.
        It selects the 2 points randomly at which crossover takes place between the pairs of parents.

        :param first_parent_chromosome: array with first parent genes
        :param second_parent_chromosome: array with second parent genes
        :return: new population of chromosomes
        """
        valid_chromosome_size = 3
        chromosome_size = first_parent_chromosome.size

        if chromosome_size < valid_chromosome_size:
            raise ValueError(f"Chromosome should have bigger size than {valid_chromosome_size} "
                             f"to perform valid crossover operation")

        first_crossover_point = np.random.randint(1, chromosome_size)
        second_crossover_point = np.random.randint(1, chromosome_size)
        while second_crossover_point == first_crossover_point:
            second_crossover_point = np.random.randint(1, chromosome_size)

        if first_crossover_point < second_crossover_point:
            lower_crossover_point, greater_crossover_point = first_crossover_point, second_crossover_point
        else:
            lower_crossover_point, greater_crossover_point = second_crossover_point, first_crossover_point

        first_child_chromosome = np.concatenate((first_parent_chromosome[0:lower_crossover_point],
                                                 second_parent_chromosome[lower_crossover_point:greater_crossover_point],
                                                 first_parent_chromosome[greater_crossover_point:]))
        second_child_chromosome = np.concatenate((second_parent_chromosome[0:lower_crossover_point],
                                                  first_parent_chromosome[lower_crossover_point:greater_crossover_point],
                                                  second_parent_chromosome[greater_crossover_point:]))

        new_population = self.create_new_population(first_child_chromosome, second_child_chromosome)

        return new_population

    def uniform_crossover(self, first_parent_chromosome: np.ndarray, second_parent_chromosome: np.ndarray) -> np.ndarray:
        """
        Applies the uniform crossover.
        For each gene, a parent out of the 2 mating parents is selected randomly and the gene is copied from it.

        :param first_parent_chromosome: array with first parent genes
        :param second_parent_chromosome: array with second parent genes
        :return: new population of chromosomes
        """
        chromosome_size = first_parent_chromosome.size
        child_chromosome = np.zeros(chromosome_size)

        for gene_cnt in range(chromosome_size):
            random_int = np.random.randint(0, 2)
            randomly_selected_parent = first_parent_chromosome if random_int else second_parent_chromosome
            child_chromosome[gene_cnt] = randomly_selected_parent[gene_cnt]

        new_population = np.zeros((self.population_size, chromosome_size))
        for population_cnt in range(self.population_size):
            new_population[population_cnt] = child_chromosome

        return new_population

    def make_mutation(self, child_generation: np.ndarray) -> np.ndarray:
        """ Make mutation on child generation after crossover.

        Args:
            child_generation (np.ndarray): Child generation after crossover.

        Returns:
            np.ndarray: Child generation after mutation.
        """
        valid_mutation_types = ["uniform", "swap", "inverse", "boundary"]

        if self.mutation_type not in valid_mutation_types:
            raise ValueError(f"Given mutation_type: {self.mutation_type} is invalid.\n Valid mutation types: "
                             f"{valid_mutation_types}")

        elif self.crossover_type == "uniform":
            child_generation = self.mutation_uniform(child_generation)
        elif self.crossover_type == "swap":
            child_generation = self.mutation_swap(child_generation)
        elif self.crossover_type == "inverse":
            child_generation = self.mutation_inverse(child_generation)
        elif self.crossover_type == "boundary":
            child_generation = self.mutation_boundary(child_generation)
        return child_generation

    def mutation_uniform(child_generation: np.ndarray) -> np.ndarray:
        """ Change value of one random gene in chromosome.

        Args:
            child_generation (np.ndarray): Child generation after crossover.

        Returns:
            np.ndarray: Child generation after mutation.
        """
        for chromosome in range(child_generation.shape[0]):
            random_index = np.randint(0, child_generation.shape[0])
            random_value = np.random.uniform(-1.0, 1.0, 1) #TODO do ustalenia jakie tu wartości
            child_generation[chromosome, random_index] = random_value
        return child_generation

    def mutation_swap(child_generation: np.ndarray) -> np.ndarray:
        """ Swap two random genes values in chromosome.

        Args:
            child_generation (np.ndarray): Child generation after crossover.

        Returns:
            np.ndarray: Child generation after mutation.
        """
        for chromosome in range(child_generation.shape[0]):
            random_indexes = random.sample(0, child_generation.shape[0], 2)
            tmp = chromosome[random_indexes[0]]
            child_generation[chromosome, random_indexes[0]] = chromosome[random_indexes[1]]
            child_generation[chromosome, random_indexes[1]] = tmp
        return child_generation

    def mutation_inverse(child_generation: np.ndarray) -> np.ndarray:
        """ Inverse one random gene value in chromosome.

        Args:
            child_generation (np.ndarray): Child generation after crossover.

        Returns:
            np.ndarray: Child generation after mutation.
        """
        for chromosome in range(child_generation.shape[0]):
            random_index = np.randint(0, child_generation.shape[0])
            child_generation[chromosome, random_index] = -(child_generation[chromosome, random_index])
        return child_generation

    def mutation_boundary(child_generation: np.ndarray) -> np.ndarray:
        """ Boundary mutation, we select a random gene from our chromosome and assign the upper bound or the lower bound to it.

        Args:
            child_generation (np.ndarray): Child generation after crossover.

        Returns:
            np.ndarray: Child generation after mutation.
        """
        lower_bound = -1.0
        upper_bound = 1.0
        for chromosome in range(child_generation.shape[0]):
            random_index = np.randint(0, child_generation.shape[0])
            tmp = chromosome[random_index]
            if tmp >= 0.5:
                child_generation[chromosome, random_index] = upper_bound
            else:
                child_generation[chromosome, random_index] = lower_bound
        return child_generation
