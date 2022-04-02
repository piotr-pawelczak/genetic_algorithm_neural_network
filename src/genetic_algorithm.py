"""
TODO module docstring
"""

import random
from typing import Tuple, List
import numpy as np
from neural_network import NeuralNetwork
from decorators import measure_time
from sklearn.model_selection import train_test_split

#pylint: disable=too-many-arguments, line-too-long, unnecessary-lambda

CHROMOSOME_SIZE = 151


class GeneticAlgorithm():
    """TODO class docstring

    Supported crossover operations:

    Single point: Implemented using the single_point_crossover() method.
    Two points: Implemented using the two_points_crossover() method.
    Uniform: Implemented using the uniform_crossover() method.


    Methods:
    - generate_population() - generate random initial population, update self.population
    - get_fitness() - get fitness of given chromosome based on type [loss, accuracy], returns float
    - select_parents() - call select_roulette() or select_elite() based on self.select_parents_type returns num_of_parents sized list 
    - 
    """

    def __init__(self, network: NeuralNetwork, X: np.ndarray, Y: np.ndarray, population_size: int = 100,
                 num_parents: int = 50, metric_type: str = "accuracy", select_parents_type: str = "elite",
                 crossover_type: str = "single_point", mutation_type: str = "uniform", iterations: int = 10):
        self.network = network
        self.population_size = population_size
        self.X = X
        self.Y = Y
        self.num_parents = num_parents
        self.metric_type = metric_type
        self.select_parents_type = select_parents_type
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.iterations = iterations

        self.population = None
        self.chromosome_size = self.network.get_total_parameters_count()

    def generate_population(self, min_value: np.float32 = -1, max_value: np.float32 = 1):
        """
        Create random population with given boundary values based on attribute population_size

        Args:
            min_value (np.float32, optional): minimum value of gene. Defaults to -1.
            max_value (np.float32, optional): maximum value of gene. Defaults to 1.
        """
        # TODO change hardcoded value (151) to class parameter or get it directly from model object
        random_population = np.random.uniform(low=min_value, high=max_value, size=(self.population_size, CHROMOSOME_SIZE))
        self.population = random_population

    def get_fitness(self, chromosome: np.ndarray, batch_size=10) -> float:
        """
        Return fitness value of given chromosome

        Args:
            chromosome (np.ndarray): chromosome contains weights which are passed to model
            batch_size (int, optional): batch_size. Defaults to 10.

        Raises:
            ValueError: Raises ValueError if metric_type is other than loss or accuracy

        Returns:
            float: value of loss or accuracy transformed to fitness
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, train_size=0.7, random_state=1)
        valid_metric_types = ['loss', 'accuracy']
        fitness_train = None
        fitness_test = None
        self.network.set_weights(chromosome)
        if self.metric_type == 'accuracy':
            fitness_train = self.network.get_accuracy(X_train, y_train, batch_size)
            fitness_test = self.network.get_accuracy(X_test, y_test, batch_size)
        elif self.metric_type == 'loss':
            loss_train = self.network.get_loss(X_train, y_train, batch_size)
            fitness_train = 1 / (loss_train + 0.000001)
            loss_test = self.network.get_loss(X_test, y_test, batch_size)
            fitness_test = 1 / (loss_test + 0.000001)
        else:
            raise ValueError(f'metric_type should be {valid_metric_types}')
        return fitness_train, fitness_test

    def find_best_chromosome(self, results) -> Tuple[float, float, np.ndarray]:
        best_chromosome = (0.0, 0.0, None)
        for i in range(len(results)):
            if results[i][0] > best_chromosome[0]:
                best_chromosome = results[i]
        return best_chromosome

    def select_parents(self) -> Tuple[np.ndarray, Tuple[float, float, np.ndarray]]:
        """
        Make parents select dependant on select_parents attribute
        :return:
        """
        selected_parents = None
        best_chromosome = None
        valid_parents_select_types = ["roulette", "elite"]

        if self.select_parents_type == "roulette":
            selected_parents = self.select_roulette()

        elif self.select_parents_type == "elite":
            selected_parents, best_chromosome = self.select_elite()

        elif self.select_parents_type not in valid_parents_select_types:
            raise ValueError(f"Given select_parents_type: {self.select_parents_type} is invalid.\n "
                             f"Valid select parents types: {valid_parents_select_types}")
        return selected_parents, best_chromosome

    def select_roulette(self) -> Tuple[np.ndarray, Tuple[float, float, np.ndarray]]:
        """
        Args:
            num_parents (int): number of parents

        Returns:
            np.ndarray: array of parents selected by roulette wheel selection method
        """
        best_chromosome = None
        population_fitness_train = list(map(lambda chromosome: self.get_fitness(chromosome)[0], self.population))
        population_fitness_test = list(map(lambda chromosome: self.get_fitness(chromosome)[1], self.population))
        total_fitness = sum(population_fitness_train)

        all_results = list(zip(population_fitness_train, population_fitness_test, self.population))
        best_chromosome = self.find_best_chromosome(all_results)

        chromosome_probabilities = [chromosome_fitness/total_fitness for chromosome_fitness in population_fitness_train]
        selected_parent_idx = np.random.choice(range(self.population_size), size=self.num_parents,
                                               p=chromosome_probabilities, replace=False)
        selected_parents = self.population[selected_parent_idx]
        return selected_parents, best_chromosome

    def select_elite(self) -> Tuple[np.ndarray, Tuple[float, float, np.ndarray]]:
        """
        Args:
            num_parents (int): number of parents

        Returns:
            np.ndarray: array of parents selected by roulette wheel selection method
        """
        elite_population = np.zeros((self.num_parents, CHROMOSOME_SIZE))
        best_chromosome = None
        population_fitness_train = list(map(lambda chromosome: self.get_fitness(chromosome)[0], self.population))
        population_fitness_test = list(map(lambda chromosome: self.get_fitness(chromosome)[1], self.population))

        all_results = list(zip(population_fitness_train, population_fitness_test, self.population))
        best_chromosome = self.find_best_chromosome(all_results)

        for i in range(self.num_parents):
            max_index = population_fitness_train.index(max(population_fitness_train))
            elite_population[i] = self.population[max_index]
            population_fitness_train[max_index] = -420
        return elite_population, best_chromosome

    # TODO consider adding crossover_probability param to class

    def make_crossover(self, selected_parents: np.ndarray):
        """
        Make crossover on selected parents array
        :param selected_parents: array with best possible parents chromosomes
        :return:
        """
        first_child, second_child = None, None
        valid_crossover_types = ["single_point", "two_points", "uniform"]

        first_parent_chromosome, second_parent_chromosome = self.get_parents_for_crossover(selected_parents)

        if self.crossover_type == "single_point":
            first_child, second_child = self.single_point_crossover(first_parent_chromosome, second_parent_chromosome)
        elif self.crossover_type == "two_points":
            first_child, second_child = self.two_points_crossover(first_parent_chromosome, second_parent_chromosome)
        elif self.crossover_type == "uniform":
            first_child = self.uniform_crossover(first_parent_chromosome, second_parent_chromosome)

        if self.crossover_type not in valid_crossover_types:
            raise ValueError(f"Given crossover_type: {self.crossover_type} is invalid.\n Valid crossover types: "
                             f"{valid_crossover_types}")

        # TODO consider changing child_generation/new_generation variables in crossover methods to self.population
        return first_child, second_child

    @staticmethod
    def get_parents_for_crossover(selected_parents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly choose parents for crossover
        :param selected_parents: array with best possible parents chromosomes
        :return: Tuple of selected parents
        """
        parents = selected_parents[np.random.randint(selected_parents.shape[0], size=2), :]
        return parents[0], parents[1]


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

        return first_child_chromosome, second_child_chromosome

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

        return first_child_chromosome, second_child_chromosome

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

        return child_chromosome


    def create_child_generation(self, selected_parents):
        new_population = np.zeros((self.population_size, self.chromosome_size))
        
        single_child_methods = ["uniform"]
        double_child_methods = ["single_point", "two_points"]
        
        if self.crossover_type in double_child_methods:
            for inx in range(0, self.population_size, 2):
                first_child, second_child = self.make_crossover(selected_parents)
                new_population[inx] = first_child
                new_population[inx+1] = second_child
        elif self.crossover_type in single_child_methods:
            for inx in range(0, self.population_size):
                child, _ = self.make_crossover(selected_parents)
                new_population[inx] = child

        return new_population


    def make_mutation(self, child_generation: np.ndarray) -> np.ndarray:
        """ Make mutation on child generation after crossover.
        Args:
            child_generation (np.ndarray): Child generation after crossover.
        Returns:
            np.ndarray: Child generation after mutation.
        """
        valid_mutation_types = ["uniform", "swap", "inverse", "boundary", "percent"]
        mutated_generation = None

        if self.mutation_type == "uniform":
            mutated_generation = self.mutation_uniform(child_generation)
        elif self.mutation_type == "swap":
            mutated_generation = self.mutation_swap(child_generation)
        elif self.mutation_type == "inverse":
            mutated_generation = self.mutation_inverse(child_generation)
        elif self.mutation_type == "boundary":
            mutated_generation = self.mutation_boundary(child_generation)
        elif self.mutation_type == "percent":
            mutated_generation = self.mutation_percent(child_generation)

        if self.mutation_type not in valid_mutation_types:
            raise ValueError(f"Given mutation_type: {self.mutation_type} is invalid.\n Valid mutation types: "
                             f"{valid_mutation_types}")

        return mutated_generation

    def mutation_uniform(self, child_generation: np.ndarray) -> np.ndarray:
        """ Change value of one random gene in chromosome.

        Args:
            child_generation (np.ndarray): Child generation after crossover.

        Returns:
            np.ndarray: Child generation after mutation.
        """
        for chromosome in range(self.population_size):
            random_index = random.randint(0, self.chromosome_size)
            random_value = np.random.uniform(-1.0, 1.0, 1) #TODO do ustalenia jakie tu wartości
            child_generation[chromosome, random_index] = random_value
        return child_generation

    def mutation_swap(self, child_generation: np.ndarray) -> np.ndarray:
        """ Swap two random genes values in chromosome.

        Args:
            child_generation (np.ndarray): Child generation after crossover.

        Returns:
            np.ndarray: Child generation after mutation.
        """
        for chromosome in range(self.population_size):
            random_indexes = random.sample(range(0, self.chromosome_size), 2)
            tmp = child_generation[chromosome, random_indexes[0]]
            child_generation[chromosome, random_indexes[0]] = child_generation[chromosome, random_indexes[1]]
            child_generation[chromosome, random_indexes[1]] = tmp
        return child_generation

    def mutation_inverse(self, child_generation: np.ndarray) -> np.ndarray:
        """ Inverse one random gene value in chromosome.

        Args:
            child_generation (np.ndarray): Child generation after crossover.

        Returns:
            np.ndarray: Child generation after mutation.
        """
        for chromosome in range(self.population_size):
            random_index = random.randint(0, self.chromosome_size)
            child_generation[chromosome, random_index] = -(child_generation[chromosome, random_index])
        return child_generation

    def mutation_boundary(self, child_generation: np.ndarray) -> np.ndarray:
        """ Boundary mutation, we select a random gene from our chromosome and assign the upper bound or the lower bound to it.

        Args:
            child_generation (np.ndarray): Child generation after crossover.

        Returns:
            np.ndarray: Child generation after mutation.
        """
        lower_bound = -1.0
        upper_bound = 1.0
        for chromosome in range(self.population_size):
            random_index = random.randint(0, self.chromosome_size)
            tmp = child_generation[chromosome, random_index]
            if tmp >= 0:
                child_generation[chromosome, random_index] = upper_bound
            else:
                child_generation[chromosome, random_index] = lower_bound
        return child_generation

    def mutation_percent(self, child_generation: np.ndarray) -> np.ndarray:
    
        for chromosome_inx in range(self.population_size):
            child_generation[chromosome_inx] *= random.uniform(0.99, 1.01)
        return child_generation

    @measure_time
    def run_algorithm(self) -> List[Tuple[float, float, np.ndarray]]:

        self.generate_population()
        best_results = []
        best_chromosome = None

        for iteration in range(self.iterations):
            print(f"Iteration: {iteration+1}")
            selected_parents, best_chromosome = self.select_parents()
            best_results.append(best_chromosome)
            crossover_generation = self.create_child_generation(selected_parents)
            mutated_generation = self.make_mutation(crossover_generation)
            self.population = mutated_generation
            print(best_chromosome[:2])
        return best_results
