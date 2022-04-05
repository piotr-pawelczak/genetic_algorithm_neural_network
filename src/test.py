from neural_network import NeuralNetwork
from genetic_algorithm import GeneticAlgorithm
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import random

random.seed(1)
np.random.seed(1)


# Load data
df = pd.read_csv('../heart.csv')
df.rename(columns={'output': 'hearth_attack_chance'}, inplace=True)
X = df.drop('hearth_attack_chance', axis=1)
y = df['hearth_attack_chance']

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)

model = NeuralNetwork(num_of_features=13)
ga = GeneticAlgorithm(model, X, y)

ga.population_size = 50
ga.num_parents = 25
ga.iterations = 3
ga.metric_type = "accuracy"
ga.select_parents_type = "elite"
ga.crossover_type = "single_point"
ga.mutation_type = "swap"

ga.run_algorithm()
ga.plot_results()
