from neural_network import NeuralNetwork
from genetic_algorithm import GeneticAlgorithm
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('../heart.csv')
df.rename(columns={'output': 'hearth_attack_chance'}, inplace=True)
x = df.drop('hearth_attack_chance', axis=1)
y = df['hearth_attack_chance']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)

# Scale data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = NeuralNetwork(num_of_features=13)
ga = GeneticAlgorithm(model, x_train, y_train)

ga.population_size = 100
ga.num_parents = 50
ga.iterations = 3
ga.metric_type = "accuracy"
ga.select_parents_type = "elite"
ga.crossover_type = "single_point"
ga.mutation_type = "uniform"

ga.run_algorithm()