from neural_network import NeuralNetwork
from genetic_algorithm import GeneticAlgorithm
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Disable CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

model = NeuralNetwork()

# testing crossover with debugger
# ga = GeneticAlgorithm(model, x_train, y_train, 100, 50, "accuracy", crossover_type="single_point")  # single point
# ga = GeneticAlgorithm(model, x_train, y_train, 100, 50, "accuracy", crossover_type="two_points")    # two points
# ga = GeneticAlgorithm(model, x_train, y_train, 100, 50, "accuracy", crossover_type="uniform")       # uniform
# ga = GeneticAlgorithm(model, x_train, y_train, 100, 50, "accuracy", crossover_type="ble")   # wrong crossover type
# ga.generate_population()
# # parents = ga.select_roulette()
# parents = np.random.rand(ga.population_size, 151)
# print(f"start population: {parents}")
# child_generation = ga.make_crossover(parents)
# print(f"child population: {child_generation}")


# chromosome = ga.population[0]
# loss = ga.get_fitness(model, chromosome, 'loss')
# accuracy = ga.get_fitness(model, chromosome, 'accuracy')
# print(f'{accuracy} - {loss}')

