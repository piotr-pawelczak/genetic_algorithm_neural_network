from neural_network import NeuralNetwork
from genetic_algorithm import GeneticAlgorithm
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

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

ga.population_size = 100
ga.num_parents = 25
ga.iterations = 15
ga.metric_type = "accuracy"
ga.select_parents_type = "elite"
ga.crossover_type = "single_point"
ga.mutation_type = "swap"

best = ga.run_algorithm()

# Plot results
plt.figure(figsize=(16, 8))
fmt = '%2.0f%%'
yticks = mtick.FormatStrFormatter(fmt)
plt.gca().yaxis.set_major_formatter(yticks)
all_k = [i for i in range(1, ga.iterations+1)]

plt.plot(all_k, [x[0]*100 for x in best], '-o', linewidth=3)
plt.plot(all_k, [x[1]*100 for x in best], '-o', linewidth=3)

plt.xticks(all_k)

plt.legend(["Train", "Test"], frameon=False, fontsize=18)
plt.xlabel("Iteration", fontsize=24)
plt.ylabel("Accuracy", fontsize=24)

plt.tick_params(axis='both', labelsize=22)
plt.savefig(r'..\\accuracy.png')
plt.show()
