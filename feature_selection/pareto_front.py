import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the columns related to pruned network performance
pruned_performance_columns = [
    'pruned_0_performance',
    'pruned_01_performance',
    'pruned_02_performance',
    'pruned_025_preformance',
    'pruned_03_performance',
    'pruned_04_performance',
    'pruned_05_preformance',
    'pruned_06_performance',
    'pruned_07_performance',
    'pruned_075_preformance',
    'pruned_08_performance',
    'pruned_09_performance'
]

df = pd.read_csv('./features.csv', usecols=pruned_performance_columns)


energy_usage = {
    'pruned_0_performance': 1.0,
    'pruned_01_performance': 0.91818182,
    'pruned_02_performance': 0.83636364,
    'pruned_025_preformance': 0.75454545,
    'pruned_03_performance': 0.67272727,
    'pruned_04_performance': 0.59090909,
    'pruned_05_preformance': 0.50909091,
    'pruned_06_performance': 0.42727273,
    'pruned_07_performance': 0.34545455,
    'pruned_075_preformance': 0.26363636,
    'pruned_08_performance': 0.18181818,
    'pruned_09_performance': 0.1
}

energy_values = list(energy_usage.values())
network_names = list(energy_usage.keys())

print('energy_usage')
print(energy_usage)

performance_means = df.mean()

print('performance_means')
print(performance_means)

performance_means_ordered = performance_means.loc[list(energy_usage.keys())]

performance_values = performance_means_ordered.values


plt.figure(figsize=(10, 6))
plt.scatter(energy_values, performance_values, color='blue')  

# Annotate each dot with the corresponding network name
for i, txt in enumerate(performance_means_ordered.index):
    plt.annotate(txt, (list(energy_usage.values())[i], performance_values[i]))


plt.title('Pruned Networks: Performance vs Energy Usage')
plt.xlabel('Energy Usage (Normalized)')
plt.ylabel('Mean Performance')
plt.grid(True)

# Save the plot as an image
plt.savefig('./pruned_networks_performance_vs_energy_annotated.png')
