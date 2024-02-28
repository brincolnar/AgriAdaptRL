import pandas as pd
import matplotlib.pyplot as plt

csv_file = './features.csv'

df = pd.read_csv(csv_file)

wins = {
    'pruned_0': 0,
    'pruned_025': 0,
    'pruned_05': 0,
    'pruned_075': 0
}

for index, row in df.iterrows():
    scores = {
        'pruned_0': row['pruned_0_performance'],
        'pruned_025': row['pruned_025_preformance'],
        'pruned_05': row['pruned_05_preformance'],
        'pruned_075': row['pruned_075_preformance']
    }
    
    best_network = max(scores, key=scores.get)
    wins[best_network] += 1

labels = wins.keys()
sizes = wins.values()
colors = ['gold', 'lightcoral', 'lightskyblue', 'lightgreen']
explode = (0.1, 0, 0, 0)  # Explode the first slice for emphasis

# Plot pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Best Performing Network (No. of Wins Across All Images)')

plt.savefig('pruned_network_performance_pie_chart.png', bbox_inches='tight')

