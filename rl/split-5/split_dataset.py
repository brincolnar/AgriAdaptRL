import pandas as pd

# Load both CSV files
df_features = pd.read_csv('features.csv')
df_performance = pd.read_csv('performance_results.csv')

# Merge the two DataFrames on a common column (ensure the 'Filename' column exists in both)
df = pd.merge(df_features, df_performance, on='Filename')

# Ensure performance columns are correctly named in the CSV file
performance_columns = ['0%', '25%', '50%', '75%']

df = df[~(df[performance_columns] == 0).all(axis=1)]

# Add a 'best_network' column that captures the column name with the highest value
df['best_network'] = df[performance_columns].idxmax(axis=1)

# Check if 'best_network' is properly calculated
print(df[['Filename', 'best_network']].head())

# Proceed with the split logic
split_size = len(df) // 5  
remainder = len(df) % 5

start_idx = 0
for i in range(5):
    if i < remainder:
        end_idx = start_idx + split_size + 1
    else:
        end_idx = start_idx + split_size

    subset_df = df.iloc[start_idx:end_idx]

    # Save each subset with the best_network column included
    subset_df.to_csv(f'./dataset_{i+1}.csv', index=False)
    
    start_idx = end_idx
