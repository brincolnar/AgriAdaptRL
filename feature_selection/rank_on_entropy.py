import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

df = pd.read_csv('./features.csv')

'''
Split feature into k quantiles 
'''
def split_feature(feature, k):
    brightness = df[[feature]].to_numpy()

    step = 1.0 / k
    quantile_values = []

    for i in range(k):
        quantile = (i + 1) * step
        quantile_value = np.quantile(brightness, quantile)
        quantile_values.append(quantile_value)
        print(f"quantile {i}: {quantile_value}")

    return quantile_values

'''
Assign integer interval based on feature value
'''
def assign_interval(feature, k):
    quantile_values = split_feature(feature, k)

    min_value = df[feature].min()
    max_value = df[feature].max()

    # Ensure max_value is slightly larger than the last quantile value if they are equal
    if quantile_values[-1] == max_value:
        max_value += 0.001 
    
    bins = [min_value] + quantile_values + [max_value]
    
    bins = np.unique(bins) 

    k_actual = len(bins) - 1 
    labels = [i+1 for i in range(k_actual)]
    df[f"{feature} Interval"] = pd.cut(df[feature], bins=bins, labels=labels, include_lowest=True)

'''
Computes average entropy for each interval
'''
def average_entropy(interval_feature, k, entropy_column):
    average_entropies = df.groupby(interval_feature)[entropy_column].mean()
    
    return average_entropies


'''
Plots mean entropies on intervalized feature range
'''
def plot_entropies(feature, k, entropy_column):
    assign_interval(feature, k)
    average_entropies = average_entropy(f'{feature} Interval', k, entropy_column)
    average_entropies = average_entropies.dropna() 
    print(average_entropies)
    plt.figure(figsize=(10, 6))
    average_entropies.plot(kind='bar', color='skyblue')
    plt.xlabel(f'{feature} Intervals')
    plt.ylabel('Average Entropy')
    plt.title(f'Average Entropy across {feature} Intervals')
    plt.xticks(rotation=45) 
    plt.ylim(1.0, 1.1)  
    plt.tight_layout() 
    
    # Save the plot as an image file
    plt.savefig(f'./plots/{feature}_average_entropy_plot.png', format='png', dpi=300)
    
    # Optionally, close the figure to free up memory
    plt.close()

'''
Rank features by sum of differences in average entropies between quantiles
'''
def rank_features_by_entropy_differences(features, k, entropy_column):
    feature_differences = {}
    for feature in features:
        assign_interval(feature, k)
        average_entropies = average_entropy(f'{feature} Interval', k, entropy_column)
        differences = np.diff(average_entropies.dropna().values)
        feature_differences[feature] = np.sum(np.abs(differences))
    
    ranked_features = sorted(feature_differences.items(), key=lambda item: item[1], reverse=True)
    
    for rank, (feature, score) in enumerate(ranked_features, 1):
        print(f"{rank}. Feature: {feature}, Score: {score}")
    
    return ranked_features

'''
Ranks features based on entropy differences across intervalized feature range
'''
def rank_features():
    features = [
        'Mean Brightness',
        'Std Brightness',
        'Max Brightness',
        'Min Brightness',
        'Hue Hist Feature 1',
        'Hue Hist Feature 2',
        'Hue Std',
        'Contrast',
        'Mean Saturation',
        'Std Saturation',
        'Max Saturation',
        'Min Saturation',
        'SIFT Features',
        'Texture Contrast',
        'Texture Dissimilarity',
        'Texture Homogeneity',
        'Texture Energy',
        'Texture Correlation',
        'Texture ASM',
        'Excess Green Index',
        'Excess Red Index',
        'CIVE',
        'ExG-ExR Ratio',
        'CIVE Ratio'
    ]

    ranked_features = rank_features_by_entropy_differences(features, 3, 'pruned_scores_entropy')
    ranked_features_df = pd.DataFrame(ranked_features, columns=['Feature', 'Entropy Difference'])

    return ranked_features_df

'''
Intervalizes chosen features only
'''
def intervalize_chosen_features(chosen_features):
    for feature in chosen_features:
        assign_interval(feature, 3)

    df.to_csv('features.csv', index=False)  

'''
Cleans all columns except for filename and intervalized chosen features
'''
def clean_csv(chosen_features):
    columns_to_keep = ['Filename'] + chosen_features
    df_cleaned = df[columns_to_keep]
    
    df_cleaned.to_csv('cleaned_features.csv', index=False)  


'''
Ranks features based on mutual information of feature and perfomance entropy
'''
def ranks_features_on_mutual_information():
    features = [
        'Mean Brightness',
        'Std Brightness',
        'Max Brightness',
        'Min Brightness',
        'Hue Hist Feature 1',
        'Hue Hist Feature 2',
        'Hue Std',
        'Contrast',
        'Mean Saturation',
        'Std Saturation',
        'Max Saturation',
        'Min Saturation',
        'SIFT Features',
        'Texture Contrast',
        'Texture Dissimilarity',
        'Texture Homogeneity',
        'Texture Energy',
        'Texture Correlation',
        'Texture ASM',
        'Excess Green Index',
        'Excess Red Index',
        'CIVE',
        'ExG-ExR Ratio',
        'CIVE Ratio'
    ]

    target = df['pruned_scores_entropy']

    mi_scores = mutual_info_regression(df[features], target)

    mi_scores_df = pd.DataFrame({"Feature": features, "MI Score": mi_scores}).sort_values(by="MI Score", ascending=False)

    return mi_scores_df


entropy_diffs_ranking = rank_features()
mi_score_rankings = ranks_features_on_mutual_information()

''' REMOVE
Plots 2 rankings next to each other
'''
def plot_rankings(ranked_features_by_entropy, ranked_features_by_mi):    
    # Merging the rankings on the 'Feature' column to compare them side by side
    merged_rankings = ranked_features_by_entropy.merge(ranked_features_by_mi, on='Feature', suffixes=('_entropy', '_mi'))
    
    # Setting up the plotting area
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Number of features to plot
    n_features = len(merged_rankings)
    index = range(n_features)
    
    # Plotting entropy difference rankings
    ax.barh(index, merged_rankings['Entropy Difference'], height=0.4, label='Entropy Difference', color='skyblue', align='edge')
    
    # Plotting mutual information score rankings
    ax.barh([p + 0.4 for p in index], merged_rankings['MI Score'], height=0.4, label='Mutual Information Score', color='lightgreen', align='edge')
    
    # Setting feature names as y-ticks
    plt.yticks([p + 0.2 for p in index], merged_rankings['Feature'])
    
    # Adding legend, title, and labels
    plt.legend()
    plt.title('Feature Rankings Comparison')
    plt.xlabel('Score')
    plt.ylabel('Feature')
    
    # Saving the plot as a PNG file
    plt.tight_layout()
    plt.savefig('feature_rankings_comparison.png')  # Specify the desired filename here

