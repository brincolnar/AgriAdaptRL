import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import _tree
import matplotlib.pyplot as plt

features = pd.read_csv('features.csv')
features['best_network'] = features[['new_unet_performance', 'pruned_06_performance', 'pruned_09_performance' ]].idxmax(axis=1)

features_ = ['Hue Std', 'Contrast', 'Texture Homogeneity']

def print_tree_structure(clf):
    tree = clf.tree_
    feature_names = features_
    class_names = clf.classes_

    def recurse(node, depth):
        indent = "  " * depth
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]
            print(f"{indent}Node: test if {name} <= {threshold}")
            recurse(tree.children_left[node], depth + 1)
            recurse(tree.children_right[node], depth + 1)
        else:
            values = tree.value[node][0]
            if len(values) > 0:
                predicted_class_index = values.argmax()
                predicted_class = class_names[predicted_class_index]
                print(f"{indent}Leaf: predict {values}, class: {predicted_class}")

    recurse(0, 0)

def extract_feature_thresholds(clf, feature, feature_names):
    tree = clf.tree_
    feature_index = feature_names.index(feature)
    thresholds = []

    def recurse(node):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            if tree.feature[node] == feature_index:
                thresholds.append(tree.threshold[node])
            recurse(tree.children_left[node])
            recurse(tree.children_right[node])

    recurse(0)
    return thresholds

def rank_features():
    sorted_features = sorted(feature_accuracies.items(), key=lambda item: item[1], reverse=True)

    print("Ranking of features by accuracy:")
    for rank, (feature, accuracy) in enumerate(sorted_features, start=1):
        print(f"{rank}. {feature}: {accuracy}")

def extract_feature_thresholds_with_depth(clf, feature, feature_names):
    tree = clf.tree_
    feature_index = feature_names.index(feature)
    thresholds_with_depth = []

    def recurse(node, depth):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            if tree.feature[node] == feature_index:
                thresholds_with_depth.append((tree.threshold[node], depth))
            recurse(tree.children_left[node], depth + 1)
            recurse(tree.children_right[node], depth + 1)

    recurse(0, 0)
    return thresholds_with_depth

def plot_feature_range_versus_performance(clf, features, feature_names, chosen_features, performance_metrics):
    color_map = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  
    
    for feature in chosen_features:
        plt.figure(figsize=(10, 6))
        for metric in performance_metrics:
            plt.scatter(features[feature], features[metric], label=metric)

        print('feature')
        print(feature)
        thresholds_with_depth = extract_feature_thresholds_with_depth(clf, feature, feature_names)

        print('thresholds_with_depth')
        print(thresholds_with_depth)

        for threshold, depth in thresholds_with_depth:
            plt.axvline(x=threshold, color=color_map[depth % len(color_map)], linestyle='--', label=f'Threshold (Depth {depth})' if f'Threshold (Depth {depth})' not in plt.gca().get_legend_handles_labels()[1] else "")

        plt.title(f'Comparison of Performance Metrics vs {feature}')
        plt.xlabel(feature)
        plt.ylabel('Performance')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'./plots_w_thresholds_highnets/{feature}_correlation_with_performance.png')
        plt.close()

feature_accuracies = {}



for feature in features_:
    X = features[[feature]] 
    y = features['best_network']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(random_state=42, max_depth=3)
    clf.fit(X_train, y_train)

    print_tree_structure(clf)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    feature_accuracies[feature] = accuracy
    print(f"Accuracy for {feature}: {accuracy}")

    # Plot
    plot_feature_range_versus_performance(clf, features, [feature], [feature], ['new_unet_performance', 'pruned_06_performance', 'pruned_09_performance'])

rank_features()