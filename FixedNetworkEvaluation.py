import pandas as pd
import os

class FixedNetworkEvaluation:
    def __init__(self, feature_path, performance_path, train_dir, test_dir, networks, verbose=False):
        # Load data
        self.features_df = pd.read_csv(feature_path)
        self.performance_df = pd.read_csv(performance_path)

        # Merge on common 'Filename' column
        self.features_and_performances_df = pd.merge(self.features_df, self.performance_df, on='Filename')
        
        # Initialize verbose flag
        self.verbose = verbose

        # Determine best network based on performance columns
        performance_columns = ['0%', '25%', '50%', '75%']
        self.features_and_performances_df['best_network'] = self.features_and_performances_df[performance_columns].idxmax(axis=1)
        
        # Set filenames for training and testing datasets
        self.train_filenames = os.listdir(train_dir)
        self.test_filenames = os.listdir(test_dir)

        self.networks = networks

    def evaluate(self):
        # Filter data for training and testing sets
        train_df = self.features_and_performances_df[self.features_and_performances_df['Filename'].isin(self.train_filenames)]
        test_df = self.features_and_performances_df[self.features_and_performances_df['Filename'].isin(self.test_filenames)]

        # Find how many times each network was best on the train set
        best_counts_train = train_df['best_network'].value_counts(normalize=True).reindex(self.networks, fill_value=0) * 100
        if self.verbose:
            print("Percentage times each network was best on train set:")
            print(best_counts_train)

        # Identify the best overall network on the train set
        best_network = best_counts_train.idxmax()
        if self.verbose:
            print(f"Best performing network on the train set: {best_network}")

        # Calculate how many times each network was the best on the test set
        best_counts_test = test_df['best_network'].value_counts(normalize=True).reindex(self.networks, fill_value=0) * 100
        if self.verbose:
            print("Percentage times each network was best on test set:")
            print(best_counts_test)

        # Evaluate how often this network is the best on the test set
        best_counts_test = test_df['best_network'].value_counts(normalize=True) * 100
        best_network_accuracy_test = best_counts_test.get(best_network, 0)
        if self.verbose:
            print(f"Percentage times {best_network} was best on test set: {best_network_accuracy_test}%")

        # Calculate average IoU and average weight
        # Prepare to calculate average IoU and weight
        iou_totals = {network: 0 for network in self.networks}
        weight_totals = {network: 0 for network in self.networks}
        count = {network: 0 for network in self.networks}

        network_weights = {'0%': 100, '25%': 75, '50%': 50, '75%': 25}  # Example weights

        # Calculate totals for IoU and weights
        for network in ["0%", "25%", "50%", "75%"]:
            total_iou = 0
            total_weight = 0
            count = 0

            for index, row in test_df.iterrows():
                iou = row[network] 
                total_iou += iou
                total_weight += network_weights[network]
                count += 1        

            # Calculate averages
            average_iou = total_iou / count
            average_weight = total_weight / count

            if self.verbose:
                print(f"Average IoU for network {network}:")
                print(average_iou)
                print(f"Average Weight for network {network}:")
                print(average_weight)

        return average_iou, average_weight

if __name__ == "__main__":

    evaluator = FixedNetworkEvaluation(feature_path='./features.csv', 
    performance_path='./performance_results.csv', 
    train_dir='./data/ordered_train_test//train/images/', 
    test_dir='./data/ordered_train_test/test/images/', 
    networks=['0%', '25%', '50%', '75%'],
    verbose=True)   

    evaluator.evaluate()