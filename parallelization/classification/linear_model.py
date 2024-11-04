import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")

def load_data(data_path):
    data_file = os.path.join(data_path, 'data.csv')
    indices_file = os.path.join(data_path, 'training_indices.csv')
    data = pd.read_csv(data_file)
    indices = pd.read_csv(indices_file, header=None)
    return data, indices

def evaluate_logistic_regression_cv(dataset):
    # Initialize LogisticRegressionCV model
    model = LogisticRegressionCV(
        Cs=list(np.power(10.0, np.arange(-10, 10))),
        penalty='l1',
        solver='saga',
        scoring='accuracy',
        cv=10,
        max_iter=10000
    )

    X_train, y_train, X_test, y_test = dataset

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit the model and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Get indices of selected features (non-zero coefficients)
    selected_features = np.where(np.any(model.coef_ != 0, axis=0))[0]
    error_test = accuracy_score(y_test, y_pred)

    model_results = (selected_features, error_test)
    return model.__class__.__name__, model_results

def simulate(data, indices, run_id):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Generate dataset
    training_indices = indices.iloc[:, run_id]

    training_set = data.iloc[training_indices, :]
    testing_set = data.drop(training_indices)

    X_train = training_set.drop('class', axis=1)
    y_train = training_set['class']
    X_test = testing_set.drop('class', axis=1)
    y_test = testing_set['class']

    dataset = (X_train, y_train, X_test, y_test)

    # Initialize results DataFrame for this run
    results_df = pd.DataFrame(index=[run_id], columns=['run_id', 'baseline error', 'LogisticRegressionCV'])
    results_df['run_id'] = run_id

    # Populate DataFrame with baseline error
    most_common_class = np.argmax(np.bincount(y_train))
    results_df.at[run_id, 'baseline error'] = accuracy_score(y_test, np.full(y_test.shape, most_common_class))

    # Evaluate the LogisticRegressionCV model and store the results
    model_name, model_results = evaluate_logistic_regression_cv(dataset)
    results_df.at[run_id, model_name] = model_results

    # Return results instead of writing to a file directly
    return results_df

if __name__ == '__main__':

    dataset_folders = ['biological_datasets', 'other_datasets', 'classical_datasets']
    for dataset_folder in dataset_folders:
        dataset_path = os.path.join("parallelization/classification", dataset_folder)
        for root, dirs, files in os.walk(dataset_path):
            if 'data.csv' in files and 'training_indices.csv' in files:

                    data, indices = load_data(root)
                    outdir = os.path.join(root, 'results/linear_model/results.csv')
                    os.makedirs(os.path.dirname(outdir), exist_ok=True)

                    print(f"Simulations started for {root}.")

                    # Run simulations in parallel, but collect all results first
                    results_list = Parallel(n_jobs=-1)(
                        delayed(simulate)(data, indices, run_id) for run_id in range(50)
                    )

                    # Concatenate all results
                    all_results_df = pd.concat(results_list)

                    # Write results to CSV file after parallel computation
                    all_results_df.to_csv(outdir, mode='w', header=True, index=False)

                    print(f"All simulations completed for {root}.")
