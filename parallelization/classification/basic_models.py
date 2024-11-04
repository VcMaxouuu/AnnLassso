import pandas as pd
import numpy as np
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import models.classification as classmodels
import utils
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


def evaluate_model(model_class, dataset):
    model = model_class()

    X_train, y_train, X_test, y_test = dataset
    model.fit(X_train, y_train)

    selected_features = model.imp_feat[1]
    error_test = accuracy_score(y_test, model.predict(X_test))

    model_results = (selected_features, error_test)
    return model.__class__.__name__, model_results


def simulate(data, indices, models, run_id):
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
    results_df = pd.DataFrame(index=[run_id], columns=['run_id'] + [m.__name__ for m in models])
    results_df['run_id'] = run_id

    # Populate DataFrame with baseline error
    most_cummon_class = np.argmax(np.bincount(y_train))
    results_df.at[run_id, 'baseline error'] = accuracy_score(y_test, np.full(y_test.shape, most_cummon_class))

    # Evaluate each model and store the results
    for model_class in models:
        model_name, model_results = evaluate_model(model_class, dataset)
        results_df.at[run_id, model_name] = model_results

    # Return results instead of writing to a file directly
    return results_df


if __name__ == '__main__':

    models = [val for _, val in classmodels.__dict__.items()
              if callable(val) and 'Ann' not in val.__name__]

    dataset_folders = ['image_datasets'] #['biological_datasets', 'other_datasets', 'classical_datasets']
    for dataset_folder in dataset_folders:
        dataset_path = os.path.join("parallelization/classification", dataset_folder)
        for root, dirs, files in os.walk(dataset_path):
            print(root)
            if 'USPS' in root:
                if 'data.csv' in files and 'training_indices.csv' in files:

                    data, indices = load_data(root)
                    outdir = os.path.join(root, 'results/basic_models/results.csv')
                    os.makedirs(os.path.dirname(outdir), exist_ok=True)

                    print(f"Simulations started for {root}.")

                    # Run simulations in parallel, but collect all results first
                    results_list = Parallel(n_jobs=-1)(
                        delayed(simulate)(data, indices, models, run_id) for run_id in range(50)
                    )

                    # Concatenate all results
                    all_results_df = pd.concat(results_list)

                    # Write results to CSV file after parallel computation
                    all_results_df.to_csv(outdir, mode='w', header=True, index=False)

                    print(f"All simulations completed for {root}.")
