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


def evaluate_model(model_class, dataset, lambda_qut, outdir, run_id):
    model = model_class(lambda_qut=lambda_qut)

    X_train, y_train, X_test, y_test = dataset
    model.fit(X_train, y_train)

    selected_features = model.imp_feat[1]
    error_test = accuracy_score(y_test, model.predict(X_test))

    # Define the model save path using the output directory and run_id
    model_save_path = os.path.join(outdir, f"{model.__class__.__name__}_run{run_id}.pth")
    model.save(model_save_path)

    model_results = (selected_features, error_test)
    return model.__class__.__name__, model_results


def simulate(data, indices, models, run_id, outdir):
    # Generate dataset
    training_indices = indices.iloc[:, run_id]

    training_set = data.iloc[training_indices, :]
    testing_set = data.drop(training_indices)

    X_train = training_set.drop('class', axis=1)
    y_train = training_set['class']
    X_test = testing_set.drop('class', axis=1)
    y_test = testing_set['class']

    dataset = (X_train, y_train, X_test, y_test)

    X_scaled = utils.StandardScaler().fit_transform(utils.X_to_tensor(X_train))
    lambda_qut = utils.lambda_qut_classification(X_scaled, utils.get_hat_p(y_train.values), utils.ShiftedReLu())

    # Initialize results DataFrame for this run
    results_df = pd.DataFrame(index=[run_id], columns=['run_id'] + [m.__name__ for m in models])
    results_df['run_id'] = run_id

    # Populate DataFrame with baseline error
    most_common_class = np.argmax(np.bincount(y_train))
    results_df.at[run_id, 'baseline error'] = accuracy_score(y_test, np.full(y_test.shape, most_common_class))

    # Evaluate each model and store the results
    for model_class in models:
        model_name, model_results = evaluate_model(model_class, dataset, lambda_qut, outdir, run_id)
        results_df.at[run_id, model_name] = model_results

    # Return results instead of writing to a file directly
    return results_df


if __name__ == '__main__':
    models = [val for _, val in classmodels.__dict__.items() if callable(val) and 'Ann' in val.__name__]

    dataset_folders = ['image_datasets'] #['biological_datasets', 'other_datasets', 'classical_datasets']
    for dataset_folder in dataset_folders:
        dataset_path = os.path.join("parallelization/classification", dataset_folder)
        for root, dirs, files in os.walk(dataset_path):
            if 'USPS' in root:
                if 'data.csv' in files and 'training_indices.csv' in files:
                    data, indices = load_data(root)
                    outdir = os.path.join(root, 'results/AnnLasso_models')
                    os.makedirs(outdir, exist_ok=True)

                    print(f"Simulations started for {root}.")

                    # Run simulations in parallel, but collect all results first
                    results_list = Parallel(n_jobs=-1)(
                        delayed(simulate)(data, indices, models, run_id, outdir) for run_id in range(50)
                    )

                    # Concatenate all results
                    all_results_df = pd.concat(results_list)

                    # Define the path for saving results
                    results_file = os.path.join(outdir, 'results.csv')
                    all_results_df.to_csv(results_file, mode='w', header=True, index=False)

                    print(f"All simulations completed for {root}. Results saved to {results_file}.")
