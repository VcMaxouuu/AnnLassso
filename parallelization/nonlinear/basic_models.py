import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import models.regression as regmodels
from joblib import Parallel, delayed

def evaluate_model(model_class, dataset):
    model = model_class()

    X_train, y_train, X_test, y_test = dataset

    model.fit(X_train, y_train)
    selected_features = model.imp_feat[1]
    mu_hat = model.predict(X_test).squeeze()
    mse = (np.square(mu_hat - y_test.squeeze())).mean(axis=0)

    model_results = (selected_features, mse)
    return model.__class__.__name__, model_results

def simulate(X_train, X_test, errors, s, models, run_id):
    # Generate dataset
    features = pd.read_csv(f"parallelization/nonlinear/data/features_{s}.csv")[f"{run_id}"]
    y_train = np.zeros(X_train.shape[0])
    y_test = np.zeros(X_test.shape[0])
    for i in range(0, s, 2):
        y_train += 10 * np.abs(X_train.iloc[:, features[i]] - X_train.iloc[:, features[i + 1]])
        y_test += 10 * np.abs(X_test.iloc[:, features[i]] - X_test.iloc[:, features[i + 1]])
    y_train += errors.iloc[:, run_id]

    dataset = (X_train, y_train, X_test, y_test)

    # Evaluate each model and collect the results
    result = {'run_id': run_id}
    for model_class in models:
        model_name, model_results = evaluate_model(model_class, dataset)
        result[model_name] = model_results

    return result

if __name__ == "__main__":
    m = 100
    s_values = range(0, 21, 2)
    X_train = pd.read_csv("parallelization/nonlinear/data/X_train.csv")
    X_test = pd.read_csv("parallelization/nonlinear/data/X_test.csv")
    errors = pd.read_csv("parallelization/nonlinear/data/errors.csv")

    models = [val for _, val in regmodels.__dict__.items()
              if callable(val) and 'Ann' not in val.__name__]

    # Ensure output directory exists
    outdir = os.path.join(os.path.dirname(__file__), 'results/basic_models/')
    os.makedirs(outdir, exist_ok=True)

    for s in s_values:
        print(f"Processing s = {s}...")

        # Run m simulations in parallel for the current s
        results = Parallel(n_jobs=-1)(
            delayed(simulate)(X_train, X_test, errors, s, models, run_id) for run_id in range(m)
        )

        # Convert list of dicts to DataFrame
        results_df = pd.DataFrame(results)

        # Define the output file name
        outname = f's{s}.csv'
        fullname = os.path.join(outdir, outname)

        # Write results to CSV
        results_df.to_csv(fullname, index=False, mode='w', header=True)

        print(f"Completed s = {s}. Results saved to {fullname}.")

    print("All simulations completed.")
