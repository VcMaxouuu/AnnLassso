import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import models.regression as regmodels
from joblib import Parallel, delayed

def evaluate_model(model, dataset, run_id, outdir):
    X_train, y_train, X_test, y_test = dataset

    model.fit(X_train, y_train)
    selected_features = model.imp_feat[1]
    mu_hat = model.predict(X_test).squeeze()
    mse = (np.square(mu_hat - y_test.squeeze())).mean(axis=0)

    filename = os.path.join(outdir, f'{model.__class__.__name__ + str(model.hidden_dims)}')
    os.makedirs(filename, exist_ok=True)

    model_save_path = os.path.join(filename, f"run{run_id}.pth")
    model.save(model_save_path)

    model_results = (selected_features, mse)
    return model.__class__.__name__+ str(model.hidden_dims), model_results

def simulate(X_train, X_test, errors, n, run_id, outdir):
    # Generate dataset
    features = pd.read_csv(f"parallelization/layers_analysis/data/features.csv")[f"{run_id}"]
    X_train_n = X_train.iloc[:n, :]

    y_train = np.abs(10 * np.abs(X_train_n.iloc[:, features[0]] - X_train_n.iloc[:, features[1]]) - 10 * np.abs(X_train_n.iloc[:, features[2]] - X_train_n.iloc[:, features[3]]))
    y_train += errors.iloc[:n, run_id]

    y_test = np.abs(10 * np.abs(X_test.iloc[:, features[0]] - X_test.iloc[:, features[1]]) - 10 * np.abs(X_test.iloc[:, features[2]] - X_test.iloc[:, features[3]]))

    dataset = (X_train_n, y_train, X_test, y_test)

    # Evaluate each model and collect the results
    models = [regmodels.AnnLassoRegressorl0(),
              regmodels.AnnLassoRegressorl0(hidden_dims=(20, 10)),
              regmodels.AnnLassoRegressorl0(hidden_dims=(20, 10, 5))
              ]

    result = {'run_id': run_id}
    for model in models:
        model_name, model_results = evaluate_model(model, dataset, run_id, outdir)
        result[model_name] = model_results

    return result

if __name__ == "__main__":
    m = 1
    X_train = pd.read_csv("parallelization/layers_analysis/data/X_train.csv")
    X_test = pd.read_csv("parallelization/layers_analysis/data/X_test.csv")
    errors = pd.read_csv("parallelization/layers_analysis/data/errors.csv")

    n_values = np.linspace(100, 10**4, num=100, dtype=int)
    for n in n_values:
        outdir = os.path.join(os.path.dirname(__file__), f'results/n{n}/')
        os.makedirs(outdir, exist_ok=True)


        print(f"Processing n = {n}...")

        # Run m simulations in parallel for the current n
        results = Parallel(n_jobs=-1)(
            delayed(simulate)(X_train, X_test, errors, n, run_id, outdir) for run_id in range(m)
        )

        # Convert list of dicts to DataFrame
        results_df = pd.DataFrame(results)

        outname = f'results.csv'
        fullname = os.path.join(outdir, outname)
        results_df.to_csv(fullname, index=False, mode='w', header=True)

        print(f"Completed s = {n}. Results saved to {fullname}.")

    print("All simulations completed.")
