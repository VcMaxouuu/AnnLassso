"""
Parallization code for following parameters:
    - Data : Mail spam (https://archive.ics.uci.edu/dataset/94/spambase)
    - Data size :  4601x57, classes = 2
    - No nan values
    - Number of simulations : 20
"""
import pandas as pd
import numpy as np
import os
import sys
from sklearn.calibration import LabelEncoder
from ucimlrepo import fetch_ucirepo
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
import models.classification as classmodels
import utils
from lassonet import LassoNetClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")


def evaluate_model(model_class, dataset, lambda_qut, cv_lambda):
    model = model_class()
    if hasattr(model, 'lambda_qut'):
        model.lambda_qut = lambda_qut
    if hasattr(model, 'cv_lambda'):
        model.cv_lambda = cv_lambda

    X, y, X_test, y_test = dataset

    t = time.time()
    model.fit(X, y)
    training_time = time.time() - t

    selected_features = model.imp_feat[1]
    error_train = accuracy_score(y, model.predict(X))
    error_test = accuracy_score(y_test, model.predict(X_test))

    model_results = (selected_features, error_train, error_test, training_time)
    return model.__class__.__name__, model_results

def simulate(X, y, models, run_id, outdir):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Generate dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    dataset = (X_train, y_train, X_test, y_test)
    most_cummon_class = np.argmax(np.bincount(y_train))

    X_scaled = StandardScaler().fit_transform(X_train)
    lambda_qut = utils.lambda_qut_classification(utils.X_to_tensor(X_scaled), utils.get_hat_p(y_train), utils.Custom_act_fun())
    LassoModel = LassoNetClassifierCV(verbose=0, cv=10)
    LassoModel.path(X_scaled, y_train)
    cv_lambda = LassoModel.best_lambda_

    # Initialize results DataFrame for this run
    columns = [m.__name__ for m in models] + ['baseline error']
    results_df = pd.DataFrame(index=[run_id], columns=columns)

    # Populate DataFrame with baseline error
    results_df.at[run_id, 'baseline error'] = accuracy_score(y_train, np.full(y_train.shape, most_cummon_class))

    # Evaluate each model and store the results
    for model_class in models:
        model_name, model_results = evaluate_model(model_class, dataset, lambda_qut, cv_lambda)
        results_df.at[run_id, model_name] = model_results

    # Save results to CSV file
    if not os.path.isfile(outdir):
        results_df.to_csv(outdir, mode='w', header=True, index=False)
    else:
        results_df.to_csv(outdir, mode='a', header=False, index=False)

if __name__ == '__main__':
    ### Parameters
    m = 20
    data = fetch_ucirepo(id=94)
    X = data.data.features
    y = data.data.targets

    data = pd.concat([X, y], axis=1)
    data = data.dropna()

    X = data.drop(columns=["Class"])
    y = data["Class"]
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])
    X = X.to_numpy()
    X_scaled = StandardScaler().fit_transform(X)
    y = LabelEncoder().fit_transform(y)

    print("Parallelization started ...")

    models = [val for _, val in classmodels.__dict__.items() if callable(val)]

    outdir = os.path.join(os.path.dirname(__file__), 'results.csv')

    # Run simulations in parallel
    Parallel(n_jobs=-1)(
        delayed(simulate)(X, y, models, run_id, outdir) for run_id in range(m)
    )

    print("All simulations completed.")
