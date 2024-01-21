import xgboost as xg
import pandas as pd
import mlflow
import numpy as np
import concurrent.futures

random_state = 42
STOCK_AGG = r"C:\Users\artem\Desktop\practice\stoks_agg"


def train_xgboost_model(process_input):
    params, forecast_range, name, days_number, X, y, X_val, y_val = process_input
    mlflow.xgboost.autolog(disable=True)
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model_type", "XGboost S10")
        mlflow.set_tag("forecast_range", forecast_range)
        mlflow.set_tag("days_number", days_number)
        xgr = xg.XGBRegressor(eval_metric=['rmse', 'mae'],
                              tree_method='gpu_hist', gpu_id=0,
                              random_state=random_state)
        mlflow.log_params(params)
        xgr.set_params(**params)
        xgr.fit(X, y, eval_set=[(X, y), (X_val, y_val)], early_stopping_rounds=10)
        results = xgr.evals_result()
        mlflow.log_metric("best_ntree_limit", xgr.best_ntree_limit)
        for i in range(len(results["validation_0"]["rmse"])):
            mlflow.log_metric("rmse_train", results["validation_0"]["rmse"][i], i)
            mlflow.log_metric("rmse_val", results["validation_1"]["rmse"][i], i)
            mlflow.log_metric("mae_train", results["validation_0"]["mae"][i], i)
            mlflow.log_metric("mae_val", results["validation_1"]["mae"][i], i)
        predictions = xgr.predict(X_val, iteration_range=(0, xgr.best_iteration + 1))
        err = ((y_val - predictions) ** 2).to_numpy()
        err_for_day = np.sqrt(err.mean(axis=0))
        err_total = np.sqrt(err.mean())
        mlflow.log_metric("final_val_rmse", err_total)
        for i in range(len(err_for_day)):
            mlflow.log_metric("rmse_in_forecast_range", err_for_day[i], i + 1)


def run_xgboost_training_with_validation(days_number, max_workers):
    params_versions = []
    np.random.seed(random_state)
    for _ in range(3):
        params = {"n_estimators": np.random.randint(300, 1501),
                  "max_depth": np.random.randint(3, 21),
                  "learning_rate": np.random.choice([0.1, 0.01, 0.005]),
                  "subsample": np.random.choice([1, 0.75, 0.5, 0.25]),
                  "colsample_bytree": np.random.choice([1, 0.75, 0.5, 0.25])}
        params_versions.append(params)

    X = pd.read_csv(f"{STOCK_AGG}\\X_train_S10.csv")
    inputs = []
    for i in range(1, 6 // days_number + 1):
        y = pd.read_csv(f"{STOCK_AGG}\\y_train_{days_number}_{i}_S10.csv", index_col="Date", parse_dates=['Date'])
        X_val = pd.read_csv(f"{STOCK_AGG}\\X_val_{days_number}_{i}_S10.csv")
        y_val = pd.read_csv(f"{STOCK_AGG}\\y_val_{days_number}_{i}_S10.csv", index_col="Date", parse_dates=['Date'])

        for j in range(len(params_versions)):
            inputs.append(
                (params_versions[j], i, f"xgboost S10 r{days_number} {i} {j}", days_number, X, y, X_val, y_val))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(train_xgboost_model, inputs)
