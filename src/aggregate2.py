import pandas as pd
import numpy as np
from prediction_models.config import STOCKS_PREPROCESSED, STOCK_AGG, val_start_date, test_start_date
import os


def multiple_observes(num_of_observes):
    countries = ["China", "Russia", "USA", "UK"]
    x_trains = []

    y_trains_6 = {"1": []}
    x_vals_6 = {"1": []}
    y_vals_6 = {"1": []}
    x_test_6 = {"1": []}
    y_test_6 = {"1": []}

    for country in countries:
        print(country)
        data_files = os.listdir(f"{STOCKS_PREPROCESSED}/{country}")
        for data_file in data_files:
            data = pd.read_csv(f"{STOCKS_PREPROCESSED}/{country}/{data_file}", index_col="Date", parse_dates=['Date'])
            data2 = data[test_start_date:]
            data = data[:test_start_date]
            target = pd.DataFrame()
            for i in range(1, 7):
                target[f"t-{i}"] = data["Close"].shift(-i)
            target = target[num_of_observes:]
            x = []
            for i in range(data.shape[0] - num_of_observes):
                x.append(data[i:i + num_of_observes].to_numpy().flatten())
            if len(x) == 0:
                continue
            x = np.array(x)

            target2 = pd.DataFrame()
            for i in range(1, 7):
                target2[f"t-{i}"] = data2["Close"].shift(-i)
            target2 = target2[num_of_observes:]
            x2 = []
            for i in range(data2.shape[0] - num_of_observes):
                x2.append(data2[i:i + num_of_observes].to_numpy().flatten())
            if len(x2) == 0:
                continue
            x2 = np.array(x2)

            val_indx = data[:val_start_date].shape[0]
            test_indx = data2[:test_start_date].shape[0]
            train_data = x[:val_indx]
            x_trains.append(train_data)
            val_data = x[val_indx:]
            train_target = target[:val_indx]
            val_target = target[val_indx:]
            test_data = x2[test_indx:]
            test_target = target2[test_indx:]
            for i in range(6):

                if i < 1:
                    y_trains_6[f"{i + 1}"].append(train_target[[f"t-{i * 3 + 1}", f"t-{i * 3 + 2}", f"t-{i * 3 + 3}",
                                                                f"t-{i * 3 + 4}", f"t-{i * 3 + 5}", f"t-{i * 3 + 6}"]])
                    y_vals_6[f"{i + 1}"].append(
                        val_target[[f"t-{i * 3 + 1}", f"t-{i * 3 + 2}", f"t-{i * 3 + 3}",
                                    f"t-{i * 3 + 4}", f"t-{i * 3 + 5}", f"t-{i * 3 + 6}"]][:-6 * (i + 1)])
                    x_vals_6[f"{i + 1}"].append(val_data[:-6 * (i + 1)])
                    y_test_6[f"{i + 1}"].append(
                        test_target[[f"t-{i * 3 + 1}", f"t-{i * 3 + 2}", f"t-{i * 3 + 3}",
                                     f"t-{i * 3 + 4}", f"t-{i * 3 + 5}", f"t-{i * 3 + 6}"]][:-6 * (i + 1)])
                    x_test_6[f"{i + 1}"].append(test_data[:-6 * (i + 1)])

    x_train = pd.DataFrame(np.concatenate(x_trains))
    x_train.to_csv(f"{STOCK_AGG}/X_train_S{num_of_observes}.csv")
    for i in range(6):

        if i < 1:
            x_val = pd.DataFrame(np.concatenate(x_vals_6[f"{i + 1}"]))
            x_val.to_csv(f"{STOCK_AGG}/X_val_6_{i + 1}_S{num_of_observes}.csv")
            y_val = pd.concat(y_vals_6[f"{i + 1}"])
            y_val.to_csv(f"{STOCK_AGG}/y_val_6_{i + 1}_S{num_of_observes}.csv")
            x_test = pd.DataFrame(np.concatenate(x_test_6[f"{i + 1}"]))
            x_test.to_csv(f"{STOCK_AGG}/X_test_6_{i + 1}_S{num_of_observes}.csv")
            y_test = pd.concat(y_test_6[f"{i + 1}"])
            y_test.to_csv(f"{STOCK_AGG}/y_test_6_{i + 1}_S{num_of_observes}.csv")
            y_train = pd.concat(y_trains_6[f"{i + 1}"])
            y_train.to_csv(f"{STOCK_AGG}/y_train_6_{i + 1}_S{num_of_observes}.csv")


if __name__ == "__main__":
    multiple_observes(10)
