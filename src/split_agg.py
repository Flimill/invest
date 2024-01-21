import pandas as pd
import numpy as np
from prediction_models.config import STOCKS_PREPROCESSED, STOCK_SPLIT_AGG, val_start_date, test_start_date
import os

num_of_observes = 10


def create_and_save_split(country: str, data_files: list, split_name: str):
    x_trains = []
    y_trains = {"1": [], "2": [], "3": []}
    x_vals = {"1": [], "2": [], "3": []}
    y_vals = {"1": [], "2": [], "3": []}

    for data_file in data_files:
        data = pd.read_csv(f"{STOCKS_PREPROCESSED}/{country}/{data_file}", index_col="Date", parse_dates=['Date'])
        data = data[:test_start_date]
        target = pd.DataFrame()
        for i in range(1, 10):
            target[f"t-{i}"] = data["Close"].shift(-i)
        target = target[num_of_observes:]
        x = []
        for i in range(data.shape[0] - num_of_observes):
            x.append(data[i:i + num_of_observes].to_numpy().flatten())
        if len(x) == 0:
            continue
        x = np.array(x)
        val_indx = data[:val_start_date].shape[0]
        train_data = x[:val_indx]
        x_trains.append(train_data)
        val_data = x[val_indx:]
        train_target = target[:val_indx]
        val_target = target[val_indx:]
        for i in range(3):
            y_trains[f"{i + 1}"].append(train_target[[f"t-{i * 3 + 1}", f"t-{i * 3 + 2}", f"t-{i * 3 + 3}"]])
            y_vals[f"{i + 1}"].append(
                val_target[[f"t-{i * 3 + 1}", f"t-{i * 3 + 2}", f"t-{i * 3 + 3}"]][:-3 * (i + 1)])
            x_vals[f"{i + 1}"].append(val_data[:-3 * (i + 1)])
    x_train = pd.DataFrame(np.concatenate(x_trains))
    x_train.to_csv(f"{split_name}\\X_train_S{num_of_observes}.csv")
    for i in range(3):
        x_val = pd.DataFrame(np.concatenate(x_vals[f"{i + 1}"]))
        x_val.to_csv(f"{split_name}\\X_val_{i + 1}_S{num_of_observes}.csv")
        y_val = pd.concat(y_vals[f"{i + 1}"])
        y_val.to_csv(f"{split_name}\\y_val_{i + 1}_S{num_of_observes}.csv")
        y_train = pd.concat(y_trains[f"{i + 1}"])
        y_train.to_csv(f"{split_name}\\y_train_{i + 1}_S{num_of_observes}.csv")


def split_agg():
    # usa_stocks_files = os.listdir(f"{STOCKS_PREPROCESSED}/USA")
    # for i in range(4):
    #     usa_data_files = usa_stocks_files[i*100:(i + 1)*100]
    #     split_name = f"{STOCK_SPLIT_AGG}\\USA_{i + 1}"
    #     create_and_save_split("USA", usa_data_files, split_name)
    #
    # create_and_save_split("USA", usa_stocks_files[400:], f"{STOCK_SPLIT_AGG}\\USA_5")

    uk_stocks_files = os.listdir(f"{STOCKS_PREPROCESSED}/UK")
    create_and_save_split("UK", uk_stocks_files, f"{STOCK_SPLIT_AGG}\\UK")

    china_stocks_files = os.listdir(f"{STOCKS_PREPROCESSED}/China")
    create_and_save_split("China", china_stocks_files, f"{STOCK_SPLIT_AGG}\\China")

    rus_stocks_files = os.listdir(f"{STOCKS_PREPROCESSED}/Russia")
    create_and_save_split("Russia", rus_stocks_files, f"{STOCK_SPLIT_AGG}\\Russia")


if __name__ == "__main__":
    split_agg()


