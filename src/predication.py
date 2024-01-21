import mlflow.xgboost
import xgboost as xg
import pandas as pd

if __name__ == "__main__":
    # Загрузите данные для предсказания (например, X_test)
    X_test = pd.read_csv(r"C:\Users\artem\Desktop\practice\my_stocks_agg\X_train_S10.csv")

    # Загрузите сохраненную модель
    model = mlflow.xgboost.load_model(r"C:\Users\artem\Desktop\practice\src\prediction_models\six_days\mlruns\0\00ad7bb513b545608da38c043d8d9a81\artifacts\xgboost_model")


    # Выполните предсказание на несколько дней вперед
    predictions = model.predict(X_test)

    # Выведите предсказанные значения
    print(predictions)
