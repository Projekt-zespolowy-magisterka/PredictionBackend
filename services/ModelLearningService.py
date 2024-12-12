from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from utilities.statisticsCalculator import calculate_r_squared
from repositories.FileModelRepository import FileModelRepository
from pymongo.errors import PyMongoError
from repositories.MongoDBModelRepository import MongoDBModelRepository
from repositories.RedisModelCacheRepository import RedisModelCacheRepository
from utilities.dataAnalyser import DataAnalyzer
from utilities.dataScaler import DataScaler
from services.DataService import DataService
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Bidirectional, Dense, Dropout
from tensorflow.python.client import device_lib
from sklearn.metrics import mean_squared_error, mean_absolute_error

AVAILABLE_MODELS_NAMES = [
    "DecisionTreeRegressor",
    "RandomForestRegressor"
    # ,
    # "LSTM",
    # "GRU",
    # "Bi-Direct"
]

def calculate_r_squared(y_test, y_pred):
    mean_y = np.mean(y_test)
    SST = np.sum((y_test - mean_y) ** 2)
    SSR = np.sum((y_test - y_pred) ** 2)
    rsquared = 1 - (SSR / SST)
    return rsquared


# TODO IMPLEMENT MODEL EFFICIENCY TESTS
class ModelLearningService:

    metrics_array = [
        mean_squared_error,
        mean_absolute_error,
        calculate_r_squared
    ]

    metrics_names = [
        "mean_squared_error",
        "mean_absolute_error",
        "r_squared"
    ]

    def __init__(self):
        self.model_repository = FileModelRepository()
        self.mongo_repo = MongoDBModelRepository()
        self.redis_cache = RedisModelCacheRepository()
        self.data_service = DataService()
        self.data_analyzer = DataAnalyzer()
        self.data_scaler = DataScaler()
        self.models = [
            DecisionTreeRegressor(),
            RandomForestRegressor()
            # ,
            # Sequential(),
            # Sequential(),
            # Sequential()
        ]
        self.metrics_array = [
            mean_squared_error,
            mean_absolute_error,
            calculate_r_squared
        ]
        self.models_names = AVAILABLE_MODELS_NAMES


    def learn_models(self, stock_symbol, interval, period):
        # TODO adding stats
        data_for_train = self.data_service.get_parquet_data(stock_symbol, interval, period)
        X, y = self.data_service.get_objectives_from_data(data_for_train)
        number_of_features = X.shape[1]

        X = X.values
        y = y.values

        # X_scaled, Y_scaled = self.data_scaler.scale_data(X, y)
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)

        learned_models = {}
        print("[learn_models] Learning data set preparing to learn models")
        current_value_index = 0
        models_size = len(self.models)
        metrics_size = len(self.metrics_array)
        cv_scores = np.zeros((models_size, metrics_size, n_splits))
        for model_index, model in enumerate(self.models):
            current_value_index = 0
            current_model_name_key = self.models_names[model_index]
            print(f"[learn_models] Starting learning process of model: {current_model_name_key}\n")
            start_time = time.time()
            if isinstance(model, Sequential):
                if current_model_name_key == 'LSTM':
                    seq_model = self.create_lstm_model(model, number_of_features)
                if current_model_name_key == 'GRU':
                    seq_model = self.create_gru_model(model, number_of_features)
                if current_model_name_key == 'Bi-Direct':
                    seq_model = self.create_bi_lstm_model(model, number_of_features)

                for train_index, test_index in tscv.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
#TODO DOPISAĆ tutaj zrobić wrzucenie danych do excela stad danych testowych i do predykcji
#TODO rzucić to razem z predykcja i dodać miare jakości predykcjy do excela
#TODO wrzucic też do excela miary jakości predykcji
#TODO zrobić wykres z danych jakie były po predykcji do tych danych testowych z tego samego okresu
                    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

                    y_train_reshaped = y_train.reshape((y_train.shape[0], 1, y_train.shape[1]))
                    y_test_reshaped = y_test.reshape((y_test.shape[0], 1, y_test.shape[1]))

                    seq_model.fit(X_train_reshaped, y_train_reshaped, epochs=50, batch_size=32, verbose=0)
                    learned_models[current_model_name_key] = seq_model

                end_time = time.time()
                elapsed_time = end_time - start_time
            else:
                for train_index, test_index in tscv.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    learned_models[current_model_name_key] = model.fit(X_train, y_train)

                    temp_model_max, temp_model_min, cv_scores = self.create_stats_of_model(X_test, model, model_index, y_test, current_value_index, cv_scores)
                current_value_index += 1
                end_time = time.time()
                elapsed_time = end_time - start_time
            print(f"Minimum predicted value {temp_model_min}")
            print(f"Maximum predicted value {temp_model_max}")
            print(f"[learn_models] Finished learning process of model: {current_model_name_key} in: {elapsed_time}\n")
        self.display_results(cv_scores)
        # self.save_models(learned_models)

    def create_stats_of_model(self, X_test, model, model_index, y_test, current_value_index, cv_scores):
        y_pred = model.predict(X_test)
        temp_model_min = min(y_pred)
        temp_model_max = max(y_pred)
        for metric_index, metric_function in enumerate(self.metrics_array):
            metric_value = metric_function(y_test, y_pred)
            cv_scores[model_index, metric_index, current_value_index] = metric_value
        return temp_model_max, temp_model_min, cv_scores

    def display_results(self, cv_scores):
        for model_index, model in enumerate(self.models):
            for metric_index, metric_function in enumerate(self.metrics_array):
                current_metric = cv_scores[model_index, metric_index]
                print(f"{self.models_names[model_index]}, metric name: {self.metrics_names[metric_index]}, mean: "
                      f"{np.mean(current_metric):.10f}, std: {np.std(current_metric):.10f}")
            print()

    def save_model(self, model, model_key):
        self.mongo_repo.save_model(model, model_key)
        self.redis_cache.cache_model(model, model_key)

    def save_models(self, learned_models):
        for model_key, model in learned_models.items():
            self.mongo_repo.save_model(model, model_key)
            self.redis_cache.cache_model(model, model_key)

    def create_lstm_model(self, model, number_of_features):
        print(f"Creating lstm")
        print(f"Creating first layer of lstm")
        model.add(LSTM(units=50, return_sequences=True, input_shape=(1, number_of_features), name="lstm_layer_1"))
        model.add(Dropout(0.2, name="dropout_1"))

        print(f"Creating second layer of lstm")
        model.add(LSTM(units=50, return_sequences=True, name="lstm_layer_2"))
        model.add(Dropout(0.2, name="dropout_2"))

        print(f"Creating third layer of lstm")
        model.add(LSTM(units=50, return_sequences=True, name="lstm_layer_3"))
        model.add(Dropout(0.2, name="dropout_3"))

        print(f"Creating fourth layer of lstm")
        model.add(LSTM(units=50, name="lstm_layer_4"))
        model.add(Dropout(0.2, name="dropout_4"))

        print(f"Creating output layer of lstm")
        model.add(Dense(units=5, name="output_layer"))

        # model.add(Dense(5))
        print(f"input finished")
        model.compile(optimizer='adam', loss='mse')
        print(f"compile finished")
        return model

    def create_gru_model(self, model, number_of_features):
        model.add(GRU(units=50, return_sequences=True, input_shape=(1, number_of_features), name="gru_layer_1"))
        model.add(Dropout(0.2, name="dropout_1"))

        model.add(GRU(units=70, return_sequences=True, name="gru_layer_2"))
        model.add(Dropout(0.2, name="dropout_2"))

        model.add(GRU(units=80, return_sequences=True, name="gru_layer_3"))
        model.add(Dropout(0.3, name="dropout_3"))

        model.add(GRU(units=50, name="gru_layer_4"))
        model.add(Dropout(0.2, name="dropout_4"))

        model.add(Dense(units=5, name="output_layer"))
        model.compile(optimizer='adam', loss='mse')
        return model

    def create_bi_lstm_model(self, model, number_of_features):
        model.add(Bidirectional(LSTM(units=20, return_sequences=True), input_shape=(1, number_of_features), name="bi_lstm_layer_1"))
        model.add(Dropout(0.3, name="dropout_1"))

        model.add(Bidirectional(LSTM(units=40, return_sequences=False), name="bi_lstm_layer_2"))
        model.add(Dropout(0.4, name="dropout_2"))
        #
        # model.add(Bidirectional(LSTM(units=50, return_sequences=True), name="bi_lstm_layer_3"))
        # model.add(Dropout(0.2, name="dropout_3"))
        #
        # model.add(Bidirectional(LSTM(units=50), name="bi_lstm_layer_4"))
        # model.add(Dropout(0.2, name="dropout_4"))

        model.add(Dense(units=5, name="output_layer"))
        model.compile(optimizer='adam', loss='mse')
        return model
