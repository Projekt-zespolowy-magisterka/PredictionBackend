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

AVAILABLE_MODELS_NAMES = [
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "LSTM",
    "GRU",
    "Bi-Direct"
]


# TODO IMPLEMENT MODEL EFFICIENCY TESTS
class ModelLearningService:
    def __init__(self):
        self.model_repository = FileModelRepository()
        self.mongo_repo = MongoDBModelRepository()
        self.redis_cache = RedisModelCacheRepository()
        self.data_service = DataService()
        self.data_analyzer = DataAnalyzer()
        self.data_scaler = DataScaler()
        self.models = [
            DecisionTreeRegressor(),
            RandomForestRegressor(),
            Sequential(),
            Sequential(),
            Sequential()
        ]
        self.metrics_array = [
            mean_squared_error,
            mean_absolute_error,
            calculate_r_squared
        ]
        self.models_names = AVAILABLE_MODELS_NAMES

    def learn_models(self, stock_symbol, interval, period):
        data_for_train = self.data_service.get_parquet_data(stock_symbol, interval, period)
        X, y = self.data_service.get_objectives_from_data(data_for_train)
        X_scaled, Y_scaled = self.data_scaler.scale_data(X, y)
        number_of_features = X.shape[1]
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)

        learned_models = {}
        print("[learn_models] Learning data set preparing to learn models")
        for model_index, model in enumerate(self.models):
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

                for train_index, test_index in tscv.split(X_scaled, y):
                    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

                    y_train_reshaped = y_train.values.reshape((y_train.shape[0], 1, y_train.shape[1]))
                    y_test_reshaped = y_test.values.reshape((y_test.shape[0], 1, y_test.shape[1]))

                    seq_model.fit(X_train_reshaped, y_train_reshaped, epochs=50, batch_size=32, verbose=2)
                    learned_models[current_model_name_key] = seq_model

                end_time = time.time()
                elapsed_time = end_time - start_time
            else:
                for train_index, test_index in tscv.split(X_scaled, y):
                    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    learned_models[current_model_name_key] = model.fit(X_train, y_train)
                end_time = time.time()
                elapsed_time = end_time - start_time
            print(f"[learn_models] Finished learning process of model: {current_model_name_key} in: {elapsed_time}\n")
        self.save_models(learned_models)

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
        model.add(Dense(units=1, name="output_layer"))

        model.add(Dense(5))
        print(f"input finished")
        model.compile(optimizer='adam', loss='mse')
        print(f"compile finished")
        return model

    def create_gru_model(self, model, number_of_features):
        model.add(GRU(units=50, return_sequences=True, input_shape=(1, number_of_features), name="gru_layer_1"))
        model.add(Dropout(0.2, name="dropout_1"))

        model.add(GRU(units=50, return_sequences=True, name="gru_layer_2"))
        model.add(Dropout(0.2, name="dropout_2"))

        model.add(GRU(units=50, return_sequences=True, name="gru_layer_3"))
        model.add(Dropout(0.2, name="dropout_3"))

        model.add(GRU(units=50, name="gru_layer_4"))
        model.add(Dropout(0.2, name="dropout_4"))

        model.add(Dense(units=5, name="output_layer"))
        model.compile(optimizer='adam', loss='mse')
        return model

    def create_bi_lstm_model(self, model, number_of_features):
        model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(1, number_of_features), name="bi_lstm_layer_1"))
        model.add(Dropout(0.2, name="dropout_1"))

        model.add(Bidirectional(LSTM(units=50, return_sequences=True), name="bi_lstm_layer_2"))
        model.add(Dropout(0.2, name="dropout_2"))

        model.add(Bidirectional(LSTM(units=50, return_sequences=True), name="bi_lstm_layer_3"))
        model.add(Dropout(0.2, name="dropout_3"))

        model.add(Bidirectional(LSTM(units=50), name="bi_lstm_layer_4"))
        model.add(Dropout(0.2, name="dropout_4"))

        model.add(Dense(units=5, name="output_layer"))
        model.compile(optimizer='adam', loss='mse')
        return model
