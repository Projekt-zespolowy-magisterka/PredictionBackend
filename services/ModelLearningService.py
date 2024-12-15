from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from repositories.FileModelRepository import FileModelRepository
from pymongo.errors import PyMongoError
from repositories.MongoDBModelRepository import MongoDBModelRepository
from repositories.RedisModelCacheRepository import RedisModelCacheRepository
from utilities.dataAnalyser import DataAnalyzer
from app_config.StatisticsConfig import metrics_array
from utilities.dataScaler import DataScaler
from services.DataService import DataService
from services.StatisticsService import StatisticsService
from app_config.ModelConfig import AVAILABLE_MODELS_NAMES, AVAILABLE_MODELS
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Bidirectional, Dense, Dropout


class ModelLearningService:

    def __init__(self):
        self.model_repository = FileModelRepository()
        self.mongo_repo = MongoDBModelRepository()
        self.redis_cache = RedisModelCacheRepository()
        self.data_service = DataService()
        self.data_analyzer = DataAnalyzer()
        self.data_scaler = DataScaler()
        self.statistics_service = StatisticsService()
        self.models = AVAILABLE_MODELS
        self.metrics_array = metrics_array
        self.models_names = AVAILABLE_MODELS_NAMES

    def learn_models(self, stock_symbol, interval, period):
        data_for_train = self.data_service.get_parquet_data(stock_symbol, interval, period)
        X, y = self.data_service.get_objectives_from_data(data_for_train)
        number_of_features = X.shape[1]
        number_of_results = y.shape[1]
        print(f"Numbers of results: {number_of_results}")
        X = X.values
        y = y.values

        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)

        learned_models = {}
        print("[learn_models] Learning data set preparing to learn models")
        models_size = len(self.models)
        metrics_size = len(self.metrics_array)

        cv_scores = np.zeros((models_size, metrics_size, n_splits, number_of_results))
        print(f"cv_scores shape: {cv_scores.shape}")
        print(f"models_size: {models_size}, metrics_size: {metrics_size}, n_splits: {n_splits}, number_of_results: {number_of_results}")

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

                    model.fit(X_train, y_train)
                    learned_models[current_model_name_key] = model
                    print("SSSS")
                    temp_model_max, temp_model_min, cv_scores, y_pred = self.statistics_service.create_stats_of_model(X_test, learned_models[current_model_name_key], model_index, y_test, current_value_index, cv_scores)
                    print("AAAAA")
                    excel_file, fold_folder, results_df = self.statistics_service.save_stats_to_excel(X_test, X_train, current_model_name_key, current_value_index, model_index, stock_symbol, y_pred, y_test, y_train, cv_scores)

                    print(f"Results saved to model_results_{stock_symbol}_{current_model_name_key}_fold_{current_value_index}.xlsx")

                    self.statistics_service.save_chart_to_excel_file(current_model_name_key, current_value_index, excel_file, fold_folder, results_df, stock_symbol)

                    current_value_index += 1
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Minimum predicted value {temp_model_min}")
                print(f"Maximum predicted value {temp_model_max}")
            print(f"[learn_models] Finished learning process of model: {current_model_name_key} in: {elapsed_time}\n")
        self.statistics_service.display_results(cv_scores)
        # self.save_models(learned_models) //TODO uncomment

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
