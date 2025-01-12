import time
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Bidirectional, Dense, Dropout
from repositories.FileModelRepository import FileModelRepository
from repositories.MongoDBModelRepository import MongoDBModelRepository
from repositories.RedisModelCacheRepository import RedisModelCacheRepository
from app_config.ModelConfig import AVAILABLE_MODELS_NAMES, AVAILABLE_MODELS
from app_config.StatisticsConfig import metrics_array
from utilities.dataScaler import DataScaler
from services.DataService import DataService
from services.StatisticsService import StatisticsService


class ModelLearningService:

    def __init__(self):
        self.model_repository = FileModelRepository()
        self.mongo_repo = MongoDBModelRepository()
        self.redis_cache = RedisModelCacheRepository()
        self.data_service = DataService()
        self.data_scaler = DataScaler()
        self.statistics_service = StatisticsService()
        self.models = AVAILABLE_MODELS
        self.metrics_array = metrics_array
        self.models_names = AVAILABLE_MODELS_NAMES

    # TODO introduce random zeroes for train data??? looks like it works well in normal models to up prediction rates
    # TODO czy zmiany reshapu wpływaja na zmiany wartości przy ponownym reshapowaniu i odwracaniu i czy 2 razy ta sama wartosc zreshapowana przez w innych zmiennych bedzie ta sama
    def learn_models(self, stock_symbol, interval, period):
        data_for_train = self.data_service.get_parquet_data(stock_symbol, interval, period)
        X, y = self.data_service.get_objectives_from_data(data_for_train)
        number_of_features = X.shape[1]
        number_of_results = y.shape[1]
        X = X.values
        y = y.values

        # TODO refactor it to data scaler
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        # TODO tutaj z tym ccalowaniem
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        assert X_scaled.shape[0] == y_scaled.shape[0], "Mismatch in X_scaled and y_scaled rows!"
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)

        learned_models = {}
        print("[learn_models] Learning data set preparing to learn models")
        models_size = len(self.models)
        metrics_size = len(self.metrics_array)

        cv_scores = np.zeros((models_size, metrics_size, n_splits, number_of_results))
        print(f"cv_scores shape: {cv_scores.shape}")
        print(f"models_size: {models_size}, metrics_size: {metrics_size}, n_splits: {n_splits}, number_of_results: {number_of_results}")
        n_timesteps = 10
        for model_index, model in enumerate(self.models):
            current_value_index = 0
            current_model_name_key = self.models_names[model_index]
            print(f"[learn_models] Starting learning process of model: {current_model_name_key}\n")
            start_time = time.time()
            if isinstance(model, Sequential):
                if current_model_name_key == 'LSTM':
                    seq_model = self.create_lstm_model(model, number_of_features, number_of_results)
                if current_model_name_key == 'GRU':
                    seq_model = self.create_gru_model(model, number_of_features, number_of_results)
                if current_model_name_key == 'Bi-Direct':
                    seq_model = self.create_bi_lstm_model(model, number_of_features, number_of_results)

                for train_index, test_index in tscv.split(X_scaled, y_scaled):
                    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                    y_train, y_test = y_scaled[train_index], y_scaled[test_index]
                    assert len(X_train) == len(y_train), "Mismatch in training set lengths!"
                    assert len(X_test) == len(y_test), "Mismatch in testing set lengths!"

                    X_train_seq, y_train_seq, train_indices = self.create_sequences(X_train, y_train, n_timesteps)
                    X_test_seq, y_test_seq, test_indices = self.create_sequences(X_test, y_test, n_timesteps)

                    assert len(X_train_seq) == len(y_train_seq), "Mismatch in X_train_seq and y_train_seq lengths!"
                    assert len(X_test_seq) == len(y_test_seq), "Mismatch in X_train_seq and y_train_seq lengths!"

                    model.fit(X_train_seq, y_train_seq, validation_data=(X_test_seq, y_test_seq), epochs=50, batch_size=32) #TODO sprawdzic tez bez validation_data performance
                    learned_models[current_model_name_key] = seq_model

                    cv_scores, y_pred = self.statistics_service.create_stats_of_sequential_model(
                        X_test_seq, learned_models[current_model_name_key], model_index, y_test_seq, current_value_index, cv_scores, scaler_y)

                    aligned_y_test = y_test[test_indices]
                    aligned_y_test_original = scaler_y.inverse_transform(aligned_y_test.reshape(-1, y_pred.shape[1]))

                    y_train_inverse = scaler_y.inverse_transform(y_train.reshape(-1, y_pred.shape[1]))

                    X_test_unscaled = scaler_X.inverse_transform(X_test[test_indices])
                    X_train_unscaled = scaler_X.inverse_transform(X_train)

                    excel_file, fold_folder, results_df = self.statistics_service.save_stats_to_excel(
                        X_test_unscaled, X_train_unscaled, current_model_name_key, current_value_index, model_index,
                        stock_symbol, y_pred, aligned_y_test_original, y_train_inverse, cv_scores)

                    print(f"Results saved to sequential model_results_{stock_symbol}_{current_model_name_key}_fold_{current_value_index}.xlsx")
                    self.statistics_service.save_chart_to_excel_file(current_model_name_key, current_value_index, excel_file, fold_folder, results_df, stock_symbol)
                    current_value_index += 1
                end_time = time.time()
                elapsed_time = end_time - start_time
            else:
                for train_index, test_index in tscv.split(X_scaled, y_scaled): #TODO dodać tutaj scalowane wartości i potem je odscalować???
                    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                    y_train, y_test = y_scaled[train_index], y_scaled[test_index]
                    assert len(X_train) == len(y_train), "Mismatch in training set lengths!"
                    assert len(X_test) == len(y_test), "Mismatch in testing set lengths!"

                    model.fit(X_train, y_train)
                    learned_models[current_model_name_key] = model

                    cv_scores, y_pred = self.statistics_service.create_stats_of_model(
                        X_test, learned_models[current_model_name_key], model_index, y_test, current_value_index, cv_scores, scaler_y)

                    y_train_inverse = scaler_y.inverse_transform(y_train.reshape(-1, y_pred.shape[1]))
                    aligned_y_test_original = scaler_y.inverse_transform(y_test)

                    X_test_unscaled = scaler_X.inverse_transform(X_test)
                    X_train_unscaled = scaler_X.inverse_transform(X_train)

                    excel_file, fold_folder, results_df = self.statistics_service.save_stats_to_excel(
                        X_test_unscaled, X_train_unscaled, current_model_name_key, current_value_index, model_index,
                        stock_symbol, y_pred, aligned_y_test_original, y_train_inverse, cv_scores)

                    print(f"Results saved to model_results_{stock_symbol}_{current_model_name_key}_fold_{current_value_index}.xlsx")
                    self.statistics_service.save_chart_to_excel_file(current_model_name_key, current_value_index, excel_file, fold_folder, results_df, stock_symbol)
                    current_value_index += 1
                end_time = time.time()
                elapsed_time = end_time - start_time
            print(f"[learn_models] Finished learning process of model: {current_model_name_key} in: {elapsed_time}\n")
        # self.statistics_service.display_results(cv_scores)
        self.save_models(learned_models)

    def save_model(self, model, model_key):
        self.mongo_repo.save_model(model, model_key)
        self.redis_cache.cache_model(model, model_key)

    def save_models(self, learned_models):
        for model_key, model in learned_models.items():
            self.mongo_repo.save_model(model, model_key)
            self.redis_cache.cache_model(model, model_key)

    def create_lstm_model(self, model, number_of_features, number_of_results):
        # print(f"Creating lstm")
        # print(f"Creating first layer of lstm")
        # model.add(LSTM(units=50, return_sequences=True, input_shape=(10, number_of_features), name="lstm_layer_1"))
        # model.add(Dropout(0.2, name="dropout_1"))
        #
        # print(f"Creating second layer of lstm")
        # model.add(LSTM(units=50, return_sequences=True, name="lstm_layer_2"))
        # model.add(Dropout(0.2, name="dropout_2"))
        #
        # print(f"Creating third layer of lstm")
        # model.add(LSTM(units=50, return_sequences=True, name="lstm_layer_3"))
        # model.add(Dropout(0.2, name="dropout_3"))
        #
        # print(f"Creating fourth layer of lstm")
        # model.add(LSTM(units=50, name="lstm_layer_4"))
        # model.add(Dropout(0.2, name="dropout_4"))
        #
        # print(f"Creating output layer of lstm")
        # model.add(Dense(units=4, name="output_layer"))
        #
        # # model.add(Dense(5))
        # print(f"input finished")
        # model.compile(optimizer='adam', loss='mse')
        # print(f"compile finished")

        model.add(LSTM(units=50, return_sequences=True, input_shape=(10, number_of_features)))
        model.add(Dropout(0.1))
        model.add(LSTM(units=50))
        model.add(Dense(units=number_of_results))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])


        return model

    def create_gru_model(self, model, number_of_features, number_of_results):
        model.add(GRU(units=50, return_sequences=True, input_shape=(1, number_of_features), name="gru_layer_1"))
        model.add(Dropout(0.2, name="dropout_1"))

        model.add(GRU(units=70, return_sequences=True, name="gru_layer_2"))
        model.add(Dropout(0.2, name="dropout_2"))

        model.add(GRU(units=80, return_sequences=True, name="gru_layer_3"))
        model.add(Dropout(0.3, name="dropout_3"))

        model.add(GRU(units=50, name="gru_layer_4"))
        model.add(Dropout(0.2, name="dropout_4"))

        model.add(Dense(units=number_of_results, name="output_layer"))
        model.compile(optimizer='adam', loss='mse')
        return model

    def create_bi_lstm_model(self, model, number_of_features, number_of_results):
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

        model.add(Dense(units=number_of_results, name="output_layer"))
        model.compile(optimizer='adam', loss='mse')
        return model

    def create_sequences(self, data, target, n_timesteps):
        sequences, labels, indices = [], [], []
        for i in range(len(data) - n_timesteps):
            sequences.append(data[i:i + n_timesteps])
            labels.append(target[i + n_timesteps])
            indices.append(i + n_timesteps)
        return np.array(sequences), np.array(labels), np.array(indices)