import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from hurst import compute_Hc
from repositories.MongoDBModelRepository import MongoDBModelRepository
from repositories.RedisModelCacheRepository import RedisModelCacheRepository
from services.DataService import DataService
from utilities.dataScaler import DataScaler
from app_config.ModelConfig import AVAILABLE_MODELS_NAMES
PREDICTION_BASED_ON_HISTORICAL_DAYS = 1


class PredictionModelService:
    def __init__(self):
        self.data_service = DataService()
        self.mongo_repo = MongoDBModelRepository()
        self.redis_cache = RedisModelCacheRepository()
        self.model_keys = AVAILABLE_MODELS_NAMES
        self.data_service = DataService()
        self.data_scaler = DataScaler()
        self.models = {}
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.trading_hours = (pd.Timestamp("09:30:00-05:00"), pd.Timestamp("16:00:00-05:00"))
        self.n_timesteps = 10

    def load_model(self, model_key):
        model = self.redis_cache.get_cached_model(model_key)
        if model:
            print("Model loaded from Redis cache.")
            return model

        try:
            model = self.mongo_repo.load_model(model_key)
            if model:
                print("Model loaded from MongoDB and cached to Redis.")
                self.redis_cache.cache_model(model, model_key)
            else:
                print("Model not found in MongoDB.")
        except PyMongoError as e:
            print(f"Failed to load model from MongoDB: {e}")
        return model

    def load_models(self):
        self.models = {}
        for model_key in self.model_keys:
            model = self.redis_cache.get_cached_model(model_key)
            if isinstance(model, (LinearRegression, DecisionTreeRegressor, Ridge, MLPRegressor, RandomForestRegressor, Sequential)):
                self.models[model_key] = model
                print(f"Loaded {model_key} from Redis cache.")
            else:
                if not model:
                    try:
                        model = self.mongo_repo.load_model(model_key)

                        if isinstance(model, (LinearRegression, DecisionTreeRegressor, Ridge, MLPRegressor, RandomForestRegressor, Sequential)):
                            self.models[model_key] = model
                            print("Model loaded from MongoDB and cached to Redis.")
                            self.redis_cache.cache_model(model_key, model)
                        else:
                            print(f"Model loaded from MongoDB is not a valid model instance")
                    except PyMongoError as e:
                        print(f"Failed to load model from MongoDB: {e}")

    def predict(self, stock_symbol, interval, period, days_ahead):
        self.load_models()
        if not self.models:
            raise ValueError("Models are not loaded.")

        all_data = self.data_service.get_parquet_data(stock_symbol, interval, period)
        X_features, y_target = self.data_service.get_objectives_from_data(all_data)

        X_scaled, y_scaled = self.scale_data(X_features.values, y_target.values)

        last_input = X_scaled[-self.n_timesteps:]


        predictions = {model_name: [] for model_name in self.model_keys}
        hours_ahead = pd.to_timedelta(days_ahead).days * 24

        for model_name, model in zip(self.model_keys, self.models.values()):
            input_data = last_input.copy()
            for _ in range(hours_ahead):
                if isinstance(model, Sequential):
                    prediction_scaled = model.predict(input_data.reshape(1, self.n_timesteps, input_data.shape[1]))[0] #TODO do ulepszenia input do takiej predykcji
                else:
                    prediction_scaled = model.predict(input_data.mean(axis=0).reshape(1, -1))[0]  #TODO do ulepszenia input do takiej predykcji
                    # prediction_scaled = model.predict(input_data[-1].reshape(1, -1))[0]
                prediction = self.scaler_y.inverse_transform([prediction_scaled])[0]
                predictions[model_name].append(prediction)

                new_row = np.zeros((input_data.shape[1],))
                new_row[:len(prediction_scaled)] = prediction_scaled
                new_row[len(prediction_scaled):] = input_data[-1, len(prediction_scaled):]
                input_data = np.vstack([input_data[1:], new_row])

        results = []
        timestamps = pd.date_range(X_features.index[-1], periods=hours_ahead + 1, freq=interval)[1:]
        for model_name, pred_list in predictions.items():
            for timestamp, values in zip(timestamps, pred_list):
                results.append({'Model': model_name, 'Timestamp': timestamp,
                                **dict(zip(['Open', 'High', 'Low', 'Close', 'Volume'], values))})

        predictions_df = pd.DataFrame(results)
        predictions_df.set_index('Timestamp', inplace=True)
        self.data_service.save_predictions_to_csv(predictions_df, stock_symbol, interval, period)
        filtered_results = [result for result in results if result["Model"] == "LSTM"]

        return filtered_results


    def append_prediction_results(self, model_name, prediction, predictions):
        predictions[model_name]['Open'].append(prediction[0])
        predictions[model_name]['High'].append(prediction[1])
        predictions[model_name]['Low'].append(prediction[2])
        predictions[model_name]['Close'].append(prediction[3])
        predictions[model_name]['Volume'].append(prediction[4])

    def create_new_row(self, next_date, prediction, temp_dataframe, original_data):
        new_row_data = dict(zip(['Open', 'High', 'Low', 'Close', 'Volume'], prediction))
        new_row_data.update({
            'Day': next_date.day,
            'Month': next_date.month,
            'Year': next_date.year,
            'Hour': next_date.hour + next_date.minute / 60,
            'DayOfWeek': next_date.weekday(),
            'IsWeekend': 1 if next_date.weekday() >= 5 else 0,
        })

        num_needed = max(100 - len(temp_dataframe), 0)
        combined_close_data = pd.concat([original_data['Close'].iloc[-num_needed:], temp_dataframe['Close']])
        H, _, _ = compute_Hc(combined_close_data)
        new_row_data['Hurst'] = round(H, 3)

        last_close = temp_dataframe['Close'].iloc[-1]
        new_row_data['Return'] = (prediction[3] - last_close) / last_close if last_close != 0 else 0

        new_row = pd.DataFrame([new_row_data], index=[next_date])
        return new_row

    def scale_data(self, X, y):
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        return X_scaled, y_scaled

    def create_sequences(self, X, y, n_timesteps):
        sequences_X, sequences_y = [], []
        for i in range(len(X) - n_timesteps + 1):
            sequences_X.append(X[i:i + n_timesteps])
            sequences_y.append(y[i + n_timesteps - 1])
        return np.array(sequences_X), np.array(sequences_y)

    def adjust_next_time(self, current_time):
        is_after_close = current_time.time() >= self.trading_hours[1].time()
        if is_after_close:
            next_time = current_time + pd.Timedelta(days=1)
            next_time = next_time.replace(hour=9, minute=30, second=0, microsecond=0)
        else:
            next_time = current_time + pd.Timedelta(hours=1)
        return next_time