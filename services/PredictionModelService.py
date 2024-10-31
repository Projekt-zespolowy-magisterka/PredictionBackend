import pickle
from services.DataService import DataService
from pymongo.errors import PyMongoError
from repositories.MongoDBModelRepository import MongoDBModelRepository
from repositories.RedisModelCacheRepository import RedisModelCacheRepository
from services.ModelLearningService import AVAILABLE_MODELS_NAMES
import pandas as pd
from tensorflow.keras.models import Sequential
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

PREDICTION_BASED_ON_HISTORICAL_DAYS = 1


class PredictionModelService:
    def __init__(self):
        self.data_service = DataService()
        self.mongo_repo = MongoDBModelRepository()
        self.redis_cache = RedisModelCacheRepository()
        self.model_keys = AVAILABLE_MODELS_NAMES


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

                        # Ensure that the loaded model is a valid instance
                        if isinstance(model, (LinearRegression, DecisionTreeRegressor, Ridge, MLPRegressor, RandomForestRegressor, Sequential)):
                            self.models[model_key] = model
                            print("Model loaded from MongoDB and cached to Redis.")
                            self.redis_cache.cache_model(model_key, model)
                        else:
                            print(f"Model loaded from MongoDB is not a valid model instance: {model}")
                    except PyMongoError as e:
                        print(f"Failed to load model from MongoDB: {e}")

    def predict(self, stock_symbol, interval, period, days_ahead):
        self.load_models()

        print(f"Self models: {self.models}")
        if not self.models:
            raise ValueError("Models are not loaded.")
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Year', 'Month', 'Day', 'DayOfWeek', 'IsWeekend']

        all_data = self.data_service.get_parquet_data(stock_symbol, interval, period)
        X_features, y = self.data_service.process_data(all_data)

        converted_features = pd.to_datetime(X_features.index)
        last_value = (converted_features[-1] - pd.Timedelta(days=PREDICTION_BASED_ON_HISTORICAL_DAYS))
        last_days_data = X_features.loc[converted_features >= last_value]
        last_days_data_copy = last_days_data
        # last_days_data_reshaped = last_days_data.values.reshape(-1, 1, last_days_data.shape[1])
        # print(f"Last days data: \n")
        print(last_days_data)
        #
        # print(f"Last days data reshaped: \n")
        # print(last_days_data_reshaped)

        print(f"Predictions setup: \n")
        # predictions = []

        predictions = {model_name: [] for model_name in self.model_keys}

        converted_days_ahead = int(''.join(filter(str.isdigit, days_ahead)))

        converted_days_ahead = 1 #TODO MOCK
        print(f"Prepering to make predictions: \n")
        for index, model in enumerate(self.models.values()):
            model_name = self.model_keys[index]
            last_days_data = last_days_data_copy
            for day in range(converted_days_ahead):
                print(f"Day: {day} \n")
                for hour in range(24):
                    print(f"Hour: {hour} \n")
                    if isinstance(model, Sequential):  # For LSTM
                        a = last_days_data.shape[0]
                        b = last_days_data.shape[1]
                        print(f"A: {a}")
                        print(f"B: {b}")
                        last_days_data_reshaped = last_days_data.values.reshape(1, last_days_data.shape[0],
                                                                                        last_days_data.shape[1])
                        raw_prediction = model.predict(last_days_data_reshaped)
                        prediction = raw_prediction.flatten()
                        predictions[model_name].append(prediction[0])

                        # Update the data for the next prediction, shifting sequence and appending new prediction
                        new_row = pd.DataFrame([last_days_data.iloc[-1].values], columns=last_days_data.columns)
                        print(f"New row lstm: {new_row}")
                        new_row.iloc[0, -1] = prediction[0]  # Update last value with prediction
                        print(f"New row lstm after iloc: {new_row}")
                        last_days_data = pd.concat([last_days_data.iloc[1:], new_row], ignore_index=True)
                        print(f"LAST DAY DATA LSTM: {last_days_data}")
                    else:  # For non-sequential models
                        prediction = model.predict(last_days_data.iloc[-1:].values)
                        predictions[model_name].append(prediction[0])

                        # Update the data for the next prediction
                        new_row = pd.DataFrame([last_days_data.iloc[-1].values], columns=last_days_data.columns)
                        print(f"New row: {new_row}")
                        new_row.iloc[0, -1] = prediction[0]  # Update last value with prediction
                        print(f"New row after iloc: {new_row}")
                        last_days_data = pd.concat([last_days_data.iloc[1:], new_row], ignore_index=True)
                        print(f"LAST DAY DATA: {last_days_data}")
                    print(f"Predictions from {model_name}: {prediction}")

        # # Convert predictions to a DataFrame or Series for easier manipulation
        # predictions_df = pd.DataFrame(predictions, columns=['Predicted_Close'])
        #
        # # Optionally, create a time index for predictions (assuming starting point is the last timestamp)
        # predictions_df.index = pd.date_range(start=latest_input.index[-1] + pd.Timedelta(hours=4), periods=len(predictions), freq='4H')

        # Convert predictions to a DataFrame
        # predictions_df = pd.DataFrame(predictions)

        # Create a time index for predictions
        print("Making dataframes")
        print(f"Predictions: {predictions}")

        # predictions_df = pd.DataFrame(predictions, columns=['Predicted_Close'])
        # print(f"Predictions df: {predictions_df}")
        # predictions_df.index = pd.date_range(start=converted_features[-1] + pd.Timedelta(hours=4), periods=len(predictions_df), freq='4h')
        # print(f"Predictions df after index: {predictions_df}")

        predictions_df = pd.DataFrame.from_dict(predictions, orient='index').transpose()
        predictions_df = predictions_df.melt(var_name='Model', value_name='Predicted_Close')

        # Generate date range for predictions
        start_date = converted_features[-1] + pd.Timedelta(hours=1)
        predictions_df['Timestamp'] = pd.date_range(start=start_date, periods=len(predictions_df), freq='1h')

        # Set the date as the index for easier viewing
        predictions_df.set_index('Timestamp', inplace=True)

        print(predictions_df)