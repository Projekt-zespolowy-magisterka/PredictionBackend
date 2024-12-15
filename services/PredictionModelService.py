from services.DataService import DataService
from pymongo.errors import PyMongoError
from repositories.MongoDBModelRepository import MongoDBModelRepository
from repositories.RedisModelCacheRepository import RedisModelCacheRepository
from app_config.ModelConfig import AVAILABLE_MODELS_NAMES
import pandas as pd
from tensorflow.keras.models import Sequential
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from utilities.dataScaler import DataScaler
from hurst import compute_Hc

PREDICTION_BASED_ON_HISTORICAL_DAYS = 1


class PredictionModelService:
    def __init__(self):
        self.data_service = DataService()
        self.mongo_repo = MongoDBModelRepository()
        self.redis_cache = RedisModelCacheRepository()
        self.model_keys = AVAILABLE_MODELS_NAMES
        self.data_service = DataService()
        self.data_scaler = DataScaler()

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
                            print(f"Model loaded from MongoDB is not a valid model instance: {model}")
                    except PyMongoError as e:
                        print(f"Failed to load model from MongoDB: {e}")

    def predict(self, stock_symbol, interval, period, days_ahead):
        self.load_models()
        # TODO ADDING STATS
        print(f"Self models: {self.models}")
        if not self.models:
            raise ValueError("Models are not loaded.")

        all_data = self.data_service.get_parquet_data(stock_symbol, interval, period)
        X_features, y = self.data_service.get_objectives_from_data(all_data)
        # X_array = np.array(X_features) if not isinstance(X_features, np.ndarray) else X_features

        original_column_names = X_features.columns
        # X_scaled, Y_scaled = self.data_scaler.scale_data(X_features, y)
        X_scaled_df = pd.DataFrame(X_features, columns=original_column_names, index=X_features.index)
        original_data = X_scaled_df.copy()

        converted_index_time_features = pd.to_datetime(X_scaled_df.index)
        last_time_value = (converted_index_time_features[-1] - pd.Timedelta(days=PREDICTION_BASED_ON_HISTORICAL_DAYS))
        last_days_data = X_scaled_df.loc[converted_index_time_features >= last_time_value]
        last_days_data_copy = last_days_data.copy()

        predictions = {model_name: {'Open': [], 'High': [], 'Low': [], 'Close': [], 'Volume': []} for model_name in self.model_keys}

        converted_days_ahead = int(''.join(filter(str.isdigit, days_ahead)))
        closing_stock_hour = pd.Timestamp('19:00:00-04:00')

        converted_days_ahead = 1 #TODO MOCK
        print(f"Prepering to make predictions: \n")
        for index, model in enumerate(self.models.values()):
            model_name = self.model_keys[index]
            last_days_data = last_days_data_copy
            for day in range(converted_days_ahead):
                for hour in range(24):
                    temp_dataframe = last_days_data.copy()
                    newest_time = last_days_data.index[-1]
                    is_last_trading_hour = (newest_time.time() >= closing_stock_hour.time())
                    if is_last_trading_hour:
                        next_date = newest_time + pd.Timedelta(days=1)
                        next_date = next_date.replace(hour=9, minute=30, second=0, microsecond=0)
                    else:
                        print("Still within trading hours; do not change the day.")
                        next_date = newest_time + pd.Timedelta(hours=1)
                    if isinstance(model, Sequential):
                        number_of_data_rows_for_prediction = last_days_data.shape[0]
                        number_of_features = last_days_data.shape[1]
                        wanted_number_of_predictions_ahead = 1

                        last_days_data_reshaped = last_days_data.values.reshape(wanted_number_of_predictions_ahead,
                                                                                number_of_data_rows_for_prediction,
                                                                                number_of_features)
                        raw_prediction = model.predict(last_days_data_reshaped)
                        print(f"Model: {model_name}, predictions raw {raw_prediction}")
                        prediction = raw_prediction[0]
                        print(f"Predictions {prediction}")

                        self.append_prediction_results(model_name, prediction, predictions)

                        new_row = self.create_new_row(next_date, prediction, temp_dataframe, original_data)
                        last_days_data = pd.concat([last_days_data.iloc[1:], new_row], axis=0)
                    else:
                        aggregated_features = {
                            'Open_mean': last_days_data['Open'].mean(),
                            'High_mean': last_days_data['High'].mean(),
                            'Low_mean': last_days_data['Low'].mean(),
                            'Close_mean': last_days_data['Close'].mean(),
                            'Volume_mean': last_days_data['Volume'].mean(),
                            'Return': last_days_data['Return'].iloc[-1],
                            'Day': last_days_data['Day'].iloc[-1],
                            'Month': last_days_data['Month'].iloc[-1],
                            'Year': last_days_data['Year'].iloc[-1],
                            'Hour': last_days_data['Hour'].iloc[-1],
                            'DayOfWeek': last_days_data['DayOfWeek'].iloc[-1],
                            'IsWeekend': last_days_data['IsWeekend'].iloc[-1],
                            'Hurst': last_days_data['Hurst'].iloc[-1]
                        }
                        aggregated_features_df = pd.DataFrame([aggregated_features])

                        raw_prediction = model.predict(aggregated_features_df.values)
                        prediction = raw_prediction[0]

                        self.append_prediction_results(model_name, prediction, predictions)

                        new_row = self.create_new_row(next_date, prediction, temp_dataframe, original_data)
                        last_days_data = pd.concat([last_days_data.iloc[1:], new_row], axis=0)
                    print(f"Predictions from {model_name}: {prediction}")
        dataframes = []
        start_date = converted_index_time_features[-1] + pd.Timedelta(hours=1)
        timestamps = pd.date_range(start=start_date, periods=24 * converted_days_ahead, freq='1h')

        for model_name, pred_values in predictions.items():
            df = pd.DataFrame(pred_values, index=timestamps)
            df['Model'] = model_name
            dataframes.append(df)

        predictions_df = pd.concat(dataframes)
        predictions_df.index.name = 'Timestamp'
        print(predictions_df)
        self.data_service.save_to_csv(predictions_df, stock_symbol, interval, period)

    def append_prediction_results(self, model_name, prediction, predictions):
        predictions[model_name]['Open'].append(prediction[0])
        predictions[model_name]['High'].append(prediction[1])
        predictions[model_name]['Low'].append(prediction[2])
        predictions[model_name]['Close'].append(prediction[3])
        predictions[model_name]['Volume'].append(prediction[4])

    def create_new_row(self, next_date, prediction, temp_dataframe, original_data):
        temp_dataframe.loc[temp_dataframe.index[-1], 'Day'] = next_date.day
        temp_dataframe.loc[temp_dataframe.index[-1], 'Month'] = next_date.month
        temp_dataframe.loc[temp_dataframe.index[-1], 'Year'] = next_date.year
        temp_dataframe.loc[temp_dataframe.index[-1], 'Hour'] = next_date.hour
        temp_dataframe.loc[temp_dataframe.index[-1], 'DayOfWeek'] = next_date.weekday()
        temp_dataframe.loc[temp_dataframe.index[-1], 'IsWeekend'] = 1 if next_date.weekday() >= 5 else 0

        if len(temp_dataframe) < 100:
            num_needed = 100 - len(temp_dataframe)
            combined_close_data = pd.concat([original_data['Close'].iloc[-num_needed:], temp_dataframe['Close']])
        else:
            combined_close_data = temp_dataframe['Close'].iloc[-100:]

        H, _, _ = compute_Hc(combined_close_data)
        temp_dataframe.loc[temp_dataframe.index[-1], 'Hurst'] = round(H, 3)

        last_close = temp_dataframe.loc[temp_dataframe.index[-1], 'Close']
        new_row = pd.DataFrame([temp_dataframe.iloc[-1].values], columns=temp_dataframe.columns, index=[next_date])

        new_row[['Open', 'High', 'Low', 'Close', 'Volume']] = prediction[:5]
        new_close = new_row.loc[next_date, 'Close']
        new_row.loc[next_date, 'Return'] = (new_close - last_close) / last_close
        print(f"NEW ROW: \n {new_row}")
        return new_row
