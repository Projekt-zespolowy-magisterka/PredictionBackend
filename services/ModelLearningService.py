from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from utilities.statisticsCalculator import calculate_r_squared, pair_test
from repositories.FileModelRepository import FileModelRepository
from pymongo.errors import PyMongoError
from repositories.MongoDBModelRepository import MongoDBModelRepository
from repositories.RedisModelCacheRepository import RedisModelCacheRepository
from utilities.dataReader import DataReader
from utilities.dataAnalyser import DataAnalyzer
from utilities.dataScaler import DataScaler

AVAILABLE_MODELS_NAMES = [
    "LinearRegression"
    , "DecisionTreeRegressor"
    , "Ridge"
    # ,"MLPRegressor"
    # ,"RandomForestRegressor"
]


class ModelLearningService:
    def __init__(self):
        self.model_repository = FileModelRepository()
        self.mongo_repo = MongoDBModelRepository()
        self.redis_cache = RedisModelCacheRepository()
        self.data_reader = DataReader()
        self.data_analyzer = DataAnalyzer()
        self.data_scaler = DataScaler()
        self.download_data = self.data_reader.get_download_data()
        self.upload_data = self.data_reader.get_upload_data()
        self.models = [
            LinearRegression()
            , DecisionTreeRegressor()
            , Ridge()
            # ,MLPRegressor()
            # ,RandomForestRegressor()
        ]
        self.metrics_array = [
            mean_squared_error,
            mean_absolute_error,
            calculate_r_squared
        ]
        self.models_names = AVAILABLE_MODELS_NAMES

    def learn_models(self):
        print()

    def learn_models_test(self):
        print()
        print("Starting learning process \n")
        all_data = [
            self.download_data,
            self.upload_data
        ]

        all_data_names = [
            "Download",
            "Upload"
        ]

        num_folds = 5
        num_repeats = 3
        rkf = RepeatedKFold(n_splits=num_folds, n_repeats=num_repeats, random_state=42)
        learned_models = {}
        for data_type_index, data_type in enumerate(all_data):
            current_data_type = all_data[data_type_index]
            for company_data_index, company_data in enumerate(current_data_type):
                X, y = DataReader.process_data(company_data, all_data_names[data_type_index])
                X_scaled, Y_scaled = self.data_scaler.scale_data(X, y)
                for model_index, model in enumerate(self.models):
                    current_model_name_key = self.models_names[model_index]
                    for train_index, test_index in rkf.split(X_scaled, y):
                        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                        y_train, y_test = y[train_index], y[test_index]
                        learned_models[current_model_name_key] = model.fit(X_train, y_train)
                self.save_models(learned_models)

    def save_model(self, model, model_key):
        self.mongo_repo.save_model(model, model_key)
        self.redis_cache.cache_model(model, model_key)

    def save_models(self, learned_models):
        for model_key, model in learned_models.items():
            self.mongo_repo.save_model(model, model_key)
            self.redis_cache.cache_model(model, model_key)
