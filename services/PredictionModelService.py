import pickle
from repositories.FileModelRepository import FileModelRepository
from pymongo.errors import PyMongoError
from repositories.MongoDBModelRepository import MongoDBModelRepository
from repositories.RedisModelCacheRepository import RedisModelCacheRepository
from services.ModelLearningService import AVAILABLE_MODELS_NAMES


class PredictionModelService:
    def __init__(self):
        self.model_repository = FileModelRepository()
        self.mongo_repo = MongoDBModelRepository()
        self.redis_cache = RedisModelCacheRepository()
        self.model_keys = AVAILABLE_MODELS_NAMES
        # self.model = self.load_model()

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
            if model:
                self.models[model_key] = model
            if not model:
                try:
                    model = self.mongo_repo.load_model(model_key)

                    if model:
                        print("Model loaded from MongoDB and cached to Redis.")
                        self.redis_cache.cache_model(model_key, model)
                    else:
                        print("Model not found in MongoDB.")
                except PyMongoError as e:
                    print(f"Failed to load model from MongoDB: {e}")

    def predict(self, data):
        if not self.model:
            raise ValueError("Model is not loaded.")
        return self.model.predict([data])[0]
