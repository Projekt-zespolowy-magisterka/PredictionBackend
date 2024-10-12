import redis
import pickle


class RedisModelCacheRepository:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.StrictRedis(host=host, port=port, db=db)

    def cache_model(self, model, model_key):
        model_data = pickle.dumps(model)
        self.redis_client.set(model_key, model_data)

    def get_cached_model(self, model_key):
        model_data = self.redis_client.get(model_key)
        if model_data:
            return pickle.loads(model_data)
        return None