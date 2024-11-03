import redis
import pickle
import time

class RedisModelCacheRepository:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.StrictRedis(host=host, port=port, db=db)

    def cache_model(self, model, model_key):
        model_data = pickle.dumps(model)
        # self.redis_client.set(model_key, model_data)
        self.redis_client.hset(model_key, mapping={
            "model": model_data,
            "cached_at": time.time()  # Store as Unix timestamp
        })

    def get_cached_model(self, model_key):
        # model_data = self.redis_client.get(model_key)
        # if model_data:
        #     return pickle.loads(model_data)
        # return None
        cached_data = self.redis_client.hgetall(model_key)
        if cached_data:
            model = pickle.loads(cached_data[b'model'])
            cached_at = float(cached_data[b'cached_at'])
            print(f"Model cached at: {time.ctime(cached_at)}")
            return model
        return None