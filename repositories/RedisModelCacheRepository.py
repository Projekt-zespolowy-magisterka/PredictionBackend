import redis
import pickle
import time
import tempfile
from tensorflow.keras.models import load_model, Sequential
import os


class RedisModelCacheRepository:
    def __init__(self):
        host = os.getenv('REDIS_HOST', 'localhost')
        port = int(os.getenv('REDIS_PORT', 6379))
        db = int(os.getenv('REDIS_DB', 0))
        self.redis_client = redis.StrictRedis(host=host, port=port, db=db)

    def cache_model(self, model, model_key):
        try:
            if isinstance(model, Sequential):
                with tempfile.TemporaryDirectory() as temp_dir:
                    model_path = f"{temp_dir}/{model_key}.h5"
                    model.save(model_path)
                    with open(model_path, "rb") as f:
                        model_data = f.read()
                    self.redis_client.set(model_key, model_data)
            else:
                model_data = pickle.dumps(model)
                self.redis_client.set(model_key, model_data)

            self.redis_client.set(f"{model_key}_cached_at", time.time())
            print(f"Model {model_key} cached in Redis.")
        except Exception as e:
            print(f"Error caching model {model_key}: {e}")

    def get_cached_model(self, model_key):
        try:
            model_data = self.redis_client.get(model_key)
            if model_data:
                if model_key.startswith("Sequential"):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        model_path = f"{temp_dir}/{model_key}.h5"
                        with open(model_path, "wb") as f:
                            f.write(model_data)
                        model = load_model(model_path)
                else:
                    model = pickle.loads(model_data)
                cached_at = self.redis_client.get(f"{model_key}_cached_at")
                if cached_at:
                    print(f"Model {model_key} retrieved from Redis, cached at: {time.ctime(float(cached_at))}")
                return model
            print(f"Model {model_key} not found in Redis.")
            return None
        except Exception as e:
            print(f"Error retrieving model {model_key}: {e}")
            return None