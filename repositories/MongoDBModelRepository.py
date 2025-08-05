from pymongo import MongoClient
import pickle
from gridfs import GridFS
import logging
from datetime import datetime
import os

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


class MongoDBModelRepository:
    def __init__(self):
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27018")
        db_name = os.getenv("MONGO_DB_NAME", "model_db")
        collection_name = os.getenv("MONGO_COLLECTION_NAME", "models")
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.fs = GridFS(self.db)

    def save_model(self, model, model_key):
        try:
            model_data = pickle.dumps(model)

            existing = self.collection.find_one({"_id": model_key})
            if existing:
                print("Model is existing")
                self.fs.delete(existing["model_file_id"])
                print("Model deleted")

            model_file_id = self.fs.put(model_data)
            print(f"Model file id: {model_file_id}")
            self.collection.replace_one(
                {"_id": model_key},
                {
                    "_id": model_key,
                    "model_file_id": model_file_id,
                    "created_at": datetime.utcnow()
                },
                upsert=True
            )
            print(f"Model {model_key} saved")
        except Exception as e:
            logger.error(f"Error saving model with key '{model_key}': string error: {str(e)},  error: {e}")

    def load_model(self, model_key):
        doc = self.collection.find_one({"_id": model_key})
        if doc:
            model_data = self.fs.get(doc["model_file_id"]).read()
            print(f"Model {model_key} loaded")
            return pickle.loads(model_data)
        print(f"Model {model_key} not found")
        return None
