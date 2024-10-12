from pymongo import MongoClient
import pickle


class MongoDBModelRepository:
    def __init__(self, uri="mongodb://localhost:27017", db_name="model_db", collection_name="models"):
        self.client = MongoClient(uri)
        self.collection = self.client[db_name][collection_name]

    def save_model(self, model, model_key):
        model_data = pickle.dumps(model)
        self.collection.replace_one(
            {"_id": model_key},
            {"_id": model_key, "model_data": model_data}
            , upsert=True)

    def load_model(self, model_key):
        record = self.collection.find_one({"_id": model_key})
        if record:
            return pickle.loads(record["model_data"])
        else:
            return None