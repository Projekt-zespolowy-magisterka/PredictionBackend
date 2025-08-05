from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential

AVAILABLE_MODELS_NAMES = [
    "DecisionTreeRegressor"
    ,
    "RandomForestRegressor"
    ,
    "LSTM"
    ,
    "GRU",
    "Bi-Direct"
]

AVAILABLE_MODELS = [
    DecisionTreeRegressor()
    ,
    RandomForestRegressor()
    ,
    Sequential()
    ,
    Sequential(),
    Sequential()
]
