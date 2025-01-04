from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

class DataScaler:
    def __init__(self):
        pass

    # @staticmethod
    # def scale_data(X, y):
    #     scaler = StandardScaler()
    #     return scaler.fit_transform(X), y



    @staticmethod
    def scale_data(X, y):
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        return X_scaled, y_scaled


    # TODO
    @staticmethod
    def inverse_transform_data(data):
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        return X_scaled, y_scaled