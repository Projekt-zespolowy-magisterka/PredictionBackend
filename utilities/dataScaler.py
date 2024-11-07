from sklearn.preprocessing import StandardScaler


class DataScaler:
    def __init__(self):
        pass

    @staticmethod
    def scale_data(X, y):
        scaler = StandardScaler()
        return scaler.fit_transform(X), y