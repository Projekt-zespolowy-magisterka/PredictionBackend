import pandas as pd
from repositories.MongoDBModelRepository import MongoDBModelRepository

core_data_path = "E:\\pliki\\uczelnia\\przedmioty\\MAG\\S1\\PAMSI\\P\\Campaign3\\"


class DataReader:
    def __init__(self):
        self.mongo_repo = MongoDBModelRepository()

    @staticmethod
    def get_download_data():
        o2_download = pd.read_csv(core_data_path + "o2_download_nexus5x.csv")
        telekom_download = pd.read_csv(core_data_path + "telekom_download_nexus5x.csv")
        vodafone_download = pd.read_csv(core_data_path + "vodafone_download_nexus5x.csv")
        download_data = [o2_download, telekom_download, vodafone_download]
        return download_data

    @staticmethod
    def get_upload_data():
        o2_upload = pd.read_csv(core_data_path + "o2_upload_nexus5x.csv")
        telekom_upload = pd.read_csv(core_data_path + "telekom_upload_nexus5x.csv")
        vodafone_upload = pd.read_csv(core_data_path + "vodafone_upload_nexus5x.csv")
        upload_data = [o2_upload, telekom_upload, vodafone_upload]
        return upload_data

    @staticmethod
    def get_company_names():
        return [
            "o2",
            "telekom",
            "vodafone"
        ]

    @staticmethod
    def process_data(data, current_data_type):
        if current_data_type == 'Download':
            mapped_data = DataReader.map_modulations_to_numbers(data)
            X = mapped_data.drop(['throughput', 'tp_cleaned', 'chipsettime', 'gpstime', 'longitude', 'latitude', 'speed'], axis=1)
            y = mapped_data['throughput']
        if current_data_type == 'Upload':
            X = data.drop(['qualitytimestamp', 'tp_cleaned', 'gpstime', 'longitude', 'latitude', 'speed'], axis=1)
            y = data['tp_cleaned']
        return X, y

    @staticmethod
    def map_modulations_to_numbers(data):
        mapping = {'QPSK': 1, '16QAM': 2, '64QAM': 3}
        data_to_map = data
        data_to_map['mcs0'] = data_to_map['mcs0'].map(mapping)
        data_to_map['mcs1'] = data_to_map['mcs1'].map(mapping)
        return data_to_map
