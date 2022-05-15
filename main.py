from data import Data

DATASET_PATH = 'C:\\Users\\anke\\OneDrive\\Документы\\ML\\heart_kaggle\\heart_cleveland_upload.csv'



if __name__ == '__main__':
    dataset = Data.Dataset(DATASET_PATH)
    print(dataset.data.head())
    dataset.preprocess()
    print(y_test[:10])


