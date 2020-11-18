import os

from sklearn.linear_model import LinearRegression

from tf_utils.taxiRoutingData import TAXIROUTING


EXCEL_FILE_PATH = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/data/taxiDataset.xlsx")


if __name__ == "__main__":
    data = TAXIROUTING(excel_file_path=EXCEL_FILE_PATH)

    x_train, y_train = data.get_train_set()
    x_test, y_test = data.get_test_set()

    num_features = data.num_features
    num_targets = data.num_targets

    regr = LinearRegression()
    regr.fit(x_train, y_train)
    print(regr.score(x_test, y_test))
