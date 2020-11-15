import os

from sklearn.linear_model import LinearRegression

from tf_utils.taxiRoutingData import TAXIROUTING


if __name__ == "__main__":
    excel_file_path = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/data/taxiDataset.xlsx")
    taxi_data = TAXIROUTING(excel_file_path=excel_file_path)
    x_train, y_train = taxi_data.x_train, taxi_data.y_train
    x_test, y_test = taxi_data.x_test, taxi_data.y_test
    num_features = taxi_data.num_features
    num_targets = taxi_data.num_targets

    regr = LinearRegression()
    regr.fit(x_train, y_train)  # Training
    print(regr.score(x_test, y_test))  # Testing
