from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras.datasets import boston_housing


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    regr = LinearRegression()
    regr.fit(x_train, y_train)
    score = regr.score(x_test, y_test)
    y_pred = regr.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Score: {score}")
    print(f"MSE: {mse}")
