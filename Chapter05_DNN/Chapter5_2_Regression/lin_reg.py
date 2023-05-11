from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def main() -> None:
    dataset = load_diabetes()
    x = dataset.data
    y = dataset.target.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

    regr = LinearRegression()
    regr.fit(x_train, y_train)
    score = regr.score(x_test, y_test)
    y_pred = regr.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Score: {score}")
    print(f"MSE: {mse}")


if __name__ == "__main__":
    main()
