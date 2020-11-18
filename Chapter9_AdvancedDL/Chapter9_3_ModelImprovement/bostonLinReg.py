from sklearn.linear_model import LinearRegression

from tf_utils.bostonData import BOSTON


if __name__ == "__main__":
    boston = BOSTON()
    x_train, y_train = boston.get_train_set()
    x_test, y_test = boston.get_test_set()

    regr = LinearRegression()
    regr.fit(x_train, y_train)
    print(regr.score(x_test, y_test))
