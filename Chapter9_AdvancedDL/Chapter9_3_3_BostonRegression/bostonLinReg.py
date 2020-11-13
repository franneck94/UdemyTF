from sklearn.linear_model import LinearRegression

from bostonData import BOSTON


if __name__ == "__main__":
    boston = BOSTON()
    x_train, y_train = boston.x_train, boston.y_train
    x_test, y_test = boston.x_test, boston.y_test
    num_features = boston.num_features
    num_targets = boston.num_targets

    regr = LinearRegression()
    regr.fit(x_train, y_train)  # Training
    print(regr.score(x_test, y_test))  # Testing
