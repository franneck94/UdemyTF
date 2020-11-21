import matplotlib.pyplot as plt
import pandas as pd

from tf_utils.caliHousingData import CALIHOUSING


if __name__ == "__main__":
    cali_data = CALIHOUSING()
    print(cali_data.x_train.shape, cali_data.y_train.shape)
    print(cali_data.x_test.shape, cali_data.y_test.shape)

    df = pd.DataFrame(data=cali_data.x, columns=cali_data.feature_names)
    df["y"] = cali_data.y

    print(cali_data.description)
    print(df.head())
    print(df.describe())
    print(df.info())

    df.hist(bins=30, figsize=(20, 15))
    plt.show()

    df.plot(
        kind="scatter",
        x="Longitude",
        y="Latitude",
        alpha=0.4,
        figsize=(10, 7),
        c="y",
        cmap=plt.get_cmap("jet"),
        colorbar=True,
    )
    plt.show()
