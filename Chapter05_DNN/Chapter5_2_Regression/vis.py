import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes


def main() -> None:
    dataset = load_diabetes()
    x = dataset.data
    y = dataset.target

    print(f"Feature names:\n{dataset.feature_names}")
    print(f"DESCR:\n{dataset.DESCR}")

    df = pd.DataFrame(x, columns=dataset.feature_names)
    df["y"] = y

    print(df.head(n=10))
    print(df.info())
    print(df.describe())

    df.hist(bins=30, figsize=(15, 15))
    plt.show()


if __name__ == "__main__":
    main()
