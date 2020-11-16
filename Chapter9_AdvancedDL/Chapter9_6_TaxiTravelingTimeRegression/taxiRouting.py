import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tf_utils.taxiRoutingData import TAXIROUTING


if __name__ == "__main__":
    excel_file_path = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/data/taxiDataset.xlsx")
    taxi_data = TAXIROUTING(excel_file_path=excel_file_path)

    df = pd.DataFrame(data=taxi_data.x, columns=taxi_data.feature_names)
    df["y"] = taxi_data.y
    print(df.head())
    print(df.describe())
    print(df.info())
    df.hist(bins=30, figsize=(20, 15))
    plt.show()
