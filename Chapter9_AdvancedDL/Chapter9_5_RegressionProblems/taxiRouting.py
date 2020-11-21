import os

import matplotlib.pyplot as plt
import pandas as pd

from tf_utils.taxiRoutingDataAdvanced import TAXIROUTING


if __name__ == "__main__":
    excel_file_path = os.path.abspath("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/data/taxiDataset.xlsx")
    data = TAXIROUTING(excel_file_path=excel_file_path)

    df = pd.DataFrame(data=data.x, columns=data.feature_names)
    df["y"] = data.y
    print(df.head())
    print(df.describe())
    print(df.info())
    df.hist(bins=30, figsize=(20, 15))
    plt.show()
