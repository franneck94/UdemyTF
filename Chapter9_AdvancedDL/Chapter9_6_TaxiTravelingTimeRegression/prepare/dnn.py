import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_excel(open(
    "dortmund_straßennamen_location_route_distanz.xlsx",
    "rb"))

df = pd.DataFrame(df, 
    columns=["uhrzeit", "Straße Start", "Nr Start", 
        "Stadt Start", "Lat Start", 
        "Lon Start", "Straße Ziel", "Nr Ziel", 
        "Stadt Ziel", "Lat Ziel", "Lon Ziel",
        "OSRM Dauer", "OSRM Distanz", "y"])

print(df.head())
print(df.describe())

sns.residplot(df["y"][:2000], df["OSRM Distanz"][:2000])
plt.show()