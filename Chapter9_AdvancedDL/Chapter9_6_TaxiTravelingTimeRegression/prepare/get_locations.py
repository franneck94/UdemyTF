import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import seaborn as sns
import pandas as pd
import itertools
from geopy.geocoders import Nominatim

geolocater = Nominatim(domain="localhost/nominatim/", scheme="http")

df = pd.read_excel(open(
    "./dortmund_straßennamen.xlsx",
    "rb"))

resulting_routes = []
i = 0
i_s = len(df)
print(i_s)
df["lat"] = ""
df["lon"] = ""
print(df.head())

for route in df.itertuples(index=True):
    if i == 0:
        i += 1
        continue

    print(i,"/",i_s)
    i+=1
    start = str(route[1]) + " " + str(route[0]) + "," + "Dortmund"

    try:
        print(start)
        print(geolocater.geocode(start))
        request = geolocater.geocode(start).raw
        df.at[i-1, "lat"] = request["lat"]
        df.at[i-1, "lon"] = request["lon"]
    except Exception as e:
        print("Error: ", e)
    print("\n")

writer = pd.ExcelWriter("dortmund_straßennamen_location.xlsx")
df.to_excel(writer, "Sheet1")
writer.save()