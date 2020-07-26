import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import seaborn as sns
import pandas as pd
import itertools
from random import randrange
from datetime import timedelta
import datetime
import random


def random_date():
    return datetime.datetime(year=2018, month=11, day=22, hour=np.random.randint(0, 23), minute=np.random.randint(0, 59)).time()

df = pd.read_excel(open(
    "dortmund_straßennamen_location.xlsx",
    "rb"))

print(df.head())

start = df.sample(n=50000, replace=True)
destination = df.sample(n=50000, replace=True)
routes = [(random_date(), s.Straße, s.Nr, s.Stadt, s.lat, s.lon, d.Straße, d.Nr, d.Stadt, d.lat, d.lon) 
    for s, d in zip(start.itertuples(), destination.itertuples()) 
    if s.lon != None and s.lat != None and d.lon != None and d.lat != None
    and np.isnan(s.lon) != True and np.isnan(s.lat) != True and np.isnan(d.lon) != True and np.isnan(d.lat) != True]
df2 = pd.DataFrame(routes, 
    columns=["uhrzeit", "Straße Start", "Nr Start", "Stadt Start", "Lat Start", "Lon Start", "Straße Ziel", "Nr Ziel", "Stadt Ziel", "Lat Ziel", "Lon Ziel"])
print(len(routes))
print(routes[:2])

writer = pd.ExcelWriter("dortmund_straßennamen_location_route.xlsx")
df2.to_excel(writer, "Sheet1")
writer.save()