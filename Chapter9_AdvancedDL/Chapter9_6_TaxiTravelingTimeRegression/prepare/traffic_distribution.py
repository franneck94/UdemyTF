import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import seaborn as sns
import pandas as pd

# Plot a 1D density example
N = 100
var1 = 3
var2 = 3
mean1 = 9
mean2 = 17
np.random.seed(1)
X = np.concatenate((np.random.normal(mean1, var1, N),np.random.normal(mean2, var2, N)))[:, np.newaxis]

X_plot = np.linspace(0, 24, 1000)[:, np.newaxis]
true_dens = (norm(mean1, var1).pdf(X_plot[:, 0]) + norm(mean2, var2).pdf(X_plot[:, 0]))*2.8+0.8

x_sampled = [i for i in range(24)]
dens_sampled = (norm(mean1, var1).pdf(x_sampled) + norm(mean2, var2).pdf(x_sampled))*2.8+0.8 + np.random.uniform(-0.015, 0.015, size=len(x_sampled))

fig, ax = plt.subplots()
ax.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2, label='input distribution')
ax.scatter(x_sampled, dens_sampled, color="red")
plt.show()

def traffic_dist(x):
    var1, var2 = 3, 3
    mean1, mean2 = 9, 17
    y = (norm(mean1, var1).pdf(x) + norm(mean2, var2).pdf(x))*2.8+0.8 + np.random.uniform(-0.015, 0.015, size=len(x))
    return y


df = pd.read_excel(open(
    "dortmund_straßennamen_location_route_distanz.xlsx",
    "rb"))

df2 = pd.DataFrame(df, 
    columns=["uhrzeit", "Straße Start", "Nr Start", 
        "Stadt Start", "Lat Start", 
        "Lon Start", "Straße Ziel", "Nr Ziel", 
        "Stadt Ziel", "Lat Ziel", "Lon Ziel",
        "OSRM Dauer", "OSRM Distanz"])

x = df["OSRM Dauer"]
noise = traffic_dist([x]).ravel()
y = np.array([x[i] * noise[i] for i in range(len(x))])
df2["y"] = y
print(df2.head())

writer = pd.ExcelWriter("dataset.xlsx")
df2.to_excel(writer, "Sheet1")
writer.save()