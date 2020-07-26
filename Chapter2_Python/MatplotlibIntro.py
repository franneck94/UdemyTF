import matplotlib.pyplot as plt

noten_jan = [56, 64, 78, 100]
noten_ben = [86, 94, 98, 90]

# Plot
#             x,            y
plt.plot([1, 2, 3, 4], noten_jan, color="blue")
plt.plot([1, 2, 3, 4], noten_ben, color="red")
plt.legend(["Jan", "Ben"])
plt.xlabel("Fach")
plt.ylabel("Note in %")
plt.title("Jan vs. Ben")
plt.show()

# Scatter Plot
#               x            y
plt.scatter([1, 2, 3, 4], noten_jan, color="blue")
plt.scatter([1, 2, 3, 4], noten_ben, color="red")
plt.legend(["Jan", "Ben"])
plt.xlabel("Fach")
plt.ylabel("Note in %")
plt.title("Jan vs. Ben")
plt.show()
