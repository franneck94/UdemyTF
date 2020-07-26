import matplotlib.pyplot as plt

x = [[1, 4, 3, 9],
     [3, 1, 5, 2]]

y = ['red', 'blue', 'blue', 'red']

# P1 (x1=1, x2=3), Klasse='red'
# P2 (x1=4, x2=1), Klasse='blue'
# P3 (x1=3, x2=5), Klasse='blue'
# P3 (x1=9, x2=2), Klasse='red'

x1 = x[0]
x2 = x[1]

plt.scatter(x1, x2, color=y)
#plt.show()

w = [1, 3, 6, 9, 7, 4]
print(w)
#                                START:END:STEP
w_squared = [val**2 for val in w[1:5]]
print(w_squared)

import numpy as np

w = np.array([1, 3, 6, 9, 7, 4])
print(w)
#                                START:END:STEP
w_squared = np.array([val**2 for val in w[1:5]])
print(w_squared)
