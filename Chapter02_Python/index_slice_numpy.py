import numpy as np


l1 = [i for i in range(20)]
print(l1)

# Start, Stop, Step
l2 = l1[0:20:2]
print(l2)

# 0:Stop:1
l3 = l1[:10]
print(l3)

my_array = np.zeros(shape=(2, 2), dtype=np.int32)
print(my_array)

my_reshaped_array = np.reshape(my_array, newshape=(4,))
print(my_reshaped_array)

#                    [0, 10]
my_random_array = np.random.randint(low=0, high=11, size=20)
print(my_random_array)

my_random_array2 = np.random.uniform(low=0.0, high=10.0, size=20)
print(my_random_array2)
