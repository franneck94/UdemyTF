import numpy as np

noten_py_list = [100, 89, 44, 78, 45, 24, 18]
noten_np_array = np.array(noten_py_list, dtype=np.int8)

print(noten_py_list)
print(noten_np_array)

noten_max = np.max(noten_np_array)
noten_min = np.min(noten_np_array)

print(noten_max)
print(noten_min)

# [100, 89, 44, 78, 45, 24, 18]
noten_arg_max = np.argmax(noten_np_array)
noten_arg_min = np.argmin(noten_np_array)

print(noten_arg_max)
print(noten_arg_min)

noten_mean = np.mean(noten_np_array)
noten_median = np.median(noten_np_array)

print(noten_mean)
print(noten_median)
