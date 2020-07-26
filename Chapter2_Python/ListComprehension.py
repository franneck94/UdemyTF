my_list = []

for i in range(10):
    my_list.append(i)

print(my_list)

# List Comprehension
#           val - iterable
my_list2 = [i for i in range(10)]
print(my_list2)

my_list3 = [i**2 for i in range(10)]
print(my_list3)

# Multi-dim List
# Matrix (3 rows, 2 columns)
M = [[1, 2], 
     [3, 4],
     [5, 6]]
print(M)

NUM_ROWS = 2
NUM_COLS = 3
M2 = [[i+j for j in range(NUM_COLS)] for i in range(NUM_ROWS)]
print(M2)
