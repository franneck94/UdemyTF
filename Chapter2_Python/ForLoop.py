grades = [1, 2, 1, 4]

num_elements = len(grades)
for idx in range(num_elements): # range(num_elements) => [0, 1, ..., num_elements-1]
    print(grades[idx])

print("\n")

for idx in range(1, 10, 1): # range(start, end, step) => [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(idx)

print("\n")

for idx in range(1, 10, 2): # range(start, end, step) => [1, 3, 5, 7, 9]
    print(idx)

print("\n")

for grade in grades:
    print(grade)
