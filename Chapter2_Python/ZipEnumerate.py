list_a = [10, 20, 30]
list_b = ["Jan", "Peter", "Max"]
list_c = [True, False, True]

for val_a, val_b, val_c in zip(list_a, list_b, list_c):
    print(val_a, val_b, val_c)

print("\n")

for i in range(len(list_a)):
    print(i, list_a[i])

print("\n")

for i, val in enumerate(list_a):
    print(i, val)
