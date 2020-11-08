names = ["Ben", "Jan", "Peter", "Melissa"]
grades = [1, 2, 1, 4]

# dict {(key, value)}
names_and_grades = {"Ben": 1, "Jan": 2, "Peter": 1, "Melissa": 4}
print(names_and_grades)

names_and_grades.update({"Pia": 3})
print(names_and_grades)

names_and_grades["Julia"] = 1
print(names_and_grades)

names_and_grades.pop("Julia")
print(names_and_grades)

print("\n")

# Keys
for element in names_and_grades:
    print(element)

print("\n")

# Keys
for k in names_and_grades.keys():
    print(k)

print("\n")

# Values
for v in names_and_grades.values():
    print(v)

print("\n")

# Keys, Values
for k, v in names_and_grades.items():
    print(k, v)

# Contains
if "Julia" in names_and_grades:
    print("Julia is present!")
else:
    print("Julia is not present")

if "Jan" in names_and_grades:
    print("Jan is present!")
else:
    print("Jan is not present")
