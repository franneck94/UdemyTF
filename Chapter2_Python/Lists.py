grade_ben = 1
grade_jan = 2
grade_peter = 1

#         0, 1, 2
grades = [1, 2, 1]

grade_melissa = 4

print(grades)
grades.append(grade_melissa)
print(grades)
grades.pop()
print(grades)

print("Bens grade: ", grades[0])
print("Jans grade: ", grades[1])
print("Peters grade: ", grades[2])

grades.pop(1)
print(grades)
