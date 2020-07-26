#### Schleifen in Python ####

# i = 0, 1, 2, 3, 4
for i in range(5):
    print("Das ist der ", i, " Durchlauf!")

print("\n")

# 1, 3, 5, 7, 9
for i in range(1, 10, 2):
    print("Das ist der ", i, " Durchlauf!")

print("\n")

# Stop (Start=0, Step=1)
# Start, Stop (Step=1)
# Start, Stop, Step
for i in range(1, 10, 2):
    print("Das ist der ", i, " Durchlauf!")

print("\n")

bin_ich_pleite = False
kontostand = 10

while bin_ich_pleite == False:
    print("Ich bin nicht pleite!", kontostand)
    kontostand -= 1
    if kontostand <= 0:
        bin_ich_pleite = True