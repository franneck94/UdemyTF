#### Schleifen (für Listen) in Python ####

noten = [1, 1, 3, 4, 5, 6, 2, 1]

for i in noten:
    print(i)

print("\n")

# len(noten) = 8
# range(8)
for i in range(len(noten)):
    print(noten[i])

print("\n")

noten = [1, 1, 3, 4, 5, 6, 2, 1]
fächer = ["Mathe", "Deutsch", "Englisch", "Reli", "Sport", "Kunst", "Informatik", "Geschichte"]

for note, fach in zip(noten, fächer):
    print(note, " - ", fach)

print("\n")

präferenzen = ["Mathe", "Deutsch", "Englisch", "Reli", "Sport", "Kunst", "Informatik", "Geschichte"]

for index, fach in enumerate(präferenzen):
    print("Das Fach: ", fach, " ist an Stelle: ", index+1)