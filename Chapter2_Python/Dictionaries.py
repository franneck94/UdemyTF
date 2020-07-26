#### Dicts in Python ####

# key: value
noten_klasse_8a = {"armin": 1, "ben": 2, "jan": 1}

armins_note = noten_klasse_8a["armin"]
print("Armins Note: ", armins_note)

print("\n")

for schüler, note in noten_klasse_8a.items():
    print(schüler, " hat als Note eine ", note)

print("\n")

for schüler in noten_klasse_8a.keys():
    print(schüler, " hat als Note eine ", noten_klasse_8a[schüler])

print("\n")

for note in noten_klasse_8a.values():
    print(" hat als Note eine ", note)