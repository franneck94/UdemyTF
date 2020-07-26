#### Abfragen und Logik in Python ####

bin_ich_pleite = None
bin_ich_reich = None

kontostand = 0

if kontostand > 0:
    bin_ich_pleite = False
elif kontostand == 0:
    print("Mies gelaufen.")
    bin_ich_pleite = True
else:
    bin_ich_pleite = True

print("Bin ich pleite?", bin_ich_pleite)