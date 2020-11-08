am_i_broke = False # bool

if am_i_broke is True:
    print("I have no money!")
else:
    print("I have money!")

my_bank_account = 1000 # int

if my_bank_account < 10000:
    print("You have to earn money!")
else:
    print("Cool for you!")

# == (Equal)
# < (Less than)
# > (Greater than)
# != (Not equal)
# <= (Less or equal than)
# >= (Greater or equal than)

my_age = 26

if my_age < 18:
    print("You are a child!")
elif my_age < 66: # else if
    print("You are an adult!")
else:
    print("You are a pensioner")
