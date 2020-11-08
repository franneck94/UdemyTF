name = "Jan"
age = 26
gender = "male"

message = "Hello my name is " + name + " i am " + str(age) + " years old and i am a " + gender + " person"
print(message)

# %-formatter
message = "Hello my name is %s i am %i years old and i am a %s person" % (name, age, gender)
print(message)

# .format
message = "Hello my name is {} i am {} years old and i am a {} person".format(name, age, gender)
print(message)

# f-String
message = f"Hello my name is {name} i am {age} years old and i am a {gender} person"
print(message)
