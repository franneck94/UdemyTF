# import MyModule # Alternative

# l1 = [-2, 1, 2, -10, 22, -10]
# l1_max = MyModule.list_max(l1)
# print(l1_max)

# l2 = [-20, 123, 112, -10, 22, -120]
# l2_max = MyModule.list_max(l2)
# print(l2_max)

# from MyModule import list_max # Alternative 2

# l1 = [-2, 1, 2, -10, 22, -10]
# l1_max = list_max(l1)
# print(l1_max)

# l2 = [-20, 123, 112, -10, 22, -120]
# l2_max = list_max(l2)
# print(l2_max)

# from MyModule import * # Alternative 3

# l1 = [-2, 1, 2, -10, 22, -10]
# l1_max = list_max(l1)
# print(l1_max)

# l2 = [-20, 123, 112, -10, 22, -120]
# l2_max = list_max(l2)
# print(l2_max)

import MyModule as mm # Alternative 4

l1 = [-2, 1, 2, -10, 22, -10]
l1_max = mm.list_max(l1)
print(l1_max)

l2 = [-20, 123, 112, -10, 22, -120]
l2_max = mm.list_max(l2)
print(l2_max)
