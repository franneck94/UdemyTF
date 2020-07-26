def list_max(my_list):
    result = my_list[0]

    for i in range(1, len(my_list)):
        if my_list[i] > result:
            result = my_list[i]

    return result
    

l1 = [-2, 1, 2, -10, 22, -10]
l1_max = list_max(l1)
print(l1_max)

l2 = [-20, 123, 112, -10, 22, -120]
l2_max = list_max(l2)
print(l2_max)
