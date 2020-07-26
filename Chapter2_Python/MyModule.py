def list_max(my_list):
    result = my_list[0]

    for i in range(1, len(my_list)):
        if my_list[i] > result:
            result = my_list[i]

    return result