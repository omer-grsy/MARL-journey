lst = [10, 15, 22, 33, 40, 55, 60]

# TODO: sadece çift sayıları al
lst_even=[]
for item in lst:
    if item % 2 == 0:
        lst_even.append(item)

print(lst_even)

############################
lst_even2 = [item for item in lst if item % 2 == 0]
print(lst_even2)