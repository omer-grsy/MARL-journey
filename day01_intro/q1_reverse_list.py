lst = [3, 7, 1, 9, 4]

# TODO: listeyi ters çevir
reversed_lst = lst[::-1]
print(lst)
print(reversed_lst)
#######################################
n=len(lst)
#print(n)
reversed_lst2= [None]*n
for i in range(n):
    reversed_lst2[i] = lst[n-1-i]

print(reversed_lst2)
