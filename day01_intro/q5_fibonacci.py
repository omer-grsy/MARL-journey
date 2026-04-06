
# TODO: fibonacci fonksiyonu yaz
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

n= 6
print(fibonacci(n)) ######### aynı hesaplar tekrar eder yavaş çalışır

############################
# Hızlı versiyon
# from functools import lru_cache
#
# @lru_cache(None)
# def fibonacci(n):
#     if n < 2:
#         return n
#     return fibonacci(n-1) + fibonacci(n-2)
#
######################################################
# En iyi çözüm
def fibonacci_best(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print(fibonacci_best(6))