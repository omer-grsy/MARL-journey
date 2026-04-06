d = {"a": 12, "b": 45, "c": 7, "d": 30}

# TODO: en büyük değeri bul
max_dict = {key: value for key, value in d.items() if value == max(d.values())}
print(max_dict)

##############################
max_key = max(d, key=d.get)  # d.get value döndürür buna göre karşılatırma yapar
                            # ve max(value)'ye karşılık gelen key'i döndürür
print(max_key, d[max_key])