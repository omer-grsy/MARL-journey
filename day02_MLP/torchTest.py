import torch

# tensor oluştur
x = torch.tensor([1, 2, 3])
print(x)

# GPU var mı kontrol et
print("GPU var mı:", torch.cuda.is_available())