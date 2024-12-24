import numpy as np
import torch

t = torch.randn(4, 9, 3, 240, 432)
t = t.permute(0 , 2 , 1, 3, 4)




print(t.shape)
t = torch.squeeze(t)
print(t.shape)
