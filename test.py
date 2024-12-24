import numpy as np
import torch

t = torch.randn(4, 9, 3, 240, 432)
t = t.permute(0 , 2 , 1, 3, 4)

conv3d = torch.nn.Conv3d(in_channels=3, out_channels=1, kernel_size=(1,1,1), padding=(0,0,0))

t = conv3d(t)
print(t.shape)
t = torch.squeeze(t)
print(t.shape)