import models.archs.PFNL_arch as PFNL

model = PFNL.PFNL()
import torch

x = torch.randn(5,7,3,64,64)
out = model(x)
print(out.shape)