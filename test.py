import torch
from mamba_ssm import Mamba, Mamba2

# Test for Mamba
batch, length, dim = 2, 64, 16
x = torch.rand(batch, length, dim).to("cuda")
model = Mamba(
    d_model=dim,
    d_state=16,
    d_conv=4,
    expand=2,
).to("cuda")

y = model(x)
print("Mamba result", y.shape)
assert y.shape == x.shape
print("Mamba works!")

# Test for Mamba2
batch, length, dim = 2, 64, 512
x = torch.rand(batch, length, dim).to("cuda")
model = Mamba2(
    d_model=dim,
    d_state=64,
    d_conv=4,
    expand=2,
).to("cuda")

y = model(x)
print("Mamba2 result", y.shape)
assert y.shape == x.shape
print("Mamba2 works!")