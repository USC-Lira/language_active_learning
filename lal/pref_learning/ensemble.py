# requires torch 2.0.0
import torch as th
import torch.nn as nn
from torch.func import stack_module_state, functional_call

import sys

import copy

vectorized = True

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2,1)
    def forward(self, x):
        return th.sigmoid(self.fc(x))


models = [Net() for _ in range(5)]
models = nn.ModuleList(models)

optimizer = th.optim.Adam(models.parameters(), lr=0.05)

if vectorized:
    base_model = copy.deepcopy(models[0])
    base_model = base_model.to('meta')
    params, buffers = stack_module_state(models)
    optimizer = th.optim.Adam(params.values(), lr=0.05)

    def fmodel(params, buffers, x):
        return functional_call(base_model, (params, buffers), x)


    for epoch in range(50):
        data = th.rand(1,2) * 2 - 1


        loss = th.vmap(fmodel, in_dims=(0, 0, None))(params, buffers, data)
        print(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

else:

    for epoch in range(50):
        data = th.rand(1,2) * 2 - 1
        data = data.to("cuda")
        for model in models:
            loss = model(data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(loss.item())