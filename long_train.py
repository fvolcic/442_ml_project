
import importlib

import constants
import models.ENet
import torch
import utils.trainers.unet_train

importlib.reload(utils.trainers.unet_train)
from dataloaders.NYU_loader import get_dataloaders
from torch import nn as nn


device = constants.DEVICE

# get the dataloaders
test_dl, train_dl = get_dataloaders(batch_size=16)

# get the model
model = models.ENet.ENet(13, 3).to(device=device)

# create an optimizer 
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# create a loss function
loss_func = nn.CrossEntropyLoss().to(device=device) 

# make a lr autodecay 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=0.01, factor=0.1, patience=10, verbose=True)

train_Unet = utils.trainers.unet_train.train_Unet

losses_total = [] 

save_model_every = 1

data = next(iter(train_dl))

in_data = [data]

for i in range(10000):
    print(f"Running epoch {i}")

    losses = train_Unet(model, train_dl, optimizer=optimizer, loss_func=loss_func, num_epochs=1, show_result_every=1)
    scheduler.step(torch.Tensor(losses).mean())
    
    losses_total += losses

    # save the model
    if i % save_model_every == 0:
        torch.save(model.state_dict(), f"saved_models/ENet_{i}.pth")
    
    # Save the losses total in losses_total.txt
    with open("saved_models/E_NET_losses_total.txt", "w") as f:
        f.write(str(losses_total))
    


    