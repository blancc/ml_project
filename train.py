# Main file: dataloader instanciation, model training, evaluation
# TODO: data parallel + device + todos & pass + verbose
import wandb
import models
import data
import torch

wandb.init(project='ml_project', entity='no_name')

train_loader, valid_loader = data.get_loaders()

model = models.Net()

criterion = None  # TODO
optimizer = None  # TODO

for i in range(wandb.config.n_epochs):
    model.train()
    for X, y in train_loader:
        pass

    model.eval()
    for X, y in valid_loader:
        with torch.no_grad():
            pass
