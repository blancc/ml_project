# Main file: dataloader instanciation, model training, evaluation
# TODO: data parallel + normalisation + hidden sizes ? + flatten direct
# Ideas: binaryce +
import models
import data
import torch
import os
from tqdm import tqdm
import wandb
import numpy as np
import matplotlib.pyplot as plt
from utils import compute_accuracy, visualize_noise
from setup import (BATCH_SIZE, MODEL, DEVICE, LEARNING_RATE, NB_EPOCHS,
                   FILE_NAME, DROPOUT, WEIGHTS_INIT, ALPHA, KEEP_WEIGHTS, SIGNAL_LENGTH, SUBSAMPLE)


train_loader, test_loader = data.get_loaders(BATCH_SIZE)

model = models.DeNoiser(dropout=DROPOUT)
if WEIGHTS_INIT:
    model.apply(models.init_weights)
model.to(DEVICE)

wandb.watch(model)

if KEEP_WEIGHTS and os.path.exists(f"weights/{FILE_NAME}.pt"):
    print("Weights found")
    model.load_state_dict(torch.load(f"weights/{FILE_NAME}.pt"))
else:
    os.makedirs("weights/", exist_ok=True)
    print("Weights not found")

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

for i in range(NB_EPOCHS):
    model.train()
    print("Training")
    train_loss = 0
    for batch in tqdm(train_loader):
        X, y = batch
        X = X.to(DEVICE)
        y = y.to(DEVICE).squeeze()

        optimizer.zero_grad()

        Y = model(X)

        loss = criterion(Y, y)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        wandb.log({"loss": loss})

    train_loss = train_loss/len(train_loader)

    print("Evaluation")
    model.eval()
    test_loss = 0
    for batch in tqdm(test_loader):
        X, y = batch
        X = X.to(DEVICE)
        y = y.to(DEVICE).squeeze()

        Y = model(X)

        loss = criterion(Y, y)

        test_loss += loss.item()

    test_loss = test_loss/len(test_loader)

    wandb.log({"train_loss": train_loss})
    wandb.log({"test_loss": test_loss})
    wandb.log({"train_results": wandb.Image(visualize_noise(model, train_loader))})
    wandb.log({"test_results": wandb.Image(visualize_noise(model, test_loader))})
    print(
        f"Epoch: {i}/{NB_EPOCHS}\n\tTrain loss: {train_loss}\n\tTest loss: {test_loss}\n\n")

    torch.save(model.state_dict(), f"weights/{FILE_NAME}.pt")
