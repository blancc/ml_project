# Main file: dataloader instanciation, model training, evaluation
import models
import data
import torch
import os
from tqdm import tqdm
import wandb
import numpy as np
import matplotlib.pyplot as plt
from utils import compute_accuracy, visualize_predictions
from setup import (BATCH_SIZE, MODEL, DEVICE, LEARNING_RATE, NB_EPOCHS,
                   FILE_NAME, DROPOUT, WEIGHTS_INIT, ALPHA, KEEP_WEIGHTS, DENOISER)


train_loader, test_loader = data.get_loaders(BATCH_SIZE)

model = models.Net(MODEL, dropout=DROPOUT)
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

if DENOISER:
    denoiser = models.DeNoiser()
    denoiser.to(DEVICE)
    denoiser.load_state_dict(torch.load("weights/Noise.pt"))
    for p in denoiser.parameters():
        p.requires_grad = False
    denoiser.eval()
else:
    denoiser = None


# criterion = torch.nn.MSELoss()


def criterion(x, y): return ((x-y)**2-ALPHA*x**2).mean()


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for i in range(NB_EPOCHS):
    model.train()
    print("Training")
    for batch in tqdm(train_loader):
        X, y = batch
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        if DENOISER:
            Y = denoiser(X)
        Y = model(Y)

        loss = criterion(Y, torch.flatten(y, 1))

        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss})

    print("Evaluation")
    train_accuracy = compute_accuracy(model, train_loader, denoiser)
    test_accuracy = compute_accuracy(model, test_loader, denoiser)

    wandb.log({"train_accuracy": train_accuracy})
    wandb.log({"test_accuracy": test_accuracy})
    wandb.log({"train_results": wandb.Image(visualize_predictions(model, train_loader, denoiser))})
    wandb.log({"test_results": wandb.Image(visualize_predictions(model, test_loader, denoiser))})
    print(
        f"Epoch: {i}/{NB_EPOCHS}\n\tTrain accuracy: {train_accuracy}\n\tTest accuracy: {test_accuracy}\n\n")

    torch.save(model.state_dict(), f"weights/{FILE_NAME}.pt")
