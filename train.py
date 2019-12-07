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
from utils import compute_accuracy, visualize_predictions
from setup import BATCH_SIZE, MODEL, DEVICE, LEARNING_RATE, NB_EPOCHS, FILE_NAME, DROPOUT


train_loader, valid_loader = data.get_loaders(BATCH_SIZE)

model = models.Net(MODEL, dropout=DROPOUT)
model.to(DEVICE)

wandb.watch(model)

if os.path.exists(f"weights/{FILE_NAME}.pt"):
    print("Weights found")
    model.load_state_dict(torch.load(f"weights/{FILE_NAME}.pt"))
else:
    os.makedirs("weights/", exist_ok=True)
    print("Weights not found")

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for i in range(NB_EPOCHS):
    model.train()
    print("Training")
    for batch in tqdm(train_loader):
        X, y = batch
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        Y = model(X)

        loss = criterion(Y, torch.flatten(y, 1))

        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss})

    print("Evaluation")
    train_accuracy = compute_accuracy(model, train_loader)
    test_accuracy = compute_accuracy(model, valid_loader)

    wandb.log({"train_accuracy": train_accuracy})
    wandb.log({"test_accuracy": test_accuracy})
    wandb.log({"train_results": wandb.Image(visualize_predictions(model, train_loader))})
    wandb.log({"test_results": wandb.Image(visualize_predictions(model, valid_loader))})
    print(
        f"Epoch: {i}/{NB_EPOCHS}\n\tTrain accuracy: {train_accuracy}\n\tTest accuracy: {test_accuracy}\n\n")

    torch.save(model.state_dict(), f"weights/{FILE_NAME}.pt")
