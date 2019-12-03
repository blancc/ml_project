# Main file: dataloader instanciation, model training, evaluation
# TODO: data parallel + device + todos & pass + verbose
import wandb
import models
import data
import torch

from utils import compute_accuracy

wandb.init(project='ml_project', entity='no_name')

LEARNING_RATE = wandb.config.learning_rate
NB_EPOCHS = wandb.config.n_epochs
MODEL = wandb.config.model
DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

train_loader, valid_loader = data.get_loaders()

model = models.Net(MODEL)
model.to(DEVICE)

wandb.watch(model)

FILE_NAME = f"{model.name}_{LEARNING_RATE}_{NB_EPOCHS}"

model.load_state_dict(torch.load(f"weights/{FILE_NAME}.pt"))

criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for i in range(NB_EPOCHS):
    model.train()
    for i_batch, batch in enumerate(train_loader):

        X, y = batch
        X.to(DEVICE)
        y.to(DEVICE)

        optimizer.zero_grad()

        Y = model(X)
        loss = criterion(Y, y)

        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss})

    train_accuracy = compute_accuracy(model, train_loader)
    test_accuracy = compute_accuracy(model, valid_loader)

    wandb.log({"train_accuracy": train_accuracy})
    wandb.log({"test_accuracy": test_accuracy})
    print(
        f"Epoch: {i}/{NB_EPOCHS}\n\tTrain accuracy: {train_accuracy}\n\tTest accuracy: {test_accuracy}\n\n")

    torch.save(model.state_dict(), f"weights/{FILE_NAME}.pt")
