import wandb
import torch

wandb.init(project='ml_project', entity='no_name')

LEARNING_RATE = wandb.config.learning_rate
NB_EPOCHS = wandb.config.n_epochs
MODEL = wandb.config.model
DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Running on {DEVICE}")
BATCH_SIZE = wandb.config.batch_size

FILE_NAME = f"{MODEL}_{LEARNING_RATE}_{NB_EPOCHS}"
