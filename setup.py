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
SUBSAMPLE = wandb.config.subsample
DROPOUT = wandb.config.dropout
WEIGHTS_INIT = wandb.config.weights_init
WORD_LENGTH = wandb.config.word_length
N_HEAD = wandb.config.n_head
N_LAYERS = wandb.config.n_layers
ALPHA = wandb.config.alpha
SIGNAL_LENGTH = wandb.config.signal_length
TARGET_LENGTH = wandb.config.target_length

FILE_NAME = f"{MODEL}"
