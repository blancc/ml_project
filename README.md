# Machine Learning Project
*Short description*

## To get started

### Installation
1. Run `pip install -f requirements.txt` to install the required python modules.
2. Run `wandb init` and follow the instructions to configure *wandb* (module used to log the runs, [more](https://docs.wandb.com/overview)).
3. The dataset should be in the `dataset/SNR10/QPSK/` folder.

### Start a run
Simply use `python train.py` to start a run. You can configure the hyperparameters by editing the `config-defaults.yaml` file.

### Run a wandb sweep
Run `wandb sweep config-sweep.yaml` to create a sweep. It will return an identification code so you can run an agent with `wandb agent *code*`.
More informations [here](https://docs.wandb.com/library/sweeps).

## References
- [LSTM](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43905.pdf)
- [LSTM (Tuto)](https://arxiv.org/pdf/1909.09586.pdf)
- [Transformer](https://arxiv.org/pdf/1706.03762.pdf)
- [WaveNet](https://arxiv.org/pdf/1609.03499.pdf)
- [Denoising](https://papers.nips.cc/paper/4686-image-denoising-and-inpainting-with-deep-neural-networks.pdf)
- [VGGNet](https://arxiv.org/pdf/1409.1556.pdf)