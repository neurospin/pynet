"""
Beta VAE disentangling
======================

Credit: A Grigis
"""

# Imports
import os
import sys
if "CI_MODE" in os.environ:
    sys.exit()
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from torch.distributions import Normal, kl_divergence
from pynet import NetParameters
from pynet.datasets import DataManager
from pynet.datasets.dsprites import DSprites
from pynet.interfaces import VAEEncoder
from pynet.plotting import Board, update_board
from pynet.models.vae.losses import get_loss
from pynet.models.vae.utils import reconstruct_traverse, make_mosaic_img, add_labels


# Global parameters
WDIR = "/tmp/beta_vae_disentangling"
BATCH_SIZE = 64
N_EPOCHS = 30
ADAM_LR = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DISPLAY = False

# Load the data
dataset = DSprites(WDIR)
manager = DataManager.from_dataset(
    train_dataset=dataset, batch_size=BATCH_SIZE, sampler="random")


# Test different losses

loss_params = {
    "betah": {"beta": 4, "steps_anneal": 0},
    "betab": {"C_init": 0.5, "C_fin": 25, "gamma": 100,
              "steps_anneal": 100000},
    "btcvae": {"dataset_size": len(dataset), "alpha": 1, "beta": 1, "gamma": 6,
               "is_mss": True, "steps_anneal": 0}
}


def plot_losses(cache, filename):
    if "kl" not in cache or "ll" not in cache:
        return
    ll = np.asarray(cache["ll"]).squeeze()
    kl = np.asarray(cache["kl"]).squeeze()
    fig, axs = plt.subplots(nrows=1, ncols=2)
    colors = list(mcolors.TABLEAU_COLORS.keys())
    for idx, dim_kl in enumerate(kl.T):
        axs[0].plot(
            dim_kl, color=colors[idx], label="dim{0}".format(idx + 1))
        axs[0].set_xlabel("Training iterations")
        axs[0].set_ylabel("KL")
        axs[1].plot(
            ll, dim_kl, color=colors[idx], label="dim{0}".format(idx + 1))
        axs[1].set_xlabel("Log Likelihood")
        axs[1].set_ylabel("KL")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(filename)


def plot_reconstructions(model, data, checkpointdir, filename=None):
    weights_files = glob.glob(os.path.join(checkpointdir, "*.pth"))
    n_plots = len(weights_files)
    original = data.cpu().numpy()
    original = np.expand_dims(original, axis=0)
    stages = [original]
    labels = ["orig"]
    for idx, path in enumerate(sorted(weights_files)):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model"])
        reconstruction = model.reconstruct(data, sample=False)
        reconstruction = np.expand_dims(reconstruction, axis=0)
        stages.append(reconstruction)
        labels.append("rec stage {0}".format(idx + 1))
    concatenated = np.concatenate(stages, axis=0)
    mosaic = make_mosaic_img(concatenated)
    concatenated = Image.fromarray(mosaic)
    concatenated = add_labels(concatenated, labels)
    if filename is not None:
        concatenated.save(filename)
    return concatenated


for loss_name in ("betah", "betab", "btcvae"):

    # Train the model
    checkpointdir = os.path.join(WDIR, "checkpoints", loss_name)
    if not os.path.isdir(checkpointdir):
        os.makedirs(checkpointdir)
    weights_filename = os.path.join(
        checkpointdir, "model_0_epoch_{0}.pth".format(N_EPOCHS))
    params = NetParameters(
        input_channels=1,
        input_dim=DSprites.img_size,
        conv_flts=[32, 32, 32, 32],
        dense_hidden_dims=[256, 256],
        latent_dim=10,
        noise_out_logvar=-3,
        noise_fixed=False,
        act_func=None,
        dropout=0,
        sparse=False)
    loss = get_loss(loss_name=loss_name, **loss_params[loss_name])
    if os.path.isfile(weights_filename):
        vae = VAEEncoder(
            params,
            optimizer_name="Adam",
            learning_rate=ADAM_LR,
            loss=loss,
            use_cuda=(DEVICE.type == "cuda"),
            pretrained=weights_filename)
    else:
        vae = VAEEncoder(
            params,
            optimizer_name="Adam",
            learning_rate=ADAM_LR,
            loss=loss,
            use_cuda=(DEVICE.type == "cuda"))
        vae.board = Board(
            port=8097, host="http://localhost", env="beta-vae")
        vae.add_observer("after_epoch", update_board)
        train_history, valid_history = vae.training(
            manager=manager,
            nb_epochs=(N_EPOCHS + 1),
            checkpointdir=checkpointdir,
            fold_index=0,
            with_validation=False,
            save_after_epochs=10)
        plot_losses(vae.loss.cache,
                    os.path.join(WDIR, "loss_{0}.png".format(loss_name)))
    print(vae.model)

    # Display results
    index = np.arange(len(dataset))
    np.random.shuffle(index)
    data = torch.unsqueeze(torch.from_numpy(
        dataset.imgs[index][:100].astype(np.float32)), dim=1).to(DEVICE)
    vae.model.eval()
    name = "traverse_posteriror_{0}".format(loss_name)
    filename = os.path.join(WDIR, "{0}.png".format(name))
    mosaic_traverse = reconstruct_traverse(
        vae.model, data, n_per_latent=8, n_latents=None, is_posterior=True,
        filename=filename)
    filename = os.path.join(
        WDIR, "reconstruction_stages_{0}.png".format(loss_name))
    plot_reconstructions(vae.model, data[:8], checkpointdir, filename=filename)

    if DISPLAY:
        plt.figure()
        plt.imshow(np.asarray(mosaic_traverse))
        plt.title(name)
        plt.axis("off")

if DISPLAY:
    plt.show()
