"""
Generation of 3D brain MRI using VAE Generative Adversial Networks
==================================================================

Credit: A Grigis

Based on:

- https://github.com/cyclomon/3dbraingen

This tutorial is for the intuition of simple Generative Adversarial Networks
(GAN) for generating  realistic  MRI images. Here, we propose a model that
can successfully generate 3D brain MRI data by integrating a code
discriminator.

Let's begin with importing stuffs:
"""

import os
import sys
if "CI_MODE" in os.environ:
    sys.exit()

import numpy as np
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
from pynet.datasets import DataManager, fetch_brats
from pynet.interfaces import DeepLearningInterface
from pynet.plotting import Board, update_board
from pynet.utils import setup_logging
from pynet.preprocessing.spatial import downsample
from pynet.models import BGDiscriminator, BGGenerator, BGCodeDiscriminator


# Global parameters
logger = logging.getLogger("pynet")
setup_logging(level="debug")


#############################################################################
# The model will be trained on BRATS
#
# We will train the model to synthesize brain disorder MRI data (Glioma).

data = fetch_brats(
    datasetdir="/neurospin/nsap/processed/deepbrain/tumor/data/brats")
batch_size = 4

def transformer(data, imgtype="flair"):
    typemap = {
        "t1": 0, "t1ce": 1, "t2": 2, "flair": 3}
    if imgtype is None:
        imgtype = range(4)
    else:
        if not isinstance(imgtype, list):   
            imgtype = [imgtype]
        imgtype = [typemap[key] for key in imgtype]
    transformed_data = []
    for channel_id in range(len(data)):
        if channel_id not in imgtype:
            continue
        arr = data[channel_id]
        transformed_data.append(downsample(arr, scale=3))
    return np.asarray(transformed_data)

manager = DataManager(
    input_path=data.input_path,
    metadata_path=data.metadata_path,
    stratify_label="grade",
    number_of_folds=10,
    batch_size=batch_size,
    test_size=0,
    input_transforms=[transformer],
    sample_size=0.2)


########################
# Loss
# ----

def calc_gradient_penalty(model, x, x_gen, w=10):
    """ WGAN-GP gradient penalty.
    """
    assert (x.size() == x_gen.size()), "Real and sampled sizes do not match."
    alpha_size = tuple((len(x), *(1, ) * (x.dim() - 1)))
    alpha_t = torch.cuda.FloatTensor if x.is_cuda else torch.Tensor
    alpha = alpha_t(*alpha_size).uniform_()
    x_hat = x.data * alpha + x_gen.data * (1 - alpha)
    x_hat = Variable(x_hat, requires_grad=True)

    def eps_norm(x):
        x = x.view(len(x), -1)
        return (x * x + eps).sum(-1).sqrt()

    def bi_penalty(x):
        return (x - 1)**2

    grad_xhat = torch.autograd.grad(
        model(x_hat).sum(), x_hat, create_graph=True, only_inputs=True)[0]

    penalty = w * bi_penalty(eps_norm(grad_xhat)).mean()

    return penalty

criterion_bce = nn.BCELoss()
criterion_l1 = nn.L1Loss()
criterion_mse = nn.MSELoss()

########################
# Training
# --------
#
# We'll train the encoder, generator and discriminator to optimize the losses 
# using Adam optimizer.

def infinite_train_generartor(data_loader):
    while True:
        for _, data in enumerate(data_loader):
            yield data.inputs

latent_dim = 1000
use_cuda = False
channels = 1
in_shape = (50, 64, 45) # (150, 190, 135)
beta = 10
eps = 1e-15
device = torch.device("cuda" if use_cuda else "cpu")
generator = BGGenerator(
    in_shape=in_shape, out_channels=channels, start_filts=64,
    latent_dim=latent_dim, mode="trilinear", with_code=True).to(device)
code_discriminator = BGCodeDiscriminator(
    out_channels=channels, code_size=latent_dim, n_units=4096).to(device)
discriminator = BGDiscriminator(
    in_shape=in_shape, in_channels=channels, out_channels=channels,
    start_filts=64, with_logit=False).to(device)
encoder = BGDiscriminator(
    in_shape=in_shape, in_channels=channels, out_channels=latent_dim,
    start_filts=64, with_logit=False).to(device)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
cd_optimizer = torch.optim.Adam(code_discriminator.parameters(), lr=0.0002)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
e_optimizer = torch.optim.Adam(encoder.parameters(), lr = 0.0002)
real_y = Variable(torch.ones((batch_size, channels)).to(
    device, non_blocking=True))
fake_y = Variable(torch.zeros((batch_size, channels)).to(
    device, non_blocking=True))
board = Board(port=8097, host="http://localhost", env="vae")
outdir = "/tmp/vae-gan/checkpoint"
if not os.path.isdir(outdir):
    os.makedirs(outdir)

g_iter = 1
d_iter = 1
cd_iter = 1
total_iter = 200000
train_loader = manager.get_dataloader(train=True, validation=False,
                                      fold_index=0).train
loader = infinite_train_generartor(train_loader)

for iteration in range(total_iter):

    # Train Encoder - Generator
    for model, with_grad in [(discriminator, False),
                             (code_discriminator, False),
                             (encoder, True),
                             (generator, True)]:
        for param in model.parameters():  
            param.requires_grad = with_grad

    for iters in range(g_iter):
        generator.zero_grad()
        encoder.zero_grad()
        real_images = loader.__next__()
        batch_size = real_images.size(0)
        real_images = Variable(real_images, volatile=True).to(
            device, non_blocking=True)
        z_rand = Variable(
            torch.randn((batch_size,latent_dim)), volatile=True).to(device)
        z_hat = encoder(real_images).view(batch_size, -1)
        x_hat = generator(z_hat)
        x_rand = generator(z_rand)
        c_loss = - code_discriminator(z_hat).mean()

        d_real_loss = discriminator(x_hat).mean()
        d_fake_loss = discriminator(x_rand).mean()
        d_loss = - d_fake_loss - d_real_loss
        l1_loss = 10 * criterion_l1(x_hat, real_images)
        loss1 = l1_loss + c_loss + d_loss

        if iters < (g_iter - 1):
            loss1.backward()
        else:
            loss1.backward(retain_graph=True)
        e_optimizer.step()
        g_optimizer.step()
        g_optimizer.step()

    # Train discriminator
    for model, with_grad in [(discriminator, True),
                             (code_discriminator, False),
                             (encoder, False),
                             (generator, False)]:
        for param in model.parameters():  
            param.requires_grad = with_grad

    for iters in range(d_iter):
        d_optimizer.zero_grad()
        real_images = loader.__next__()
        batch_size = real_images.size(0)
        z_rand = Variable(
            torch.randn((batch_size, latent_dim)),volatile=True).to(device)
        real_images = Variable(real_images, volatile=True).to(
            device, non_blocking=True)
        z_hat = encoder(real_images).view(batch_size,-1)
        x_hat = generator(z_hat)
        x_rand = generator(z_rand)
        x_loss2 = (-2 * discriminator(real_images).mean() +
                   discriminator(x_hat).mean() +
                   discriminator(x_rand).mean())
        gradient_penalty_r = calc_gradient_penalty(
            discriminator, real_images.data, x_rand.data)
        gradient_penalty_h = calc_gradient_penalty(
            discriminator, real_images.data, x_hat.data)

        loss2 = x_loss2 + gradient_penalty_r + gradient_penalty_h
        loss2.backward(retain_graph=True)
        d_optimizer.step()

    # Train code discriminator
    for model, with_grad in [(discriminator, False),
                             (code_discriminator, True),
                             (encoder, False),
                             (generator, False)]:
        for param in model.parameters():  
            param.requires_grad = with_grad

    for iters in range(cd_iter):
        cd_optimizer.zero_grad()
        z_rand = Variable(
            torch.randn((batch_size, latent_dim)), volatile=True).to(device)
        gradient_penalty_cd = calc_gradient_penalty(
            code_discriminator, z_hat.data, z_rand.data)
        loss3 = (- code_discriminator(z_rand).mean() -
                 c_loss + gradient_penalty_cd)

        loss3.backward(retain_graph=True)
        cd_optimizer.step()

    # Visualization 
    if iteration % 4 == 0:
        print("[{0}/{1}]".format(iteration, total_iter),
              "D: {:<8.3}".format(loss2.item()), 
              "En Ge: {:<8.3}".format(loss1.item()),
              "Code: {:<8.3}".format(loss3.item()))
        
        for name, data in [("X_real", real_images), ("X_dec", x_hat),
                           ("X_rand", x_rand)]:
            featmask = (0.5 * data[0] + 0.5).data.cpu().numpy()
            img = featmask[..., featmask.shape[-1] // 2]
            img = np.expand_dims(img, axis=1)
            img = (img / img.max()) * 255
            board.viewer.images(
                img,
                opts={
                    "title": name,
                    "caption": name},
                win=name)

    # Save model
    if (iteration + 1) % 100 == 0: 
        for name, model in [("generator", generator),
                            ("code_discriminator", code_discriminator),
                            ("discriminator", discriminator),
                            ("encoder", encoder)]:
            fname = os.path.join(
                outdir, name + "_epoch_" + str(iteration + 1) + ".pth")
            torch.save(model.state_dict(), fname)


#############################################################################
# Conclusion
# ----------
#
# Variational Auto-Encoder(VAE) GAN are free from mode collapse but outputs
# are characterized with blurriness. In order to effectively address the
# problems of both mode collapse of GANs and blurriness of VAEs, we will
# use α-GAN, a solution born by combining both models, in the next tutorial.

