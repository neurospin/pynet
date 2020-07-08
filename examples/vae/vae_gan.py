"""
Generation of 3D brain MRI using VAE Generative Adversial Networks
==================================================================

Credit: A Grigis

Based on:

- https://github.com/cyclomon/3dbraingen

This tutorial is for the intuition of simple Generative Adversarial Networks
(GAN) for generating  realistic  MRI images. Here, we propose a novel that
can successfully generate 3D brain MRI data from random vectors by learning
the data distribution.
After reading this tutorial, you'll understand the technical details needed to
implement VAE-GAN.

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
from pynet.models import BGDiscriminator, BGEncoder, BGGenerator


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

criterion_bce = nn.BCELoss()
criterion_l1 = nn.L1Loss()


########################
# Training
# --------
#
# We'll train the encoder, generator and discriminator to optimize the losses 
# using Adam optimizer.

n_epochs = 100
latent_dim = 1000
use_cuda = False
channels = 1
in_shape = (50, 64, 45) # (64, 64, 64) # (150, 190, 135)
gamma = 20
beta = 10
device = torch.device("cuda" if use_cuda else "cpu")
generator = BGGenerator(
    in_shape=in_shape, out_channels=channels, start_filts=64,
    latent_dim=latent_dim, mode="trilinear", with_code=False).to(device)
discriminator = BGDiscriminator(
    in_shape=in_shape, in_channels=channels, out_channels=channels,
    start_filts=64, with_logit=True).to(device)
encoder = BGEncoder(
    in_shape=in_shape, in_channels=channels, start_filts=64,
    latent_dim=latent_dim).to(device)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
e_optimizer = torch.optim.Adam(encoder.parameters(), lr = 0.0001)
real_y = Variable(torch.ones((batch_size, channels)).to(device))
fake_y = Variable(torch.zeros((batch_size, channels)).to(device))
board = Board(port=8097, host="http://localhost", env="vae")
outdir = "/tmp/vae-gan/checkpoint"
if not os.path.isdir(outdir):
    os.makedirs(outdir)

for epoch in range(n_epochs):
    loaders = manager.get_dataloader(train=True, validation=False,
                                     fold_index=0)
    for iteration, item in enumerate(loaders.train):
        real_images = item.inputs.to(device)
        batch_size = real_images.size(0)
        real_images = Variable(real_images,requires_grad=False).to(device)
        z_rand = Variable(torch.randn(
            (batch_size, latent_dim)), requires_grad=False).to(device)
        mean, logvar, code = encoder(real_images)
        x_rec = generator(code)
        x_rand = generator(z_rand)
        logger.debug("X_real: {0}".format(real_images.shape))
        logger.debug("X_rand: {0}".format(x_rand.shape))
        logger.debug("X_rec: {0}".format(x_rec.shape))

        # Train discriminator 
        d_optimizer.zero_grad()
        d_real_loss = criterion_bce(
            discriminator(real_images), real_y[:batch_size])
        d_recon_loss = criterion_bce(discriminator(x_rec), fake_y[:batch_size])
        d_fake_loss = criterion_bce(discriminator(x_rand), fake_y[:batch_size])
        dis_loss = d_recon_loss + d_real_loss + d_fake_loss
        dis_loss.backward(retain_graph=True)
        d_optimizer.step()
        
        # Train generator
        g_optimizer.zero_grad()
        output = discriminator(real_images)
        d_real_loss = criterion_bce(output, real_y[:batch_size])
        output = discriminator(x_rec)
        d_recon_loss = criterion_bce(output, fake_y[:batch_size])
        output = discriminator(x_rand)
        d_fake_loss = criterion_bce(output, fake_y[:batch_size])
        d_img_loss = d_real_loss + d_recon_loss + d_fake_loss
        gen_img_loss = -d_img_loss
        rec_loss = ((x_rec - real_images)**2).mean()
        err_dec = gamma * rec_loss + gen_img_loss
        err_dec.backward(retain_graph=True)
        g_optimizer.step()

        # Train encoder
        prior_loss = 1 + logvar-mean.pow(2) - logvar.exp()
        prior_loss = (-0.5 * torch.sum(prior_loss)) / torch.numel(mean.data)
        err_enc = prior_loss + beta * rec_loss
        e_optimizer.zero_grad()
        err_enc.backward()
        e_optimizer.step()

        # Visualization 
        if iteration % 4 == 0:
            print("[{0}/{1}]".format(epoch, n_epochs),
                  "D: {:<8.3}".format(dis_loss.item()), 
                  "En: {:<8.3}".format(err_enc.item()),
                  "De: {:<8.3}".format(err_dec.item()))
            
            for name, data in [("X_real", real_images), ("X_dec", x_rec),
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

    for name, model in [("generator", generator),
                        ("discriminator", discriminator),
                        ("encoder", encoder)]:
        fname = os.path.join(
            outdir, name + "_epoch_" + str(epoch + 1) + ".pth")
        torch.save(model.state_dict(), fname)


#############################################################################
# Conclusion
# ----------
#
# Variational Auto-Encoder(VAE) GAN are free from mode collapse but outputs
# are characterized with blurriness. In order to effectively address the
# problems of both mode collapse of GANs and blurriness of VAEs, we will
# use α-GAN, a solution born by combining both models, in the next tutorial.

