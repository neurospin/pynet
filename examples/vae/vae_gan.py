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
import torch.nn.functional as func
from torch.autograd import Variable
from pynet.datasets import DataManager, fetch_brats
from pynet.interfaces import DeepLearningInterface
from pynet.plotting import Board, update_board
from pynet.utils import setup_logging
from pynet.preprocessing.spatial import downsample


# Global parameters
logger = logging.getLogger("pynet")
setup_logging(level="info")


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
# The Model
# ---------

class Discriminator(nn.Module):
    """ This is the discriminator part of VAE-GAN.
    """
    def __init__(self, in_shape, in_channels=1, out_channels=1,
                 start_filts=64):
        """ Init class.

        Parameters
        ----------
        in_shape: uplet
            the input tensor data shape (X, Y, Z).
        in_channels: int, default 1
            number of channels in the input tensor.
        out_channels: int, default 1
            number of channels in the output tensor.
        start_filts: int, default 64
            number of convolutional filters for the first conv.
        """
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_filts = start_filts
        self.in_shape = in_shape
        self.shapes = _downsample_shape(
            self.in_shape, nb_iterations=4, scale_factor=2)
        self.conv1 = nn.Conv3d(
            self.in_channels, self.start_filts, kernel_size=4, stride=2,
            padding=1)
        self.conv2 = nn.Conv3d(
            self.start_filts, self.start_filts * 2, kernel_size=4, stride=2,
            padding=1)
        self.bn2 = nn.BatchNorm3d(self.start_filts * 2)
        self.conv3 = nn.Conv3d(
            self.start_filts * 2, self.start_filts * 4, kernel_size=4,      
            stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(self.start_filts * 4)
        self.conv4 = nn.Conv3d(
            self.start_filts * 4, self.start_filts * 8, kernel_size=4,
            stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(self.start_filts * 8)
        self.conv5 = nn.Conv3d(
            self.start_filts * 8, self.out_channels, kernel_size=self.shapes[-1],
            stride=1, padding=0)

    def forward(self, x):
        logger.debug("VAE-GAN Discriminator...")
        self.debug("input", x)
        h1 = func.leaky_relu(self.conv1(x), negative_slope=0.2)
        self.debug("conv1", h1)
        h2 = func.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        self.debug("conv2", h2)
        h3 = func.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        self.debug("conv3", h3)
        h4 = func.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        self.debug("conv4", h4)
        h5 = self.conv5(h4)
        self.debug("conv5", h5)
        output = torch.sigmoid(h5.view(h5.size(0), -1))
        self.debug("output", output)
        logger.debug("Done.")
        return output

    def debug(self, name, tensor):
        logger.debug("  {3}: {0} - {1} - {2}".format(
            tensor.shape, tensor.get_device(), tensor.dtype, name))

class Encoder(nn.Module):
    """ This is the encoder part of VAE-GAN.
    """
    def __init__(self, in_shape, in_channels=1, start_filts=64,
                 latent_dim=1000):
        """ Init class.

        Parameters
        ----------
        in_shape: uplet
            the input tensor data shape (X, Y, Z).
        in_channels: int, default 1
            number of channels in the input tensor.
        start_filts: int, default 64
            number of convolutional filters for the first conv.
        latent_dim: int, default 1000
            the latent variable sizes.
        """
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.latent_dim = latent_dim
        self.in_shape = in_shape
        self.shapes = _downsample_shape(
            self.in_shape, nb_iterations=4, scale_factor=2)
        self.dense_features = np.prod(self.shapes[-1])
        logger.debug("VAE-GAN Encoder shapes: {0}".format(self.shapes))
        self.conv1 = nn.Conv3d(
            self.in_channels, self.start_filts, kernel_size=4, stride=2,
            padding=1)
        self.conv2 = nn.Conv3d(
            self.start_filts, self.start_filts * 2, kernel_size=4, stride=2,      
            padding=1)
        self.bn2 = nn.BatchNorm3d(self.start_filts * 2)
        self.conv3 = nn.Conv3d(
            self.start_filts * 2, self.start_filts * 4, kernel_size=4,
            stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(self.start_filts * 4)
        self.conv4 = nn.Conv3d(
            self.start_filts * 4, self.start_filts * 8, kernel_size=4,
            stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(self.start_filts * 8)
        self.mean = nn.Sequential(
            nn.Linear(self.start_filts * 8 * self.dense_features, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, self.latent_dim))
        self.logvar = nn.Sequential(
            nn.Linear(self.start_filts * 8 * self.dense_features, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, self.latent_dim))
        
    def forward(self, x):
        logger.debug("VAE-GAN Encoder...")
        batch_size = x.size(0)
        logger.debug("  batch_size: {0}".format(batch_size))
        self.debug("input", x)
        h1 = func.leaky_relu(self.conv1(x), negative_slope=0.2)
        self.debug("conv1", h1)
        h2 = func.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        self.debug("conv2", h2)
        h3 = func.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        self.debug("conv3", h3)
        h4 = func.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        self.debug("conv4", h4)
        mean = self.mean(h4.view(batch_size, -1))
        self.debug("mean", mean)
        logvar = self.logvar(h4.view(batch_size, -1))
        self.debug("logvar", logvar)
        std = logvar.mul(0.5).exp_()
        reparametrized_noise = Variable(
            torch.randn((batch_size, self.latent_dim))).to(x.device)
        reparametrized_noise = mean + std * reparametrized_noise
        self.debug("reparametrization", reparametrized_noise)
        logger.debug("Done.")
        return mean, logvar, reparametrized_noise

    def debug(self, name, tensor):
        logger.debug("  {3}: {0} - {1} - {2}".format(
            tensor.shape, tensor.get_device(), tensor.dtype, name))
    
class Generator(nn.Module):
    """ This is the generator part of VAE-GAN.
    """
    def __init__(self, in_shape, out_channels=1, start_filts=64,
                 latent_dim=1000, mode="trilinear"):
        """ Init class.

        Parameters
        ----------
        in_shape: uplet
            the input tensor data shape (X, Y, Z).
        out_channels: int, default 1
            number of channels in the output tensor.
        start_filts: int, default 64
            number of convolutional filters for the first conv.
        latent_dim: int, default 1000
            the latent variable sizes.
        mode: str, default 'trilinear'
            the interpolation mode.
        """
        super(Generator, self).__init__()
        self.out_channels = out_channels
        self.start_filts = start_filts
        self.latent_dim = latent_dim
        self.in_shape = in_shape
        self.mode = mode
        self.shapes = _downsample_shape(
            self.in_shape, nb_iterations=4, scale_factor=2)
        self.dense_features = np.prod(self.shapes[-1])
        logger.debug("VAE-GAN Generator shapes: {0}".format(self.shapes))
        self.fc = nn.Linear(
            self.latent_dim, self.start_filts * 8 * self.dense_features)
        self.bn1 = nn.BatchNorm3d(self.start_filts * 8)

        self.tp_conv2 = nn.Conv3d(
            self.start_filts * 8, self.start_filts * 4, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(self.start_filts * 4)

        self.tp_conv3 = nn.Conv3d(
            self.start_filts * 4, self.start_filts * 2, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(self.start_filts * 2)

        self.tp_conv4 = nn.Conv3d(
            self.start_filts * 2, self.start_filts, kernel_size=3, stride=1,    
            padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(self.start_filts)

        self.tp_conv5 = nn.Conv3d(
            self.start_filts, self.out_channels, kernel_size=3, stride=1,
            padding=1, bias=False)
        
    def forward(self, noise):
        logger.debug("VAE-GAN Generator...")
        self.debug("input", noise)
        noise = noise.view(-1, self.latent_dim)
        self.debug("view", noise)
        h = self.fc(noise)
        self.debug("dense", h)
        h = h.view(-1, self.start_filts * 8, *self.shapes[-1])
        self.debug("view", h)
        h = func.relu(self.bn1(h))

        h = nn.functional.interpolate(
            h, size=self.shapes[-2], mode=self.mode, align_corners=False)
        h = self.tp_conv2(h)
        h = func.relu(self.bn2(h))
        self.debug("tp_conv2", h)

        h = nn.functional.interpolate(
            h, size=self.shapes[-3], mode=self.mode, align_corners=False)
        h = self.tp_conv3(h)
        h = func.relu(self.bn3(h))
        self.debug("tp_conv3", h)

        h = nn.functional.interpolate(
            h, size=self.shapes[-4], mode=self.mode, align_corners=False)
        h = self.tp_conv4(h)
        h = func.relu(self.bn4(h))
        self.debug("tp_conv4", h)

        h = nn.functional.interpolate(
            h, size=self.shapes[-5], mode=self.mode, align_corners=False)
        h = self.tp_conv5(h)
        self.debug("tp_conv5", h)

        h = torch.tanh(h)
        self.debug("output", h)
        logger.debug("Done.")
        return h

    def debug(self, name, tensor):
        logger.debug("  {3}: {0} - {1} - {2}".format(
            tensor.shape, tensor.get_device(), tensor.dtype, name))

def _downsample_shape(shape, nb_iterations=1, scale_factor=2):
    shape = np.asarray(shape)
    all_shapes = [shape.astype(int).tolist()]
    for idx in range(nb_iterations):
        shape = np.floor(shape / scale_factor)
        all_shapes.append(shape.astype(int).tolist())
    return all_shapes

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
generator = Generator(
    in_shape=in_shape, out_channels=channels, start_filts=64,
    latent_dim=latent_dim, mode="trilinear").to(device)
discriminator = Discriminator(
    in_shape=in_shape, in_channels=channels, out_channels=channels,
    start_filts=64).to(device)
encoder = Encoder(
    in_shape=in_shape, in_channels=channels, start_filts=64,
    latent_dim=latent_dim).to(device)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
e_optimizer = torch.optim.Adam(encoder.parameters(), lr = 0.0001)
real_y = Variable(torch.ones((batch_size, channels)).to(device))
fake_y = Variable(torch.zeros((batch_size, channels)).to(device))
board = Board(port=8097, host="http://localhost", env="vae")
outdir = "/tmp/vae-gan"
if not os.path.isdir(outdir):
    os.mkdir(outdir)

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
            outdir, "chechpoint", name + "_epoch_" + str(epoch + 1) + ".pth")
        torch.save(model.state_dict(), fname)


#############################################################################
# Conclusion
# ----------
#
# Variational Auto-Encoder(VAE) GAN are free from mode collapse but outputs
# are characterized with blurriness. In order to effectively address the
# problems of both mode collapse of GANs and blurriness of VAEs, we will
# usee α-GAN, a solution born by combining both models, in the net tutorial.

