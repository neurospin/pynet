# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import unittest
import copy
import torch

# Package import
import pynet
from pynet.losses import get_vae_loss


class TestModels(unittest.TestCase):
    """ Test the models defined in pynet.
    """
    def setUp(self):
        """ Setup test.
        """
        self.networks = pynet.get_tools(tool_name="networks")
        self.losses = pynet.get_tools(tool_name="losses")
        self.x1 = torch.randn(3, 1, 64)
        self.x2 = torch.randn(1, 1, 127, 128)
        self.x3 = torch.randn(1, 1, 64, 64, 64)

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_unet(self):
        """ Test the UNet.
        """
        params = {
            "num_classes": 2,
            "in_channels": 1,
            "depth": 3,
            "start_filts": 16,
            "up_mode": "upsample",
            "merge_mode": "concat",
            "batchnorm": False,
            "input_shape": self.x2.shape[2:]
        }
        net = self.networks["UNet"](dim="2d", **params)
        y = net(self.x2)

    def test_nvnet(self):
        """ Test the NvNet.
        """
        params = {
            "input_shape": self.x3.shape[2:],
            "in_channels": 1,
            "num_classes": 2,
            "activation": "relu",
            "normalization": "group_normalization",
            "mode": "trilinear",
            "with_vae": False
        }
        net = self.networks["NvNet"](**params)
        y = net(self.x3)

    def test_vtnet(self):
        """ Test the VTNet.
        """
        params = {
            "input_shape": self.x3.shape[2:],
            "in_channels": 2,
            "kernel_size": 3,
            "padding": 1,
            "flow_multiplier": 1,
            "nb_channels": 16
        }
        net = self.networks["VTNet"](**params)
        y = net(torch.cat((self.x3, self.x3), dim=1))

    def test_addnet(self):
        """ Test the ADDNet.
        """
        params = {
            "input_shape": self.x3.shape[2:],
            "in_channels": 2,
            "kernel_size": 3,
            "padding": 1,
            "flow_multiplier": 1
        }
        net = self.networks["ADDNet"](**params)
        y = net(torch.cat((self.x3, self.x3), dim=1))

    def test_voxelmorphnet(self):
        """ Test the VoxelMorphNet.
        """
        params = {
            "vol_size": self.x3.shape[2:],
            "enc_nf": [16, 32, 32, 32],
            "dec_nf": [32, 32, 32, 32, 32, 16, 16],
            "full_size": True
        }
        net = self.networks["VoxelMorphNet"](**params)
        y = net(torch.cat((self.x3, self.x3), dim=1))

    def test_rcnet(self):
        """ Test the RCNet.
        """
        params = {
            "input_shape": self.x3.shape[2:],
            "in_channels": 1,
            "base_network": "VTNet",
            "n_cascades": 1,
            "rep": 1
        }
        net = self.networks["RCNet"](**params)
        y = net(self.x3)

    def test_brainnetcnn(self):
        """ Test the BrainNetCNN.
        """
        params = {
            "input_shape": self.x3.shape[2: -1],
            "in_channels": 1,
            "num_classes": 2,
            "nb_e2e": 32,
            "nb_e2n": 64,
            "nb_n2g": 30,
            "dropout": 0.5,
            "leaky_alpha": 0.33,
            "twice_e2e": False,
            "dense_sml": True
        }
        net = self.networks["BrainNetCNN"](**params)
        y = net(self.x3[..., 0])

    def test_pspnet(self):
        """ Test the PSPNet.
        """
        params = {
            "n_classes": 2,
            "sizes": (1, 2, 3, 6),
            "psp_size": 512,
            "deep_features_size": 256,
            "backend": "resnet18",
            "drop_rate": 0
        }
        net = self.networks["PSPNet"](**params)
        y = net(self.x3[..., 0])

    def test_deeplabnet(self):
        """ Test the DeepLabNet.
        """
        params = {
            "n_classes": 2,
            "drop_rate": 0
        }
        net = self.networks["DeepLabNet"](**params)
        y = net(self.x3[..., 0])

    def test_vae(self):
        """ Test the VAENet.
        """
        # Dense network only
        params = {
            "input_channels": self.x1.shape[1],
            "input_dim": self.x1.shape[2],
            "conv_flts": None,
            "dense_hidden_dims": None,
            "latent_dim": 10}
        loss_params = {
            "betah": {"beta": 4, "steps_anneal": 0, "use_mse": True},
            "betab": {"C_init": 0.5, "C_fin": 25, "gamma": 100,
                      "steps_anneal": 100000, "use_mse": True},
            "btcvae": {"dataset_size": self.x1.shape[0], "alpha": 1,
                       "beta": 1, "gamma": 6, "is_mss": True,
                       "steps_anneal": 0, "use_mse": True}}
        for dense_hidden_dims in (None, [256], [128, 256]):
            c_params = copy.deepcopy(params)
            c_params["dense_hidden_dims"] = dense_hidden_dims
            net = self.networks["VAENet"](**c_params)
            p, dists = net(self.x1)
            outputs = self.networks["VAENet"].p_to_prediction(p)
        for loss_name in ("betah", "betab", "btcvae"):
            loss_instance = get_vae_loss(
                loss_name=loss_name, **loss_params[loss_name])
            loss_instance.layer_outputs = dists
            loss = loss_instance(p, self.x1)

        # Dense + Conv Nd
        for dim in (1, 2, 3):
            if dim == 1:
                x = self.x1
            elif dim == 2:
                x = self.x2
            else:
                x = self.x3
            c_params = copy.deepcopy(params)
            c_params["conv_flts"] = [128, 64, 64]
            c_params["input_dim"] = x.shape[2:]
            net = self.networks["VAENet"](**c_params)
            p, dists = net(x)
            outputs = self.networks["VAENet"].p_to_prediction(p)

    def test_sparsevae(self):
        """ Test the sVAENet.
        """
        # Dense network only
        params = {
            "input_channels": self.x1.shape[1],
            "input_dim": self.x1.shape[2],
            "conv_flts": None,
            "dense_hidden_dims": None,
            "latent_dim": 10,
            "noise_fixed": False,
            "sparse": True}
        net = self.networks["VAENet"](**params)
        p, dists = net(self.x1)
        outputs = self.networks["VAENet"].p_to_prediction(p)
        loss_instance = self.losses["SparseLoss"](beta=4)
        loss_instance.layer_outputs = dists
        loss = loss_instance(p, self.x1)

    def test_gmvae(self):
        """ Test the GMVAENet.
        """
        params = {
            "input_dim": self.x1.shape[2],
            "latent_dim": 10,
            "n_mix_components": 3,
            "dense_hidden_dims": [256]
        }
        net = self.networks["GMVAENet"](**params)
        p_x_given_z, dists = net(self.x1)
        loss_instance = self.losses["GMVAELoss"]()
        loss_instance.layer_outputs = dists
        loss = loss_instance(p_x_given_z, self.x1)

    def test_vaegmp(self):
        """ Test the VAEGMPNet.
        """
        params = {
            "input_dim": self.x1.shape[2],
            "latent_dim": 10,
            "n_mix_components": 3,
            "dense_hidden_dims": [256]
        }
        net = self.networks["VAEGMPNet"](**params)
        p_x_given_z, dists = net(self.x1)
        loss_instance = self.losses["VAEGMPLoss"]()
        loss_instance.layer_outputs = dists
        loss = loss_instance(p_x_given_z, self.x1)

    def test_vade(self):
        """ Test the VaDE.
        """
        params = {
            "n_classes": 2,
            "input_dim": self.x1.shape[2],
            "latent_dim": 10,
            "hidden_dims": [500, 500, 2000],
            "binary": True
        }
        net = self.networks["VaDENet"](**params)
        x = self.x1[:, 0]
        recon_x, dists = net(x)
        loss_instance = self.losses["VaDELoss"](alpha=1)
        loss_instance.layer_outputs = dists
        loss = loss_instance(recon_x, x)

    def test_mcvae(self):
        """ Test the MCVAE.
        """
        params = {
            "n_channels": 2,
            "n_feats": [self.x1.shape[2], self.x1.shape[2]],
            "latent_dim": 10
        }
        net = self.networks["MCVAE"](**params)
        x = [self.x1, self.x1]
        p_x_given_z, dists = net(x)
        loss_instance = self.losses["MCVAELoss"](
            n_channels=params["n_channels"], beta=1.)
        loss_instance.layer_outputs = dists
        loss = loss_instance(p_x_given_z)

    def test_sphericalunet(self):
        """ Test the SphericalUNet.
        """
        params = {
            "in_order": 3,
            "in_channels": 2,
            "out_channels": 3,
            "depth": 2,
            "start_filts": 32,
            "conv_mode": "1ring",
            "up_mode": "transpose"
        }
        net = self.networks["SphericalUNet"](**params)
        vertices = net.ico[params["in_order"]].vertices
        x = torch.randn(1, params["in_channels"], len(vertices))
        y = net(x)


if __name__ == "__main__":
    from pynet.utils import setup_logging
    setup_logging(level="debug")
    unittest.main()
