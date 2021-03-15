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
import torch

# Package import
import pynet


class TestModels(unittest.TestCase):
    """ Test the models defined in pynet.
    """
    def setUp(self):
        """ Setup test.
        """
        self.networks = pynet.get_tools(tool_name="networks")
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
        n_samples = 10
        n_channels = 3
        n_dims = [33, 64, 35]
        inputs = torch.zeros(n_samples, n_channels, n_dims[0])
        for dense_hidden_dims in (None, [256], [128, 256]):
            model = self.networks["VAE"](
                input_channels=inputs.shape[1], input_dim=inputs.shape[2],
                conv_flts=None, dense_hidden_dims=dense_hidden_dims,
                latent_dim=10, noise_out_logvar=-3, noise_fixed=False,
                act_func=None, dropout=0, sparse=False)
            print(model)
            p, dist_extra = model(inputs)
            outputs = VAE.p_to_prediction(p)
            print("-- dense_hidden_dims", dense_hidden_dims)
            print("-- results", inputs.shape, outputs.shape,
                  dist_extra["z"].shape)
        for dim in (1, 2, 3):
            inputs = torch.zeros(n_samples, n_channels, *n_dims[:dim])
            model = self.networks["VAE"](
                input_channels=inputs.shape[1], input_dim=inputs.shape[2:],
                conv_flts=[128, 64, 64], dense_hidden_dims=None,
                latent_dim=10, noise_out_logvar=-3, noise_fixed=False,
                act_func=None, dropout=0, sparse=False)
            print(model)
            p, dist_extra = model(inputs)
            outputs = VAE.p_to_prediction(p)
            print("-- dim", dim)
            print("-- results", inputs.shape, outputs.shape,
                  dist_extra["z"].shape)

    def test_vae(self):
        """ Test the GMVAENet.
        """
        n_samples = 5
        n_channels = 2
        n_data = 33
        inputs = torch.zeros(n_samples, n_channels, n_data)
        gmvae = self.networks["GMVAENet"](
            input_dim=inputs.shape[2], latent_dim=10, n_mix_components=3,
            dense_hidden_dims=[256], sigma_min=0.001, raw_sigma_bias=0.25,
            dropout=0, temperature=1, gen_bias_init=0.)
        print(gmvae)
        p_x_given_z, dists = gmvae.forward(inputs)


if __name__ == "__main__":
    from pynet.utils import setup_logging
    setup_logging(level="debug")
    unittest.main()
