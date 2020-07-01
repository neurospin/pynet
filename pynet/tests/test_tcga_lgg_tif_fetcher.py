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
import numpy as np
import pandas as pd
import unittest.mock as mock
from unittest.mock import patch, mock_open
from collections import namedtuple

# Package import
from pynet.datasets import fetch_tcga_lgg_tif
from pynet.datasets.tcga_lgg_tif import get_subjects_files
from pynet.datasets.tcga_lgg_tif import read_metadata
from pynet.datasets.tcga_lgg_tif import get_slice_id


class TestTcgaLggFetcher(unittest.TestCase):
    """ Test the tcga_lgg_tif fetcher.
    """

    def setUp(self):
        """ Setup test.
        """
        self.Item = namedtuple("Item", ["input_path", "output_path",
                                        "metadata_path", "height", "width"])
        self.height = 256
        self.width = 256
        self.datasetdir = "./kaggle_3m"
        self.desc_path = "./kaggle_3m/pynet_tgca-lgg-tif.tsv"
        self.input_path = "./kaggle_3m/pynet_tgca-lgg-tif_inputs.npy"
        self.output_path = "./kaggle_3m/pynet_tgca-lgg-tif_outputs.npy"
        self.metadata_file = "./kaggle_3m/data.csv"
        self.tif_files = [[("./kaggle_3m/TCGA_HT_A61B_19991127/"
                            "TCGA_HT_A61B_19991127_80_mask.tif"),
                           ("./kaggle_3m/TCGA_HT_A61B_19991127/"
                            "TCGA_HT_A61B_19991127_81_mask.tif")]]
        self.raw_metadata = """
        Patient,RNASeqCluster,MethylationCluster
        TCGA_HT_A61B,2,4
        TCGA_CS_4942,1,5
        """
        self.csv_dict = (
            {"Patient": "TCGA_HT_A61B",
             "RNASeqCluster": "2",
             "MethylationCluster": "4"},
            {"Patient": "TCGA_CS_4942",
             "RNASeqCluster": "1",
             "MethylationCluster": "5"}
        )
        self.csv_metadata = {
            "A61B": {"Patient": "TCGA_HT_A61B",
                     "RNASeqCluster": "2",
                     "MethylationCluster": "4"},
            "4942": {"Patient": "TCGA_CS_4942",
                     "RNASeqCluster": "1",
                     "MethylationCluster": "5"}
        }
        self.final_metadata = {
            'participant_id': ['A61B', 'A61B'],
            'slice_id': [80, 81],
            'center': ['HT', 'HT'],
            'serie': ['19991127', '19991127'],
            'Patient': ['TCGA_HT_A61B', 'TCGA_HT_A61B'],
            'RNASeqCluster': ['2', '2'],
            'MethylationCluster': ['4', '4']}
        self.dataframe = pd.DataFrame.from_dict(self.final_metadata)
        self.subjects_data = {
            "A61B": {"center": "HT",
                     "serie": "19991127",
                     "masks": [("./kaggle_3m/TCGA_HT_A61B_19991127/"
                                "TCGA_HT_A61B_19991127_80_mask.tif"),
                               ("./kaggle_3m/TCGA_HT_A61B_19991127/"
                                "TCGA_HT_A61B_19991127_81_mask.tif")],
                     "images": [('./kaggle_3m/TCGA_HT_A61B_19991127/'
                                 'TCGA_HT_A61B_19991127_80.tif'),
                                ('./kaggle_3m/TCGA_HT_A61B_19991127/'
                                 'TCGA_HT_A61B_19991127_81.tif')]}}
        self.flair1 = np.random.rand(self.height, self.width, 3)
        self.flair2 = np.random.rand(self.height, self.width, 3)
        self.mask1 = np.zeros((self.height, self.width))
        self.mask1[self.height//2, self.width//2] = 255
        self.mask2 = np.zeros((self.height, self.width))

    def tearDown(self):
        """ Run after each test.
        """
        pass

    @mock.patch("os.path")
    def test_invalid_dir_throws_exception(self, mock_os):
        """ Test an exception is raised when datasetdir do not exists.
        """
        # Set the mocked function returned values.
        mock_os.isdir.return_value = False
        self.assertRaises(ValueError, fetch_tcga_lgg_tif, self.datasetdir)

    @mock.patch("os.path.isfile")
    @mock.patch("os.path.isdir")
    def test_existing_tsv_file_returns(self, mock_isdir, mock_isfile):
        """ Test returns immediately when tsv file already exists.
        """
        # Set the mocked function returned values.
        mock_isdir.return_value = True
        mock_isfile.return_value = True
        res = fetch_tcga_lgg_tif(self.datasetdir)
        self.assertEqual(
            res.input_path, self.input_path)
        self.assertEqual(
            res.output_path, self.output_path)
        self.assertEqual(res.metadata_path, self.desc_path)
        self.assertEqual(res.height, self.height)
        self.assertEqual(res.width, self.width)

    @mock.patch("glob.glob")
    def test_get_subjects_files(self, mock_glob):
        """ Test the function to get subject images.
        """
        # Set the mocked function returned values.
        mock_glob.side_effect = self.tif_files
        res = get_subjects_files(self.datasetdir)
        self.assertDictEqual(res, self.subjects_data)

    @mock.patch("csv.DictReader")
    def test_read_metadata(self, mock_csvdictreader):
        """ Test the function to read genetics csv file.
        """
        mock_csvdictreader.return_value = self.csv_dict
        with patch("builtins.open", mock_open(read_data=self.raw_metadata)) \
                as mock_file:
            res = read_metadata(self.metadata_file)
            mock_file.assert_called_with(self.metadata_file)
            self.assertDictEqual(res, self.csv_metadata)

    def test_get_slice_id_with_mask(self):
        """ Test the function to get slice id from mask image.
        """
        fp = ("./kaggle_3m/TCGA_HT_A61B_19991127/"
              "TCGA_HT_A61B_19991127_80_mask.tif")
        self.assertEqual(get_slice_id(fp), 80)

    def test_get_slice_id_with_flair(self):
        """ Test the function to get slice id from flair image.
        """
        fp = './kaggle_3m/TCGA_HT_A61B_19991127/TCGA_HT_A61B_19991127_81.tif'
        self.assertEqual(get_slice_id(fp), 81)

    @mock.patch("pandas.DataFrame.from_dict")
    @mock.patch("pandas.DataFrame.to_csv")
    @mock.patch("numpy.save")
    @mock.patch("skimage.io.imread")
    @mock.patch("pynet.datasets.tcga_lgg_tif.read_metadata")
    @mock.patch("pynet.datasets.tcga_lgg_tif.get_subjects_files")
    @mock.patch("os.path.isfile")
    @mock.patch("os.path.isdir")
    def test_not_existing_tsv_file(self, mock_isdir, mock_isfile,
                                   mock_getsubjectsfiles, mock_readmetadata,
                                   mock_imread, mock_save, mock_tocsv,
                                   mock_fromdict):
        """ Test the global behaviour.
        """

        def check_arrays(save_path, array):
            self.assertTrue(("inputs" in save_path)
                            or ("outputs" in save_path))
            if "inputs" in save_path:
                self.assertEqual(array.shape, (2, 3, self.height, self.width))
            elif "outputs" in save_path:
                self.assertEqual(array.shape, (2, 1, self.height, self.width))

        def check_metadata(input_dict):
            self.assertDictEqual(input_dict, self.final_metadata)
            return self.dataframe

        def check_tocsv(desc, sep, index):
            self.assertEqual(desc, self.desc_path)
            self.assertEqual(sep, "\t")
            self.assertEqual(index, False)

        mock_isdir.return_value = True
        mock_isfile.return_value = False
        mock_getsubjectsfiles.return_value = self.subjects_data
        mock_readmetadata.return_value = self.csv_metadata
        mock_imread.side_effect = [self.flair1,
                                   self.flair2,
                                   self.mask1,
                                   self.mask2]
        mock_save.side_effect = check_arrays
        mock_fromdict.side_effect = check_metadata
        mock_tocsv.side_effect = check_tocsv
        res = fetch_tcga_lgg_tif(self.datasetdir)
        self.assertEqual(
            res.input_path, self.input_path)
        self.assertEqual(
            res.output_path, self.output_path)
        self.assertEqual(res.metadata_path, self.desc_path)
        self.assertEqual(res.height, self.height)
        self.assertEqual(res.width, self.width)


if __name__ == "__main__":
    from pynet.utils import setup_logging
    setup_logging(level="debug")
    unittest.main()
