# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
# @Author: Zhou Kai
# @GitHub: https://github.com/athon2
# @Date: 2018-11-30 09:53:44
##########################################################################

"""
Module that provides functions to prepare the lpsnc dataset.
"""

# Imports
from collections import namedtuple
import os
import logging
import numpy as np
import pandas as pd
import nibabel as nib
import progressbar
from pynet.datasets import Fetchers

# Global parameters
MODALITIES = ('T2w', 'T1w', 'ce-GADOLINIUM_T1w', 'FLAIR')
Item = namedtuple("Item", ["input_path", "output_path", "metadata_path"])
logger = logging.getLogger("pynet")


# datasetdir="/neurospin/radiomics/workspace_LPSNC/LPSNC_data"
@Fetchers.register
def fetch_lpsnc(datasetdir, modality=0):
    """ Fetch/prepare the lpsnc dataset for pynet.

    Parameters
    ----------
    datasetdir: str
        the dataset destination folder.

    Returns
    -------
    item: namedtuple
        a named tuple containing 'input_path', 'output_path', and
        'metadata_path'.
    """
    if modality not in MODALITIES and modality != 0:
        raise ValueError(
                "Expect modality==0 for all modalities "
                +"or modality in ('T2w', 'T1w', 'ce-GADOLINIUM_T1w', 'FLAIR')")
    logger.info("Loading lpsnc dataset.")

    def _crop(arr):
        # return arr[45: 195, 30: 220, 10: 145]
        return arr

    def _norm(arr):
        arr = np.nan_to_num(arr)
        logical_mask = (arr != 0)
        mean = arr[logical_mask].mean()
        std = arr[logical_mask].std()
        return ((arr - mean) / std).astype(np.single)

    def merge_rois(roifiles, v=1):
        """ Merge the input ROIs.

        Parameters
        ----------
        roifiles: list of str
            the ROI images to be merged.
        outdir: str
            the destination folder.
        fmaskname: str
            the filename of the mask to be saved.

        Returns
        -------
        merged_file: str
            the merged ROI.
        """
        # Check inputs
        if len(roifiles) == 0:
            raise ValueError("Expect at least one ROI.")

        # Merge ROIS
        im = nib.load(roifiles[0])
        ref_affine = im.affine
        merged_data = np.nan_to_num(im.get_data())
        if merged_data.ndim != 3:
            raise ValueError("Expect 3d-ROIS.")
        if len(roifiles) > 1:
            for path in roifiles[1:]:
                im = nib.load(path)
                affine = im.affine
                if not np.allclose(ref_affine, affine):
                    raise ValueError(
                            "ROI files have different affine matrices.")
                data = np.nan_to_num(im.get_data())
                if data.ndim != 3:
                    raise ValueError("Expect 3d-ROIS.")
                merged_data += data
        merged_data[merged_data > 0] = v
        return merged_data

    mapping_path = os.path.join(datasetdir, "data.json")
    zero_path = os.path.join(datasetdir, "zero.nii.gz")  # path for empty image of correct dimensions
    zero = nib.load(zero_path).get_data()
    if not os.path.isfile(mapping_path):
        raise ValueError(
            "Are you in the right folder? Your folder= '{0}' ".format(datasetdir)
            +"You may need special access for LPSNC dataset")
    desc_path = os.path.join(datasetdir, "pynet_lpsnc_"+str(modality)+".tsv")
    input_path = os.path.join(datasetdir,
                              "pynet_lpsnc_inputs_"+str(modality)+".npy")
    output_path = os.path.join(datasetdir,
                               "pynet_lpsnc_outputs_"+str(modality)+".npy")
    if not os.path.isfile(desc_path):
        df = pd.read_json(mapping_path)

        arr = df[["sub", "ses"]].values
        input_dataset = []
        output_dataset = []
        nb_subjects = len(arr)
        arr1 = []
        with progressbar.ProgressBar(max_value=nb_subjects,
                                     redirect_stdout=True) as bar:
            for cnt, (sub, ses) in enumerate(arr):
                logger.debug("Processing {0}...".format(sub))

                subdata = df.loc[df["sub"] == sub]
                if (modality == 0):
                    dataout = []
                    arrs = []
                    paths = subdata["lesionfiles"].values[0]

                    for (v, c) in enumerate(['edema', 'enh', 'necrosis']):
                        cat = paths[c]

                        if len(cat) > 0:
                            _arr = _crop(merge_rois(cat, v+1))
                        else:
                            _arr = _crop(zero)
                        dataout.append(_arr == v+1)
                        arrs.append(_arr)
                    allmasks = arrs[0]+arrs[1]+arrs[2]
                    if (len(np.unique(allmasks)) > 0):
                        dataout.insert(0, allmasks == 0)
                        dataout = np.asarray(dataout)
                        output_dataset.append(dataout)
                        datain = []
                        for mod in MODALITIES:
                            if mod in subdata["mod"].values[0]:
                                path = subdata["Files"].values[0][mod]
                                datain.append(
                                        _norm(_crop(
                                                nib.load(path).get_data())))

                            else:
                                datain.append(_crop(zero))
                        arr1.append(subdata[["sub", "ses"]].values[0])
                        datain = np.asarray(datain)
                        input_dataset.append(datain)

                else:
                    if modality in subdata["mod"].values[0]:
                        dataout = []
                        arrs = []
                        paths = subdata["lesionfiles"].values[0]
                        for (v, c) in enumerate(['edema', 'enh', 'necrosis']):
                            cat = paths[c]
                            if len(cat) > 0:
                                _arr = _crop(merge_rois(cat, v+1))
                            else:
                                _arr = _crop(zero)
                            arrs.append(_arr)
                            dataout.append(_arr == v+1)
                        allmasks = arrs[0]+arrs[1]+arrs[2]
                        if (len(np.unique(allmasks)) > 0):
                            dataout.insert(0, allmasks == 0)      
                            dataout = np.asarray(dataout)
                            output_dataset.append(dataout)
                            datain = []
                            path = subdata["Files"].values[0][modality]
                            datain.append(
                                    _norm(_crop(nib.load(path).get_data())))
                            datain = np.asarray(datain)
                            input_dataset.append(datain) 
                            arr1.append(subdata[["sub", "ses"]].values[0])

                bar.update(cnt)

            input_dataset = np.asarray(input_dataset)
            output_dataset = np.asarray(output_dataset)

            dataset_desc = pd.DataFrame(
                    arr1, columns=["sub", "ses"])

            np.save(input_path, input_dataset)
            np.save(output_path, output_dataset)
            dataset_desc.to_csv(desc_path, sep="\t")
    return Item(input_path=input_path, output_path=output_path,
                metadata_path=desc_path)
