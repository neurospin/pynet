# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
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


@Fetchers.register
def fetch_lpsnc(datasetdir="/neurospin/radiomics/workspace_LPSNC/LPSNC_data",
                modality=None):
    """ Fetch/prepare the lpsnc dataset for pynet.

    Parameters
    ----------
    datasetdir: str
        the dataset destination folder.
    modality: str
        the modality to be fetched, or all modalities if none.
    Returns
    -------
    item: namedtuple
        a named tuple containing 'input_path', 'output_path', and
        'metadata_path'.
    """
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

    def merge_rois(roifiles, mask_value=1):
        """ Merge the input ROIs.

        Parameters
        ----------
        roifiles: list of str
            the ROI images to be merged.
        mask_value: int
            the value of the mask voxels.

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
        merged_data[merged_data > 0] = mask_value
        return merged_data
    if modality not in MODALITIES and modality is not None:
        raise ValueError(
                "Expect modality==None for all modalities "
                + "or modality in ('T2w','T1w','ce-GADOLINIUM_T1w','FLAIR')")
    modalities = MODALITIES
    if modality is not None:
        modalities = [modality]
    mapping_path = os.path.join(datasetdir, "data.json")
    zero = np.zeros((182, 218, 182), dtype=np.float)
    if not os.path.isfile(mapping_path):
        raise ValueError(
            "Are you in the right folder? "
            + "You may need special access for LPSNC dataset")
    fileid = str(modality).replace("None", "All")
    desc_path = os.path.join(datasetdir, "pynet_lpsnc_" + fileid + ".tsv")
    input_path = os.path.join(datasetdir,
                              "pynet_lpsnc_inputs_" + fileid + ".npy")
    output_path = os.path.join(datasetdir,
                               "pynet_lpsnc_outputs_" + fileid + ".npy")
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
                _arr1 = {}
                _arr1["sub"] = sub
                _arr1["ses"] = ses
                subdata = df.loc[df["sub"] == sub]
                for mod in modalities:
                    dataout = []
                    arrs = []
                    paths = subdata["lesionfiles"].values[0]

                    for (count, tissue_type) in enumerate([
                            'edema', 'enh', 'necrosis']):
                        cat = paths[tissue_type]

                        if len(cat) > 0:
                            _arr = _crop(merge_rois(cat, count+1))
                        else:
                            _arr = _crop(zero)
                        dataout.append(_arr == count+1)
                        arrs.append(_arr)
                    allmasks = arrs[0] + arrs[1] + arrs[2]
                    if (len(np.unique(allmasks)) > 0):
                        dataout.insert(0, allmasks == 0)
                        dataout = np.asarray(dataout)
                        output_dataset.append(dataout)
                        datain = []

                        if mod in subdata["mod"].values[0]:
                            path = subdata["Files"].values[0][mod]
                            datain.append(
                                    _norm(_crop(
                                            nib.load(path).get_data())))
                            _arr1[mod] = 1

                        else:
                            datain.append(_crop(zero))
                            _arr1[mod] = 0
                        arr1.append(_arr1)
                        datain = np.asarray(datain)
                        input_dataset.append(datain)

                bar.update(cnt)

            input_dataset = np.asarray(input_dataset)
            output_dataset = np.asarray(output_dataset)

            dataset_desc = pd.DataFrame(arr1)

            np.save(input_path, input_dataset)
            np.save(output_path, output_dataset)
            dataset_desc.to_csv(desc_path, sep="\t")
    return Item(input_path=input_path, output_path=output_path,
                metadata_path=desc_path)
