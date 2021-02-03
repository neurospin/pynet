# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides functions to prepare the PRIME-DE dataset.
"""

# Imports
import os
import re
import glob
import copy
import json
import logging
import subprocess
import lxml.html as lh
from pprint import pprint
from collections import namedtuple
from collections import OrderedDict
import requests
import nibabel
import numpy as np
import pandas as pd
from pynet.datasets import Fetchers


# Global parameters
QC = [
    "016032099-001",
    "025032241-001",
    "016032098-001",
    "016032098-002",
    "016032103-001",
    "016032103-002",
    "016032097-001",
    "016032104-001",
    "016032104-002",
    "016032100-001",
    "016032100-002",
    "016032101-001",
    "016032102-001"
]
URL = "https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/PRIME/{0}.tar.gz"
DESC_URL = "http://fcon_1000.projects.nitrc.org/indi/PRIME/files/{0}.csv"
HOME_URL = "http://fcon_1000.projects.nitrc.org/indi/indiPRIME.html"
SITES = [
    "amu",
    "caltech",
    "ecnu-chen",
    "ecnu",
    "ion",
    "iscmj",
    "mcgill",
    "lyon",
    "mountsinai-P",
    "mountsinai-S",
    "nki",
    "NIMH_encrypted",
    "NIMH-CT_encrypted",
    "nin",
    "neurospin",
    "newcastle",
    "ohsu",
    "princeton",
    "rockefeller",
    "sbri",
    "ucdavis",
    "uminn",
    "oxford_encrypted",
    "oxford-PM",
    "NINPBBUtrecht",
    "uwo",
    "georgetown"
]
TRANSCODING = dict((name, "site-{0}".format(name)) for name in SITES)
TRANSCODING["NINPBBUtrecht"] = "site-utrecht"
TRANSCODING["georgetown"] = "Archakov2020"
TRANSCODING["oxford-encrypted"] = "site-oxford"
HTML_SITES = {
    "amu": "Aix-Marseille Université",
    "caltech": "California Institute of Technology",
    "ecnu-chen": "East China Normal University - Chen",
    "ecnu": "East China Normal University - Kwok",
    "ion": "Institute of Neuroscience",
    "iscmj": "Institut des Sciences Cognitives Marc Jeannerod",
    "mcgill": "McGill University",
    "lyon": "Lyon Neuroscience Research Center",
    "mountsinai-P": "Mount Sinai School of Medicine",
    "mountsinai-S": "Mount Sinai School of Medicine",
    "nki": "Nathan Kline Institute",
    "NIMH-encrypted": "National Institute of Mental Health",
    "NIMH-CT-encrypted": "National Institute of Mental Health",
    "nin": "Netherlands Institute for Neuroscience",
    "neurospin": "NeuroSpin",
    "newcastle": "Newcastle University",
    "ohsu": "Oregon Health and Science University",
    "princeton": "Princeton University",
    "rockefeller": "Rockefeller University",
    "sbri": "Stem Cell and Brain Research Institute",
    "ucdavis": "University of California, Davis",
    "uminn": "University of Minnesota",
    "oxford-encrypted": "University of Oxford",
    "oxford-PM": "University of Oxford (PM)",
    "NINPBBUtrecht": "NIN Primate Brain Bank/Utrecht University",
    "uwo": "University of Western Ontario",
    "georgetown": "Georgetown University"
}
EXTRA_SITE = dict((name, "{0}".format(name)) for name in SITES)
EXTRA_SITE["ecnu"] = "ecnu-kwok"
EXTRA_SITE["NIMH-encrypted"] = "NIMH-L"
EXTRA_SITE["NIMH-CT-encrypted"] = "NIMH-M"
EXTRA_SITE["sbri"] = "sbri_pheno"
EXTRA_SITE["oxford-encrypted"] = "oxford"
DATADIR = "/neurospin/lbi/monkeyfmri/PRIME_DE_database"
Item = namedtuple("Item", ["input_path", "output_path", "metadata_path"])
logger = logging.getLogger("pynet")


def download_primede(datasetdir):
    """ Download the PRIME-DE dataset.

    Reference: http://fcon_1000.projects.nitrc.org/indi/PRIMEdownloads.html

    Parameters
    ----------
    datasetdir: str
        the dataset destination folder.
    """
    logger.info("Download PRIME-DE dataset.")
    if not os.path.isdir(datasetdir):
        os.mkdir(datasetdir)
    downloaddir = os.path.join(datasetdir, "download")
    if not os.path.isdir(downloaddir):
        os.mkdir(downloaddir)
    for site in SITES:
        localfile = os.path.join(downloaddir, "{0}.tar.gz".format(site))
        if os.path.isfile(localfile):
            logger.info("  - {0}".format(localfile))
            continue
        url = URL.format(site)
        logger.info("  - {0}".format(url))
        cmd = ["wget", "-P", downloaddir, url]
        logger.debug(" ".join(cmd))
        subprocess.check_call(cmd)
    for site in SITES:
        site = site.replace("_encrypted", "-encrypted")
        localfile = os.path.join(downloaddir, "{0}.csv".format(
            EXTRA_SITE[site]))
        if os.path.isfile(localfile):
            logger.info("  - {0}".format(localfile))
            continue
        url = DESC_URL.format(EXTRA_SITE[site])
        logger.info("  - {0}".format(url))
        cmd = ["wget", "-P", downloaddir, url]
        logger.debug(" ".join(cmd))
        try:
            subprocess.check_call(cmd)
        except:
            pass
    for site in SITES:
        site = site.replace("_encrypted", "-encrypted")
        tarballfile = os.path.join(downloaddir, "{0}.tar.gz".format(site))
        if site not in TRANSCODING:
            logger.info("  - {0}".format(site))
            continue
        localdir = os.path.join(downloaddir, "{0}".format(TRANSCODING[site]))
        if os.path.isdir(localdir):
            logger.info("  - {0}".format(localdir))
            continue
        cmd = ["tar", "-zxvf", tarballfile, "--directory", downloaddir]
        logger.debug(" ".join(cmd))
        subprocess.check_call(cmd)
    infofile = os.path.join(downloaddir, "info.json")
    info = convert_html_table(HOME_URL)
    with open(infofile, "wt") as open_file:
        json.dump(info, open_file, indent=4)


def convert_html_table(url):
    """ Web scraping: HTML tables.
    """
    page = requests.get(url)
    doc = lh.fromstring(page.content)
    tr_elements = doc.xpath("//tr")
    assert all(len(tr_elements[0]) == len(row) for row in tr_elements)
    data = []
    for cnt, item in enumerate(tr_elements[0]):
        name = item.text_content()
        data.append((name, []))
    for row in tr_elements[1:]:
        for cnt, item in enumerate(row.iterchildren()):
            value = item.text_content()
            data[cnt][1].append(value)
    return dict(data)


def organize_primede(datasetdir):
    """ Organize the PRIME-DE dataset.

    Put all the data in the same BIDS organized folder.

    Parameters
    ----------
    datasetdir: str
        the dataset destination folder.
    """
    logger.info("Download PRIME-DE dataset.")
    downloaddir = os.path.join(datasetdir, "download")
    rawdir = os.path.join(datasetdir, "rawdata")
    if not os.path.isdir(rawdir):
        os.mkdir(rawdir)
    infofile = os.path.join(downloaddir, "info.json")
    with open(infofile, "rt") as open_file:
        info = json.load(open_file)
    col_names = info.pop("")
    info = dict((key, dict((_key, _val) for _key, _val in zip(col_names, val)))
                for key, val in info.items())
    metadata = OrderedDict(
        (key, []) for key in ("participant_id", "site", "site_index",
                              "species", "scanner", "state", "age", "weight",
                              "housing", "sex", "implant", "usage_agreement"))
    for site_idx, site in enumerate(SITES):
        extrafile = os.path.join(downloaddir, "{0}.csv".format(
            EXTRA_SITE[site]))
        if os.path.isfile(extrafile):
            df = pd.read_csv(extrafile, dtype=str)
            if "SubID" in df.columns:
                df["Subject ID"] = df["SubID"]
        else:
            df = pd.DataFrame.from_dict({"Subject ID": []})
        if "Subject ID" not in df.columns:
            raise ValueError("A 'Subject ID' column is mandatory in "
                             "'{0}'.".format(extrafile))
        site_idx = str(site_idx + 1).zfill(3)
        site = site.replace("_encrypted", "-encrypted")
        if site not in TRANSCODING:
            logger.info("  - {0}".format(site))
            continue
        localdir = os.path.join(downloaddir, "{0}".format(TRANSCODING[site]))
        if not os.path.isdir(localdir):
            logger.info("  - {0}".format(site))
            continue
        for sid in os.listdir(localdir):
            if not sid.startswith("sub-"):
                logger.info("  - {0}".format(sid))
                continue
            _sid = sid.replace("sub-", "")
            _sid = re.sub("[^0-9]", "", _sid)
            sidinfo = {}
            if _sid in df["Subject ID"].values:
                sidinfo = df[df["Subject ID"] == _sid]
            elif _sid.lstrip("0") in df["Subject ID"].values:
                sidinfo = df[df["Subject ID"] == _sid.lstrip("0")]
            if len(sidinfo) > 1:
                raise ValueError("Multiple match for '{0}' in '{1}'.".format(
                    _sid, extrafile))
            elif len(sidinfo) > 0:
                sidinfo = sidinfo.to_dict(orient="list")
                sidinfo = dict((key.split(" ")[0].lower(), val[0])
                               for key, val in sidinfo.items())
                if "sexe" in sidinfo:
                    sidinfo["sex"] = sidinfo["sexe"]
            _sid = "sub-{0}{1}".format(site_idx, _sid)
            siddir = os.path.join(localdir, sid)
            subject = _sid.replace("sub-", "")
            if subject in metadata["participant_id"]:
                raise ValueError("Subject '{0}' is not unique.".format(sid))
            metadata["participant_id"].append(subject)
            metadata["site"].append(site)
            metadata["site_index"].append(site_idx)
            metadata["species"].append(info[HTML_SITES[site]]["Species"])
            metadata["scanner"].append(info[HTML_SITES[site]]["Scanner"])
            metadata["state"].append(info[HTML_SITES[site]]["State"])
            metadata["age"].append(sidinfo.get("age", "nc"))
            metadata["weight"].append(sidinfo.get("weight", "nc"))
            metadata["housing"].append(sidinfo.get("housing", "nc"))
            metadata["sex"].append(sidinfo.get("sex", "nc"))
            metadata["implant"].append(sidinfo.get("implant", "nc"))
            metadata["usage_agreement"].append(info[HTML_SITES[site]][
                "Usage Agreement"])
            cmd = ["mv", siddir, os.path.join(rawdir, _sid)]
            logger.info(" ".join(cmd))
            subprocess.check_call(cmd)
    participantsfile = os.path.join(rawdir, "participants.tsv")
    df = pd.DataFrame.from_dict(metadata)
    df.to_csv(participantsfile, sep="\t", index=False)
    desc = {
        "Name": "primede",
        "BIDSVersion": "1.0.2"
    }
    descfile = os.path.join(rawdir, "dataset_description.json")
    with open(descfile, "wt") as open_file:
        json.dump(desc, open_file, indent=4)


@Fetchers.register
def fetch_primede(datasetdir, maskdirname="brainmask"):
    """ Fetch/prepare the PRIME-DE dataset for pynet.

    Parameters
    ----------
    datasetdir: str
        the dataset destination folder.
    maskdirname: str
        name of the folder that contains the brain masks.

    Returns
    -------
    item: namedtuple
        a named tuple containing 'input_path', 'output_path', and
        'metadata_path'.
    """
    logger.info("Loading PRIME-DE dataset.")
    imdirname = os.path.split(os.sep)[-1]
    if not os.path.isdir(datasetdir):
        os.mkdir(datasetdir)
    desc_path = os.path.join(datasetdir, "pynet_primede.tsv")
    input_path = os.path.join(datasetdir, "pynet_primede_inputs.npy")
    output_path = os.path.join(datasetdir, "pynet_primede_outputs.npy")
    if not os.path.isfile(desc_path):
        metadata = OrderedDict(
            (key, []) for key in ("participant_id", "site", "with_mask",
                                  "valid", "session", "run"))
        anat_files = glob.glob(os.path.join(
            datasetdir, "sub-*", "ses-*", "anat", "*acq-nc1iso*.nii.gz"))
        if len(anat_files) == 0:
            raise ValueError("Your dataset directory must contain the Prime "
                             "DE data organized with the function provided in "
                             "this module and preprocessed.")
        inputs = []
        outputs = []
        for path in anat_files:
            sid = path.split(os.sep)[-4].replace("sub-", "")
            ses = path.split(os.sep)[-3].replace("ses-", "")
            inputs.append(nibabel.load(path).get_data())
            mask_path = path.replace(imdirname, maskdirname).replace(
                "acq-nc1iso", "acq-c1iso").replace(".nii.gz", "_mask.nii.gz")
            with_mask = 0
            if os.path.isfile(mask_path):
                outputs.append(nibabel.load(mask_path).get_data())
                with_mask = 1
            else:
                outputs.append(np.zeros((90, 90, 60), dtype=int))
            basename = os.path.basename(path)
            match = re.findall("run-(\d+)_", basename)
            if len(match) == 1:
                run = match[0]
            else:
                run = "nc"
            valid = 1
            if "{0}-{1}".format(sid, ses) in QC:
                valid = 0
            metadata["participant_id"].append(sid)
            metadata["site"].append(sid[:3])
            metadata["with_mask"].append(with_mask)
            metadata["valid"].append(valid)
            metadata["session"].append(ses)
            metadata["run"].append(run)
        inputs = np.asarray(inputs)
        outputs = np.asarray(outputs)
        inputs_im = nibabel.Nifti1Image(
            inputs.transpose(1, 2, 3, 0), np.eye(4))
        outputs_im = nibabel.Nifti1Image(
            outputs.transpose(1, 2, 3, 0), np.eye(4))
        inputs = np.expand_dims(inputs, axis=1)
        outputs = np.expand_dims(outputs, axis=1)
        np.save(input_path, inputs)
        np.save(output_path, outputs)
        nibabel.save(inputs_im, input_path.replace(".npy", ".nii.gz"))
        nibabel.save(outputs_im, output_path.replace(".npy", ".nii.gz"))
        df = pd.DataFrame.from_dict(metadata)
        df.to_csv(desc_path, sep="\t", index=False)
    return Item(input_path=input_path, output_path=output_path,
                metadata_path=desc_path)
