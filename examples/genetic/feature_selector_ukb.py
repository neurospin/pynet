import os
from pynet.datasets import DataManager
from pynet.feature_selection import PlinkSelector
from pynet.utils import setup_logging
import numpy as np
import pandas as pd
from pandas_plink import read_plink
import statsmodels.api as sm
import warnings
import progressbar
import shutil
from collections import namedtuple

setup_logging(level="info")

data_path = '/neurospin/brainomics/tmp/CA263211'

Item = namedtuple("Item", ["input_path", "output_path", "metadata_path",
                           "labels"])

fam = pd.read_csv('/neurospin/brainomics/tmp/CA263211/full_data_qc.fam', sep=' ',
    names=['FID', 'IID', 'father', 'mother', 'sex', 'trait'])
labels = pd.read_csv('/neurospin/brainomics/tmp/CA263211/ukb_rfmri_height_bmi.pheno', sep='\t')

labels = fam[['FID', 'IID']].join(labels.set_index(['FID', 'IID']), on=['FID', 'IID'])

labels.to_csv(os.path.join(data_path, 'full_data_qc.pheno'), index=False, sep='\t')


data = Item(input_path=os.path.join(data_path, 'full_data_qc'),
    output_path=None, labels=None,
    metadata_path=os.path.join(data_path, 'full_data_qc.pheno'))
# fetch_ukb('nicodep_nd_aa', data_path, treat_nans=None)

labels = ['height']

feature_selector = PlinkSelector(
    kbest=2000, data_path=data_path, data_file='full_data_qc',
    pheno_file='ukb_rfmri_height_bmi.pheno', cov_file='ukb_age_10pc.cov',
    pheno_name=labels[0], save_res_to='res_assoc_{}_{}_folds_test_{}'.format(
        labels[0], '4', '0.2'
    ))

manager = DataManager(
    input_path=data.input_path,
    labels=labels,
    metadata_path=data.metadata_path,
    number_of_folds=4,
    batch_size=16,
    test_size=0.2,
    feature_selector=feature_selector)
