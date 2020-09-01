import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from pandas_plink import read_plink
from pynet.feature_selection import FeatureSelector


class PlinkSelector(FeatureSelector):
    def __init__(self, kbest, data_path, data_file,
        pheno_file, cov_file, pheno_name,
        method='linear', impute_na=None,
        plink_path='/neurospin/brainomics/ig_install/plink2-v1.9/plink',
        save_res_to=None,
        **plink_args):
        """ Class instantiation.

        Parameters
        ----------
        kbest: int
            number of best feature to select.
        data_path: str
            path to the folder containing the data.
        data_file: str
            prefix of the files containing the data.
        pheno_file: str
            name of the file containing the phenotypes.
        cov_file: str
            name of the file containing the covariates for the model.
        pheno_name: str
            name of the phenotype to associate, as provided in the
            file containing the phenotypes.
        method: str, default 'linear
            the name of plink's method, one of 'linear' or 'logistic'.
        plink_path: str, default
            '/neurospin/brainomics/ig_install/plink2-v1.9/plink'
            path to the folder containing the executable plink.
        plink_args: dict
            additionnal named parameters to provide to the plink
            command. It can be anything as long as it is valid
            for plink, for instance maf=0.05 if you only want to
            fit the model on common variations. If you want to provide
            a flag to plink, ie a command that does not take any argument,
            give an argument with None as value, like allow-no-sex=None.
        """
        super().__init__(kbest=kbest)
        self.method = method
        self.data_path = data_path
        self.plink_path = plink_path
        self.file_path = os.path.join(data_path, data_file)
        self.pheno_path = os.path.join(data_path, pheno_file)
        self.cov_path = os.path.join(data_path, cov_file)
        self.pheno_name = pheno_name
        self.feature_idx = []
        self.impute_na = impute_na
        self.plink_args = plink_args
        self.save_res_to = save_res_to

    def fit(self, train_indices, save_name=None, verbose=False):
        """ Fit the model using plink

        Parameters
        ----------
        train_indices: list of int
            indices of the samples to use for training the model.
        save_name: str, default None
            name of the file in which we want to store the
            result of association.
        Returns
        -------
        self: instance of PlinkSelector
            the instance that was just fitted to the data.

        """
        if self.save_res_to and save_name is None:
            raise ValueError(
                'If you want to save the association file, you need to provide\
                a file name to save to.')
        bim, fam, _ = read_plink(self.file_path, verbose=verbose)

        indiv_to_keep = fam.loc[fam['i'].isin(train_indices), ['fid', 'iid']]

        if not os.path.isdir(os.path.join(self.data_path, 'tmp')):
            os.mkdir(os.path.join(self.data_path, 'tmp'))


        if self.save_res_to:
            res_path = os.path.join(self.data_path, self.save_res_to)
            if not os.path.isdir(res_path):
                os.mkdir(res_path)

        indiv_to_keep.to_csv(os.path.join(self.data_path, 'tmp', 'indivs.txt'),
            header=False, index=False, sep=' ')

        out = None
        if not verbose:
            out = subprocess.DEVNULL

        list_args = []
        for key, value in self.plink_args.items():
            if value is not None:
                list_args += ['--{}'.format(str(key)), str(value)]
            else:
                list_args.append('--{}'.format(str(key)))

        if self.save_res_to:
            res_path =  os.path.join(self.data_path,
                self.save_res_to, '{}.assoc.{}'.format(save_name, self.method))
            if not os.path.exists(res_path):
                subprocess.run([
                    self.plink_path,
                    '--bfile', self.file_path, '--allow-no-sex',
                    '--keep', os.path.join(self.data_path, 'tmp', 'indivs.txt'),
                    '--pheno', self.pheno_path, '--pheno-name', self.pheno_name,
                    '--{}'.format(self.method), '--covar', self.cov_path,
                    '--out', os.path.join(
                        self.data_path,
                        self.save_res_to,
                        save_name)] + list_args,
                    stdout=out, stderr=out)

        else:
            subprocess.run([
                self.plink_path,
                '--bfile', self.file_path, '--allow-no-sex',
                '--keep', os.path.join(self.data_path, 'tmp', 'indivs.txt'),
                '--pheno', self.pheno_path, '--pheno-name', self.pheno_name,
                '--{}'.format(self.method), '--covar', self.cov_path,
                '--out', os.path.join(self.data_path, 'tmp', 'res')]+list_args,
                stdout=out, stderr=out)
            res_path = os.path.join(self.data_path,
                'tmp', 'res.assoc.{}'.format(self.method))

        res = pd.read_csv(res_path, delim_whitespace=True)
        res = res[res['TEST'] == 'ADD']

        ordered_best_res = res.sort_values('P')
        best_snp_rs = res['SNP'].iloc[:self.kbest]

        best_snps = bim.loc[bim['snp'].isin(best_snp_rs), ['chrom', 'pos', 'i']]
        best_snps = best_snps.sort_values(['chrom', 'pos'])['i'].tolist()

        self.feature_idx = best_snps

        shutil.rmtree(os.path.join(self.data_path, 'tmp'), ignore_errors=True)

        return self

    def transform(self, save_name=None, verbose=False):
        """ Transform the data by selecting the best features

        Parameters
        ----------
        save_name: str, default None
            name of the file in which we want to store the
            transformed data.
        verbose: bool, default False
            if set to true, information will be provided while
            fitting the model.

        Returns
        -------
        data: np array or np array mapping
            the transformed data.
        """
        if len(self.feature_idx) == 0:
            raise AttributeError('You must fit your model before transforming the data.')

        bim, fam, bed = read_plink(self.file_path, verbose=verbose)

        bed = bed[self.feature_idx]

        data = np.transpose(bed.compute())

        if save_name is not None:
            save_path = os.path.join(self.data_path, save_name)
            np.save(save_path, data.astype(float))
            return np.load(save_path, mmap_mode='r')
        return data


    def fit_transform(self, train_indices, save_res_name=None,
        save_data_name=None, verbose=False):
        """ Fit the model using plink, then transform the data

        Parameters
        ----------
        train_indices: list of int
            indices of the samples to use for training the model,
            and for which we want to select the best features.
        save_name: str, default None
            name of the file in which we want to store the
            result of association.
        save_data_name: str, default None
            name of the file in which we want to store the
            transformed data.
        verbose: bool, default False
            if set to true, information will be provided while
            fitting the model.

        Returns
        -------
        data: np array or np array mapping
            the transformed data.
        """

        self.fit(train_indices, save_res_name, verbose)

        return self.transform(save_data_name, verbose)
