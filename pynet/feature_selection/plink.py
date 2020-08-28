import os
import subprocess
from panadas_plink import read_plink
from pynet.feature_selection import FeatureSelector


class PlinkSelector(FeatureSelector):
    def __init__(self, kbest, data_path,
        pheno_file, cov_file,
        method='linear',
        plink_path='/neurospin/brainomics/ig_install/plink2-v1.9/plink'):
        """ Class instantiation.

        Parameters
        ----------
        kbest: int
            number of best feature to select.
        data_path: str
            path to the folder containing the data.
        pheno_file: str
            name of the file containing the phenotypes.
        cov_file: str
            name of the file containing the covariates for the model.
        method: str, default 'linear
            the name of plink's method, one of 'linear' or 'logistic'.
        plink_path: str, default '/neurospin/brainomics/ig_install/plink2-v1.9/plink'
            path to the folder containing the executable plink.
        """
        super().__init__(kbest=kbest)
        self.method = method
        self.data_path = data_path
        self.plink_path = plink_path
        self.pheno_path = os.path.join(data_path, pheno_file)
        self.cov_path = os.path.join(data_path, cov_file)
        self.feature_idx = []
        self.impute_na = impute_na
    
    def fit(file_name, pheno_name,
        train_indices, verbose=False,
        **plink_args):
        """ Fit the model using plink

        Parameters
        ----------
        file_name: str
            prefix of the files containing the data.
        pheno_name: str
            name of the phenotype to associate, as provided in the
            file containing the phenotypes.
        train_indices: list of int
            indices of the samples to use for training the model.
        verbose: bool, default False
            if set to true, information will be provided while
            fitting the model.
        plink_args: dict
            additionnal named parameters to provide to the plink
            command. It can be anything as long as it is valid
            for plink, for instance maf=0.05 if you only want to
            fit the model on common variations.
        """
        file_path = os.path.join(self.data_path, file_name)

        bim, fam, _ = read_plink(file_path, verbose=verbose)

        indiv_to_keep = fam.loc[fam['i'].isin(train_indices), ['fid', 'iid']]

        if not os.path.isdir(os.path.join(self.data_path, 'tmp')):
            os.mkdir(os.path.join(data_path, 'tmp'))

        indiv_to_keep.to_csv(os.path.join(data_path, 'tmp', 'indivs.txt'),
            header=False, index=False, sep=' ')
        
        out = None
        if not verbose:
            out = subprocess.DEVNULL

        list_args = []
        for key, value in plink_args.items():
            list_args += ['--{}'.format(str(key)), str(value)]

        subprocess.run([
            os.path.join(plink_path, 'plink'),
            '--bfile', file_path,
            '--keep', os.path.join(data_path, 'tmp', 'indivs.txt'),
            '--{}'.format(self.method), '--covar', self.cov_path,
            '--out', os.path.join(data_path, 'tmp', 'res')] + list_args,
            stdout=out, stderr=out)

        res = pd.read_csv(
            os.path.join(data_path, 
                'tmp', 'res.assoc.{}'.format(self.method)),
            delim_whitespace=True)
        res = res[res['TEST'] == 'ADD']

        ordered_best_res = res.sort_values('P')
        best_snp_rs = res['SNP'].iloc[:self.kbest]

        best_snps = bim.loc[bim['snp'].isin(best_snp_rs), ['chrom', 'pos', 'i']]
        best_snps = best_snps.sort_values(['chrom', 'pos'])['i'].tolist()
        
        self.feature_idx = best_snps

        return self

    def transform(file_name, indices, save_name=None):
         """ Transform the data by selecting the best features

        Parameters
        ----------
        file_name: str
            prefix of the files containing the data.
        indices: list
            indices of the samples for which we want to select
            features.
        save_name: str, default None
            name of the file in which we want to store the
            transformed data.
        """
        if len(self.feature_idx) == 0:
            raise AttributeError('You must fit you model before transforming the data.')
    
        file_path = os.path.join(self.data_path, file_name)

        bim, fam, bed = read_plink(file_path, verbose=verbose)

        bed = bed[self.feature_idx]

        data = np.transpose(bed.compute())

        if save_name is not None:
            save_path = os.path.join(self.data_path, 'tmp', save_name)
            np.save(save_path, data.astype(float))


    def fit_transform(file_name, pheno_name,
        train_indices, verbose=False, **plink_args):
        """ Fit the model using plink

        Parameters
        ----------
        file_name: str
            prefix of the files containing the data.
        pheno_name: str
            name of the phenotype to associate, as provided in the
            file containing the phenotypes.
        train_indices: list of int
            indices of the samples to use for training the model,
            and for which we want to select the best features.
        verbose: bool, default False
            if set to true, information will be provided while
            fitting the model.
        plink_args: dict
            additionnal named parameters to provide to the plink
            command. It can be anything as long as it is valid
            for plink, for instance maf=0.05 if you only want to
            fit the model on common variations.
        """

        self.fit(file_name, pheno_name,
            train_indices, verbose, **plink_args)

        return self.transform(file_name, train_indices)

