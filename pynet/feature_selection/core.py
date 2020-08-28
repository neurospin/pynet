

class FeatureSelector(object):
    """ Base class to perform Feature Selection.
    """
    def __init__(self, kbest):
        """ Class instantiation.

        Parameters
        ----------
        kbest: int, number of best features to select
        """
        self.kbest = kbest

    def fit(self, **kwargs):
        """ Fit the model.

        Must be implemented in the descendant class

        Parameters
        ----------
        kwargs: dict
            contains named arguments
        """

        raise NotImplementedError('The class does not have a fit method yet.')

    def transform(self, **kwargs):
        """ Transform the data by selecting the best feature with the model 
        previously fitted.

        Must be implemented in the descendant class

        Parameters
        ----------
        kwargs: dict
            contains named arguments
        """
        raise NotImplementedError('The class does not have a transform method yet.')

    def fit_transform(self, **kwargs):
        """ Fit the model and then transform the data

        Parameters
        ----------
        kwargs: dict
            contains named arguments
        """
        
        self.fit()

        return self.transform()