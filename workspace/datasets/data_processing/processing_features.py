import warnings
from typing import TypeAlias

import cupy as cp
import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

DataFrame: TypeAlias = pd.DataFrame | pl.DataFrame
ndarray: TypeAlias = cp.ndarray | np.ndarray

class FeaturesDropCorr(TransformerMixin, BaseEstimator):
    """
    Removes highly correlated features and those with zero standard deviation.
    Support polars and pandas input only, and GPU computation.
    """
    def __init__(
        self, 
        threshold: float = 0.95, 
        use_gpu: bool = False
    ) -> None: 
        """
        Initializes the transformer.

        Args:
            threshold (float): Correlation threshold for feature removal (0 to 1).
            use_gpu (bool): If True, uses GPU for computation.
        """
        self.threshold = threshold
        self.columns_to_drop = []
        self.use_gpu = use_gpu

    
    def fit(
        self, 
        features: DataFrame | ndarray, 
        y: None = None, 
        **params
    ) -> None:
        """
        Identifies features to drop based on correlation and standard deviation.

        Args:
            features (DataFrame | ndarray): Input feature dataframe or matrix.
            y: Unused, for compatibility.
        """
        # cpu/gpu agnostic code
        # .to_numpy() and .columns with np.array(., dtype=object ) allow polars/pandas compatible
        if isinstance(features, DataFrame):
            features_arr = features.to_numpy()
            features_col = np.array(features.columns, dtype=object)
        else: 
            features_arr = cp.asnumpy(features)  # make sure starts on numpy
            features_col = np.array([str(i) for i in range(features.shape[1])],
                                    dtype=object)
    
        if self.use_gpu:
            features_arr = cp.asarray(features_arr)
        xp = cp.get_array_module(features_arr)
        print("Using:", xp.__name__)
        
        ## Compute correlation matrix
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", 
                category=RuntimeWarning, 
                message=".*invalid value encountered in divide.*")
            cov = xp.abs(xp.corrcoef(features_arr, rowvar=False))
        std_drop = set(features_col[cp.asnumpy(xp.isnan(cov).sum(axis=1) == len(cov))])

        ## Compute correlation matrix mean
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", 
                category=RuntimeWarning, 
                message=".*Mean of empty slice.*")
            cov_mean = xp.nanmean(cov, axis=1)
        cov_mean_col = cov_mean.reshape(1, -1)
        cov_mean_row = cov_mean.reshape(-1, 1)

        
        ## Compute whether the columns features has a higher correlation average
        mean_col = cp.asnumpy(cov_mean_row <= cov_mean_col)
        mean_row = cp.asnumpy(cov_mean_row > cov_mean_col)
        
        ## Apply the threshold on the correlation matrix
        mask = xp.triu(xp.ones(cov.shape), k=1)
        cov_thre = cp.asnumpy(cov*mask > self.threshold)
        
        ## Create a table res
        ### v1, v2 store row, col features from the correlation matrix above threshold
        ### drop either store v1 or v2 if mean_row or mean_col above thre.         
        res = {
            "v1": ' '.join((cov_thre * features_col.reshape(-1,1)).flatten()).split(), 
            "v2": ' '.join((cov_thre * features_col.reshape(1,-1)).flatten()).split(),
            "drop": (' '.join((
                (cov_thre*mean_col) * features_col.reshape(1,-1) + 
                (cov_thre*mean_row) * features_col.reshape(-1, 1)
            ).flatten()).split()
                       )
        }
        
        ## all_corr_vars, is every considered pair
        ## poss_drop potential, features to remove
        ## keep, features that are never possibly droped
        all_corr_vars = set(res["v1"] + res["v2"])
        poss_drop = set(res['drop'])
        keep = all_corr_vars.difference(poss_drop)
        
        ## drop, features in pair with kept features
        p = list(filter(lambda x: x[0] in keep or x[1] in keep, zip(res["v1"], res["v2"])))
        q = set([item for pair in p for item in pair])
        drop = q.difference(keep)
        poss_drop = poss_drop.difference(drop)
        
        ## more_drop, features possibly droped but not paired with surely drop
        ## select the drop features according res(drop)
        m = filter(
            lambda x: 
            (x[0] in poss_drop and x[1] not in drop) or 
            (x[1] in poss_drop and x[0] not in drop), 
            zip(res["v1"], res["v2"], res["drop"])
        )
        more_drop = set(map(lambda x: x[2], m))
        #Finally merge with columns to drop while removing std=0
        self.columns_to_drop = list(std_drop | drop | more_drop)
        
    def transform(
        self, 
        features: DataFrame | ndarray, 
        y: None = None, 
        **params
    ) -> DataFrame | ndarray:
        """
        Drops identified features from the dataset.

        Args:
            features (DataFrame | ndarray): Input feature dataframe or matrix.
            y: Unused, for compatibility.

        Returns:
            DataFrame | ndarray: Transformed dataset with dropped features.
        """
        if isinstance(features, pd.DataFrame):
            return features.drop(columns=self.columns_to_drop)
        elif isinstance(features, pl.DataFrame):
            return features.drop(self.columns_to_drop)
        else:
            xp = cp.get_array_module(features)
            return xp.delete(features, [int(i) for i in self.columns_to_drop], axis=1)

    
    def fit_transform(
        self, 
        features: DataFrame | ndarray, 
        y: None = None, 
        **params
    ) -> DataFrame | ndarray:
        """
        Fits and transforms the dataframe or matrix.
        """
        self.fit(features, y, **params)
        return self.transform(features, y, **params)

