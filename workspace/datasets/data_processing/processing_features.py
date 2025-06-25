import warnings
from typing import TypeAlias

import cupy as cp
import numpy as np
import pandas as pd
import polars as pl
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby, starmap
from scipy.stats import median_abs_deviation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing._data import _handle_zeros_in_scale
from os import cpu_count


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
    ) -> "FeaturesDropCorr":
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
        return self
        
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


class MadScaler(BaseEstimator, TransformerMixin):
    """
    Scales data using the Median Absolute Deviation (MAD), making it robust to outliers.
    """
    
    def __init__(
        self, 
        **kwargs
    ) -> None:
        pass
    
    def fit(
        self, 
        X: np.ndarray, 
        y: None = None
    ) -> "MadScaler":
        """
        Compute the median and MAD of the training data.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            y (None, optional): Ignored.
        
        Returns:
            MadScaler: Fitted scaler instance.
        """
        self.median = np.median(X, axis=0)
        self.scale = median_abs_deviation(X, axis=0, scale=1.0)
        self.scale = _handle_zeros_in_scale(self.scale, copy=False)
        return self
    
    def transform(
        self, 
        X: np.ndarray
    ) -> np.ndarray:
        """
        Scale the data using the previously computed median and MAD.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
        
        Returns:
            np.ndarray: Scaled data.
        """
        return (X - self.median) / self.scale
    
    def fit_transform(
        self, 
        X: np.ndarray, 
        y: None = None
    ) -> np.ndarray:
        """
        Fit the scaler and transform the data in one step.
        """
        return self.fit(X).transform(X)

def _group_fitter(
    scaler_group_X
) -> dict:
    """ 
    Helper function.
    Apply fitter function to a group.
    """
    
    scaler, group, X = scaler_group_X
    return {group: scaler.fit(X)}

def _group_transformer(
    scaler_index_X
) -> dict:
    """ 
    Helper function.
    Apply fitter function to a group.
    """
    
    scaler, index, X = scaler_index_X
    return [index, scaler.transform(X)]
    

class GroupScaling():
    """
    Scales data using a specified scaler for each group in the dataset.
    """
    def __init__(self, scaler, **scaler_kwargs) -> None:
        """
        Initialize with a scaler class and its arguments.

        Args:
            scaler (class): Scaler class (e.g., StandardScaler).
            **scaler_kwargs: Additional arguments for the scaler.
        """
        self._scaler = scaler
        self._scaler_kwargs = scaler_kwargs

    def fit(
        self, 
        X: np.ndarray, 
        groups: np.ndarray
    ) -> "GroupScaling":
        """
        Fit a scaler for each group in the data.

        Args:
            X (np.ndarray): Data to scale.
            groups (np.ndarray): Group labels for each data point.

        Returns:
            GroupScaling: Fitted instance.
        """
        data = groupby(
            sorted(zip(X, groups), key=lambda tup: tup[1]), 
            key=lambda tup: tup[1]
        )
        unique_keys = np.unique(groups).size
        scaler_groups = zip(
            [self._scaler(**self._scaler_kwargs) for _ in range(unique_keys)], 
            data
        )
        iterator = list(
            starmap(
                lambda scaler, groups: (scaler, groups[0], np.stack(list(zip(*groups[1]))[0])), 
                scaler_groups))
        
        with ProcessPoolExecutor(min(cpu_count(), 50)) as ex:
            _list_scalers = list(ex.map(_group_fitter, iterator))
        self.scalers = {k: v for d in _list_scalers for k,v in d.items()}
        return self
        
    def transform(
        self, 
        X: np.ndarray, 
        groups: np.ndarray
    ) -> np.ndarray:
        """
        Transform data using the fitted scalers for each group.

        Args:
            X (np.ndarray): Data to transform.
            groups (np.ndarray): Group labels for each data point.

        Returns:
            np.ndarray: Scaled data.

        Raises:
            AssertionError: If groups are not a subset of fitted groups.
        """
        if not set(groups).issubset(self.scalers.keys()):
            raise AssertionError(
                "GroupScaling can only occur if the groups of transformed data are a subset of fitted groups."
            )
            
        data = groupby(
            sorted(zip(X, groups, np.arange(len(groups))), 
                   key=lambda tup: tup[1]), 
            key=lambda tup: tup[1]
        )
        iterator = list(
            starmap(
                lambda _k, _groups : (
                    self.scalers[_k], 
                    *list(
                        starmap(lambda _X, _gr, _index: (np.stack(_index), np.stack(_X)), 
                                [list(zip(*_groups))])
                    )[0]), 
                data)
        )        
        with ProcessPoolExecutor(min(cpu_count(), 50)) as ex:
            _list_index_X = list(ex.map(_group_transformer, iterator))
        scaled_X = np.zeros_like(X)
        for _index, _X in _list_index_X:
            scaled_X[_index, :] = _X
        
        # for k, g in groupby(data, key=lambda tup: tup[1]):
        #     _X, _, _index = list(zip(*g))
        #     _X, _index = np.stack(_X), np.stack(_index)
        #     scaled_X[_index, :] = self.scalers[k].transform(_X)
        return scaled_X

    def fit_transform(
        self, 
        X: np.ndarray, 
        groups: np.ndarray
    ) -> np.ndarray: 
        """
        Fits and transforms the dataframe or matrix.
        """
        self.fit(X, groups)
        return self.transform(X, groups)
        


















