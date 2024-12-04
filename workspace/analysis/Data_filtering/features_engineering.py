import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import cupy as cp

class features_drop_corr(TransformerMixin, BaseEstimator):
    """
    Class that to remove features too correlated with 
    each others and with a null standard deviation
    """
    def __init__(self, threshold=0.95):
        """
        threshold(float): must be between 0 and 1
            threshold from which to remove too correlated features
        """
        self.threshold = threshold
        self.columns_to_drop = []

    
    def fit(self, feat_table, y=None, **params):
        """
        feat_table(pd.DataFrame): table of features 
        """
        # Remove features with std = 0:
        self.columns_to_drop = feat_table.columns[feat_table.std() == 0]
        feat_table_test = feat_table.drop(self.columns_to_drop, axis=1)
    
        # Remove features whose correlation above threshold

        ## Compute correlation matrix on GPU
        cov_mat = feat_table_test.corr()

        ## Compute correlation matrix mean
        cov_mat_mean = cov_mat.mean(axis=1).values
        cov_mat_mean_row = cov_mat_mean.reshape(-1,1)
        cov_mat_mean_col = cov_mat_mean.reshape(1,-1)
    
        ## Compute whether the columns features has a higher correlation average
        mean_col = cov_mat_mean_row <= cov_mat_mean_col
        ## Compute whether the columns features has a higher correlation average
        mean_row = cov_mat_mean_row > cov_mat_mean_col
    
        ## Apply the threshold on the correlation matrix
        mask = np.triu(np.ones(cov_mat.shape), k=1).astype(bool)
        cov_mat_thre = (cov_mat.where(mask) > self.threshold)
    
        ## Get the list of features
        features_arr = cov_mat_thre.columns.values
    
        ## Create a table res
        res = {}
        ### v1 store row features from the correlation matrix above threshold
        res["v1"] = (' '.join((cov_mat_thre * features_arr.reshape(-1,1)).to_numpy().flatten()).split())
        ### v2 store col features from the correlation matrix above threshold
        res["v2"] = (' '.join((cov_mat_thre * features_arr.reshape(1,-1)).to_numpy().flatten()).split())
    
        ### drop store either row or col features from the correlation matrix above threshold
        ### based on which (row or col) features has the highest correlation average
        res["drop"] = ((cov_mat_thre * mean_col) * features_arr.reshape(1,-1) + 
                (cov_mat_thre * mean_row) * features_arr.reshape(-1, 1))
        res["drop"] = ' '.join(res["drop"].to_numpy().flatten()).split()
        res = pd.DataFrame(res)
        
        ## all_corr_vars gather all features considered in pair v1 or v2
        all_corr_vars = set(res['v1'].tolist() + res['v2'].tolist()) 
        ## poss_drop potential features to remove 
        poss_drop = set(res['drop'].tolist())
        ## keep features involve in the pair of features but which is never 
        ## the highest correlation average: features that wont be drop
        keep = all_corr_vars.difference(poss_drop)
    
        ## drop the features that are paired with kept features
        p = res[ res['v1'].isin(keep)  | res['v2'].isin(keep) ][['v1', 'v2']]
        q = set(p['v1'].tolist() + p['v2'].tolist())
        drop = q.difference(keep)
        poss_drop = poss_drop.difference(drop)
    
        ## m store the features that can still be droped, and which are in paired with droped features
        m = res[ res['v1'].isin(poss_drop)  | res['v2'].isin(poss_drop) ][['v1', 'v2','drop']]
        ## we consider only the pair of features made of poss drop, poss drop. From those pair, 
        ## the column drop tells which one to drop
        more_drop = set(m[~m['v1'].isin(drop) & ~m['v2'].isin(drop)]['drop'])
        
        #Finally merge with columns to drop
        self.columns_to_drop = list(set(self.columns_to_drop) | drop | more_drop)

    def transform(self, feat_table, y=None, **params):
        return feat_table.drop(self.columns_to_drop, axis=1)

    
    def fit_transform(self, feat_table, y=None, **params):
        self.fit(feat_table, y, **params)
        return self.transform(feat_table, y, **params)




class StandardScaler_group(TransformerMixin, BaseEstimator):
    def __init__(self, metadata):
        self.metadata = metadata
        self.scaler_dict = {k: StandardScaler() 
                            for k in self.metadata.unique()}


    def fit(self, feat_table, y=None, **params):
        grouped = feat_table.groupby(self.metadata).apply(lambda x: x)
        for k in self.metadata.unique():
            self.scaler_dict[k].fit(grouped.loc[k])

    def transform(self, feat_table, y=None, **params):
        index_save = feat_table.index.values
        index_name = feat_table.index.name
        grouped = feat_table.groupby(self.metadata).apply(lambda x: x)
        for k in self.metadata.unique():
            grouped.loc[k] = self.scaler_dict[k].transform(grouped.loc[k])  
            
        _, index = zip(*grouped.index)
        grouped = grouped.set_index(pd.Index(index))
        grouped.index.name = index_name
        return grouped.loc[index_save]

    def fit_transform(self, feat_table, y=None, **params):
        self.fit(feat_table, y=None, **params)
        return self.transform(feat_table, y, **params)


class StandardScaler_pandas(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit(self, feat_table, y=None, **params):
        self.scaler.fit(feat_table)

    def transform(self, feat_table, y=None, **params):
        feat_table.loc[:, :] = self.scaler.transform(feat_table)
        return feat_table

    def fit_transform(self, feat_table, y=None, **params):
        self.fit(feat_table, y=None, **params)
        return self.transform(feat_table, y=None, **params)
"""
-------------- GPU version of the above code ---------------------------
"""
class features_drop_corr_gpu(TransformerMixin, BaseEstimator):
    """
    Class that to remove features too correlated with 
    each others and with a null standard deviation
    """
    def __init__(self, threshold=0.95):
        """
        threshold(float): must be between 0 and 1
            threshold from which to remove too correlated features
        """
        self.threshold = threshold
        self.columns_to_drop = []

    
    def fit(self, feat_table, y=None, **params):
        """
        feat_table(pd.DataFrame): table of features 
        """

        # Remove features whose correlation above threshold
        
        ## Compute correlation matrix on GPU

        cov = cp.abs(cp.corrcoef(cp.array(feat_table.values), rowvar=False))
        features_arr = feat_table.columns.values
        
        
        ## Compute correlation matrix mean
        cov_mean = cp.nanmean(cov, axis=1)
        cov_mean_row = cov_mean.reshape(-1,1)
        cov_mean_col = cov_mean.reshape(1,-1)
        
        ## Compute whether the columns features has a higher correlation average
        mean_col = (cov_mean_row <= cov_mean_col).get()
        ## Compute whether the columns features has a higher correlation average
        mean_row = (cov_mean_row > cov_mean_col).get()
        
        ## Apply the threshold on the correlation matrix
        mask = cp.triu(cp.ones(cov.shape), k=1)
        cov_thre = (cov * mask > self.threshold).get()
        
        ## Get the list of features
        #features_arr = cov_mat_thre.columns.values
        
        ## Create a table res
        res = {}
        ### v1 store row features from the correlation matrix above threshold
        res["v1"] = (' '.join((cov_thre * features_arr.reshape(-1,1)).flatten()).split())
        ### v2 store col features from the correlation matrix above threshold
        res["v2"] = (' '.join((cov_thre * features_arr.reshape(1,-1)).flatten()).split())
        
        ### drop store either row or col features from the correlation matrix above threshold
        ### based on which (row or col) features has the highest correlation average
        res["drop"] = ((cov_thre * mean_col) * features_arr.reshape(1,-1) + 
                (cov_thre * mean_row) * features_arr.reshape(-1, 1))
        res["drop"] = ' '.join(res["drop"].flatten()).split()
        
        res = pd.DataFrame(res)

        
        ## all_corr_vars gather all features considered in pair v1 or v2
        all_corr_vars = set(res['v1'].tolist() + res['v2'].tolist()) 
        ## poss_drop potential features to remove 
        poss_drop = set(res['drop'].tolist())
        ## keep features involve in the pair of features but which is never 
        ## the highest correlation average: features that wont be drop
        keep = all_corr_vars.difference(poss_drop)
        
        ## drop the features that are paired with kept features
        p = res[ res['v1'].isin(keep)  | res['v2'].isin(keep) ][['v1', 'v2']]
        q = set(p['v1'].tolist() + p['v2'].tolist())
        drop = q.difference(keep)
        poss_drop = poss_drop.difference(drop)
        
        ## m store the features that can still be droped, and which are in paired with droped features
        m = res[ res['v1'].isin(poss_drop)  | res['v2'].isin(poss_drop) ][['v1', 'v2','drop']]
        ## we consider only the pair of features made of poss drop, poss drop. From those pair, 
        ## the column drop tells which one to drop
        more_drop = set(m[~m['v1'].isin(drop) & ~m['v2'].isin(drop)]['drop'])
        
        #Finally merge with columns to drop
        self.columns_to_drop = list(set(list(features_arr[(cp.isnan(cov).sum(axis=1) == len(cov)).get()])) 
                               | drop | more_drop)
        

    def transform(self, feat_table, y=None, **params):
        return feat_table.drop(self.columns_to_drop, axis=1)

    
    def fit_transform(self, feat_table, y=None, **params):
        self.fit(feat_table, y, **params)
        return self.transform(feat_table, y, **params)

